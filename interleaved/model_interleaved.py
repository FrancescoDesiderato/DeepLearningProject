import torch
import torch.nn as nn
import math
from utils.positional_encoding import PositionalEncoding

def _generate_local_attention_mask(seq_len: int, window_size: int, device=None, dtype=torch.float32):
    """
    Genera una maschera di attenzione locale per l'attenzione multi-testa.
    La maschera consente l'attenzione solo alle posizioni entro una finestra
    specificata intorno alla posizione corrente.
    """
    # mask additiva: 0 = consentito, -1e9 = mascherato
    mask = torch.full((seq_len, seq_len), fill_value=-1e9, dtype=dtype, device=device)
    # per ogni posizione i nella sequenza
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        # consenti l'attenzione alle posizioni nella finestra
        mask[i, start:end] = 0.0
    return mask

class LocalAttention(nn.Module):
    """
    Modulo per l'implementazione dell'attenzione locale in un layer dell'encoder Transformer.
    Combina attenzione multi-testa locale con una feed-forward network.
    """
    def __init__(self, d_model, n_heads, ffn_hid_dim, window_size, dropout=0.1):
        super().__init__()
        self.window_size = window_size # dimensione della finestra di attenzione locale
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        # due layer lineari per la feed-forward network
        self.linear1 = nn.Linear(d_model, ffn_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_hid_dim, d_model)
        # due norm layer per le residual connection
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        # src: [S, B, E]; src_key_padding_mask: [B, S] (True = PAD)
        S, B, _ = src.shape
        # maschera di attenzione locale
        local_mask = _generate_local_attention_mask(S, self.window_size, device=src.device, dtype=src.dtype)

        # applicazione attenzione locale senza key_padding_mask per evitare righe totalmente mascherate
        # i 3 src sono query, key, value
        # [0] per prendere solo l'output e ignorare i pesi
        src2 = self.self_attn(src, src, src, attn_mask=local_mask, key_padding_mask=None)[0]

        # gestione manuale del padding
        if src_key_padding_mask is not None:
            pad = src_key_padding_mask.transpose(0, 1).unsqueeze(-1)  # [S, B, 1]
            src2 = src2.masked_fill(pad, 0.0)
            src = src.masked_fill(pad, 0.0)

        src = self.norm1(src + self.dropout1(src2)) # residual connection + norm

        # feed-forward network + secondo residual connection + norm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        if src_key_padding_mask is not None:
            pad = src_key_padding_mask.transpose(0, 1).unsqueeze(-1)
            src2 = src2.masked_fill(pad, 0.0)
            src = src.masked_fill(pad, 0.0)

        src = self.norm2(src + self.dropout2(src2))

        if src_key_padding_mask is not None:
            pad = src_key_padding_mask.transpose(0, 1).unsqueeze(-1)
            src = src.masked_fill(pad, 0.0)
        return src


class InterleavedEncoder(nn.Module):
    """
    Encoder Transformer con layer di attenzione locale e globale alternati.
    """
    def __init__(self, num_layers, d_model, n_heads, ffn_hid_dim, window_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList() # lista di layer
        # costruzione dei layer alternati
        for i in range(num_layers):
            if i % 2 == 0:
                # layer con attenzione locale custom
                layer = LocalAttention(
                    d_model, n_heads, ffn_hid_dim, window_size, dropout
                )
            else:
                # layer con attenzione globale standard
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=ffn_hid_dim,
                    dropout=dropout,
                    batch_first=False
                )
            self.layers.append(layer)
        self.norm = nn.LayerNorm(d_model) # normalizzazione finale

    def forward(self, src, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)

class NanoSocratesTransformerInterleaved(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 n_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 ffn_hid_dim,
                 local_attention_window_size,
                 dropout=0.1,
                 padding_idx=0):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        # encoder con layer interleaved
        self.encoder = InterleavedEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            ffn_hid_dim=ffn_hid_dim,
            window_size=local_attention_window_size,
            dropout=dropout
        )

        # decoder standard di PyTorch
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_hid_dim,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

    def forward(self, src, tgt):
        # src/tgt: [S, B]
        # padding
        src_padding_mask = (src == self.embedding.padding_idx).transpose(0, 1)  # [B, S]
        tgt_padding_mask = (tgt == self.embedding.padding_idx).transpose(0, 1)  # [B, T]

        # Evita righe completamente mascherate
        if src_padding_mask.all(dim=1).any():
            idx = torch.where(src_padding_mask.all(dim=1))[0]
            src_padding_mask[idx, 0] = False
        if tgt_padding_mask.all(dim=1).any():
            idx = torch.where(tgt_padding_mask.all(dim=1))[0]
            tgt_padding_mask[idx, 0] = False

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(src.device)

        encoder = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        out = self.decoder(
            tgt=tgt_emb,
            memory=encoder,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.output(out)

    def encoder_only_forward(self, input_ids, attention_mask=None):
        # del tutto analogo al modello originale
        src = input_ids.transpose(0, 1)
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = (input_ids == self.embedding.padding_idx)
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        encoding = self.encoder(src=src_emb, src_key_padding_mask=src_key_padding_mask)
        logits = self.output(encoding).transpose(0, 1)
        return logits