import torch
import torch.nn as nn
import math
from utils.positional_encoding import PositionalEncoding

def _generate_local_attention_mask(seq_len, window_size):
    """
    Genera una maschera per l'attenzione locale a finestra.
    I token possono "vedere" solo i token entro `window_size` a sinistra e a destra.
    """
    mask = torch.full((seq_len, seq_len), float('-inf'))
    # Crea una banda diagonale di zeri
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 0
    return mask


class LocalAttentionTransformerEncoderLayer(nn.Module):
    """
    Un TransformerEncoderLayer che forza l'uso di una maschera di attenzione locale.
    """

    def __init__(self, d_model, n_heads, ffn_hid_dim, window_size, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, ffn_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_hid_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        seq_len = src.shape[0]
        # Genera la maschera di attenzione locale ad ogni forward
        local_mask = _generate_local_attention_mask(seq_len, self.window_size).to(src.device)

        # Self-attention locale
        src2 = self.self_attn(src, src, src,
                              attn_mask=local_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class InterleavedEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, ffn_hid_dim, window_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.globalAttentionTransformerEncoderLayer = nn.TransformerEncoderLayer
        for i in range(num_layers):
            if i % 2 == 0:
                # Layer pari: Attenzione Locale
                layer = LocalAttentionTransformerEncoderLayer(
                    d_model, n_heads, ffn_hid_dim, window_size, dropout
                )
            else:
                # Layer dispari: Attenzione Globale
                layer = self.globalAttentionTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=ffn_hid_dim,
                    dropout=dropout,
                    batch_first=False
                )
            self.layers.append(layer)
        self.norm = nn.LayerNorm(d_model)

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
                 local_attention_window_size,  # <-- Nuovo parametro!
                 dropout=0.1,
                 padding_idx=0):  # Aggiunto per gestire il padding
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        # --- MODIFICA CHIAVE ---
        # Sostituiamo nn.Transformer con i nostri moduli personalizzati

        # 1. Il nostro nuovo Encoder Interleaved
        self.encoder = InterleavedEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            ffn_hid_dim=ffn_hid_dim,
            window_size=local_attention_window_size,
            dropout=dropout
        )

        # 2. Un Decoder standard (potremmo personalizzare anche questo, ma iniziamo così)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_hid_dim,
            dropout=dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ------------------------

        self.output = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src, tgt):
        """
        src: Tensor, shape [src_seq_len, batch_size]
        tgt: Tensor, shape [tgt_seq_len, batch_size]
        """
        # Creazione maschere
        tgt_seq_len = tgt.shape[0]
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        src_padding_mask = (src == self.embedding.padding_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == self.embedding.padding_idx).transpose(0, 1)

        # Embedding + Positional Encoding
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        # --- NUOVO FLUSSO FORWARD ---
        # 1. Passa l'input attraverso l'encoder
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # 2. Il decoder usa l'output dell'encoder ('memory') per la cross-attention
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask  # Maschera per la memoria dell'encoder
        )
        # --------------------------

        return self.output(output)

    # La funzione encoder_only_forward ora è molto più semplice
    def encoder_only_forward(self, input_ids, attention_mask=None):
        """
        input_ids: LongTensor [B, T]
        """
        src = input_ids.transpose(0, 1)  # [T, B]

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = (input_ids == self.embedding.padding_idx)

        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))  # [T, B, D]

        encoding = self.encoder(src=src_emb, src_key_padding_mask=src_key_padding_mask)  # [T, B, D]

        logits = self.output(encoding).transpose(0, 1)  # [B, T, V]
        return logits