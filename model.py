import torch
import torch.nn as nn
import math
from utils.positional_encoding import PositionalEncoding

class NanoSocratesTransformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 n_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 ffn_hid_dim,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. Embedding Layer (condiviso tra encoder e decoder)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Il cuore del modello: il Transformer di PyTorch
        # Questo modulo si occupa di creare gli stack di EncoderLayer e DecoderLayer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_hid_dim,
            dropout=dropout,
            batch_first=False  # PyTorch di default usa [seq_len, batch_size, dim]
        )

        # 4. Output Layer
        # Un layer lineare che proietta l'output del decoder sulla dimensione del vocabolario
        self.output = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        # Genera una maschera per il decoder per prevenire che "veda" il futuro
        # Es: per una sequenza di lunghezza 3, la maschera è:
        # [[0, -inf, -inf],
        #  [0,   0,  -inf],
        #  [0,   0,    0 ]]
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src, tgt):
        """
        src: Tensor, shape [src_seq_len, batch_size] - La sequenza di input
        tgt: Tensor, shape [tgt_seq_len, batch_size] - La sequenza target
        """
        # Creazione delle maschere
        # Maschera per il target (decoder) per la causalità
        tgt_seq_len = tgt.shape[0]
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        # Maschere per il padding (per ignorare i token <PAD>)
        src_padding_mask = (src == self.embedding.padding_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == self.embedding.padding_idx).transpose(0, 1)

        # 1. Applica l'embedding e il positional encoding
        # Moltiplichiamo per sqrt(d_model) come da paper originale
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        # 2. Passa tutto al modulo Transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # 3. Applica il layer di output finale
        return self.output(output)

    def encoder_only_forward(self, input_ids, attention_mask=None):
        """
        Esegue solo l'encoder per l'MLM.
        input_ids: LongTensor [B, T]
        attention_mask: LongTensor [B, T] (1=token valido, 0=pad) opzionale
        Ritorna logits [B, T, vocab_size]
        """
        # [B, T] -> [T, B]
        src = input_ids.transpose(0, 1)

        # padding mask
        pad_id = self.embedding.padding_idx if self.embedding.padding_idx is not None else -1
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            if pad_id >= 0:
                src_key_padding_mask = (input_ids == pad_id)
            else:
                src_key_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # Embedding + PositionalEncoding
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))  # [T, B, D]

        # Encoder‑only
        encoding = self.transformer.encoder(src=src_emb, src_key_padding_mask=src_key_padding_mask)  # [T, B, D]

        # Proiezione a vocab e ritorno a [B, T, V]
        logits = self.output(encoding).transpose(0, 1)  # [B, T, V]
        return logits
