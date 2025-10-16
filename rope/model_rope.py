import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, max_leght, head_dim: int):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        self.head_dim = head_dim
        self.theta = 10000
        self.max_position_embeddings = max_leght
        # Creo onde sinusoidali di varie velocità usate per la rotazione
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_leght, dtype=torch.float32)
        # Creazione della matrice theta, con posizione-theta
        theta = torch.outer(positions, inv_freq)
        # Precomputazione e caching di seno e coseno così non verrà creato dopo
        self.register_buffer("cos_cached", theta.cos(), persistent=False)
        self.register_buffer("sin_cached", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # Recuperiamo cos e sin
        cos = self.cos_cached[position_ids, :]
        sin = self.sin_cached[position_ids, :]
        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        # Rotazione in-place
        x_rotated = torch.empty_like(x) #Creo un tensore di forma x e successivamente lo riempio
        x_rotated[..., ::2] = x[..., ::2] * cos - x[..., 1::2] * sin
        x_rotated[..., 1::2] = x[..., ::2] * sin + x[..., 1::2] * cos

        return x_rotated #[batch, n_heads, seq_len, head_dim]


class ROPEAttention(nn.Module):
    """
    Modulo per l'implementazione dell'attenzione locale in un layer dell'encoder Transformer.
    Combina attenzione multi-testa locale con una feed-forward network.
    """
    def __init__(self, d_model, n_heads, ffn_hid_dim, window_size, dropout=0.1,max_lenght=256):
        super().__init__()
        self.window_size = window_size # dimensione della finestra di attenzione locale
        self.self_attn = ROPEAttentionBlock(d_model, n_heads, dropout=dropout, max_lenght=max_lenght)
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

        # applicazione attenzione locale senza key_padding_mask per evitare righe totalmente mascherate
        # i 3 src sono query, key, value
        # [0] per prendere solo l'output e ignorare i pesi
        src2 = self.self_attn(src, src, src, key_padding_mask=None)[0]

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



# Creating the Multi-Head Attention block
class ROPEAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float,max_lenght:int) -> None: # h = number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h

        # We ensure that the dimensions of the model is divisible by the number of heads
        assert d_model % h == 0, 'd_model is not divisible by h'

        # d_k is the dimension of each attention head's key, query, and value vectors
        self.d_k = d_model // h # d_k formula, like in the original "Attention Is All You Need" paper

        # Defining the weight matrices
        self.w_q = nn.Linear(d_model, d_model) # W_q
        self.w_k = nn.Linear(d_model, d_model) # W_k
        self.w_v = nn.Linear(d_model, d_model) # W_v
        self.w_o = nn.Linear(d_model, d_model) # W_o

        self.rope_q = RotaryPositionalEmbedding(max_lenght, d_model)
        self.rope_k = RotaryPositionalEmbedding(max_lenght, d_model)

        self.dropout = nn.Dropout(dropout) # Dropout layer to avoid overfitting


    @staticmethod
    def attention(query, key, value, attn_mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # Calcola attention scores: (batch, heads, seq_len_q, seq_len_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Applica attn_mask causale (deve avere shape compatibile con attention_scores)
        if attn_mask is not None:
            # attn_mask dovrebbe essere (seq_len_q, seq_len_k) o (batch, heads, seq_len_q, seq_len_k)
            #attention_scores = attention_scores.masked_fill(attn_mask == 0, -1e9)
            attention_scores.masked_fill_(attn_mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, position_ids,attn_mask=None):

        query = self.w_q(q) # Q' matrix
        key = self.w_k(k) # K' matrix
        value = self.w_v(v) # V' matrix

        #Applying Rope
        query = self.rope_q(query,position_ids)
        key = self.rope_k(key,position_ids)

        # Splitting results into smaller matrices for the different heads
        # Splitting embeddings (third dimension) into h parts
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension

        # Obtaining the output and the attention scores
        x, self.attention_scores = ROPEAttentionBlock.attention(query, key, value, attn_mask, self.dropout)

        # Obtaining the H matrix
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x) # Multiply the H matrix by the weight matrix W_o, resulting in the MH-A matrix


class EncoderBlock(nn.Module):

    # This block takes in the MultiHeadAttentionBlock and FeedForwardBlock, as well as the dropout rate for the residual connections
    def __init__(self,d_model,ffn_hid_dim,n_heads, dropout,max_lenght=256):
        super().__init__()
        # due layer lineari per la feed-forward network
        self.rope_attention = ROPEAttentionBlock(d_model, n_heads, dropout,max_lenght)
        self.linear1 = nn.Linear(d_model, ffn_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_hid_dim, d_model)
        # due norm layer per le residual connection
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x,position_ids,mask=None):
        # Multi-Head Attention sublayer con residual connection
        attn_output = self.rope_attention(x, x, x, position_ids,attn_mask=None)
        # gestione manuale del padding
        if mask is not None:
            pad = mask.transpose(0, 1).unsqueeze(-1)  # [S, B, 1]
            attn_output = attn_output.masked_fill(pad, 0.0)
            x = x.masked_fill(pad, 0.0)

        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-Forward Network sublayer con residual connection
        # feed-forward network + secondo residual connection + norm
        ffn_output = self.linear1(x)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.linear2(ffn_output)
        if mask is not None:
            pad = mask.transpose(0, 1).unsqueeze(-1)
            ffn_output = ffn_output.masked_fill(pad, 0.0)
            x = x.masked_fill(pad, 0.0)

        x = x + self.dropout2(ffn_output) #Residual connections
        if mask is not None:
            pad = mask.transpose(0, 1).unsqueeze(-1)
            x = x.masked_fill(pad, 0.0)
        x = self.norm2(x)

        return x

class Encoder(nn.Module):
    """
    Encoder Transformer con layer di attenzione locale e globale alternati.
    """
    def __init__(self, num_layers, d_model, n_heads, ffn_hid_dim, dropout=0.1,max_lenght=256):
        super().__init__()
        self.layers = nn.ModuleList() # lista di layer
        # costruzione dei layer alternati
        for i in range(num_layers):
            layer = EncoderBlock(d_model,ffn_hid_dim, n_heads, dropout,max_lenght)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(d_model) # normalizzazione finale

    def forward(self, src, src_key_padding_mask=None):
        output = src
        batch_size, seq_len, _ = src.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=src.device).unsqueeze(0).repeat(batch_size, 1)

        for layer in self.layers:
            output = layer(output, position_ids,mask=src_key_padding_mask)
        return self.norm(output)

class Decoder(nn.Module):
    def __init__(self,num_layers, d_model, n_heads, ffn_hid_dim, dropout=0.1,max_lenght=256) -> None:
        super().__init__()
        self.layers = nn.ModuleList() # lista di layer
        for i in range(num_layers):
            layer = DecoderBlockRope(d_model,ffn_hid_dim, n_heads, dropout,max_lenght)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(d_model) # normalizzazione finale

    def forward(self, x, encoder_output, src_mask, tgt_mask,tgt_key_padding_mask):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        for layer in self.layers:
            x = layer(x, encoder_output, position_ids, src_mask, tgt_mask,tgt_key_padding_mask)
        return self.norm(x)

class DecoderBlockRope(nn.Module):
    def __init__(self, d_model,ffn_hid_dim,n_heads, dropout,max_lenght=256) -> None:
        super().__init__()
        self.self_attention_block = ROPEAttentionBlock(d_model, n_heads, dropout,max_lenght)
        self.cross_attention_block = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, ffn_hid_dim)
        self.linear2 = nn.Linear(ffn_hid_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, encoder_output, position_ids, src_mask, tgt_mask_in,tgt_key_padding_mask):
        # Self-Attention con MLA
        h = self.self_attention_block(self.norm1(x),self.norm1(x),self.norm1(x), position_ids, tgt_mask_in)
        x = x + self.dropout(h)
        # Cross-Attention con MHA originale
        h = self.cross_attention_block(self.norm2(x), encoder_output, encoder_output, src_mask)
        x = x + self.dropout(h)
        # Feed-Forward
        h = self.linear1(self.norm3(x))
        h = self.linear2(self.activation(h))
        x = x + self.dropout(h)

        return x

class NanoSocratesTransformerROPE(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 n_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 ffn_hid_dim,
                 max_lenght,
                 dropout=0.1,
                 padding_idx=0,
                        ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # encoder con layer interleaved
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            ffn_hid_dim=ffn_hid_dim,
            dropout=dropout,
            max_lenght=max_lenght
        )

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            ffn_hid_dim=ffn_hid_dim,
            dropout=dropout,
            max_lenght=max_lenght
        )

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

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(src.device)

        encoder = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        out = self.decoder(
            x=tgt_emb,
            encoder_output=encoder,
            src_mask=src_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.output(out)