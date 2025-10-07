import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # Argomenti del modello standard
    dim: int = 1024  # D_MODEL
    vocab_size: int = 30000  # Esempio
    n_layers: int = 6  # NUM_ENCODER/DECODER_LAYERS
    n_heads: int = 4  # N_HEADS
    n_kv_heads: int = 4  # Per semplicità, usiamo MHA (n_heads == n_kv_heads)
    padding_idx: int = 0 # Indice del Token di padding
    norm_eps: float = 1e-6  # Modificato da 1e-8 per coerenza con l'originale
    intermediate_size: int = 256  # FFN_HID_DIM
    rope_theta: float = 10000 # Parametro di frequenza di ROPE
    max_position_embeddings: int = 256 # Max Lenght in Input, per semplicità uguale alla dim degli embedding in ingresso

    # Argomenti specifici per MLA
    q_compressed_dim: int = 512 # Dimensione compressa di q, in questo caso sarà la metà
    q_nope_head_dim: int = 192 # Parte dell'embedding totale che fa parte di NOPE
    q_rope_head_dim: int = 64 # Parte dell'embedding totale che fa parte di ROPE
    kv_compressed_dim: int = 512 # dimensione compressa di k e v, in questo caso sarà la metà
    k_nope_head_dim: int = 192 # Parte dell'embedding totale che fa parte di NOPE
    k_rope_head_dim: int = 64 # Parte dell'embedding totale che fa parte di ROPE
    v_head_dim: int = 128  # Dimensione della testa di V, solitamente dim / n_heads, ed anche in questo caso

    #Controlli vari ed eventiali sulle dimensioni
    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        assert self.q_nope_head_dim + self.q_rope_head_dim == self.k_nope_head_dim + self.k_rope_head_dim, \
            "La somma delle dimensioni delle head RoPE e NoPE deve essere uguale per Q e K"
        assert self.v_head_dim * self.n_heads == self.dim, \
            "v_head_dim * n_heads deve corrispondere a dim"

    @property
    def gqa_factor(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

# Check sulla dim del tensore, in particolare che siano indicate le teste
def repeat_kv_heads(x: Tensor, n_rep: int) -> Tensor:
    if x.dim() != 4: # 4 non indica il numero di teste
        raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
    if n_rep == 1:
        return x
    return torch.repeat_interleave(x, n_rep, dim=1)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, args: ModelArgs, head_dim: int):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        self.head_dim = head_dim
        self.max_position_embeddings = args.max_position_embeddings
        # Creo onde sinusoidali di varie velocità usate per la rotazione
        inv_freq = 1.0 / (args.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(args.max_position_embeddings, dtype=torch.float32)
        # Creazione della matrice theta, con posizione-theta
        theta = torch.outer(positions, inv_freq)
        # Precomputazione e caching di seno e coseno così non verrà creato dopo
        self.register_buffer("cos_cached", theta.cos(), persistent=False)
        self.register_buffer("sin_cached", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # Recuperiamo cos e sin
        cos = self.cos_cached[position_ids, :]
        sin = self.sin_cached[position_ids, :]
        # Lo adattiamo a Tensore per pytorch
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)
        # Separiamo l'input in posizione pari e dispari
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        # Applico la rotazione nel piano 2D
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        # Torch stack li metto nello stesso vettore
        x_rotated = torch.stack((rot_even, rot_odd), dim=-1) # [batch, n_heads, seq_len, head_dim/2, 2]
        # Fondiamo gli ultimi 2 assi affinchè abbia la stessa dim di x in entrata
        x_rotated = x_rotated.flatten(-2) # [batch, n_heads, seq_len, head_dim]
        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class MultiLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        # Comprimiamo i vettori x in uno spazio latente ridotto e successivamente viene normalizzato
        self.w_dq = nn.Linear(args.dim, args.q_compressed_dim)
        self.q_norm = RMSNorm(args.q_compressed_dim, args.norm_eps)
        # Pesi per la decompressione di nope e rope
        self.w_uq = nn.Linear(args.q_compressed_dim, args.n_heads * args.q_nope_head_dim)
        self.w_qr = nn.Linear(args.q_compressed_dim, args.n_heads * args.q_rope_head_dim)
        # Applicazione del rope su q e k, solo per le parti interessate
        self.rope_q = RotaryPositionalEmbedding(args, args.q_rope_head_dim)
        self.rope_k = RotaryPositionalEmbedding(args, args.k_rope_head_dim)
        # Pesi per la compressione kv
        self.w_dkv = nn.Linear(args.dim, args.kv_compressed_dim)
        self.kv_norm = RMSNorm(args.kv_compressed_dim, args.norm_eps)
        # Pesi per la decompressione di k
        self.w_uk = nn.Linear(args.kv_compressed_dim, args.n_kv_heads * args.k_nope_head_dim)
        self.w_kr = nn.Linear(args.dim, 1 * args.k_rope_head_dim)
        # Produce la decompressione di V, inoltre ricordiamo che non viene applicato il rope su questa parte
        self.w_uv = nn.Linear(args.kv_compressed_dim, args.n_kv_heads * args.v_head_dim)
        # Pesi per il feed forward net
        self.w_o = nn.Linear(args.n_heads * args.v_head_dim, args.dim)

    def forward(self, x: Tensor, position_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, q_seq_len, _ = x.shape # Prendo info sull'input x, l'ultima dim contiene i dati veri e propri
        # Comprimo q e kv + normalizzazione
        compressed_q = self.q_norm(self.w_dq(x))
        compressed_kv = self.kv_norm(self.w_dkv(x))
        # Applicazione del Rope, solo sulla porzione indicata dal costruttore della classe
        k_rope = self.w_kr(x)
        k_rope = k_rope.view(batch_size, q_seq_len, 1, self.args.k_rope_head_dim).transpose(1, 2)
        k_rope = self.rope_k(k_rope, position_ids)
        k_seq_len = compressed_kv.shape[-2]
        # Decomprimo la parte di rope e nope
        q_nope = self.w_uq(compressed_q)
        q_rope = self.w_qr(compressed_q)
        q_nope = q_nope.view(batch_size, q_seq_len, self.args.n_heads, self.args.q_nope_head_dim).transpose(1, 2)
        q_rope = q_rope.view(batch_size, q_seq_len, self.args.n_heads, self.args.q_rope_head_dim).transpose(1, 2)
        # Applico il rope su q
        q_rope = self.rope_q(q_rope, position_ids)
        # Riunisco sotto un unito vettore
        query_states = torch.cat((q_nope, q_rope), dim=-1)
        k_nope = self.w_uk(compressed_kv)
        k_nope = k_nope.view(batch_size, k_seq_len, self.args.n_kv_heads, self.args.k_nope_head_dim).transpose(1, 2)
        # Replico per le varie teste
        k_rope = repeat_kv_heads(k_rope, self.args.n_kv_heads)
        k_states = torch.cat((k_nope, k_rope), dim=-1)
        # Replico per le varie teste
        k_states = repeat_kv_heads(k_states, self.args.gqa_factor)
        # Decomprimo v
        v_states = self.w_uv(compressed_kv)
        v_states = v_states.view(batch_size, k_seq_len, self.args.n_kv_heads, self.args.v_head_dim).transpose(1, 2)
        v_states = repeat_kv_heads(v_states, self.args.gqa_factor)
        # Applico il meccanismo di attenzione
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states, key=k_states, value=v_states, attn_mask=attention_mask
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, q_seq_len,
                                                          self.args.n_heads * self.args.v_head_dim)
        # Ritorno l'output
        return self.w_o(attn_output)
# Non viene applicato il positional Embedding perchè utilizziamo il ROPE
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(args.dim, args.intermediate_size)
        self.act_fn = nn.ReLU()  # o nn.SiLU() come in transformer.py
        self.linear_2 = nn.Linear(args.intermediate_size, args.dim)

    def forward(self, x):
        return self.linear_2(self.act_fn(self.linear_1(x)))


# Blocco di attenzione originale, mantenuto per la Cross-Attention nel Decoder
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, _ = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

# Blocco di Encoder che implementa la MLA
class EncoderBlockMLA(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, dropout: float):
        super().__init__()
        self.norm_attention = RMSNorm(args.dim, args.norm_eps)
        self.attention = MultiLatentAttention(args, layer_idx)
        self.norm_mlp = RMSNorm(args.dim, args.norm_eps)
        self.mlp = FeedForwardBlock(args)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, position_ids: Tensor, mask: Tensor) -> Tensor:
        # Prima connessione residuale (Attenzione)
        h = self.attention(self.norm_attention(x), position_ids, mask)
        x = x + self.dropout(h)
        # Seconda connessione residuale (MLP)
        h = self.mlp(self.norm_mlp(x))
        x = x + self.dropout(h)
        return x


class Encoder(nn.Module):
    def __init__(self, args: ModelArgs, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        for layer in self.layers:
            x = layer(x, position_ids, mask)
        return self.norm(x)

# Blocco di Decoder che implementa la MLA
class DecoderBlockMLA(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, cross_attention_block: MultiHeadAttentionBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = MultiLatentAttention(args, layer_idx)
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = FeedForwardBlock(args)
        self.norm1 = RMSNorm(args.dim, args.norm_eps)
        self.norm2 = RMSNorm(args.dim, args.norm_eps)
        self.norm3 = RMSNorm(args.dim, args.norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, position_ids, src_mask, tgt_mask):
        # Self-Attention con MLA
        h = self.self_attention_block(self.norm1(x), position_ids, tgt_mask)
        x = x + self.dropout(h)
        # Cross-Attention con MHA originale
        h = self.cross_attention_block(self.norm2(x), encoder_output, encoder_output, src_mask)
        x = x + self.dropout(h)
        # Feed-Forward
        h = self.feed_forward_block(self.norm3(x))
        x = x + self.dropout(h)
        return x


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        for layer in self.layers:
            x = layer(x, encoder_output, position_ids, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Esegue il forward pass completo del modello Encoder-Decoder.
        """
        # 1. Codifica la sequenza di input
        # La maschera per l'encoder (src_mask) di solito serve a ignorare il padding
        encoder_output = self.encode(src, src_mask)

        # 2. Decodifica usando l'output dell'encoder e la sequenza target
        # La maschera per il target (tgt_mask) è solitamente una maschera causale per prevenire
        # che il modello "veda" i token futuri durante il training.
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)

        # 3. Proietta l'output del decoder nello spazio del vocabolario
        return self.project(decoder_output)


def build_transformer_with_mla(src_vocab_size: int, tgt_vocab_size: int, dropout: float = 0.1) -> Transformer:
    args = ModelArgs(
        vocab_size=max(src_vocab_size, tgt_vocab_size),
        # Gli altri parametri usano i default di ModelArgs
    )

    # Create the embedding layers
    src_embed = InputEmbeddings(args.dim, src_vocab_size)
    tgt_embed = InputEmbeddings(args.dim, tgt_vocab_size)

    # Create the encoder blocks
    encoder_blocks = []
    for i in range(args.n_layers):
        encoder_block = EncoderBlockMLA(args, layer_idx=i, dropout=dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for i in range(args.n_layers):
        # Cross-attention usa ancora il vecchio MHA
        decoder_cross_attention_block = MultiHeadAttentionBlock(args.dim, args.n_heads, dropout)
        decoder_block = DecoderBlockMLA(args, layer_idx=i, cross_attention_block=decoder_cross_attention_block,
                                        dropout=dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(args, nn.ModuleList(encoder_blocks))
    decoder = Decoder(args, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(args.dim, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer