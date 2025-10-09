import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1) # tensore colonna [max_len, 1] che rappresenta le posizioni della sequenza
        # termine divisione della codifica posizionale
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Crea la matrice di positional encoding
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # 'register_buffer' salva il tensore nello state_dict del modello,
        # ma non lo considera un parametro da addestrare.
        self.register_buffer('pe', pe)
        self.pe = pe

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, d_model]
        """
        # Aggiunge il positional encoding all'input
        x = x + self.pe[:x.size(0)]
        return x