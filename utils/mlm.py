import math
from functools import reduce
import torch
from torch.utils.data import Dataset
from torch import nn
from tqdm.auto import tqdm
import torch.nn.functional as F

def prob_mask_like(t, prob):
    """
    Genera una maschera booleana con la stessa forma di t,
    dove ogni elemento è True con probabilità prob.
    """
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    """
    Crea una maschera booleana per i token specificati in token_ids.
    """
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    # Per ogni token ID (el), controlla dove t == el
    # e combina il risultato con l'accumulatore (acc) usando l'operatore OR.
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    """
    Seleziona casualmente un sottoinsieme di posizioni mascherabili
    rispettando la probabilità prob.
    """
    batch, seq_len, device = *mask.shape, mask.device
    # Calcola numero massimo di token da mascherare
    max_masked = math.ceil(prob * seq_len)

    # Conta quanti token mascherabili ci sono per ogni sequenza del batch
    num_tokens = mask.sum(dim=-1, keepdim=True)
    # Identifica token in eccesso (posizioni che superano il numero di token da mascherare per ciascuna sequenza del batch)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    # Genera numeri casuali per le posizioni mascherabili, escludendo le posizioni non mascherabili
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    # Seleziona i top-k=max_masked indici con valori random più alti
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    # Incrementa gli indici di 1 e maschera con 0 le posizioni in eccesso
    # Le posizioni in eccesso vengono azzerate, garantendo che ogni sequenza mascheri
    # esattamente la percentuale corretta di token, rispettando il vincolo prob.
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    # Posiziona '1' negli indici campionati usando scatter
    new_mask.scatter_(-1, sampled_indices, 1)
    # Rimuove la prima colonna (buffer) e converte a booleano
    # Risultato: [B, seq_len] - True per token da mascherare, False altrimenti
    return new_mask[:, 1:].bool()

class MLM(nn.Module):
    def __init__(
        self,
        transformer,
        mask_prob=0.15, # mascheramento del 15%
        replace_prob=0.9, # 90% delle volte con <MASKMLM>, 10% inalterato o token random
        num_tokens=None,
        random_token_prob=0.1,
        mask_token_id=2,
        pad_token_id=0,
        mask_ignore_token_ids=None # token speciali da non mascherare
    ):
        super().__init__()
        self.transformer = transformer
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        # set con token da ignorare durante mascheramento (padding + token speciali)
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, seq, attention_mask=None, **kwargs):
        # input può essere dict (con attention_mask) o tensor
        if isinstance(seq, dict):
            attention_mask = seq.get("attention_mask", attention_mask)
            seq = seq["input_ids"]
        if seq.dtype != torch.long:
            seq = seq.long()

        # maschera per token non mascherabili
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)        # [B, T]
        # subset casuale di token da mascherare
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)         # [B, T]

        masked_seq = seq.clone() # copia della sequenza originale
        # Crea labels: mantiene token originali solo per posizioni mascherate, padding altrove
        labels = seq.masked_fill(~mask, self.pad_token_id)                 # [B, T]

        # Eventuale sostituzione con token random
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, "num_tokens richiesto per la sostituzione random."
            random_token_prob = prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device=seq.device)
            # Assicura che token random non siano token speciali
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)
            # Rimuove dalle posizioni da mascherare quelle già sostituite con random
            mask = mask & ~random_token_prob

        # Sostituzione con <MASKMLM>
        replace_prob = prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask & replace_prob, self.mask_token_id)

        # Passaggio encoder‑only per MLM
        logits = self.transformer.encoder_only_forward(masked_seq, attention_mask=attention_mask)  # [B, T, V]

        # Calcola loss: cross-entropy tra logits e labels, ignorando padding
        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),  # [B, V, T]
            labels,                  # [B, T]
            ignore_index=self.pad_token_id
        )
        return mlm_loss

def train_mlm(data, model, epochs=1, device=None, lr=3e-4):
    """
    Funzione di training per il modello MLM.
    """

    optimizer = torch.optim.AdamW(model.transformer.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(data, 1),
                    total=len(data),
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    ncols=100)
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)

            loss = model(input_ids, attention_mask=attention_mask)
            loss.backward()
            running_loss += loss.item()


            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{running_loss / step:.4f}"
            })

        torch.save(model.state_dict(), "mlm_model.pt")

    return model

class CorpusMLMDataset(Dataset):
    """
    Classe per il dataset di MLM basato su un corpus di testo.
    """
    def __init__(self, corpus_path: str, tokenizer, max_length: int):
        # Carica il corpus da file, rimuovendo righe vuote
        # Ogni riga è considerata un esempio separato
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        # tokenizza il testo di una riga
        ids = self.tok.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}

class MLMPadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        input_ids_list = [item["input_ids"] for item in batch]
        # Padding delle sequenze alla lunghezza massima nel batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_id
        )
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

