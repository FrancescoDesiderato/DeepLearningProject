import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class NanoSocratesDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.no_padding()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["input"]
        target_text = sample["target"]

        input_encoding = self.tokenizer.encode(input_text)
        target_encoding = self.tokenizer.encode(target_text)

        return {
            "input_ids": torch.tensor(input_encoding.ids, dtype=torch.long),
            "labels": torch.tensor(target_encoding.ids, dtype=torch.long)
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.token_to_id("<PAD>")

    def __call__(self, batch):
        # batch è una lista di dizionari, es: [{'input_ids': tensor, 'labels': tensor}, ...]

        # Separiamo gli input e i target
        input_ids_list = [item['input_ids'] for item in batch]
        labels_list = [item['labels'] for item in batch]

        # Usiamo una funzione di PyTorch per fare il padding dinamico
        # Riempie le sequenze con `pad_token_id` fino alla lunghezza della sequenza più lunga nel batch
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=self.pad_token_id
        )

        # Creiamo la maschera di attenzione manualmente
        # 1 dove ci sono token reali, 0 dove c'è padding
        attention_mask = (input_ids_padded != self.pad_token_id).long()

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'labels': labels_padded
        }


def dataLoaderFromCSV(csv_file,tokenizer_path,MAX_LENGTH,BATCH_SIZE):
    data = pd.read_csv(csv_file)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    transformed_data = data[['input', 'target']].to_dict('records')
    data = NanoSocratesDataset(transformed_data, tokenizer, MAX_LENGTH)

    data_collator = DataCollator(tokenizer)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(data, [0.8, 0.1, 0.1])

    train_dataloader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )
    evaluation_dataloader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )

    test_dataloader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )
    return train_dataloader, evaluation_dataloader, test_dataloader