import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

class NanoSocratesDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.no_truncation() # disabilita il troncamento automatico per gestirlo dopo manualmente
        self.tokenizer.no_padding()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        task = sample["task"]
        input_text = sample["input"]
        target_text = sample["target"]

        input_encoding = self.tokenizer.encode(input_text)
        target_encoding = self.tokenizer.encode(target_text)

        # Assicura che EOS sia sempre presente
        eos_id = self.tokenizer.token_to_id("<EOS>")
        if len(target_encoding.ids) >= self.max_length and eos_id is not None:
            # Tronca a max_length-1 e aggiungi EOS
            target_ids = target_encoding.ids[:self.max_length - 1] + [eos_id]
        else:
            target_ids = target_encoding.ids

        return {
            "task": task,
            "input_ids": torch.tensor(input_encoding.ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long)
        }


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.token_to_id("<PAD>")

    def __call__(self, batch):
        # batch è una lista di dizionari, es: [{'input_ids': tensor, 'labels': tensor}, ...]

        # Separiamo gli input e i target
        task_ids_list = [item['task'] for item in batch]
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

        # maschera di attenzione per il padding
        attention_mask = (input_ids_padded != self.pad_token_id).long()

        return {
            'task': task_ids_list,
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'labels': labels_padded
        }

def dataLoaderFromCSV(csv_file, tokenizer_path, MAX_LENGTH, BATCH_SIZE, SEED: int = 42):
    # da utilizzare quando tutti gli elementi necessari alla creazione del dataset sono già disponibili
    data = pd.read_csv(csv_file)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    train_dataloader, evaluation_dataloader, test_dataloader = dataset_split(data, tokenizer, MAX_LENGTH, BATCH_SIZE, SEED)

    return tokenizer, train_dataloader, evaluation_dataloader, test_dataloader

def dataset_split(data, tokenizer, MAX_LENGTH, BATCH_SIZE, SEED: int = 42):
    X = data[['input', 'target']].to_dict('records')
    y = data['task'].tolist()

    # train (80%) vs temp (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # validation (10%) vs test (10%) dal temp (20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    # Ricostruisci i dataset con i task
    def create_samples(X_data, y_data):
        return [{'task': task, 'input': sample['input'], 'target': sample['target']}
                for sample, task in zip(X_data, y_data)]

    train_samples = create_samples(X_train, y_train)
    val_samples = create_samples(X_val, y_val)
    test_samples = create_samples(X_test, y_test)

    train_ds = NanoSocratesDataset(train_samples, tokenizer, MAX_LENGTH)
    val_ds = NanoSocratesDataset(val_samples, tokenizer, MAX_LENGTH)
    test_ds = NanoSocratesDataset(test_samples, tokenizer, MAX_LENGTH)

    data_collator = DataCollator(tokenizer)

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
        shuffle=False,
        collate_fn=data_collator
    )

    print("Train task distribution:", Counter(y_train))
    print("Validation task distribution:", Counter(y_val))
    print("Test task distribution:", Counter(y_test))

    return train_dataloader, evaluation_dataloader, test_dataloader

