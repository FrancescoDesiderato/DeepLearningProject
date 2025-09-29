import json
import random
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from dataset_utils.dataset import NanoSocratesDataset, DataCollator

def serialize_triple(triple_dict):
    s = triple_dict.get('subject', '')
    p = triple_dict.get('predicate', '')
    o = triple_dict.get('object', '')
    return f"<SOT> <SUBJ> {s} <PRED> {p} <OBJ> {o} <EOT>"

class DatasetFormatting:
    def __init__(self,dataset_filename,tokenizer_path,csv_filename, MAX_LENGTH, BATCH_SIZE):
        self.dataset_filename = dataset_filename
        self.tokenizer_path = tokenizer_path
        self.MAX_LENGTH = MAX_LENGTH
        self.csv_filename = csv_filename
        self.BATCH_SIZE = BATCH_SIZE

    def compute(self, seed=42):
        with open(self.dataset_filename, "r", encoding="utf-8") as f:
            original_dataset = json.load(f)

        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        processed_samples = []

        print("Inizio la formattazione degli esempi per i 4 task...")

        for item in original_dataset:
            text = item["text"]
            triples = item["triples"]

            if not text or not triples:
                continue

            # Text2RDF - Add SOS/EOS tokens to target
            input_text1 = f"<Text2RDF> {text}"
            target_text1 = f"<SOS> {' '.join([serialize_triple(t) for t in triples])} <EOS>"
            if target_text1:
                processed_samples.append({"task":"Text2RDF","input": input_text1, "target": target_text1})

            # RDF2Text - Add SOS/EOS tokens to target
            input_text2 = f"<RDF2Text> {' '.join([serialize_triple(t) for t in triples])}"
            target_text2 = f"<SOS> {text} <EOS>"
            if input_text2:
                processed_samples.append({"task":"RDF2Text","input": input_text2, "target": target_text2})

            # RDF Completion 1 (Masking) - Add SOS/EOS tokens to target
            for triple in triples:
                components = ["subject", "predicate", "object"]

                # maschera un solo componente
                component_to_mask = random.choice(components)
                masked_triple_single = triple.copy()
                masked_triple_single[component_to_mask] = "<MASK>"

                input_text3_single = f"<MASKTASK> {serialize_triple(masked_triple_single)}"
                target_text3_single = f"<SOS> {serialize_triple(triple)} <EOS>"
                processed_samples.append(
                    {"task":"MASK","input": input_text3_single, "target": target_text3_single})

                # maschera due componenti
                if random.random() < 0.3 and len(components) >= 2:
                    components_to_mask = random.sample(components, 2)
                    masked_triple_double = triple.copy()
                    for comp in components_to_mask:
                        masked_triple_double[comp] = "<MASK>"

                    input_text3_double = f"<MASKTASK> {serialize_triple(masked_triple_double)}"
                    target_text3_double = f"<SOS> {serialize_triple(triple)} <EOS>"
                    processed_samples.append(
                        {"task":"MASK","input": input_text3_double, "target": target_text3_double})

                # RDF Completion 2 (Continuation) - Add SOS/EOS tokens to target
                if len(triples) >= 2:
                    available_indices = list(range(len(triples)))
                    context_idx = random.choice(available_indices)
                    target_candidates = [idx for idx in available_indices if idx != context_idx]
                    target_idx = random.choice(target_candidates)

                    input_text4 = f"<CONTINUERDF> {serialize_triple(triples[context_idx])}"
                    target_text4 = f"<SOS> {serialize_triple(triples[target_idx])} <EOS>"
                    processed_samples.append(
                        {"task":"CONTINUERDF","input": input_text4, "target": target_text4})


        print(f"Creati {len(processed_samples)} esempi di addestramento totali.")
        print("\n--- Esempi del dataset formattato ---")
        for i, task_type in enumerate(["Text2RDF", "RDF2Text", "RDF Completion (Masking)", "RDF Completion (Continuation)"]):
            if i < len(processed_samples):
                sample = processed_samples[i]
                print(f"\n{task_type}:")
                print(f"Input: {sample['input'][:200]}{'...' if len(sample['input']) > 200 else ''}")
                print(f"Target: {sample['target'][:200]}{'...' if len(sample['target']) > 200 else ''}")
                print("-" * 80)


        df = pd.DataFrame(processed_samples)
        df.to_csv(self.csv_filename, index=False, encoding='utf-8', columns=["task", "input", "target"])
        print(f"Dataset salvato come CSV: {self.csv_filename}")

        dataset = NanoSocratesDataset(processed_samples, tokenizer, self.MAX_LENGTH)
        data_collator = DataCollator(tokenizer)
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [0.8, 0.1, 0.1], generator=generator
        )

        train_dataloader = DataLoader(
            train_ds,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator
        )
        evaluation_dataloader = DataLoader(
            val_ds,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator
        )

        test_dataloader = DataLoader(
            test_ds,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            collate_fn=data_collator
        )

        return tokenizer, train_dataloader, evaluation_dataloader, test_dataloader
