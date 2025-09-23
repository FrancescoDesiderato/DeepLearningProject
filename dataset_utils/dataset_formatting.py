import json
import random
import pandas as pd
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from dataset_utils.dataset import NanoSocratesDataset,DataCollator

def serialize_triple(triple_dict):
    s = triple_dict.get('subject', '')
    p = triple_dict.get('predicate', '')
    o = triple_dict.get('object', '')
    return f"<SOT> <SUBJ> {s} <PRED> {p} <OBJ> {o} <EOT>"

class DatasetFormatting:
    def __init__(self,dataset_filename,tokenizer_path,csv_filename,MAX_LENGTH,BATCH_SIZE):
        self.dataset_filename = dataset_filename
        self.tokenizer_path = tokenizer_path
        self.MAX_LENGTH = MAX_LENGTH
        self.csv_filename = csv_filename
        self.BATCH_SIZE = BATCH_SIZE

    def compute(self):
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

            # Text2RDF
            input_text1 = f"<Text2RDF> {text}"
            target_text1 = " ".join([serialize_triple(t) for t in triples])
            if target_text1:
                processed_samples.append({"input": input_text1, "target": target_text1})

            # RDF2Text
            input_text2 = f"<RDF2Text> {' '.join([serialize_triple(t) for t in triples])}"
            target_text2 = text
            if input_text2:
                processed_samples.append({"input": input_text2, "target": target_text2})

            # RDF Completion 1 (Masking)
            # TODO capire se abbia senso considerare la probabilità oppure fare direttamente due esempi per tripla
            for triple in triples:
                components = ["subject", "predicate", "object"]

                # maschera un solo componente
                component_to_mask = random.choice(components)
                masked_triple_single = triple.copy()
                masked_triple_single[component_to_mask] = "<MASK>"

                input_text3_single = serialize_triple(masked_triple_single)
                target_text3_single = serialize_triple(triple)
                processed_samples.append(
                    {"input": input_text3_single, "target": target_text3_single})

                # maschera due componenti
                if random.random() < 0.3 and len(components) >= 2:
                    components_to_mask = random.sample(components, 2)
                    masked_triple_double = triple.copy()
                    for comp in components_to_mask:
                        masked_triple_double[comp] = "<MASK>"

                    input_text3_double = serialize_triple(masked_triple_double)
                    target_text3_double = serialize_triple(triple)
                    processed_samples.append(
                        {"input": input_text3_double, "target": target_text3_double})

                # RDF Completion 2 (Continuation)
                #TODO capire se va bene
                if len(triples) == 1:
                    for i in range(len(triples) - 1):
                        input_text4 = f"<CONTINUERDF> {serialize_triple(triples[i])}"
                        target_text4 = serialize_triple(triples[i + 1])
                        processed_samples.append({"input": input_text4, "target": target_text4})
                else:
                    available_indices = list(range(len(triples)))
                    context_idx = random.choice(available_indices)
                    target_candidates = [idx for idx in available_indices if idx != context_idx]
                    target_idx = random.choice(target_candidates)

                    input_text4_random = f"<CONTINUERDF> {serialize_triple(triples[context_idx])}"
                    target_text4_random = serialize_triple(triples[target_idx])
                    processed_samples.append(
                        {"input": input_text4_random, "target": target_text4_random})


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
        df.to_csv(self.csv_filename, index=False, encoding='utf-8', columns=["input", "target"])
        print(f"Dataset salvato come CSV: {self.csv_filename}")

        train_dataset = NanoSocratesDataset(processed_samples, tokenizer, self.MAX_LENGTH)
        data_collator = DataCollator(tokenizer)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator
        )

        return train_dataloader

