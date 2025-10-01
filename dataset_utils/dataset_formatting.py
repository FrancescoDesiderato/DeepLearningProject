import json
import random
import pandas as pd
from tokenizers import Tokenizer
from dataset_utils.dataset import dataset_split

def serialize_triple(triple_dict):
    s = triple_dict.get('subject', '')
    p = triple_dict.get('predicate', '')
    o = triple_dict.get('object', '')
    return f"<SOT> <SUBJ> {s} <PRED> {p} <OBJ> {o} <EOT>"

class DatasetFormatting:
    def __init__(self,dataset_filename,tokenizer_path,csv_filename, MAX_LENGTH, BATCH_SIZE, full_balancing, dataset_size = None):
        self.dataset_filename = dataset_filename
        self.tokenizer_path = tokenizer_path
        self.MAX_LENGTH = MAX_LENGTH
        self.csv_filename = csv_filename
        self.BATCH_SIZE = BATCH_SIZE
        self.full_balancing = full_balancing
        self.dataset_size = dataset_size

    def compute(self, seed=42):
        with open(self.dataset_filename, "r", encoding="utf-8") as f:
            original_dataset = json.load(f)

        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        processed_samples = []

        print("Inizio la formattazione degli esempi per i 4 task...")

        if self.dataset_size:
            original_dataset = original_dataset[:self.dataset_size]

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
            if not self.full_balancing:
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
                        indices = list(range(len(triples)))
                        # scegli 1..len(triples)-1 triple di contesto
                        ctx_count = random.randint(1, len(triples) - 1)
                        ctx_indices = sorted(random.sample(indices, ctx_count))
                        remaining = [i for i in indices if i not in ctx_indices]

                        # scegli 1..K triple target (limita K per controllare la lunghezza)
                        max_target = min(len(remaining), 3)  # limite pratico
                        tgt_count = random.randint(1, max_target)
                        tgt_indices = sorted(random.sample(remaining, tgt_count))

                        context_triples = [serialize_triple(triples[i]) for i in ctx_indices]
                        target_triples = [serialize_triple(triples[i]) for i in tgt_indices]

                        input_text4 = f"<CONTINUERDF> {' '.join(context_triples)}"
                        target_text4 = f"<SOS> {' '.join(target_triples)} <EOS>"

                        processed_samples.append(
                            {"task": "CONTINUERDF", "input": input_text4, "target": target_text4}
                        )
            else:
                triple = random.choice(triples)
                components = ["subject", "predicate", "object"]

                if random.random() < 0.3 and len(components) >= 2:
                    # Con una certa probabilità maschera 2 componenti
                    components_to_mask = random.sample(components, 2)
                    masked_triple_double = triple.copy()
                    for comp in components_to_mask:
                        masked_triple_double[comp] = "<MASK>"

                    input_text3_double = f"<MASKTASK> {serialize_triple(masked_triple_double)}"
                    target_text3_double = f"<SOS> {serialize_triple(triple)} <EOS>"
                    processed_samples.append(
                        {"task": "MASK", "input": input_text3_double, "target": target_text3_double})

                else:
                    # maschera un solo componente
                    component_to_mask = random.choice(components)
                    masked_triple_single = triple.copy()
                    masked_triple_single[component_to_mask] = "<MASK>"

                    input_text3_single = f"<MASKTASK> {serialize_triple(masked_triple_single)}"
                    target_text3_single = f"<SOS> {serialize_triple(triple)} <EOS>"
                    processed_samples.append(
                        {"task": "MASK", "input": input_text3_single, "target": target_text3_single})

                if len(triples) >= 2:
                    indices = list(range(len(triples)))
                    # scegli 1..len(triples)-1 triple di contesto
                    ctx_count = random.randint(1, len(triples) - 1)
                    ctx_indices = sorted(random.sample(indices, ctx_count))
                    remaining = [i for i in indices if i not in ctx_indices]

                    # scegli 1..K triple target (limita K per controllare la lunghezza)
                    max_target = min(len(remaining), 3)  # limite pratico
                    tgt_count = random.randint(1, max_target)
                    tgt_indices = sorted(random.sample(remaining, tgt_count))

                    context_triples = [serialize_triple(triples[i]) for i in ctx_indices]
                    target_triples = [serialize_triple(triples[i]) for i in tgt_indices]

                    input_text4 = f"<CONTINUERDF> {' '.join(context_triples)}"
                    target_text4 = f"<SOS> {' '.join(target_triples)} <EOS>"

                    processed_samples.append(
                        {"task": "CONTINUERDF", "input": input_text4, "target": target_text4}
                    )

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

        train_dataloader, evaluation_dataloader, test_dataloader = dataset_split(df, tokenizer, self.MAX_LENGTH, self.BATCH_SIZE, seed)
        return tokenizer, train_dataloader, evaluation_dataloader, test_dataloader