import json
def serialize_triple(triple_dict):
    s = triple_dict['subject']
    p = triple_dict['predicate']
    o = triple_dict['object']
    return f"<SOT> <SUBJ> {s} <PRED> {p} <OBJ> {o} <EOT>"


class Corpus:
    def __init__(self, dataset_filename, corpus_filename):
        self.dataset_filename = dataset_filename
        self.corpus_filename = corpus_filename

    def compute(self):
        print(f"Caricamento del dataset da '{self.dataset_filename}'...")
        with open(self.dataset_filename, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        print("Creazione del file di corpus in corso...")
        with open(self.corpus_filename, "w", encoding="utf-8") as f:
            # iterazione su ogni elemento del dataset
            for item in dataset:
                # testo dell'abstract
                text = item.get("text")
                if text:
                    f.write(text + "\n")

                # iterazione sulle triple associate al film
                triples = item.get("triples", [])
                for triple in triples:
                    serialized_str = serialize_triple(triple)
                    f.write(serialized_str + "\n")

        print(f"Corpus creato con successo e salvato in '{self.corpus_filename}'.")