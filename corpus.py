import json

dataset_filename = "final_paired_dataset.json"
corpus_filename = "corpus.txt"


# funzione helper per trasformare una tripla da dizionario a stringa formattata
def serialize_triple(triple_dict):
    s = triple_dict['subject']
    p = triple_dict['predicate']
    o = triple_dict['object']
    return f"<SOT> <SUBJ> {s} <PRED> {p} <OBJ> {o} <EOT>"


print(f"Caricamento del dataset da '{dataset_filename}'...")
with open(dataset_filename, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("Creazione del file di corpus in corso...")
with open(corpus_filename, "w", encoding="utf-8") as f:
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

print(f"Corpus creato con successo e salvato in '{corpus_filename}'.")