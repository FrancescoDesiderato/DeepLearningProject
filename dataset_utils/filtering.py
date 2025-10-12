import pandas as pd
import json
import re
import unicodedata

"""
PIPELINE:

Eliminare i film con caratteri speciali

Eliminare ciò che è scritto in parentesi nel text(solitamente c'è il titolo in lingua originale)

Normalizzazione utf8
"""

def to_ascii_equivalent(text):
    # NFD decompone i caratteri (separa base da accenti)
    nfd_text = unicodedata.normalize('NFD', text)
    # Rimuove i caratteri di categoria 'Mn' (segni diacritici non-spaziati)
    ascii_text = ''.join(c for c in nfd_text if unicodedata.category(c) != 'Mn') #Trasforma gli accenti nella lettera base
    return ascii_text

# Apertura e parsing del file
with open('5000_dataset/final_paired_dataset_5000.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data_tot = []
count = 0
# Iterare attraverso tutti i dizionari se sono in una lista
for item in data:
    title = item["uri"].split('/')[-1] # Prendo solo l'effettivo titolo del film
    pattern = "^[a-zA-Z0-9 _]+$" # Regex per eliminare i caratteri speciali
    new_data = {}
    #Verifichiamo che il titolo non contenga caratteri speciali [già questo potrebbe complicare l'apprendimento]
    if re.match(pattern, title):
        pattern_sub = r'\([^)]*\)' # Regex per eliminare il testo presente in parentesi tonda
        new_text = re.sub(pattern_sub, "", item["text"])
        new_text = to_ascii_equivalent(new_text)
        new_data["uri"] = item["uri"]
        new_data["text"] = new_text
        new_triples = []
        for triple in item["triples"]:
            new_triple = {}
            triple["object"] = re.sub(pattern_sub,"",triple["object"])
            triple["subject"] = to_ascii_equivalent(triple["subject"])
            triple["predicate"] = to_ascii_equivalent(triple["predicate"])
            triple["object"] = to_ascii_equivalent(triple["object"])

            new_triples.append(
                    {"subject": triple["subject"], "predicate": triple["predicate"], "object": triple["object"]})

            new_data["triples"] = new_triples

        new_data_tot.append(new_data)

with open('5000_dataset/final_paired_dataset_5000_rielaborate_no_nums.json', 'w', encoding='utf-8') as file:
    json.dump(new_data_tot, file, ensure_ascii=False, indent=4)