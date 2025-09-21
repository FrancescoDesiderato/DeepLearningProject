import json

import pandas as pd


# funzione helper per trasformare una tripla da dizionario a stringa formattata
def serialize_triple(triple_dict):
    s = triple_dict['subject']
    p = triple_dict['predicate']
    o = triple_dict['object']
    return f"<SOT> <SUBJ> {s} <PRED> {p} <OBJ> {o} <EOT>"

def mask_triple(uri,triple_dict):
    rows = pd.DataFrame(columns=df.columns)
    s = triple_dict['subject']
    p = triple_dict['predicate']
    o = triple_dict['object']

    for i in range(8):
        if i == 0 or i == 7:
            continue

        text = "<SOT> "
        pos = bin(i)[2:].zfill(3)

        if pos[0] == "1": first = f"<SUBJ> {s} "
        else: first ="<MASK> "
        if pos[1] == "1":
            second = f"<PRED> {p} "
        else: second ="<MASK> "
        if pos[2] == "1":
            third = f"<OBJ> {o} "
        else: third ="<MASK> "
        text += first + second + third + "<EOT>"
        row = pd.DataFrame([[uri,"Mask", text]],columns=df.columns)
        rows = pd.concat([rows,row],ignore_index=True)

    return rows

def gpt_task(uri,serialized_text):
    return pd.DataFrame([[uri, "ContinuerDf", serialized_text + " <Text2RDF>"]],columns=df.columns)

def text_task(uri,serialized_text):
    return pd.DataFrame([[uri,"RDF2Text",serialized_text+" <RDF2Text>"]],columns=df.columns)

dataset_filename = "final_paired_dataset.json"

print(f"Caricamento del dataset da '{dataset_filename}'...")
with open(dataset_filename, "r", encoding="utf-8") as f:
    dataset = json.load(f)

df = pd.DataFrame(columns=["URI", "Task", "Content"])
for item in dataset:
    # testo dell'abstract
    text = item.get("text")
    uri = item.get("uri")
    if text:
        row_df = pd.DataFrame([[uri, "Text2RDF", text + " <Text2RDF>"]], columns=df.columns)
        df = pd.concat([df, row_df], ignore_index=True)

    # iterazione sulle triple associate al film
    triples = item.get("triples", [])
    for triple in triples:
        serialized_str = serialize_triple(triple)

        RDFText = text_task(uri,serialized_str)
        gptText = gpt_task(uri,serialized_str)
        maskText = mask_triple(uri,triple)

        df = pd.concat([df,RDFText,gptText,maskText],ignore_index=True)

df.to_json("final_annotated_dataset.json")
df.to_csv("final_annotated_dataset.csv")

