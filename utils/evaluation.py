import os
from typing import List, Dict, Tuple
import re
import torch
from tqdm import tqdm
import evaluate
from utils.train import greedy_decode

def _normalize_tokens(text: str) -> List[str]:
    """
    Tokenizza la stringa `text` in una lista di token:
    - Restituisce [] se `text` è None.
    - Converte in str, fa strip e split su spazi bianchi (collassando spazi multipli).
    """
    if text is None:
        return []
    # split() senza argomenti rimuove token vuoti e separa su qualsiasi whitespace
    return str(text).strip().split()

def _extract_triples(seq: str) -> List[Tuple[str, str, str]]:
    """
    Parsing delle triple nel formato: "dbr:subject dbo:predicate dbr:object"
    """
    if seq is None or not seq.strip():
        return []
    # elimina token SOS/EOS
    seq = re.sub(r'<SOS>|<EOS>', '', seq).strip()

    tokens = _normalize_tokens(seq)
    triples = []
    i = 0
    while i < len(tokens):
        # ricerca start e end of triple
        if tokens[i] == "<SOT>":
            subj, pred, obj = [], [], []
            i += 1
            mode = None
            while i < len(tokens) and tokens[i] != "<EOT>":
                if tokens[i] == "<SUBJ>":
                    mode = "subj"
                elif tokens[i] == "<PRED>":
                    mode = "pred"
                elif tokens[i] == "<OBJ>":
                    mode = "obj"
                else:
                    # dopo l'idendificatore, retrieve del token successivo
                    if mode == "subj":
                        subj.append(tokens[i])
                    elif mode == "pred":
                        pred.append(tokens[i])
                    elif mode == "obj":
                        obj.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i] == "<EOT>":
                i += 1
            # costruisce tupla (soggetto, predicato, oggetto)
            triples.append((" ".join(subj).strip(), " ".join(pred).strip(), " ".join(obj).strip()))
        else:
            i += 1
    return triples


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def _triples_prf(preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    """
    Calcola precision, recall e F1 basandosi sul matching esatto delle triple estratte.
    È invariante rispetto all'ordine delle triple nella stessa stringa.
    """
    tp = fp = fn = 0
    for pred_str, ref_str in zip(preds, refs):
        # Estrae triple da predizione e riferimento
        pred_triples = set(_extract_triples(pred_str))
        ref_triples = set(_extract_triples(ref_str))
        # True positives: triple comuni
        tp += len(pred_triples & ref_triples)
        # False positives: triple predette ma non presenti nel riferimento
        fp += len(pred_triples - ref_triples)
        # False negatives: triple nel riferimento non predette
        fn += len(ref_triples - pred_triples)
    # Restituisce (precision, recall, f1)
    return _precision_recall_f1(tp, fp, fn)

def _mask_accuracy(preds: List[str], refs: List[str]) -> float:
    """
    Accuracy per RDF Completion 1 (MASK): la predizione è corretta se la tripla
    estratta dalla predizione corrisponde esattamente a quella del riferimento.
    """
    correct = 0
    total = 0
    for pred_str, ref_str in zip(preds, refs):
        pred_triples = _extract_triples(pred_str)
        ref_triples = _extract_triples(ref_str)
        # per questo task ci si aspetta esattamente una tripla predetta e una di riferimento
        if len(pred_triples) != 1 or len(ref_triples) != 1:
            total += 1
            continue
        if pred_triples[0] == ref_triples[0]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def _mask_accuracy_all(preds: List[str], refs: List[str], inps:List[str]) -> float:
    """Valuta se la tripla predetta esiste nel corpus originale."""
    correct = 0
    total = 0

    corpus_path = "./dataset_utils/outputs/corpus.txt"

    with open(corpus_path, "r") as corpus_file:
        txt = corpus_file.read()

    # Estrae tutte le triple dal corpus
    pattern = r'<SOT>\s*<SUBJ>\s*([^<]+)\s*<PRED>\s*([^<]+)\s*<OBJ>\s*([^<]+)\s*<EOT>'
    matches = re.findall(pattern, txt)

    matches_triples = [{"subj": m[0].strip(), "pred": m[1].strip(), "obj": m[2].strip()} for m in matches]
    matches_triples = [{k: v.replace("_", " ") for k, v in triple.items()} for triple in matches_triples]

    for i, p in zip(inps, preds):
        total += 1 # conta i sample presenti
        pattern = re.compile(
            r'<SOT>\s*'
            r'<SUBJ>\s*(?P<subj>[^<]+)\s*'
            r'<PRED>\s*(?P<pred>[^<]+)\s*'
            r'<OBJ>\s*(?P<obj>[^<]+)\s*'
            r'<EOT>'
        )
        pattern_mask = re.compile(
            r'<SOT>\s*'
            r'<SUBJ>\s*(?P<subj>(?:<MASK>|[^<]+))\s*'
            r'<PRED>\s*(?P<pred>(?:<MASK>|[^<]+))\s*'
            r'<OBJ>\s*(?P<obj>(?:<MASK>|[^<]+))\s*'
            r'<EOT>',
            flags=re.DOTALL
        )

        m = pattern.search(p)
        if m:
            # tripla predetta
            triple_p = {k: v.strip() for k, v in m.groupdict().items()}  # {'subj': 'dbr:$1,000_a_Touchdown', 'pred': 'dbo:starring', 'obj': 'dbr:Joe_E._Brown'}
        m = pattern_mask.search(i)
        if m:
            # tripla input con <MASK>
            triple_i = {k: v.strip() for k, v in m.groupdict().items()} # {'subj': '<MASK>', 'pred': 'dbo:starring', 'obj': 'dbr:Joe_E._Brown'}

        # verifica che la predizione non abbia cambiato gli elementi non mascherati
        continueFlag = True
        for el_p, el_i in zip(triple_p, triple_i):
            if el_i == "<MASK>":
                continue
            elif el_i == el_p:
                continue
            else:
                continueFlag = False

        # Controlla se esiste la tripla predetta nel corpus
        if continueFlag:
            for triple_m in matches_triples:
                t_m = {k:t.replace(" ","") for k,t in triple_m.items()}
                t_p = {k:t.replace(" ","") for k,t in triple_p.items()}

                if all(k in t_m and t_m[k] == v for k, v in t_p.items()):
                    correct += 1
                    continue

    return correct / total if total > 0 else 0.0


def _text_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """
    Calcola BLEU, ROUGE, METEOR tra predizioni e riferimenti testuali per RDF2Text.
    """
    clean_preds = []
    clean_refs = []

    for pred, ref in zip(preds, refs):
        # rimuove token speciali
        clean_pred = re.sub(r'<[^>]+>', ' ', pred).strip()
        clean_ref = re.sub(r'<[^>]+>', ' ', ref).strip()
        clean_preds.append(clean_pred)
        clean_refs.append(clean_ref)

    try:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")

        bleu_res = bleu.compute(predictions=clean_preds, references=[[r] for r in clean_refs])
        rouge_res = rouge.compute(predictions=clean_preds, references=clean_refs)
        meteor_res = meteor.compute(predictions=clean_preds, references=clean_refs)

        out = {
            "bleu": float(bleu_res.get("bleu", 0.0)),
            "meteor": float(meteor_res.get("meteor", 0.0)),
        }
        # aggiungi varianti ROUGE se presenti
        for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            if k in rouge_res:
                out[k] = float(rouge_res[k])
        return out
    except Exception as e:
        print(f"Warning: Error computing text metrics: {e}")
        return {"bleu": 0.0, "meteor": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def normalize_triple_text(text: str) -> str:
    """
    Normalizza il testo delle triple in un formato standard:
    - Aggiunge spazi attorno ai tag speciali.
    - Collassa spazi multipli in uno singolo.
    - Rimuove spazi iniziali e finali.
    """
    tag_pattern = re.compile(r"(<SOT>|<SUBJ>|<PRED>|<OBJ>|<EOT>|<MASK>|<MASKTASK>|<SOS>|<EOS>)")
    if text is None:
        return ""
    text = tag_pattern.sub(r" \1 ", text)
    return " ".join(text.strip().split())


def evaluate_tasks(
    task_list: List[str],
    predictions: List[str],
    references: List[str],
    inputs: List[str],
    output_dir: str,
) -> Dict[str, Dict[str, float]]:

    assert len(task_list) == len(predictions) == len(references), "Mismatched lengths"

    os.makedirs(output_dir, exist_ok=True)

    # normalizzazione (spazi attorno ai tag, rimozione spazi multipli)
    inputs = [normalize_triple_text(i) for i in inputs]
    predictions = [normalize_triple_text(p) for p in predictions]
    references = [normalize_triple_text(r) for r in references]

    # dizionario per raggruppare gli esempi in base al task
    buckets: Dict[str, Dict[str, List[str]]] = {} # task -> {"preds": [], "refs": [], "inputs":[]}
    for t, p, r, i in zip(task_list, predictions, references, inputs):
        if t not in buckets:
            buckets[t] = {"preds": [], "refs": [],"inputs":[]}
        buckets[t]["preds"].append(p)
        buckets[t]["refs"].append(r)
        buckets[t]["inputs"].append(i)
    # dizionario per i risultati
    results: Dict[str, Dict[str, float]] = {} # task -> {metric_name: value}

    # valutazione per ogni task
    for task, data in buckets.items():
        preds = data["preds"]
        refs = data["refs"]
        inps = data["inputs"]

        print(f"\n=== Evaluating {task} ({len(preds)} samples) ===")

        if task == "RDF2Text":
            metrics = _text_metrics(preds, refs)
        elif task == "Text2RDF":
            p, r, f = _triples_prf(preds, refs)
            metrics = {"precision": p, "recall": r, "f1": f}
        elif task == "MASK":  # RDF Completion 1
            acc = _mask_accuracy(preds, refs)
            acc_all = _mask_accuracy_all(preds, refs,inps)
            metrics = {"accuracy": acc, "accuracy_wrt_all":acc_all}
        elif task == "CONTINUERDF":  # RDF Completion 2
            p, r, f = _triples_prf(preds, refs)
            metrics = {"precision": p, "recall": r, "f1": f}
        else:
            # Fallback
            p, r, f = _triples_prf(preds, refs)
            metrics = {"precision": p, "recall": r, "f1": f}

        results[task] = metrics

        # salvataggio su txt
        report_path = os.path.join(output_dir, f"metrics_{task}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Task: {task}\n")
            f.write(f"Number of samples: {len(preds)}\n")
            f.write("="*50 + "\n")
            for mk, mv in metrics.items():
                f.write(f"{mk}: {mv:.6f}\n")

            # esempi
            f.write("\n" + "="*50 + "\n")
            f.write("Sample Predictions vs References:\n")
            f.write("="*50 + "\n")
            for i in range(min(5, len(preds))):
                f.write(f"\nExample {i+1}:\n")
                f.write(f"Prediction: {preds[i][:300]}{'...' if len(preds[i]) > 300 else ''}\n")
                f.write(f"Reference:  {refs[i][:300]}{'...' if len(refs[i]) > 300 else ''}\n")
                f.write("-" * 50 + "\n")

        for mk, mv in metrics.items():
            print(f"{mk}: {mv:.6f}")

    # salvataggio summary complessivo
    summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("OVERALL EVALUATION SUMMARY\n")
        f.write("="*50 + "\n")
        for task, metrics in results.items():
            f.write(f"\n{task}:\n")
            for mk, mv in metrics.items():
                f.write(f"  {mk}: {mv:.6f}\n")

    return results

def run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH, k_top=False, k_words = 5):
    """
    Esegue la valutazione del modello sul test set.
    """
    model.eval()
    all_input = []
    all_tasks = []
    all_predictions = []
    all_references = []
    total_examples = 0
    examples_shown = 0
    num_examples = 5

    print(f"Starting test over {len(test_dataset)} batches...")

    sos_id = tokenizer.convert_tokens_to_ids("<SOS>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    pad_id = tokenizer.convert_tokens_to_ids("<PAD>")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataset, desc="Testing", unit="batch")):
            tasks = batch['task']
            src = batch['input_ids'].to(device)         # [B, S_in]
            tgt = batch['labels'].to(device)            # [B, S_out]

            src_t = src.transpose(0, 1)                 # [S_in, B]

            # generazione predizioni
            predictions = greedy_decode(model, src_t, tokenizer, max_len=MAX_LENGTH // 2, device=device, top_k=k_top,
                                        k_words=k_words)

            # preparazione references (rimuove SOS dall'inzio, EOS/PAD dalla fine)
            references = []

            for i in range(tgt.size(0)):
                ref_tokens = tgt[i].tolist()

                if ref_tokens and ref_tokens[0] == sos_id:
                    ref_tokens = ref_tokens[1:]

                if pad_id is not None:
                    ref_tokens = [t for t in ref_tokens if t != pad_id]

                if ref_tokens and eos_id is not None and ref_tokens[-1] == eos_id:
                    ref_tokens = ref_tokens[:-1]
                # decodifica reference
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=False) if ref_tokens else ""
                references.append(ref_text)

            # estrazione e decodifica input
            for i in range(src.size(0)):  # Itera su ogni esempio nel batch
                src_tokens = src[i].tolist()
                if pad_id is not None:
                    src_tokens = [t for t in src_tokens if t != pad_id]
                input_text = tokenizer.decode(src_tokens, skip_special_tokens=False) if src_tokens else ""
                all_input.append(input_text)

            all_tasks.extend(tasks)
            all_predictions.extend(predictions)
            all_references.extend(references)
            total_examples += len(tasks)

            # esempi di test
            if examples_shown < num_examples and batch_idx < 3:
                for i in range(min(3, len(tasks))):
                    if examples_shown >= num_examples:
                        break

                    src_tokens = src_t[:, i].tolist()
                    if pad_id is not None:
                        src_tokens = [t for t in src_tokens if t != pad_id]
                    input_text = tokenizer.decode(src_tokens, skip_special_tokens=False)

                    print(f"\nTest Example {examples_shown + 1} - Task: {tasks[i]}")
                    print(f"Input: {input_text[:200]}{'...' if len(input_text) > 200 else ''}")
                    print(f"Reference: {references[i][:200]}{'...' if len(references[i]) > 200 else ''}")
                    print(f"Prediction: {predictions[i][:200]}{'...' if len(predictions[i]) > 200 else ''}")
                    print("-" * 80)
                    examples_shown += 1

    # metriche
    print(f"\nEvaluating {total_examples} examples across {len(set(all_tasks))} task types...")
    evaluate_tasks(all_tasks, all_predictions, all_references, all_input, output_dir="test_results")

    print(f"Results saved in 'test_results/' directory")