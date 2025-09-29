import os
from typing import List, Dict, Tuple
import re
import torch
from tqdm import tqdm
import evaluate
import time

def _safe_tokens(text: str) -> List[str]:
    """
    Tokenizza in modo sicuro:
    - Restituisce [] se `text` è None.
    - Converte in str, fa strip e split su spazi bianchi (collassando spazi multipli).
    """
    if text is None:
        return []
    # split() senza argomenti rimuove token vuoti e separa su qualsiasi whitespace
    return str(text).strip().split()

def _extract_triples(seq: str) -> List[Tuple[str, str, str]]:
    """
    Parse triples from RDF format: "dbr:subject dbo:predicate dbr:object"
    """
    if seq is None or not seq.strip():
        return []

    seq = re.sub(r'<SOS>|<EOS>', '', seq).strip()

    tokens = _safe_tokens(seq)
    triples = []
    i = 0
    while i < len(tokens):
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
                    if mode == "subj":
                        subj.append(tokens[i])
                    elif mode == "pred":
                        pred.append(tokens[i])
                    elif mode == "obj":
                        obj.append(tokens[i])
                i += 1
            if i < len(tokens) and tokens[i] == "<EOT>":
                i += 1
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
    Accuracy for RDF Completion 1 (MASK): a prediction is correct if the
    fully reconstructed triple matches the ground-truth fully reconstructed triple.
    """
    correct = 0
    total = 0
    for pred_str, ref_str in zip(preds, refs):
        pred_triples = _extract_triples(pred_str)
        ref_triples = _extract_triples(ref_str)
        if len(pred_triples) == 0 or len(ref_triples) == 0:
            total += 1
            continue
        # for this dataset, masking examples produce a single triple target
        if pred_triples[0] == ref_triples[0]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def _text_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    # Clean predictions and references from special tokens for text metrics
    clean_preds = []
    clean_refs = []

    for pred, ref in zip(preds, refs):
        # Remove all special tokens for text evaluation
        clean_pred = re.sub(r'<[^>]+>', ' ', pred).strip()
        clean_ref = re.sub(r'<[^>]+>', ' ', ref).strip()
        clean_preds.append(clean_pred)
        clean_refs.append(clean_ref)

    try:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")

        # evaluate expects list[str] and list[list[str]] for references
        bleu_res = bleu.compute(predictions=clean_preds, references=[[r] for r in clean_refs])
        rouge_res = rouge.compute(predictions=clean_preds, references=clean_refs)
        meteor_res = meteor.compute(predictions=clean_preds, references=clean_refs)

        out = {
            "bleu": float(bleu_res.get("bleu", 0.0)),
            "meteor": float(meteor_res.get("meteor", 0.0)),
        }
        # pick common ROUGE variants if present
        for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            if k in rouge_res:
                out[k] = float(rouge_res[k])
        return out
    except Exception as e:
        print(f"Warning: Error computing text metrics: {e}")
        return {"bleu": 0.0, "meteor": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


_TAG_PATTERN = re.compile(r"(<SOT>|<SUBJ>|<PRED>|<OBJ>|<EOT>|<MASK>|<MASKTASK>|<SOS>|<EOS>)")

def normalize_triple_text(text: str) -> str:
    if text is None:
        return ""
    text = _TAG_PATTERN.sub(r" \1 ", text)
    return " ".join(text.strip().split())

def evaluate_tasks(
    task_list: List[str],
    predictions: List[str],
    references: List[str],
    output_dir: str = "altri_output",
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per task and save/print results.

    task_list: list of task identifiers matching dataset formatting: "RDF2Text", "Text2RDF", "MASK", "CONTINUERDF"
    predictions: model decoded strings
    references: ground-truth target strings
    output_dir: directory where per-task txt reports will be saved
    """
    assert len(task_list) == len(predictions) == len(references), "Mismatched lengths"

    os.makedirs(output_dir, exist_ok=True)

    # normalize potential tag spacing issues
    predictions = [normalize_triple_text(p) for p in predictions]
    references = [normalize_triple_text(r) for r in references]

    # Group by task
    buckets: Dict[str, Dict[str, List[str]]] = {}
    for t, p, r in zip(task_list, predictions, references):
        if t not in buckets:
            buckets[t] = {"preds": [], "refs": []}
        buckets[t]["preds"].append(p)
        buckets[t]["refs"].append(r)

    results: Dict[str, Dict[str, float]] = {}

    for task, data in buckets.items():
        preds = data["preds"]
        refs = data["refs"]

        print(f"\n=== Evaluating {task} ({len(preds)} samples) ===")

        if task == "RDF2Text":
            metrics = _text_metrics(preds, refs)
        elif task == "Text2RDF":
            p, r, f = _triples_prf(preds, refs)
            metrics = {"precision": p, "recall": r, "f1": f}
        elif task == "MASK":  # RDF Completion 1
            acc = _mask_accuracy(preds, refs)
            metrics = {"accuracy": acc}
        elif task == "CONTINUERDF":  # RDF Completion 2
            p, r, f = _triples_prf(preds, refs)
            metrics = {"precision": p, "recall": r, "f1": f}
        else:
            # Fallback: try triple PRF as it's the safest for RDF-like outputs
            p, r, f = _triples_prf(preds, refs)
            metrics = {"precision": p, "recall": r, "f1": f}

        results[task] = metrics

        # Save detailed report to txt
        report_path = os.path.join(output_dir, f"metrics_{task}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Task: {task}\n")
            f.write(f"Number of samples: {len(preds)}\n")
            f.write("="*50 + "\n")
            for mk, mv in metrics.items():
                f.write(f"{mk}: {mv:.6f}\n")

            # Add some examples
            f.write("\n" + "="*50 + "\n")
            f.write("Sample Predictions vs References:\n")
            f.write("="*50 + "\n")
            for i in range(min(5, len(preds))):
                f.write(f"\nExample {i+1}:\n")
                f.write(f"Prediction: {preds[i][:300]}{'...' if len(preds[i]) > 300 else ''}\n")
                f.write(f"Reference:  {refs[i][:300]}{'...' if len(refs[i]) > 300 else ''}\n")
                f.write("-" * 50 + "\n")

        # Print summary
        for mk, mv in metrics.items():
            print(f"{mk}: {mv:.6f}")

    # Save overall summary
    summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("OVERALL EVALUATION SUMMARY\n")
        f.write("="*50 + "\n")
        for task, metrics in results.items():
            f.write(f"\n{task}:\n")
            for mk, mv in metrics.items():
                f.write(f"  {mk}: {mv:.6f}\n")

    return results

def run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH):
    """
    Esegue la valutazione del modello sul test set usando greedy decoding.
    """
    model.eval()
    all_tasks = []
    all_predictions = []
    all_references = []
    total_examples = 0
    examples_shown = 0
    num_examples = 5

    print(f"Starting test with GREEDY DECODING over {len(test_dataset)} batches...")

    # Use proper greedy decoding for test evaluation
    sos_id = tokenizer.convert_tokens_to_ids("<SOS>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    pad_id = tokenizer.convert_tokens_to_ids("<PAD>")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataset, desc="Testing", unit="batch")):
            tasks = batch['task']
            src = batch['input_ids'].to(device)         # [B, S_in]
            tgt = batch['labels'].to(device)            # [B, S_out]

            # Model expects [seq_len, batch]; transpose
            src_t = src.transpose(0, 1)                 # [S_in, B]

            # Greedy decode
            from utils.train import greedy_decode
            predictions = greedy_decode(model, src_t, tokenizer, max_len=MAX_LENGTH//2, device=device)

            # Prepare references (remove SOS token from beginning, EOS/PAD from end)
            references = []

            for i in range(tgt.size(0)):
                ref_tokens = tgt[i].tolist()

                # Remove SOS from beginning if present
                if ref_tokens and ref_tokens[0] == sos_id:
                    ref_tokens = ref_tokens[1:]

                # Remove PAD tokens
                if pad_id is not None:
                    ref_tokens = [t for t in ref_tokens if t != pad_id]

                # Remove EOS from end if present
                if ref_tokens and eos_id is not None and ref_tokens[-1] == eos_id:
                    ref_tokens = ref_tokens[:-1]

                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=False) if ref_tokens else ""
                references.append(ref_text)

            # Collect results
            all_tasks.extend(tasks)
            all_predictions.extend(predictions)
            all_references.extend(references)
            total_examples += len(tasks)

            # Show progress with some examples
            if examples_shown < num_examples and batch_idx < 3:
                for i in range(min(3, len(tasks))):
                    if examples_shown >= num_examples:
                        break

                    # Decode input
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

    # Compute and save metrics per task
    print(f"\nEvaluating {total_examples} examples across {len(set(all_tasks))} task types...")
    evaluate_tasks(all_tasks, all_predictions, all_references, output_dir="test_results")

    print(f"Results saved in 'test_results/' directory")