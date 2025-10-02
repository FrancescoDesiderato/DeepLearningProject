import os
from typing import List, Dict, Tuple
import re
import torch
from tqdm import tqdm
import evaluate
from mla_version.train_mla import greedy_decode


def _safe_tokens(text: str) -> List[str]:
    if text is None:
        return []
    return str(text).strip().split()


def _extract_triples(seq: str) -> List[Tuple[str, str, str]]:
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
    tp = fp = fn = 0
    for pred_str, ref_str in zip(preds, refs):
        pred_triples = set(_extract_triples(pred_str))
        ref_triples = set(_extract_triples(ref_str))
        tp += len(pred_triples & ref_triples)
        fp += len(pred_triples - ref_triples)
        fn += len(ref_triples - pred_triples)
    return _precision_recall_f1(tp, fp, fn)


def _mask_accuracy(preds: List[str], refs: List[str]) -> float:
    correct, total = 0, 0
    for pred_str, ref_str in zip(preds, refs):
        pred_triples = _extract_triples(pred_str)
        ref_triples = _extract_triples(ref_str)
        if not pred_triples or not ref_triples:
            total += 1
            continue
        if pred_triples[0] == ref_triples[0]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def _mask_accuracy_all(preds: List[str], refs: List[str],inps:List[str]) -> float:
    """The accuracy metric is calculated wrt all possible matches in the corpus.
    Furthermore this metric is not token based but triples based"""
    correct = 0
    total = 0

    corpus_path = "../dataset_utils/outputs/corpus.txt"

    with open(corpus_path, "r") as corpus_file:
        txt = corpus_file.read()

    pattern = r'<SOT>\s*<SUBJ>\s*([^<]+)\s*<PRED>\s*([^<]+)\s*<OBJ>\s*([^<]+)\s*<EOT>'
    matches = re.findall(pattern, txt)

    matches_triples = [{"subj": m[0].strip(), "pred": m[1].strip(), "obj": m[2].strip()} for m in matches]
    matches_triples = [{k: v.replace("_", " ") for k, v in triple.items()} for triple in matches_triples]

    for i,p in zip(inps, preds):
        total += 1 #Count dei sample presenti
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
            triple_p = {k: v.strip() for k, v in m.groupdict().items()}
        m = pattern_mask.search(i)
        if m:
            triple_i = {k: v.strip() for k, v in m.groupdict().items()}
        continueFlag = True
        for el_p,el_i in zip(triple_p, triple_i):
            if el_i == "<MASK>":
                continue
            elif el_i == el_p:
                continue
            else:
                continueFlag = False
                #print("Error in triple_i")
        #Controllo
        if continueFlag: #TODO: Controllo con i match nel corpus
            for triple_m in matches_triples:
                t_m = {k:t.replace(" ","") for k,t in triple_m.items()}
                t_p = {k:t.replace(" ","") for k,t in triple_p.items()}

                if all(k in t_m and t_m[k] == v for k, v in t_p.items()):
                    #print(f"Match: {t_m}")
                    #print(f"Match prediction: {t_p}")
                    correct += 1
                    continue

    return correct / total if total > 0 else 0.0


def _text_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    clean_preds = [re.sub(r'<[^>]+>', ' ', p).strip() for p in preds]
    clean_refs = [re.sub(r'<[^>]+>', ' ', r).strip() for r in refs]

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
        for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            if k in rouge_res:
                out[k] = float(rouge_res[k])
        return out
    except Exception as e:
        print(f"Warning: Error computing text metrics: {e}")
        return {"bleu": 0.0, "meteor": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


_TAG_PATTERN = re.compile(r"(<SOT>|<SUBJ>|<PRED>|<OBJ>|<EOT>|<MASK>|<MASKTASK>|<SOS>|<EOS>)")


def normalize_triple_text(text: str) -> str:
    if text is None: return ""
    text = _TAG_PATTERN.sub(r" \1 ", text)
    return " ".join(text.strip().split())


def evaluate_tasks(task_list: List[str], predictions: List[str], references: List[str], inputs: List[str],
                   output_dir: str = "altri_output") -> Dict[str, Dict[str, float]]:
    assert len(task_list) == len(predictions) == len(references) == len(inputs), "Mismatched lengths"
    os.makedirs(output_dir, exist_ok=True)

    inputs_norm = [normalize_triple_text(i) for i in inputs]
    predictions_norm = [normalize_triple_text(p) for p in predictions]
    references_norm = [normalize_triple_text(r) for r in references]

    buckets = {}
    for t, p, r, i in zip(task_list, predictions_norm, references_norm, inputs_norm):
        if t not in buckets:
            buckets[t] = {"preds": [], "refs": [], "inputs": []}
        buckets[t]["preds"].append(p)
        buckets[t]["refs"].append(r)
        buckets[t]["inputs"].append(i)

    results = {}
    for task, data in buckets.items():
        preds, refs, inps = data["preds"], data["refs"], data["inputs"]
        print(f"\n=== Evaluating {task} ({len(preds)} samples) ===")

        if task == "RDF2Text":
            metrics = _text_metrics(preds, refs)
        elif task == "Text2RDF":
            p, r, f = _triples_prf(preds, refs); metrics = {"precision": p, "recall": r, "f1": f}
        elif task == "MASK":
            acc = _mask_accuracy(preds, refs); metrics = {"accuracy": acc}
        elif task == "CONTINUERDF":
            p, r, f = _triples_prf(preds, refs); metrics = {"precision": p, "recall": r, "f1": f}
        else:
            p, r, f = _triples_prf(preds, refs); metrics = {"precision": p, "recall": r, "f1": f}

        results[task] = metrics
        report_path = os.path.join(output_dir, f"metrics_{task}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Task: {task}\n")
            f.write(f"Number of samples: {len(preds)}\n")
            f.write("=" * 50 + "\n")
            for mk, mv in metrics.items():
                f.write(f"{mk}: {mv:.6f}\n")

            # Add some examples
            f.write("\n" + "=" * 50 + "\n")
            f.write("Sample Predictions vs References:\n")
            f.write("=" * 50 + "\n")
            for i in range(min(5, len(preds))):
                f.write(f"\nExample {i + 1}:\n")
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
        f.write("=" * 50 + "\n")
        for task, metrics in results.items():
            f.write(f"\n{task}:\n")
            for mk, mv in metrics.items():
                f.write(f"  {mk}: {mv:.6f}\n")
    return results


def run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH):
    """
    Metodo modificato per lavorare con tensori BATCH-FIRST [batch, seq_len].
    """
    model.eval()
    all_inputs, all_tasks, all_predictions, all_references = [], [], [], []
    total_examples, examples_shown = 0, 0

    print(f"Starting test with GREEDY DECODING over {len(test_dataset)} batches...")

    sos_id = tokenizer.convert_tokens_to_ids("<SOS>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    pad_id = tokenizer.convert_tokens_to_ids("<PAD>")

    with torch.no_grad():
        for batch in tqdm(test_dataset, desc="Testing", unit="batch"):
            tasks = batch['task']
            src = batch['input_ids'].to(device)  # [B, S_in]
            tgt = batch['labels'].to(device)  # [B, S_out]

            predictions = greedy_decode(model, src, tokenizer, max_len=MAX_LENGTH // 2, device=device)

            references = []
            for i in range(tgt.size(0)):
                ref_tokens = tgt[i].tolist()
                if ref_tokens and ref_tokens[0] == sos_id: ref_tokens = ref_tokens[1:]
                if pad_id is not None: ref_tokens = [t for t in ref_tokens if t != pad_id]
                if ref_tokens and eos_id is not None and eos_id in ref_tokens:
                    ref_tokens = ref_tokens[:ref_tokens.index(eos_id)]
                references.append(tokenizer.decode(ref_tokens, skip_special_tokens=False))

            inputs = []
            for i in range(src.size(0)):
                input_tokens = src[i].tolist()
                if pad_id is not None: input_tokens = [t for t in input_tokens if t != pad_id]
                inputs.append(tokenizer.decode(input_tokens, skip_special_tokens=False))

            all_inputs.extend(inputs)
            all_tasks.extend(tasks)
            all_predictions.extend(predictions)
            all_references.extend(references)
            total_examples += len(tasks)

            if examples_shown < 5:
                for i in range(min(len(tasks), 5 - examples_shown)):
                    print(f"\nTest Example {examples_shown + 1} - Task: {tasks[i]}")
                    print(f"Input:      {inputs[i][:200]}")
                    print(f"Reference:  {references[i][:200]}")
                    print(f"Prediction: {predictions[i][:200]}")
                    print("-" * 80)
                    examples_shown += 1

    print(f"\nEvaluating {total_examples} examples...")
    evaluate_tasks(all_tasks, all_predictions, all_references, all_inputs, output_dir="./test_results")
    print(f"Results saved in 'test_results/' directory")