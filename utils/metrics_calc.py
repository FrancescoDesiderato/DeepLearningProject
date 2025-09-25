import evaluate
import re

def evaluate_text_generation_metrics(predictions, references):
    """Calcola BLEU, ROUGE e METEOR per la generazione di testo."""

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    # Alcune metriche come BLEU si aspettano una lista di riferimenti
    # Noi ne abbiamo solo uno, quindi lo mettiamo in una lista
    list_of_references = [[ref] for ref in references]

    bleu_score = bleu_metric.compute(predictions=predictions, references=list_of_references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    meteor_score = meteor_metric.compute(predictions=predictions, references=references)

    return {
        "bleu": bleu_score,
        "rouge": rouge_score,
        "meteor": meteor_score
    }

def evaluate_completion_accuracy(predictions, references):
    """Calcola l'accuracy (Exact Match) per il task di completamento."""

    correct_predictions = 0
    for pred, ref in zip(predictions, references):
        # Confrontiamo le stringhe dopo aver rimosso spazi iniziali/finali
        if pred.strip() == ref.strip():
            correct_predictions += 1

    accuracy = correct_predictions / len(predictions) if len(predictions) > 0 else 0

    return {"accuracy": accuracy}

def parse_triples_from_string(text):
    """Estrae le triple da una stringa generata e le restituisce come un set di tuple."""
    # Regex per catturare S, P, O da una tripla serializzata
    triple_pattern = re.compile(r"<SUBJ>\s*(.*?)\s*<PRED>\s*(.*?)\s*<OBJ>\s*(.*?)\s*<EOT>")
    found_triples = triple_pattern.findall(text)
    # Rimuoviamo spazi extra e creiamo un set di tuple
    return {(s.strip(), p.strip(), o.strip()) for s, p, o in found_triples}


def evaluate_triple_metrics(predictions, references):
    """Calcola Precision, Recall e F1-score per la generazione di triple."""

    total_tp, total_fp, total_fn = 0, 0, 0

    for pred_str, ref_str in zip(predictions, references):
        pred_set = parse_triples_from_string(pred_str)
        ref_set = parse_triples_from_string(ref_str)

        tp = len(pred_set.intersection(ref_set))
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Calcolo delle metriche aggregate
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1
    }

def metrics_calculation(tasks, labels, preds):
    predictions = {}
    references = {}

    # Raggruppa per task
    for task, label, pred in zip(tasks, labels, preds):
        if task not in predictions:
            predictions[task] = []
            references[task] = []
        predictions[task].append(pred)
        references[task].append(label)

    # Inizializza risultati vuoti
    first_task = second_task = third_task = fourth_task = None

    # Calcola metriche solo per i task presenti
    if 'RDF2Text' in predictions:
        first_task = evaluate_text_generation_metrics(predictions['RDF2Text'], references['RDF2Text'])

    if 'Text2RDF' in predictions:
        second_task = evaluate_triple_metrics(predictions['Text2RDF'], references['Text2RDF'])

    if 'MASK' in predictions:
        third_task = evaluate_completion_accuracy(predictions['MASK'], references['MASK'])

    if 'CONTINUERDF' in predictions:
        fourth_task = evaluate_triple_metrics(predictions['CONTINUERDF'], references['CONTINUERDF'])

    return first_task, second_task, third_task, fourth_task
