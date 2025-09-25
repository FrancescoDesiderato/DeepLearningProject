import torch

from utils.metrics_calc import metrics_calculation


def train_model(model, train_loader, val_loader, num_epochs, device, VOCAB_SIZE):

    # --- Test di Sanità Mentale (Sanity Check) ---
    # Crea dei tensori di input fittizi
    BATCH_SIZE = 8
    SRC_SEQ_LEN = 100
    TGT_SEQ_LEN = 80

    src_dummy = torch.randint(0, VOCAB_SIZE, (SRC_SEQ_LEN, BATCH_SIZE)).to(device)
    tgt_dummy = torch.randint(0, VOCAB_SIZE, (TGT_SEQ_LEN, BATCH_SIZE)).to(device)

    # Esegui il forward pass
    output = model(src_dummy, tgt_dummy)

    print("Modello istanziato con successo.")
    print(f"Shape dell'output: {output.shape}")
    print(f"Shape attesa: [{TGT_SEQ_LEN}, {BATCH_SIZE}, {VOCAB_SIZE}]")

    # Verifica che la shape sia corretta
    assert output.shape == (TGT_SEQ_LEN, BATCH_SIZE, VOCAB_SIZE)
    print("La shape dell'output è corretta!")

    sanity_check_passed = True
    if not sanity_check_passed:
        print("Sanity check fallito. Interrompo l'addestramento.")
        return

    # --- Configurazione dell'Addestramento ---
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.embedding.padding_idx, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            src = batch['input_ids'].transpose(0, 1).to(device)  # [src_seq_len, batch_size]
            tgt = batch['labels'].transpose(0, 1).to(device)  # [tgt_seq_len, batch_size]

            # Shift del target per il teacher forcing
            tgt_input = tgt[:-1, :]  # Input al decoder (tgt senza l'ultimo token)
            tgt_output = tgt[1:, :]  # Target vero (tgt senza il primo token)

            optimizer.zero_grad()
            output = model(src, tgt_input)  # [tgt_seq_len-1, batch_size, vocab_size]

            loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validazione
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['input_ids'].transpose(0, 1).to(device)
                tgt = batch['labels'].transpose(0, 1).to(device)

                tgt_input = tgt[:-1, :]
                tgt_output = tgt[1:, :]

                output = model(src, tgt_input) # il modello prende la sorgente e il target sfasato e produce logits di dimensione [T-1,B.VOCAB_SIZE]
                loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        model.train()


def test_model(model, test_loader, device, tokenizer):
    model.eval()

    # Accumula tutti i risultati
    all_tasks = []
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            tsk = batch['task']  # task del testo
            src = batch['input_ids'].transpose(0, 1).to(device)
            tgt = batch['labels'].transpose(0, 1).to(device)

            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            # Decodifica i target
            tgt_output_decoded = [tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                                  for seq in tgt_output.transpose(0, 1)]  # Trasposizione per avere (B, T-1)

            output = model(src, tgt_input)  # (T-1, B, V)
            pred_ids = output.argmax(dim=-1)  # (T-1, B)
            pred_decoded = [tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                            for seq in pred_ids.transpose(0, 1)]  # Trasposizione per avere (B, T-1)

            # Accumula i risultati
            all_tasks.extend(tsk)
            all_targets.extend(tgt_output_decoded)
            all_predictions.extend(pred_decoded)

    # Calcola le metriche una sola volta su tutti i dati
    first_task, second_task, third_task, fourth_task = metrics_calculation(
        all_tasks, all_targets, all_predictions
    )

    return first_task, second_task, third_task, fourth_task,all_predictions


def print_test_results(results):
    """Stampa i risultati del test in formato leggibile"""
    first_task, second_task, third_task, fourth_task,all_predictions = results

    print("\n" + "=" * 60)
    print("RISULTATI DEL TEST")
    print("=" * 60)

    # RDF2Text (generazione testo)
    if first_task:
        print("\nRDF2Text (Generazione Testo):")
        print("\nNumero di esempi testati:", len(first_task))
        print(f"  BLEU: {first_task['bleu']['bleu']:.4f}")
        print(f"  ROUGE-1: {first_task['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2: {first_task['rouge']['rouge2']:.4f}")
        print(f"  ROUGE-L: {first_task['rouge']['rougeL']:.4f}")
        print(f"  METEOR: {first_task['meteor']['meteor']:.4f}")
    else:
        print("\nRDF2Text: Non presente nel dataset di test")

    # Text2RDF (generazione triple)
    if second_task:
        print("\nText2RDF (Generazione Triple):")
        print("\nNumero di esempi testati:", len(second_task))
        print(f"  Precisione: {second_task['precision']:.4f}")
        print(f"  Recall: {second_task['recall']:.4f}")
        print(f"  F1-Score: {second_task['f1-score']:.4f}")
    else:
        print("\nText2RDF: Non presente nel dataset di test")

    # MASK (completamento RDF)
    if third_task:
        print("\nMASK (Completamento RDF):")
        print("\nNumero di esempi testati:", len(third_task))
        print(f"  Accuratezza: {third_task['accuracy']:.4f}")
    else:
        print("\nMASK: Non presente nel dataset di test")

    # CONTINUERDF (continuazione RDF)
    if fourth_task:
        print("\nCONTINUERDF (Continuazione RDF):")
        print("\nNumero di esempi testati:", len(fourth_task))
        print(f"  Precisione: {fourth_task['precision']:.4f}")
        print(f"  Recall: {fourth_task['recall']:.4f}")
        print(f"  F1-Score: {fourth_task['f1-score']:.4f}")
    else:
        print("\nCONTINUERDF: Non presente nel dataset di test")

    print("\n" + "=" * 60)

    print(all_predictions)



def overfit_single_batch(model, train_loader, device, VOCAB_SIZE, num_iterations=1000):
    """
    Sanity check: testa se il modello può fare overfit su un singolo batch
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK: OVERFITTING SU SINGOLO BATCH")
    print("=" * 60)

    # Prendi solo il primo batch
    single_batch = next(iter(train_loader))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.embedding.padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  # Learning rate più alto

    model.train()
    initial_loss = None

    for iteration in range(num_iterations):
        src = single_batch['input_ids'].transpose(0, 1).to(device)
        tgt = single_batch['labels'].transpose(0, 1).to(device)

        # Shift del target per il teacher forcing
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        if iteration == 0:
            initial_loss = loss.item()

        if loss.item() < 0.1:
            print(f"Obiettivo di loss raggiunto a iterazione {iteration + 1}")
            break

        # Stampa ogni 10 iterazioni
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iterazione {iteration + 1:3d}: Loss = {loss.item():.6f}")

    final_loss = loss.item()
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100

    print(f"\nRisultati del Sanity Check:")
    print(f"Loss iniziale: {initial_loss:.6f}")
    print(f"Loss finale:   {final_loss:.6f}")
    print(f"Riduzione:     {loss_reduction:.2f}%")

    # Verifica se il modello ha fatto overfit correttamente
    if final_loss < 0.1:
        print("SANITY CHECK PASSATO: Il modello può fare overfit su un singolo batch")
    else:
        print("SANITY CHECK FALLITO: Il modello non riesce a fare overfit")
    print("=" * 60)
    return final_loss < 0.1




