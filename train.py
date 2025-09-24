import torch

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
    test_text = []
    with torch.no_grad():
        for batch in test_loader:
            src = batch['input_ids'].transpose(0, 1).to(device)
            tgt = batch['labels'].transpose(0, 1).to(device)

            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]

            #Tokenizer non lavora con tensori quindi lo faccio diventare una lista di interi
            #TODO:Utilizzare questo per le metriche
            tgt_output_decoded = [tokenizer.decode(seq.tolist(),skip_special_tokens=True) for seq in tgt_output] # Riconverto gli output in testo

            output = model(src, tgt_input) # (T-1,B,V) con tutti i logits di probabilità
            print(output.shape)
            #TODO: Temperature e quindi considerare le k migliori
            pred_ids = output.argmax(dim=-1)  # (T-1, B) ritorno solo la parola con prob più alta
            pred_decoded = [tokenizer.decode(seq.tolist(),skip_special_tokens=True) for seq in pred_ids] # Ritrasformiamo in testo per poter fare le metriche
            test_text.append(pred_decoded)

    return test_text



