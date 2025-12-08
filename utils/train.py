import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

def greedy_decode(model, src, tokenizer, max_len=128, device='cuda', top_k=False, k_words = 5):
    """
    Generazione di sequenze con greedy decoding o top-k sampling.
    """
    model.eval()

    sos_id = tokenizer.convert_tokens_to_ids("<SOS>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    pad_id = tokenizer.convert_tokens_to_ids("<PAD>")

    with torch.no_grad():
        # src shape: [src_len, batch_size]
        batch_size = src.size(1)

        # Inizializza la sequenza target con il token SOS per ogni elemento del batch
        tgt = torch.full((1, batch_size), sos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            # Predizione del prossimo token del modello
            output = model(src, tgt)  # [tgt_len, batch_size, vocab_size]

            if top_k:
                # approccio top-k sampling
                logits = output[-1, :, :]
                probs = torch.softmax(logits, dim=-1)
                # Prendo le prime k prob con i rispettivi indici
                top_k_probs, top_k_indices = torch.topk(probs, k=k_words, dim=-1) # [batch_size,kwords]

                # Scegli in maniera proporzionale alle probabilità uno dei token per ogni elemento del batch
                next_token_index = torch.multinomial(top_k_probs, num_samples=1) # [batch_size,1]
                # gather preleva da top_k_indices usando next_token_index come indice
                next_token = torch.gather(top_k_indices, dim=1, index=next_token_index)
            else:
                # approccio greedy: seleziona il token con la massima probabilità
                next_token = output[-1, :, :].argmax(dim=-1, keepdim=True)  # [batch_size, 1]

            next_token = next_token.transpose(0, 1)  # [1, batch_size]

            # Append alla sequenza target
            tgt = torch.cat([tgt, next_token], dim=0)

            # Verifica se tutti gli elementi del batch hanno generato EOS: se sì, interrompi
            if eos_id is not None and (next_token.squeeze(0) == eos_id).all():
                break

        # rimuove SOS
        generated = tgt[1:, :].transpose(0, 1)  # [batch_size, seq_len]

        # conversione in testo
        results = []
        for i in range(batch_size):
            tokens = generated[i].tolist()
            # rimuove PAD ed EOS
            if eos_id is not None and eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id)]
            if pad_id is not None:
                tokens = [t for t in tokens if t != pad_id]

            # decodifica i risultati
            text = tokenizer.decode(tokens, skip_special_tokens=False) if tokens else ""
            results.append(text)

        return results

def run_validation(model, val_loader, tokenizer, device, num_examples=5):
    """
    Validazione con greedy decoding.
    La loss viene calcolata con teacher forcing mentre gli esempi vengono mostrati applicando il greedy decoding.
    """
    model.eval()
    val_loss = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.embedding.padding_idx)

    examples_shown = 0

    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            src = batch['input_ids'].transpose(0, 1).to(device)
            tgt = batch['labels'].transpose(0, 1).to(device)
            tasks = batch['task']

            # Standard validation loss (teacher forcing)
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), tgt_output.reshape(-1))
            val_loss += loss.item()

            # decodifica greedy (senza teacher forcing) per ottenere predizioni reali
            predictions = greedy_decode(model, src, tokenizer, device=device)

            # lista per i riferimenti (ground truth)
            references = []
            for i in range(tgt.size(1)):
                tgt_tokens = tgt[1:, i].tolist()  # salta SOS token
                pad_id = tokenizer.convert_tokens_to_ids("<PAD>")
                eos_id = tokenizer.convert_tokens_to_ids("<EOS>")

                # rimuove PAD ed EOS
                if pad_id is not None:
                    tgt_tokens = [t for t in tgt_tokens if t != pad_id]
                if eos_id is not None and eos_id in tgt_tokens:
                    tgt_tokens = tgt_tokens[:tgt_tokens.index(eos_id)]

                # decodifica i token del target
                ref_text = tokenizer.decode(tgt_tokens, skip_special_tokens=False) if tgt_tokens else ""
                references.append(ref_text)

            # esempi decodificati
            if examples_shown < num_examples and batch_idx < 3:
                for i in range(min(3, len(tasks))):
                    if examples_shown >= num_examples:
                        break

                    # estrae l'input per questo esempio
                    src_tokens = src[:, i].tolist()
                    pad_id = tokenizer.convert_tokens_to_ids("<PAD>")
                    if pad_id is not None:
                        src_tokens = [t for t in src_tokens if t != pad_id]
                    # decodifica l'input
                    input_text = tokenizer.decode(src_tokens, skip_special_tokens=False)

                    print(f"\nExample {examples_shown + 1} - Task: {tasks[i]}")
                    print(f"Input: {input_text[:200]}{'...' if len(input_text) > 200 else ''}")
                    print(f"Reference: {references[i][:200]}{'...' if len(references[i]) > 200 else ''}")
                    print(f"Prediction: {predictions[i][:200]}{'...' if len(predictions[i]) > 200 else ''}")
                    print("-" * 80)
                    examples_shown += 1

    avg_val_loss = val_loss / len(val_loader)
    print(f"\nValidation Loss: {avg_val_loss:.4f}")

    return avg_val_loss

def train_model(model, train_loader, val_loader, num_epochs, device, tokenizer, scheduler_flag):

    # cross entropy loss con ignore index per il padding
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.embedding.padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.1)
    # scheduler che riduce il learning rate con cosine annealing
    # T_max: numero totale di epoche, eta_min: LR minimo (1e-5)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-5)

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

            # criterion, confronta ogni token predetto con il token vero
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # validation
        if (epoch + 1) % 5 == 0:
            avg_val_loss = run_validation(model, val_loader, tokenizer, device)
            torch.save(model.state_dict(), "nanosocrates_transformer.pkl") # Checkpoint periodico dei pesi

        if scheduler_flag:
            scheduler.step()
        model.train()
