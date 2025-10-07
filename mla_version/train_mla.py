import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def greedy_decode(model, src, tokenizer, max_len=128, device='cuda'):
    """
    Metodo modificato per lavorare con tensori BATCH-FIRST [batch, seq_len].
    """
    model.eval()

    sos_id = tokenizer.convert_tokens_to_ids("<SOS>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    pad_id = tokenizer.convert_tokens_to_ids("<PAD>")

    with torch.no_grad():
        # src shape: [batch_size, src_len]
        batch_size = src.size(0)

        # Crea la maschera per l'encoder (per ignorare il padding)
        src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).to(device)

        # Codifica l'input una sola volta
        encoder_output = model.encode(src, src_mask)

        # Inizializza l'input del decoder con il token SOS
        tgt = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)

        # Flag per tenere traccia delle sequenze che hanno già generato EOS
        finished_seqs = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # Crea la maschera causale per il target
            tgt_mask = torch.tril(torch.ones((tgt.size(1), tgt.size(1)), device=device)).bool()

            # Decodifica
            output = model.decode(encoder_output, src_mask, tgt, tgt_mask)

            # Proietta per ottenere i logits
            logits = model.project(output)  # [batch_size, tgt_len, vocab_size]

            # Prendi il token successivo (greedy) dall'ultimo step temporale
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [batch_size, 1]

            # Aggiungi il nuovo token alla sequenza target
            tgt = torch.cat([tgt, next_token], dim=1)

            # Aggiorna le sequenze terminate
            finished_seqs |= (next_token.squeeze(-1) == eos_id)

            # Interrompi se tutte le sequenze nel batch sono terminate
            if finished_seqs.all():
                break

        # Rimuovi il token SOS iniziale
        generated = tgt[:, 1:]

        # Converte gli ID in testo
        results = []
        for i in range(batch_size):
            tokens = generated[i].tolist()
            # Rimuovi i token dopo EOS
            if eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id)]

            text = tokenizer.decode(tokens, skip_special_tokens=False)
            results.append(text)

        return results


def run_validation(model, val_loader, tokenizer, device, pad_idx, num_examples=5):
    """
    Metodo modificato per usare pad_idx e input batch-first.
    """
    model.eval()
    val_loss = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    actual_vocab_size = tokenizer.vocab_size

    all_tasks, all_predictions, all_references = [], [], []
    examples_shown = 0

    print("\n" + "=" * 80 + "\nVALIDATION\n" + "=" * 80)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            tasks = batch['task']

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, actual_vocab_size), tgt_output.reshape(-1))
            val_loss += loss.item()

            predictions = greedy_decode(model, src, tokenizer, device=device)

            references = []
            for i in range(tgt.size(0)):
                ref_tokens = tgt[i, 1:].tolist()
                if pad_idx in ref_tokens:
                    ref_tokens = ref_tokens[:ref_tokens.index(pad_idx)]
                if tokenizer.eos_token_id in ref_tokens:
                    ref_tokens = ref_tokens[:ref_tokens.index(tokenizer.eos_token_id)]
                references.append(tokenizer.decode(ref_tokens, skip_special_tokens=False))

            all_tasks.extend(tasks)
            all_predictions.extend(predictions)
            all_references.extend(references)

            if examples_shown < num_examples and batch_idx < 3:
                for i in range(min(3, len(tasks))):
                    if examples_shown >= num_examples:
                        break

                    # Decode input
                    src_tokens = src[i].tolist()
                    pad_id = tokenizer.convert_tokens_to_ids("<PAD>")
                    if pad_id is not None:
                        src_tokens = [t for t in src_tokens if t != pad_id]
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


def train_model(model, train_loader, val_loader, num_epochs, device, tokenizer, warm_restart, pad_idx):
    """
    Esegue il training. MODIFICATO per input batch-first.
    """
    actual_vocab_size = tokenizer.vocab_size
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)

            # Teacher forcing con formato batch-first
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)

            loss = criterion(output.reshape(-1, actual_vocab_size), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            avg_val_loss = run_validation(model, val_loader, tokenizer, device, pad_idx)
            torch.save(model.state_dict(), "../nanosocrates_transformer.pkl")

        if warm_restart:
            scheduler.step()

