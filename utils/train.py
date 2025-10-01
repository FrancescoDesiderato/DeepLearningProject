import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def greedy_decode(model, src, tokenizer, max_len=128, device='cuda'):
    """
    Perform greedy decoding for a single source sequence.
    """
    model.eval()

    sos_id = tokenizer.convert_tokens_to_ids("<SOS>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    pad_id = tokenizer.convert_tokens_to_ids("<PAD>")

    with torch.no_grad():
        # src shape: [src_len, batch_size]
        batch_size = src.size(1)

        # Initialize decoder input with SOS token
        tgt = torch.full((1, batch_size), sos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            # Get model predictions
            output = model(src, tgt)  # [tgt_len, batch_size, vocab_size]

            # Get next token (greedy)
            next_token = output[-1, :, :].argmax(dim=-1, keepdim=True)  # [batch_size, 1]
            next_token = next_token.transpose(0, 1)  # [1, batch_size]

            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=0)

            # Check if all sequences have generated EOS
            if eos_id is not None and (next_token.squeeze(0) == eos_id).all():
                break

        # Remove SOS token from beginning
        generated = tgt[1:, :].transpose(0, 1)  # [batch_size, seq_len]

        # Convert to text
        results = []
        for i in range(batch_size):
            tokens = generated[i].tolist()
            # Remove EOS and PAD tokens
            if eos_id is not None and eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id)]
            if pad_id is not None:
                tokens = [t for t in tokens if t != pad_id]

            text = tokenizer.decode(tokens, skip_special_tokens=False) if tokens else ""
            results.append(text)

        return results

def run_validation(model, val_loader, tokenizer, device, num_examples=5):
    """
    Run validation with greedy decoding and visualization of examples.
    """
    model.eval()
    val_loss = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.embedding.padding_idx)
    actual_vocab_size = tokenizer.vocab_size

    # Collect examples for evaluation
    all_tasks = []
    all_predictions = []
    all_references = []

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
            loss = criterion(output.reshape(-1, actual_vocab_size), tgt_output.reshape(-1))
            val_loss += loss.item()

            # Greedy decoding for evaluation
            predictions = greedy_decode(model, src, tokenizer, device=device)

            # Prepare references (remove SOS token from target)
            references = []
            for i in range(tgt.size(1)):
                tgt_tokens = tgt[1:, i].tolist()  # Skip SOS token
                pad_id = tokenizer.convert_tokens_to_ids("<PAD>")
                eos_id = tokenizer.convert_tokens_to_ids("<EOS>")

                if pad_id is not None:
                    tgt_tokens = [t for t in tgt_tokens if t != pad_id]
                if eos_id is not None and eos_id in tgt_tokens:
                    tgt_tokens = tgt_tokens[:tgt_tokens.index(eos_id)]

                ref_text = tokenizer.decode(tgt_tokens, skip_special_tokens=False) if tgt_tokens else ""
                references.append(ref_text)

            # Collect for evaluation metrics
            all_tasks.extend(tasks)
            all_predictions.extend(predictions)
            all_references.extend(references)

            # Show some examples
            if examples_shown < num_examples and batch_idx < 3:
                for i in range(min(3, len(tasks))):
                    if examples_shown >= num_examples:
                        break

                    # Decode input
                    src_tokens = src[:, i].tolist()
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

    # Calculate metrics
    avg_val_loss = val_loss / len(val_loader)
    print(f"\nValidation Loss: {avg_val_loss:.4f}")

    return avg_val_loss

def train_model(model, train_loader, val_loader, num_epochs, device, tokenizer, warm_restart):

    actual_vocab_size = tokenizer.vocab_size
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.embedding.padding_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.01)
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

            loss = criterion(output.reshape(-1, actual_vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Enhanced validation with greedy decoding
        if (epoch + 1) % 5 == 0:
            avg_val_loss = run_validation(model, val_loader, tokenizer, device)
            torch.save(model.state_dict(), "nanosocrates_transformer.pkl")

        if warm_restart:
            scheduler.step()
        model.train()


def overfit_single_batch(model, train_loader, device, tokenizer, num_iterations=1000):
    """
    Sanity check: testa se il modello può fare overfit su un singolo batch
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK: OVERFITTING SU SINGOLO BATCH")
    print("=" * 60)

    # Prendi solo il primo batch
    single_batch = next(iter(train_loader))

    # Usa la dimensione reale del vocabolario
    actual_vocab_size = tokenizer.get_vocab_size()

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

        loss = criterion(output.reshape(-1, actual_vocab_size), tgt_output.reshape(-1))
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

    # Show greedy decoding example from overfitted batch
    if final_loss < 0.5:
        print("\nGreedy decoding example from overfitted batch:")
        model.eval()
        src = single_batch['input_ids'].transpose(0, 1).to(device)
        predictions = greedy_decode(model, src[:, :1], tokenizer, device=device)  # Just first example

        # Show input and prediction
        src_tokens = src[:, 0].tolist()
        pad_id = tokenizer.token_to_id("<PAD>")
        if pad_id is not None:
            src_tokens = [t for t in src_tokens if t != pad_id]
        input_text = tokenizer.decode(src_tokens, skip_special_tokens=False)

        print(f"Input: {input_text[:200]}{'...' if len(input_text) > 200 else ''}")
        print(f"Greedy Output: {predictions[0][:200]}{'...' if len(predictions[0]) > 200 else ''}")
        model.train()

    # Verifica se il modello ha fatto overfit correttamente
    if final_loss < 0.1:
        print("SANITY CHECK PASSATO: Il modello può fare overfit su un singolo batch")
    else:
        print("SANITY CHECK FALLITO: Il modello non riesce a fare overfit")
    print("=" * 60)
    return final_loss < 0.1
