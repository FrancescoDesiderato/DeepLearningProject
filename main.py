
from transformers import PreTrainedTokenizerFast

from dataset_construction import DatasetConstruction
from dataset_utils.dataset import dataLoaderFromCSV
from model import NanoSocratesTransformer
import torch
from utils.train import *
from utils.evaluation import evaluate_tasks, run_test_evaluation


page_size = 5000            # Max number of pages
test_enable = True          # Toy Dataset Flag
underscoreRemoval = True    # The Tokenizer breaks word every _ too
VOCAB_SIZE = 32000          # Max Vocabulary Size
MAX_LENGTH = 256            # Max Seq length
BATCH_SIZE = 64              # Batch Size for Training
NUM_EPOCHS = 100            # Number of Epochs for Training

csv_file = "processed_samples.csv"
tokenizer_path = "tokenizer.json"

dataset_created = False     # Set to TRUE if you have the csv data
overfit_test = False      # Set to TRUE if you want to overfit on a small dataset
test_flag = False            # Set to TRUE if you want to test
model_training = False      # Set to TRUE if you need to train the model, FALSE if you already have the weights

D_MODEL = 256               # Dimensione nascosta (embedding dimension)
N_HEADS = 4                 # Numero di teste di attenzione (deve dividere D_MODEL)
NUM_ENCODER_LAYERS = 4      # Numero di layer nell'encoder
NUM_DECODER_LAYERS = 4      # Numero di layer nel decoder
FFN_HID_DIM = 256           # Dimensione del layer nascosto nella Feed-Forward Network
DROPOUT = 0.3
weight_path = "nanosocrates_transformer.pkl"


if __name__ == '__main__':
    if dataset_created:
        tokenizer, train_dataset, val_dataset, test_dataset = dataLoaderFromCSV(csv_file,tokenizer_path,MAX_LENGTH,BATCH_SIZE)
        PAD_IDX = tokenizer.token_to_id("<PAD>")
    else:
        dataset = DatasetConstruction(page_size, test_enable, underscoreRemoval, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)
        tokenizer, train_dataset, val_dataset, test_dataset = dataset.pipeline()
        PAD_IDX = tokenizer.token_to_id("<PAD>")

    # Debug tokenizer
    sample_text = "<SOS> <SOT> <SUBJ> dbr :' If Only ' Jim <PRED> dbo : director <OBJ> dbr : Jacques Jaccard <EOT> <EOS>"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokens = tokenizer.encode(sample_text)
    print(f"Original: {sample_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {tokenizer.decode(tokens, skip_special_tokens=False)}")

    # Sposta il modello sulla GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_training:
        model = NanoSocratesTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM
        )
        # Informa il layer di embedding quale ID è per il padding
        model.embedding.padding_idx = PAD_IDX
        model.to(device)

        if overfit_test:
            sanity_passed = overfit_single_batch(model, train_dataset, device, tokenizer)
            if not sanity_passed:
                print("Sanity check fallito.")
                exit(1)

        train_model(model=model,
                    train_loader=train_dataset,
                    val_loader=val_dataset,
                    tokenizer=tokenizer,
                    num_epochs=NUM_EPOCHS,
                    device=device)

        # Salva il modello addestrato
        torch.save(model.state_dict(), "nanosocrates_transformer.pkl")

        # Test subito dopo il training se richiesto
        if test_flag:
            print("\n" + "="*80)
            print("STARTING TEST EVALUATION AFTER TRAINING")
            print("="*80)
            run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH)

    else:
        # Carica modello pre-addestrato
        model = NanoSocratesTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM
        )
        # Informa il layer di embedding quale ID è per il padding
        model.embedding.padding_idx = PAD_IDX
        model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
        model.to(device)

        # Test con modello pre-caricato se richiesto
        if test_flag:
            print("\n" + "="*80)
            print("STARTING TEST EVALUATION WITH PRE-TRAINED MODEL")
            print("="*80)
            run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH)
