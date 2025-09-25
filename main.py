from dataset_construction import DatasetConstruction
from dataset_utils.dataset import dataLoaderFromCSV
from model import NanoSocratesTransformer
import torch
from utils.train import *

page_size = 5000            # Max number of pages
test_enable = True          # Toy Dataset Flag
underscoreRemoval = True    # The Tokenizer breaks word every _ too
VOCAB_SIZE = 32000          # Max Vocabulary Size
MAX_LENGTH = 512            # Max Seq length
BATCH_SIZE = 8              # Batch Size for Training

csv_file = "processed_samples.csv"
tokenizer_path = "tokenizer.json"

dataset_created = True     # Set to TRUE if you have the csv data
overfit_test = True      # Set to TRUE if you want to overfit on a small dataset
test_flag = False            # Set to TRUE if you want to test
model_training = True      # Set to TRUE if you need to train the model, FALSE if you already have the weights

D_MODEL = 512               # Dimensione nascosta (embedding dimension)
N_HEADS = 8                 # Numero di teste di attenzione (deve dividere D_MODEL)
NUM_ENCODER_LAYERS = 4      # Numero di layer nell'encoder
NUM_DECODER_LAYERS = 4      # Numero di layer nel decoder
FFN_HID_DIM = 512           # Dimensione del layer nascosto nella Feed-Forward Network
DROPOUT = 0.1
weight_path = "nanosocrates_transformer.pkl"


if __name__ == '__main__':
    if dataset_created:
        tokenizer,train_dataset, val_dataset, test_dataset = dataLoaderFromCSV(csv_file,tokenizer_path,MAX_LENGTH,BATCH_SIZE)
        PAD_IDX = tokenizer.token_to_id("<PAD>")
    else:
        dataset = DatasetConstruction(page_size, test_enable, underscoreRemoval, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)
        tokenizer, train_dataset, val_dataset, test_dataset = dataset.pipeline()
        PAD_IDX = tokenizer.token_to_id("<PAD>")

    if model_training:
        model = NanoSocratesTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM
        )
        # Informa il layer di embedding quale ID è per il padding
        model.embedding.padding_idx = PAD_IDX

        # Sposta il modello sulla GPU se disponibile
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if overfit_test:
            sanity_passed = overfit_single_batch(model, train_dataset, device, VOCAB_SIZE)
            if not sanity_passed:
                print("Sanity check fallito.")
                exit(1)

        train_model(model=model,
                    train_loader=train_dataset,
                    val_loader=val_dataset,
                    VOCAB_SIZE=VOCAB_SIZE,
                    num_epochs=10,
                    device=device)

        # Salva il modello addestrato
        torch.save(model.state_dict(), "nanosocrates_transformer.pkl")

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NanoSocratesTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM
        )
        # Informa il layer di embedding quale ID è per il padding
        model.embedding.padding_idx = PAD_IDX
        model.load_state_dict(torch.load(weight_path, weights_only=True))

    #test phase
    if test_flag:
        predictions = test_model(model, test_dataset,device, tokenizer)
        print_test_results(predictions)







