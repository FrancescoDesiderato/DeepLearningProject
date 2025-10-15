from transformers import PreTrainedTokenizerFast
import torch
from dataset_utils.dataset import dataLoaderFromCSV
from rope.model_rope import NanoSocratesTransformerROPE
from utils.evaluation import run_test_evaluation
from utils.train import train_model

csv_file = "../processed_samples.csv"
tokenizer_path = "../tokenizer.json"
weight_path = "nanosocrates_transformer_rope_444_150.pkl"
model_training = False
test_flag = True

MAX_LENGTH = 256
BATCH_SIZE = 64
D_MODEL = 256               # Dimensione nascosta (embedding dimension)
N_HEADS = 4                 # Numero di teste di attenzione (deve dividere D_MODEL)
NUM_ENCODER_LAYERS = 4      # Numero di layer nell'encoder
NUM_DECODER_LAYERS = 4      # Numero di layer nel decoder
FFN_HID_DIM = 256           # Dimensione del layer nascosto nella Feed-Forward Network
DROPOUT = 0.3
NUM_EPOCHS = 200            # Number of Epochs for Training

warm_restart = False        # Set to TRUE if you want to use warm restarts

_, train_dataset, val_dataset, test_dataset = dataLoaderFromCSV(csv_file, tokenizer_path, MAX_LENGTH, BATCH_SIZE)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
PAD_IDX = tokenizer.convert_tokens_to_ids("<PAD>")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = NanoSocratesTransformerROPE(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM,
            max_lenght=MAX_LENGTH,
            dropout=DROPOUT,
            padding_idx=PAD_IDX
        )
model.embedding.padding_idx = PAD_IDX
model.to(device)

if model_training:

    train_model(model=model,
                train_loader=train_dataset,
                val_loader=val_dataset,
                tokenizer=tokenizer,
                num_epochs=NUM_EPOCHS,
                device=device,
                scheduler_flag=warm_restart)

if test_flag:

    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
    model.to(device)

    run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH)
