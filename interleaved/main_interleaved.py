from transformers import PreTrainedTokenizerFast
import torch
from dataset_utils.dataset import dataLoaderFromCSV
from utils.evaluation import run_test_evaluation
from utils.train import train_model
from model_interleaved import NanoSocratesTransformerInterleaved

csv_file = "../150_dataset/processed_samples_150.csv"
tokenizer_path = "../tokenizer.json"
weight_path = "../models/nanosocrates_transformer_interleaved_444_150(1).pkl"

MAX_LENGTH = 256
BATCH_SIZE = 64
D_MODEL = 256
N_HEADS = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
FFN_HID_DIM = 256
DROPOUT = 0.3
NUM_EPOCHS = 100

scheduler_flag = True
model_training = False
test_flag = True

_, train_dataset, val_dataset, test_dataset = dataLoaderFromCSV(csv_file, tokenizer_path, MAX_LENGTH, BATCH_SIZE)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
PAD_IDX = tokenizer.convert_tokens_to_ids("<PAD>")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = NanoSocratesTransformerInterleaved(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM,
            dropout=DROPOUT,
            local_attention_window_size=5,
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
                scheduler_flag=scheduler_flag,
                )

if test_flag:

    model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
    model.to(device)

    run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH)

