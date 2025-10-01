import torch
from torch.utils.data import DataLoader
from utils.mlm import CorpusMLMDataset, MLMPadCollator
from transformers import PreTrainedTokenizerFast
from dataset_construction import DatasetConstruction
from dataset_utils.dataset import dataLoaderFromCSV
from model import NanoSocratesTransformer
"""
Function to call to build the Transformer
from model_from_scratch import build_transformer"""
from utils.train import *
import random
import os
from utils.mlm import MLM, train_mlm
import numpy as np
from utils.evaluation import run_test_evaluation


page_size = 5000            # Max number of pages
test_enable = False          # Toy Dataset Flag
underscoreRemoval = True    # The Tokenizer breaks word every _ too
VOCAB_SIZE = 32000          # Max Vocabulary Size
MAX_LENGTH = 256            # Max Seq length
BATCH_SIZE = 64              # Batch Size for Training
NUM_EPOCHS = 150            # Number of Epochs for Training
dataset_size = 1500        # Set to a number to limit the dataset size (for testing purposes)

csv_file = "processed_samples.csv"
tokenizer_path = "tokenizer.json"

dataset_created = True     # Set to TRUE if you have the csv data
enable_mlm = True          # Set to TRUE if you want to use MLM during training
mlm_trained = False         # Set to TRUE if you want to load a pre-trained MLM model
full_balancing = True       # Set to TRUE if you want truly balanced dataset (only 1 sample for masking and continuerdf)
warm_restart = True        # Set to TRUE if you want to use warm restarts
overfit_test = False      # Set to TRUE if you want to overfit on a small dataset
test_flag = True            # Set to TRUE if you want to test
model_training = True      # Set to TRUE if you need to train the model, FALSE if you already have the weights

D_MODEL = 256               # Dimensione nascosta (embedding dimension)
N_HEADS = 4                 # Numero di teste di attenzione (deve dividere D_MODEL)
NUM_ENCODER_LAYERS = 6      # Numero di layer nell'encoder
NUM_DECODER_LAYERS = 6      # Numero di layer nel decoder
FFN_HID_DIM = 256           # Dimensione del layer nascosto nella Feed-Forward Network
DROPOUT = 0.3
weight_path = "nanosocrates_transformer.pkl"

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

if __name__ == '__main__':
    if dataset_created:
        tokenizer, train_dataset, val_dataset, test_dataset = dataLoaderFromCSV(csv_file, tokenizer_path, MAX_LENGTH, BATCH_SIZE)
        PAD_IDX = tokenizer.token_to_id("<PAD>")
    else:
        dataset = DatasetConstruction(page_size, test_enable, underscoreRemoval,
                                      VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE, full_balancing, dataset_size)
        tokenizer, train_dataset, val_dataset, test_dataset = dataset.pipeline()
        PAD_IDX = tokenizer.token_to_id("<PAD>")

    # Debug tokenizer
    sample_text = "<SOS> <SOT> <SUBJ> dbr :' If Only ' Jim <PRED> dbo : director <OBJ> dbr : Jacques Jaccard <EOT> <EOS>"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    # PAD_IDX = tokenizer.convert_tokens_to_ids("<PAD>")
    tokens = tokenizer.encode(sample_text)
    print(f"Original: {sample_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {tokenizer.decode(tokens, skip_special_tokens=False)}")

    # Sposta il modello sulla GPU se disponibile
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)

    if model_training:
        SEED = 42
        # set_global_seed(SEED)

        model = NanoSocratesTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM
        )
        model.embedding.padding_idx = PAD_IDX
        model.to(device)

        if overfit_test:
            sanity_passed = overfit_single_batch(model, train_dataset, device, tokenizer)
            if not sanity_passed:
                print("Sanity check fallito.")
                exit(1)

        if enable_mlm:
            model = MLM(
                transformer=model,
                mask_prob=0.15,
                replace_prob=0.9,
                num_tokens=tokenizer.vocab_size,
                random_token_prob=0.1,
                mask_token_id=tokenizer.convert_tokens_to_ids("<MASKMLM>"),
                pad_token_id=PAD_IDX,
                mask_ignore_token_ids=[
                    tokenizer.convert_tokens_to_ids("<PAD>"),
                    tokenizer.convert_tokens_to_ids("<SOS>"),
                    tokenizer.convert_tokens_to_ids("<EOS>"),
                    tokenizer.convert_tokens_to_ids("<SOT>"),
                    tokenizer.convert_tokens_to_ids("<EOT>"),
                    tokenizer.convert_tokens_to_ids("<SUBJ>"),
                    tokenizer.convert_tokens_to_ids("<PRED>"),
                    tokenizer.convert_tokens_to_ids("<OBJ>"),
                    tokenizer.convert_tokens_to_ids("<Text2RDF>"),
                    tokenizer.convert_tokens_to_ids("<RDF2Text>"),
                    tokenizer.convert_tokens_to_ids("<CONTINUERDF>"),
                    tokenizer.convert_tokens_to_ids("<MASKTASK>"),
                    tokenizer.convert_tokens_to_ids("<MASK>"),
                ]
            )

            corpus_path = "dataset_utils/outputs/corpus.txt"
            mlm_dataset = CorpusMLMDataset(corpus_path, tokenizer, MAX_LENGTH)
            mlm_collator = MLMPadCollator(PAD_IDX)
            mlm_loader = DataLoader(mlm_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=mlm_collator)

            model.to(device)
            model = train_mlm(mlm_loader, model, epochs=100)
            torch.save(model, "nanosocrates_mlm.pkl")
            model = model.transformer

        if mlm_trained:
            model.load_state_dict(torch.load("nanosocrates_mlm.pkl", map_location=device).state_dict())
            model.to(device)

        train_model(model=model,
                    train_loader=train_dataset,
                    val_loader=val_dataset,
                    tokenizer=tokenizer,
                    num_epochs=NUM_EPOCHS,
                    device=device,
                    warm_restart=warm_restart)

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
