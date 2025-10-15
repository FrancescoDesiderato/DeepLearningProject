from torch.utils.data import DataLoader
from utils.mlm import CorpusMLMDataset, MLMPadCollator
from transformers import PreTrainedTokenizerFast
from dataset_construction import DatasetConstruction
from dataset_utils.dataset import dataLoaderFromCSV
from model import NanoSocratesTransformer
from utils.train import *
from utils.mlm import MLM, train_mlm
from utils.evaluation import run_test_evaluation

# CONFIGURAZIONE DATASET
page_size = 5000            # Numero massimo di pagine da scaricare da DBPedia
test_enable = True          # Toy Dataset Flag: se TRUE compone un JSON che comprende solo n_film film
n_film = 150                # Numero di film da scaricare (se test_enable è TRUE)
underscoreRemoval = True    # Il tokenizer separa le parole anche in corrispondenza del carattere underscore
VOCAB_SIZE = 32000          # Max Vocabulary Size
dataset_size = 1500         # Limita il numero di samples nel dataset
full_balancing = True       # TRUE se si vuole bilanciare il dataset in modo che che ogni task abbia lo stesso numero di occorrenze

csv_file = "processed_samples.csv" # Percorso del file CSV
tokenizer_path = "500_dataset/tokenizer_500.json" # Percorso del tokenizer

# CONFIGURAZIONE TRAINING
dataset_created = True      # TRUE se il dataset è già stato creato e salvato in CSV, FALSE altrimenti
enable_mlm = False          # TRUE se si vuole abilitare il Masked Language Modeling durante l'addestramento
mlm_trained = False         # TRUE se si vuole caricare un modello MLM già addestrato
scheduler_flag = True       # TRUE se si vuole abilitare il learning rate scheduler
overfit_test = False        # TRUE se si vuole fare un overfit test su un singolo batch
test_flag = True            # TRUE se si vuole eseguire la valutazione sul test set
model_training = False      # TRUE se si vuole addestrare il modello, FALSE se si vuole caricare un modello pre-addestrato
NUM_EPOCHS = 150            # Epoche di addestramento
k_top = False               # TRUE se si vuole abilitare la valutazione K-top
k_words = 3                 # Numero di predizioni da considerare nella valutazione K-top

# CONFIGURAZIONE MODELLO
BATCH_SIZE = 64             # Batch Size
D_MODEL = 256               # Dimensione del modello (attenzione)
MAX_LENGTH = 256            # Max Seq length
N_HEADS = 4                 # Numero di teste di attenzione (deve dividere D_MODEL)
NUM_ENCODER_LAYERS = 4      # Numero di layer nell'encoder
NUM_DECODER_LAYERS = 4      # Numero di layer nel decoder
FFN_HID_DIM = 256           # Dimensione del layer nascosto nella Feed-Forward Network
DROPOUT = 0.3               # Dropout rate
weight_path = "models/nanosocrates_transformer_PRETRAINED_444_150.pkl" # Percorso del modello pre-addestrato

if __name__ == '__main__':

    if dataset_created:
        _, train_dataset, val_dataset, test_dataset = dataLoaderFromCSV(csv_file, tokenizer_path, MAX_LENGTH, BATCH_SIZE)
    else:
        dataset = DatasetConstruction(page_size, test_enable, underscoreRemoval,
                                      VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE, full_balancing, dataset_size, n_film)
        _, train_dataset, val_dataset, test_dataset = dataset.pipeline()


    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    PAD_IDX = tokenizer.convert_tokens_to_ids("<PAD>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if model_training:

        model = NanoSocratesTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM,
            dropout=DROPOUT,
            max_len=MAX_LENGTH
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
            model = train_mlm(mlm_loader, model, epochs=100, device=device)
            torch.save(model, "nanosocrates_mlm.pkl")
            model = model.transformer

        if mlm_trained:
            model.load_state_dict(torch.load("mlm_model.pt"), strict=False)
            model.to(device)

        train_model(model=model,
                    train_loader=train_dataset,
                    val_loader=val_dataset,
                    tokenizer=tokenizer,
                    num_epochs=NUM_EPOCHS,
                    device=device,
                    scheduler_flag=scheduler_flag,
                    )

        torch.save(model.state_dict(), "nanosocrates_transformer.pkl")

        if test_flag:
            print("\n" + "="*80)
            print("STARTING TEST EVALUATION AFTER TRAINING")
            print("="*80)
            run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH)

    else:

        # modello pre-addestrato
        model = NanoSocratesTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            ffn_hid_dim=FFN_HID_DIM,
            dropout=DROPOUT,
            max_len=MAX_LENGTH
        )

        model.embedding.padding_idx = PAD_IDX
        model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=device))
        model.to(device)

        if test_flag:
            print("\n" + "="*80)
            print("STARTING TEST EVALUATION WITH PRE-TRAINED MODEL")
            print("="*80)
            run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH, k_top=k_top, k_words=k_words)
