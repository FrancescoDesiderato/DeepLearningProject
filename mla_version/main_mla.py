import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from dataset_construction import DatasetConstruction
from dataset_utils.dataset import dataLoaderFromCSV
from mla_version.train_mla import train_model
from eval_mla import run_test_evaluation
from utils.mlm import CorpusMLMDataset, MLMPadCollator, MLM, train_mlm
from mla_version.model_mla import build_transformer_with_mla, ModelArgs

page_size = 5000
test_enable = True
underscoreRemoval = True
VOCAB_SIZE = 32000
MAX_LENGTH = 256
dataset_size = 1500
csv_file = "../processed_samples.csv"
tokenizer_path = "../tokenizer.json"
dataset_created = True
full_balancing = True

BATCH_SIZE = 64
NUM_EPOCHS = 1
DROPOUT = 0.3
warm_restart = True

model_training = True
test_flag = True
overfit_test = False
enable_mlm = False
mlm_trained = False

weight_path = "nanosocrates_transformer_mla.pkl"
mlm_weight_path = "nanosocrates_mlm_mla.pkl"


if __name__ == '__main__':
    SEED = 42

    if dataset_created:
        tokenizer, train_dataset, val_dataset, test_dataset = dataLoaderFromCSV(csv_file, tokenizer_path, MAX_LENGTH,
                                                                                BATCH_SIZE)
    else:
        dataset = DatasetConstruction(page_size, test_enable, underscoreRemoval,
                                      VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE, full_balancing, dataset_size)
        tokenizer, train_dataset, val_dataset, test_dataset = dataset.pipeline()

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    PAD_IDX = tokenizer.convert_tokens_to_ids("<PAD>")
    actual_vocab_size = tokenizer.vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_transformer_with_mla(
        src_vocab_size=actual_vocab_size,
        tgt_vocab_size=actual_vocab_size,
        dropout=DROPOUT
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

    if model_training:

        if enable_mlm:
            print("Starting MLM pre-training...")
            mlm_model = MLM(
                transformer=model,
                mask_prob=0.15,
                replace_prob=0.9,
                num_tokens=actual_vocab_size,
                random_token_prob=0.1,
                mask_token_id=tokenizer.convert_tokens_to_ids("<MASKMLM>"),
                pad_token_id=PAD_IDX,
                mask_ignore_token_ids=[tokenizer.token_to_id(t) for t in
                                       ["<PAD>", "<SOS>", "<EOS>", "<SOT>", "<EOT>", "<SUBJ>", "<PRED>", "<OBJ>",
                                        "<Text2RDF>", "<RDF2Text>", "<CONTINUERDF>", "<MASKTASK>", "<MASK>"]]
            ).to(device)

            corpus_path = "../dataset_utils/outputs/corpus.txt"
            mlm_dataset = CorpusMLMDataset(corpus_path, tokenizer, MAX_LENGTH)
            mlm_collator = MLMPadCollator(PAD_IDX)
            mlm_loader = DataLoader(mlm_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=mlm_collator)

            mlm_model = train_mlm(mlm_loader, mlm_model, epochs=100)
            torch.save(mlm_model.state_dict(), mlm_weight_path)
            model = mlm_model.transformer
            print("MLM pre-training finished.")

        if mlm_trained:
            print(f"Loading MLM pre-trained weights from {mlm_weight_path}")
            mlm_wrapper_state_dict = torch.load(mlm_weight_path, map_location=device)
            transformer_state_dict = {k.replace('transformer.', ''): v for k, v in mlm_wrapper_state_dict.items() if
                                      k.startswith('transformer.')}
            model.load_state_dict(transformer_state_dict)
            model.to(device)
            print("MLM weights loaded successfully.")

        print("Starting main training loop...")
        train_model(
            model=model,
            train_loader=train_dataset,
            val_loader=val_dataset,
            tokenizer=tokenizer,
            num_epochs=NUM_EPOCHS,
            device=device,
            warm_restart=warm_restart,
            pad_idx=PAD_IDX
        )

        torch.save(model.state_dict(), weight_path)
        print(f"Model saved to {weight_path}")

    else:
        print(f"Loading pre-trained model from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)

    if test_flag:
        print("\n" + "=" * 80)
        print("STARTING TEST EVALUATION")
        print("=" * 80)
        run_test_evaluation(model, test_dataset, tokenizer, device, MAX_LENGTH)