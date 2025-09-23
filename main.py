from dataset_construction import DatasetConstruction
from dataset_utils.dataset import dataLoaderFromCSV

page_size = 5000 # Max number of pages
test_enable = True # Toy Dataset Flag
underscoreRemoval = True # The Tokenizer breaks word every _ too
VOCAB_SIZE = 32000 # Max Vocabulary Size
MAX_LENGTH = 512 # Max Seq length
BATCH_SIZE = 8 # Batch Size for Training
csv_file = "processed_samples.csv"
tokenizer_path = "tokenizer.json"
dataset_created = True # Set to TRUE if you have the csv data

if __name__ == '__main__':
    if dataset_created:
        train_dataset, eval_dataset, test_dataset = dataLoaderFromCSV(csv_file,tokenizer_path,MAX_LENGTH,BATCH_SIZE)
    else:
        dataset = DatasetConstruction(page_size, test_enable, underscoreRemoval, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)
        train_dataset,eval_dataset,test_dataset = dataset.pipeline()

    # TODO:Make Model
