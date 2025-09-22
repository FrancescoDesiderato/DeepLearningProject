from dataset_construction import DatasetConstruction
page_size = 5000 #Max number of pages
test_enable = True #Toy Dataset Flag
underscoreRemoval = True #The Tokenizer breaks word every _ too
VOCAB_SIZE = 32000 #Max Vocabulary Size
MAX_LENGTH = 512 #Max Seq Lenght
BATCH_SIZE = 8 #Batch Size for Training

if __name__ == '__main__':
    dataset = DatasetConstruction(page_size, test_enable, underscoreRemoval, VOCAB_SIZE, MAX_LENGTH, BATCH_SIZE)
    train_dataset = dataset.pipeline()

    #TODO:Make Model