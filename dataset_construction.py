from dataset_utils.endpoint import Endpoint
from dataset_utils.json_retrieve import JSONRetrieve
from dataset_utils.corpus import Corpus
from dataset_utils.bpe import BPECustom
from dataset_utils.dataset_formatting import DatasetFormatting

class DatasetConstruction:
    def __init__(self,page_size = 5000,test_enable = True,underscoreRemoval = True,VOCAB_SIZE = 32000,MAX_LENGTH = 512,BATCH_SIZE = 8,full_balancing = False,):
        self.page_size = page_size
        self.test_enable = test_enable
        self.underscoreRemoval = underscoreRemoval
        self.VOCAB_SIZE = VOCAB_SIZE
        self.MAX_LENGTH = MAX_LENGTH
        self.BATCH_SIZE = BATCH_SIZE
        self.full_balancing = full_balancing

    def pipeline(self):
        #1-STEP: ENDPOINT
        output_filename_uri = "1500_dataset/film_uris_1500.txt"
        """endpointClass = Endpoint(self.page_size,output_filename_uri)
        endpointClass.compute()"""

        #2-STEP: JSON RETRIEVE
        output_filename_json = "1500_dataset/final_paired_dataset_1500.json"
        jsonRetrieveClass = JSONRetrieve(self.test_enable, output_filename_uri,output_filename_json)
        jsonRetrieveClass.compute()

        #3-STEP: CORPUS
        corpus_filename = "dataset_utils/outputs/corpus.txt"
        corpusClass = Corpus(output_filename_json,corpus_filename)
        corpusClass.compute()

        #4-BPE
        tokenizer_path = "tokenizer.json"
        bpeCustomClass = BPECustom(self.underscoreRemoval, corpus_filename, tokenizer_path, self.VOCAB_SIZE)
        tokenizer = bpeCustomClass.compute()

        #5-DATASET
        csv_filename = "processed_samples.csv"
        datasetFormattingClass = DatasetFormatting(output_filename_json, tokenizer_path, csv_filename,
                                                   self.MAX_LENGTH, self.BATCH_SIZE,self.full_balancing)
        tokenizer, train_dataset, val_dataset, test_dataset = datasetFormattingClass.compute()

        return tokenizer, train_dataset, val_dataset, test_dataset