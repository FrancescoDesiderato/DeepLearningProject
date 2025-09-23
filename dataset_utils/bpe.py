from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Split, Sequence

SPECIAL_TOKENS = [
    "<UNK>", "<PAD>", "<MASK>",
    "<SOT>", "<EOT>",
    "<SUBJ>", "<PRED>", "<OBJ>",
    "<Text2RDF>", "<RDF2Text>", "<CONTINUERDF>"
]

class BPECustom:
    def __init__(self,underscoreRemoval,corpus_filename,tokenizer_path,VOCAB_SIZE):
        self.underscoreRemoval = underscoreRemoval
        self.VOCAB_SIZE = VOCAB_SIZE
        self.corpus_filename = corpus_filename
        self.tokenizer_path = tokenizer_path

    def compute(self):
        # inizializza un Tokenizer vuoto che userà il modello BPE
        # il `unk_token` è il token che verrà usato se incontra qualcosa di sconosciuto
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

        # imposta un "pre-tokenizer" che divide il testo in parole basandosi sugli spazi
        if self.underscoreRemoval:
            tokenizer.pre_tokenizer = Sequence([
                Split(pattern="_", behavior="removed"),
                Whitespace()
            ])
        else:
            tokenizer.pre_tokenizer = Whitespace()


        # definizione dei token speciali che vogliamo includere nel vocabolario
        # includiamo anche i token standard come PAD (per il padding) e UNK (sconosciuto)


        # crea un trainer per il tokenizzatore BPE
        trainer = BpeTrainer(vocab_size=self.VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

        # addestramento tramite il file di corpus
        print(f"Inizio addestramento del tokenizzatore dal file '{self.corpus_filename}'...")
        tokenizer.train(files=[self.corpus_filename], trainer=trainer)
        print("Addestramento del tokenizzatore completato.")

        # salvataggio del tokenizer
        tokenizer.save(self.tokenizer_path)

        print(f"Tokenizzatore salvato con successo in '{self.tokenizer_path}'.")

        return tokenizer

