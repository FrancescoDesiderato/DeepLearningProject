from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace,Split

underscoreRemoval = True
# inizializza un Tokenizer vuoto che userà il modello BPE
# il `unk_token` è il token che verrà usato se incontra qualcosa di sconosciuto
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
corpus_filename = "corpus.txt"

# imposta un "pre-tokenizer" che divide il testo in parole basandosi sugli spazi
tokenizer.pre_tokenizer = Whitespace()
#TODO:Add a pretokenizer even for _
if underscoreRemoval:
    pretokenizer = Split(pattern="_", behavior="removed")
# limitazione della dimensione del vocabolario
VOCAB_SIZE = 32000

# definizione dei token speciali che vogliamo includere nel vocabolario
# includiamo anche i token standard come PAD (per il padding) e UNK (sconosciuto)
SPECIAL_TOKENS = [
    "<UNK>", "<PAD>", "<MASK>",
    "<SOT>", "<EOT>",
    "<SUBJ>", "<PRED>", "<OBJ>",
    "<Text2RDF>", "<RDF2Text>", "<CONTINUERDF>"
]

# crea un trainer per il tokenizzatore BPE
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

# addestramento tramite il file di corpus
print(f"Inizio addestramento del tokenizzatore dal file '{corpus_filename}'...")
tokenizer.train(files=[corpus_filename], trainer=trainer)
print("Addestramento del tokenizzatore completato.")

# salvataggio del tokenizer
tokenizer_path = "tokenizer.json"
tokenizer.save(tokenizer_path)

print(f"Tokenizzatore salvato con successo in '{tokenizer_path}'.")

