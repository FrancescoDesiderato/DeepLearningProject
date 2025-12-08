# 🎬 NanoSocrates

**NanoSocrates** è un modello di deep learning che combina la potenza dei Transformer T5 con la rappresentazione simbolica RDF per comprendere e generare informazioni sul cinema. Addestrato esclusivamente sul dominio dei **Film** di DBpedia, il modello impara a navigare fluidamente tra il linguaggio naturale e la struttura di triple RDF.

---

## 🎯 Task Principali

NanoSocrates affronta un **pipeline multi-task** composto da quattro task interconnessi:

### 📝 **Text2RDF**
Dato un testo descrittivo su un film, il modello estrae e genera triple RDF strutturate secondo lo schema DBpedia. Ogni tripla cattura una relazione semantica (soggetto → predicato → oggetto) che arricchisce il grafo di conoscenza del film.

```
Input: "!Oka Tokat was a Philippine paranormal horror-action-thriller drama which originally aired on ABS-CBN from June 24, 1997 to May 7, 2002 every Tuesday night..."
Output: <SOT> <SUBJ> dbr:!Oka_Tokat <PRED> dbo:director <OBJ> dbr:Rez_Cortez <EOT> <SOT> <SUBJ> dbr:!Oka_Tokat <PRED> dbo:director <OBJ> dbr:Ricky_Davao <EOT> <SOT>...
```

### 📖 **RDF2Text**
Inverso di Text2RDF: partendo da un insieme di triple RDF, il modello genera una descrizione testuale fluente e coerente, perfetta per verbalizzare grafi di conoscenza o creare abstract leggibili da umani.

```
Input: <SOT> <SUBJ> dbr:!Women_Art_Revolution <PRED> dbo:director <OBJ> dbr:Lynn_Hershman_Leeson <EOT> <SOT> ...
Output: "!Women Art Revolution is a 2010 documentary film directed by Lynn Hershman Leeson and distributed by Zeitgeist Films..."
```

### 🎭 **Masking**
Il modello impara a completare testi o triple con elementi mascherati, migliorando robustezza e capacità predittive. Questo task simula un obiettivo di masked language modeling adattato alla coppia testo-RDF.

```
Input: <SOT> <SUBJ> dbr : 01 January <PRED> <MASK> <OBJ> dbr : Yoosuf Shafeeu <EOT>
Output: <SOT> <SUBJ> dbr : 01 January <PRED> dbo : director <OBJ> dbr : Yoosuf Shafeeu <EOT>
```

### 🔗 **Generazione di Triple Successive**
Data una sequenza di triple RDF relative a un film, il modello predice ulteriori triple plausibili e coerenti con lo stesso film. 

```
Input: <SOT> <SUBJ> dbr : 008 : Operation Exterminate <PRED> dbo : cinematography <OBJ> dbr : Augusto Tiezzi <EOT> <SOT> <SUBJ> dbr : 008 : Operation Exterminate <PRED> dbo : director <OBJ> dbr : Umberto Lenzi <EOT>
Output: <SOT> <SUBJ> dbr : 008 : Operation Exterminate <PRED> dbo : writer <OBJ> dbr : Umberto Lenzi <EOT>
```

---

## 🧠 Architettura del Modello

### **Base T5**
L'architettura principale si fonda su un **Transformer encoder-decoder T5**, un modello robusto e versatile per task di sequence-to-sequence. T5 mapps sequenze testuali in sequenze di token RDF e viceversa, sfruttando l'attenzione multi-head per catturare relazioni complesse tra testo e struttura simbolica.

### **Interleaved Attention** 🔄
Oltre al T5 standard, il progetto sperimenta schemi di **interleaved attention** che alternano e combinano attenzione globale e locale. Questo approccio ibrido mira a mitigare la complessitò computazionale del modello conservando le performance.

---

## 📊 Dati e Dominio

I testi elaborati dal modello sono:

- **Triple RDF**: Serializzate in formato **soggetto, predicato, oggetto**.
- **Testi**: **Abstract** estratti delle pagine DBpedia.

---

## 🛠️ Tecnologie Utilizzate

| Componente | Tecnologia                          |
|---|-------------------------------------|
| **Linguaggio** | Python 3.8+                         |
| **Deep Learning Framework** | PyTorch + Hugging Face Transformers |
| **Data Processing** | Pandas, Numpy, JSON                 |
| **Versionamento** | Git + GitHub                        |

---

## 🚀 Getting Started

### 1️⃣ **Clonare il Repository**
```bash
git clone https://github.com/FrancescoDesiderato/DeepLearningProject.git
cd DeepLearningProject
```

### 2️⃣ **Installazione Dipendenze**
```bash
pip install -r requirements.txt
```

---

## ⚠️ Limitazioni e Training

### **Risorse Computazionali**
Per vincoli computazionali dovuti all'ambiente di training, il modello è stato addestrato su **GPU Kaggle**, soggette a limiti di:
- ⏱️ Tempo di esecuzione massimo per sessione
- 🧠 Memoria GPU limitata (~16GB)
- 💾 Storage e banda di rete ristretti

Questo ha imposto scelte conservative su:
- **Dimensione del modello**: Transformer T5 con al massimo 6 encoder, 6 decoder e 8 teste di attenzione
- **Dataset**: Subset significativo del corpus DBpedia completo
---

## 📈 Metriche di Valutazione

| Task | Metriche Principali            |
|---|--------------------------------|
| **Text2RDF** | Precision,Recall, F1           |
| **RDF2Text** | BLEU, METEOR, ROUGE            |
| **Masking** | Accuracy   |
| **Triple Generation** | Precision,Recall, F1  |

---

## 👨‍💻 Autori

### **Francesco Desiderato**,**Bianca Di Bitetto**

