# 🎬 NanoSocrates

**NanoSocrates** is a deep learning model that combines the power of T5 Transformers with symbolic RDF representation to understand and generate information about cinema. Trained exclusively on the DBpedia **Films** domain, the model learns to navigate fluidly between natural language and RDF triple structure.

---

## 🎯 Main Tasks

NanoSocrates tackles a **multi-task pipeline** made up of four interconnected tasks:

### 📝 **Text2RDF**
Given a descriptive text about a film, the model extracts and generates structured RDF triples according to the DBpedia schema. Each triple captures a semantic relation (subject → predicate → object) that enriches the film's knowledge graph.

```
Input: "!Oka Tokat was a Philippine paranormal horror-action-thriller drama which originally aired on ABS-CBN from June 24, 1997 to May 7, 2002 every Tuesday night..."
Output: <SOT> <SUBJ> dbr:!Oka_Tokat <PRED> dbo:director <OBJ> dbr:Rez_Cortez <EOT> <SOT> <SUBJ> dbr:!Oka_Tokat <PRED> dbo:director <OBJ> dbr:Ricky_Davao <EOT> <SOT>...
```

### 📖 **RDF2Text**
The inverse of Text2RDF: starting from a set of RDF triples, the model generates a fluent, coherent textual description, ideal for verbalizing knowledge graphs or creating human-readable abstracts.

```
Input: <SOT> <SUBJ> dbr:!Women_Art_Revolution <PRED> dbo:director <OBJ> dbr:Lynn_Hershman_Leeson <EOT> <SOT> ...
Output: "!Women Art Revolution is a 2010 documentary film directed by Lynn Hershman Leeson and distributed by Zeitgeist Films..."
```

### 🎭 **Masking**
The model learns to complete texts or triples with masked elements, improving robustness and predictive ability. This task simulates a masked language modeling objective adapted to the text-RDF pair.

```
Input: <SOT> <SUBJ> dbr : 01 January <PRED> <MASK> <OBJ> dbr : Yoosuf Shafeeu <EOT>
Output: <SOT> <SUBJ> dbr : 01 January <PRED> dbo : director <OBJ> dbr : Yoosuf Shafeeu <EOT>
```

### 🔗 **Next Triple Generation**
Given a sequence of RDF triples related to a film, the model predicts further triples that are plausible and consistent with the same film.

```
Input: <SOT> <SUBJ> dbr : 008 : Operation Exterminate <PRED> dbo : cinematography <OBJ> dbr : Augusto Tiezzi <EOT> <SOT> <SUBJ> dbr : 008 : Operation Exterminate <PRED> dbo : director <OBJ> dbr : Umberto Lenzi <EOT>
Output: <SOT> <SUBJ> dbr : 008 : Operation Exterminate <PRED> dbo : writer <OBJ> dbr : Umberto Lenzi <EOT>
```

---

## 🧠 Model Architecture

### **T5 Base**
The main architecture is built on a **T5 encoder-decoder Transformer**, a robust and versatile model for sequence-to-sequence tasks. T5 maps textual sequences into sequences of RDF tokens and vice versa, leveraging multi-head attention to capture complex relationships between text and symbolic structure.

### **Interleaved Attention** 🔄
Beyond the standard T5, the project experiments with **interleaved attention** schemes that alternate and combine global and local attention. This hybrid approach aims to mitigate the model's computational complexity while preserving performance.

---

## 📊 Data and Domain

The texts processed by the model are:

- **RDF Triples**: Serialized in **subject, predicate, object** format.
- **Texts**: **Abstracts** extracted from DBpedia pages.

---

## 🛠️ Technologies Used

| Component | Technology                          |
|---|-------------------------------------|
| **Language** | Python 3.8+                         |
| **Deep Learning Framework** | PyTorch + Hugging Face Transformers |
| **Data Processing** | Pandas, Numpy, JSON                 |
| **Version Control** | Git + GitHub                        |

---

## 🚀 Getting Started

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/FrancescoDesiderato/DeepLearningProject.git
cd DeepLearningProject
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## ⚠️ Limitations and Training

### **Computational Resources**
Due to computational constraints in the training environment, the model was trained on **Kaggle GPUs**, which are subject to limits on:
- ⏱️ Maximum execution time per session
- 🧠 Limited GPU memory (~16GB)
- 💾 Restricted storage and network bandwidth

This imposed conservative choices regarding:
- **Model size**: T5 Transformer with a maximum of 6 encoder layers, 6 decoder layers, and 8 attention heads
- **Dataset**: A significant subset of the full DBpedia corpus

---

## 📈 Evaluation Metrics

| Task | Main Metrics                   |
|---|--------------------------------|
| **Text2RDF** | Precision, Recall, F1          |
| **RDF2Text** | BLEU, METEOR, ROUGE            |
| **Masking** | Accuracy                        |
| **Triple Generation** | Precision, Recall, F1  |

---

## 👨‍💻 Authors

### **Francesco Desiderato**, **Bianca Di Bitetto**
