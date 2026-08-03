# Sentiment — IMDB Movie Review Sentiment Classification Using RNNs

An end-to-end deep learning project that classifies IMDB movie reviews as **positive** or **negative** using a Recurrent Neural Network built in PyTorch. The pipeline combines text preprocessing, TF-IDF feature extraction, and a custom RNN architecture to automatically detect sentiment from raw, unstructured review text.

---

## 🔍 Project Overview

Sentiment analyzes **~50,000 IMDB movie reviews** and classifies each one as **positive** or **negative** based on the language used in the review. The goal is to move from manual review reading to automated, scalable sentiment detection that can be applied to new, unseen reviews.

The notebook walks through the full NLP + deep learning workflow:

> **Raw text → Cleaning → Preprocessing → Deduplication → Feature Extraction (TF-IDF) → Train/Test Split → RNN Model Build → Training → Evaluation**

---

## 💼 Business Problem

Platforms with large volumes of user-generated text (reviews, comments, feedback) can't manually read every entry to gauge audience sentiment. Automated sentiment classification allows a business to:

- **Track** overall audience sentiment at scale without manual tagging
- **Flag** negative reviews early for customer service or content teams
- **Feed** sentiment signals into recommendation or ranking systems
- **Monitor** sentiment trends over time for a product, film, or brand

---

## 📊 Dataset

**File:** `IMDB Dataset.csv` (50,000 rows × 2 columns)

| Category | Column | Description |
|----------|--------|--------------|
| Text | `review` | Raw, unstructured movie review text (may include HTML tags, punctuation, URLs) |
| Label | `sentiment` | Target class — `positive` or `negative` |

---

## 🛠️ Requirements

- **Python** 3.8+
- **Jupyter** Notebook or JupyterLab
- Python packages:

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `re` | Regex-based text cleaning |
| `scikit-learn` | TF-IDF vectorization, train/test split |
| `torch` | Model building, training, and evaluation (PyTorch) |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/sentimint.git
cd sentiment

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## Download the dataset

Grab it directly from the GitHub Release:

[⬇️ Download IMDB Dataset (CSV)](https://github.com/kalyanitangade4/sentiment/releases/download/v1.0/IMDB_Dataset.csv)

---

## 🧠 Methodology

### 1. Data Preprocessing
- Load `IMDB Dataset.csv` into a pandas DataFrame
- Drop duplicate reviews (`df.drop_duplicates()`) — reduces **50,000 → 49,582** rows

### 2. Text Cleaning

| Cleaning Step | Method | Purpose |
|----------------|--------|---------|
| **Lowercasing** | `str.lower()` | Normalize casing so *"Great"* and *"great"* are treated the same |
| **URL removal** | Regex `http\S+` | Strip out links that carry no sentiment signal |
| **Punctuation removal** | Regex `[^A-Za-z0-9\s]` | Remove noise that inflates the vocabulary |
| **HTML tag removal** | Regex `<.*?>` | Strip leftover markup (e.g. `<br />`) from scraped review text |

### 3. Label Encoding
- Mapped sentiment labels (`positive` / `negative`) to binary numeric targets (**1 / 0**) for use in a binary classification model

### 4. Feature Extraction (TF-IDF)
- Vectorized cleaned review text using `TfidfVectorizer` (top **5,000** features)
- Converts variable-length text into fixed-length numeric vectors weighted by term importance across the corpus

### 5. Train/Test Split
- Split the TF-IDF feature matrix and labels using `train_test_split` (**80% train / 20% test**, `random_state=42`)
- Resulting shapes: `X_train` **(39,665 × 5,000)**, `X_test` **(9,917 × 5,000)**

### 6. Dataset & DataLoader
- Wrapped train/test tensors in PyTorch `TensorDataset` objects
- Loaded via `DataLoader` with `shuffle=True` and `batch_size=64` for mini-batch training

### 7. Model Architecture (RNN)

| Layer | Configuration | Purpose |
|-------|----------------|---------|
| **RNN** | `input_size=5000, hidden_size=128, num_layers=1, batch_first=True` | Processes the TF-IDF vector as a sequence input and learns sentiment-relevant patterns |
| **Fully Connected** | `Linear(128 → 1)` | Maps the final hidden state to a single sentiment logit |
| **Sigmoid** | Applied at inference/loss step | Converts the logit into a probability for binary classification |

```
Input (TF-IDF vector, 5000 dims)
        │
        ▼
   RNN (hidden_size=128, num_layers=1)
        │
        ▼
   Fully Connected (128 → 1)
        │
        ▼
   Sigmoid → Binary Prediction
```

### 8. Training
- **Loss function:** `BCELoss` (Binary Cross-Entropy)
- **Optimizer:** `Adam`
- **Epochs:** 10
- **Batch size:** 64

### 9. Evaluation
- Ran the trained model on the held-out test set in `eval()` mode with `no_grad()`
- Applied a **0.5 threshold** on sigmoid outputs to get binary predictions
- Computed overall classification accuracy

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **85.73%** |
| Epochs | 10 |
| Optimizer | Adam |
| Loss Function | BCELoss |
| Train samples | 39,665 |
| Test samples | 9,917 |

> Loss generally trended downward across epochs, with some fluctuation typical of vanilla RNNs on high-dimensional sparse (TF-IDF) input — see the training log in the notebook for per-epoch values.

---

## ▶️ Usage

```bash
jupyter notebook sentiment_rnn.ipynb
```

Run all cells sequentially (**Kernel → Restart & Run All**). 

---

## 🔮 Future Improvements

- [ ] Replace TF-IDF with learned word embeddings (`nn.Embedding`, GloVe, Word2Vec) to capture word order and semantics
- [ ] Swap vanilla RNN for **LSTM/GRU** to better handle long-range dependencies in longer reviews
- [ ] Add a validation split and early stopping to guard against overfitting
- [ ] Track **precision, recall, and F1-score** in addition to accuracy
- [ ] Add a `predict.py` script for running inference on new, custom review text
- [ ] Build a simple dashboard (e.g., **Streamlit**) to test the model interactively

---

## 👨‍💻 Author

**Kalyani Tangade** | B.Tech IT Graduate | Aspiring Data Scientist
📍 Mumbai, India
