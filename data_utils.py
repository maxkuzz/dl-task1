import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import gensim.downloader as api

# -------------------
# 1) Load dataset
# -------------------
col_names = ['label', 'ids', 'date', 'flag', 'user', 'text']

df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin1",
    names=col_names,
    header=None
)

# Convert labels: 0=negative, 4=positive
df['label'] = df['label'].replace({0: 0, 4: 1})

# -------------------
# 2) Text preprocessing
# -------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(t):
    t = t.lower()
    t = re.sub(r'http\S+|www\S+', '', t)   # remove urls
    t = re.sub(r'@\w+', '', t)            # remove @mentions
    t = re.sub(r'#[A-Za-z0-9_]+', '', t)  # remove hashtags
    t = re.sub(r'<.*?>','', t)            # remove html
    t = re.sub(r'[^a-z\s]', '', t)        # keep only letters
    t = " ".join([w for w in t.split() if w not in stop_words])
    return t

df['text'] = df['text'].astype(str).apply(clean_text)

texts = df['text'].tolist()
labels = df['label'].values

# -------------------
# 3) Train/Val/Test Split (64/20/16)
# -------------------
X_train_texts, X_temp_texts, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.36, random_state=42, stratify=labels
)

X_val_texts, X_test_texts, y_val, y_test = train_test_split(
    X_temp_texts, y_temp, test_size=0.4444, random_state=42, stratify=y_temp
)

print("Samples:")
print("Train:", len(X_train_texts))
print("Val:  ", len(X_val_texts))
print("Test: ", len(X_test_texts))

# -------------------
# 4) Fit tokenizer ONLY on train (no leakage!)
# -------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_texts)
vocab_size = len(tokenizer.word_index) + 1

# Tokenize all splits
MAX_LEN = 50
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_texts), maxlen=MAX_LEN, padding='post', truncating='post')
X_val   = pad_sequences(tokenizer.texts_to_sequences(X_val_texts),   maxlen=MAX_LEN, padding='post', truncating='post')
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test_texts),  maxlen=MAX_LEN, padding='post', truncating='post')

# -------------------
# 5) Load GloVe via API
# -------------------
print("📥 Loading GloVe 100d model...")
glove = api.load("glove-wiki-gigaword-100")
print("Loaded. Vocab size:", len(glove))

# -------------------
# 6) Create embedding matrix
# -------------------
EMB_DIM = 100
embedding_matrix = np.zeros((vocab_size, EMB_DIM))

for word, idx in tokenizer.word_index.items():
    if word in glove:
        embedding_matrix[idx] = glove[word]

print("Embedding matrix shape:", embedding_matrix.shape)

# -------------------
# 7) Save prepared data
# -------------------
with open("data_prepared.pkl", "wb") as f:
    pickle.dump({
        "tokenizer": tokenizer,
        "embedding_matrix": embedding_matrix,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }, f)

print("✅ Done — data_prepared.pkl saved")
