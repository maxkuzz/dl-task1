## 📁 Project Structure

```text
sentiment_project/
│
├── data/                        # Data directory
│   ├── raw/                     # Original dataset (unprocessed)
│   │   └── training.1600000.processed.noemoticon.csv
│   └── processed/               # Preprocessed and encoded data
│       ├── train_data.pkl
│       ├── val_data.pkl
│       ├── test_data.pkl
│       ├── vocab.pkl
│       └── embedding_matrix.pt
│
├── notebooks/
│   └── train_textcnn.ipynb      # Main notebook: data prep, model training, evaluation
│
├── data_utils.py                # Functions for text cleaning, tokenization, encoding, and GloVe embedding
├── model.py                     # CNN model architecture (TextCNN)
├── requirements.txt             # Python dependencies
└── README.md                    # Project description
