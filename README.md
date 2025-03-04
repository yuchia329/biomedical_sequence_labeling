# Bi-LSTM with CRF for BIO Tagging

## Overview
This project implements a **Bi-LSTM with a Viterbi Decoder (CRF)** for the **BIO tagging task** using the **BioNLP/NLPBA 2004 corpus**. The model is trained for named entity recognition (NER) in biomedical text.

The system utilizes:
- **BiLSTM** for sequence modeling
- **Conditional Random Fields (CRF)** for sequence decoding
- **Pre-trained GloVe embeddings** for word representations
- **Character-level CNNs** for additional feature extraction

## Project Structure
```
BIOMEDICAL_SEQUENCE_TAGGER/
│── A2-data/
│── test_answers/
│   ├── test.answers
│   ├── dev
│   ├── dev.answers
│   ├── test
│   ├── train
│── .gitignore
│── baseline.py
│── evaluation.py
│── LICENSE
│── main_cnn.py
│── main.py
│── old.py
│── output_softmax_margin.txt
│── output_svm.txt
│── output_wo_cnn.txt
│── README.md
│── show_tag_dist.py
```
### Key Files
- `main.py` - **Main training script** for Bi-LSTM + CRF on the BIO tagging task
- `evaluation.py` - **Computes token-level F1-score and accuracy** for evaluation
- `main_cnn.py` - Alternative implementation using **CNN features**
- `baseline.py` - A baseline model for comparison
- `show_tag_dist.py` - Displays the distribution of tags in the dataset
- `output_*.txt` - Contains model predictions using different configurations
- `README.md` - Project documentation
- `.gitignore` - Ignore unnecessary files
- `LICENSE` - License information

## Dataset
The dataset used is the **BioNLP/NLPBA 2004 corpus**, a standard benchmark for biomedical named entity recognition. It consists of sequences labeled with BIO tags indicating named entity spans.

### Data Format
The dataset follows a CoNLL-style format:
```
word1    B-Protein
word2    I-Protein
word3    O
```
Each word is followed by its BIO label. Sentences are separated by blank lines.

## Model Architecture
The **BiLSTM-CRF model** consists of:
1. **Word Embeddings**: Pre-trained **GloVe embeddings** from `gensim`
2. **Character-level CNNs**: Extract sub-word information
3. **BiLSTM**: Contextual encoding of token sequences
4. **CRF Layer**: Enforces valid label transitions using **Viterbi decoding**

## Training & Evaluation

### Training the Model
To train the model, use:
```sh
python main_cnn.py train_file.txt val_file.txt test_file.txt output_predictions.txt --loss_function nll
```

Arguments:
- `train_file.txt`: Path to the training data
- `val_file.txt`: Path to validation data
- `test_file.txt`: Path to test data
- `output_predictions.txt`: Path to save model predictions
- `--loss_function`: Choose between **nll (Negative Log-Likelihood), svm (Structured Hinge), and sm (Softmax Margin)**

### Evaluating Predictions
To evaluate the model, run:
```sh
python evaluation.py --pred output_predictions.txt --gold test.answers
```
This script calculates:
- **Per-label precision, recall, F1-score, and accuracy**
- **Micro and Macro F1-scores**

## Results & Performance
The model is benchmarked using **F1-score and accuracy**. The inclusion of **character embeddings and CRF decoding** improves performance over baseline models.

## Dependencies
Install required dependencies with:
```sh
pip install torch numpy tqdm gensim
```

## Future Work
- **Hyperparameter tuning** (hidden size, dropout, optimizer settings)
- **Exploring transformer-based models** (e.g., BERT for token embeddings)
- **Experimenting with more loss functions**

## License
This project is licensed under the MIT License.

---
This README provides an overview of the Bi-LSTM-CRF tagging model, detailing the dataset, model structure, training, evaluation, and future improvements. 🚀

