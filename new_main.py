# main.py
import torch
import torch.optim as optim
from model import BiLSTM_CRF
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from load_data import load_data, createTagSets
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Special symbols
START_TAG = "<START>"
STOP_TAG  = "<STOP>"
PAD_TAG   = "<PAD>"
UNK_TAG   = "<UNK>"

torch.manual_seed(1)

# Hyperparameters
EMBEDDING_DIM = 300
HIDDEN_DIM    = 256
BATCH_SIZE    = 128
NUM_EPOCHS    = 1

# File paths (adjust as needed)
train_file_path = "A2-data/train"
vali_file_path  = "A2-data/dev.answers"
test_file_path  = "A2-data/test.answers"

# Load data from disk
training_data   = load_data(train_file_path)
validation_data = load_data(vali_file_path)
test_data       = load_data(test_file_path)

# Create dataset objects (implement createTagSets accordingly)
train_dataset, val_dataset, test_dataset = createTagSets(
    training_data, validation_data, test_data
)

# Ensure these special symbols exist in your dataset’s vocab:
# (If your createTagSets() already adds them, you can skip this.)
if START_TAG not in train_dataset.tag_vocab:
    train_dataset.tag_vocab[START_TAG] = len(train_dataset.tag_vocab)
if STOP_TAG not in train_dataset.tag_vocab:
    train_dataset.tag_vocab[STOP_TAG]  = len(train_dataset.tag_vocab)
if PAD_TAG not in train_dataset.tag_vocab:
    train_dataset.tag_vocab[PAD_TAG]   = len(train_dataset.tag_vocab)

if UNK_TAG not in train_dataset.token_vocab:
    train_dataset.token_vocab[UNK_TAG] = len(train_dataset.token_vocab)

# Simple collate function that pads each batch
def collate_fn(batch):
    # batch is a list of (token_ids, tag_ids) pairs
    token_ids = [item[0] for item in batch]
    tag_ids   = [item[1] for item in batch]

    # Pad sequences to the same length in this batch
    pad_token_idx = train_dataset.token_vocab[PAD_TAG]
    pad_tag_idx   = train_dataset.tag_vocab[PAD_TAG]

    sentences_padded = pad_sequence(
        token_ids, batch_first=True, padding_value=pad_token_idx
    )
    tags_padded = pad_sequence(
        tag_ids, batch_first=True, padding_value=pad_tag_idx
    )
    return sentences_padded, tags_padded

# DataLoaders for training, validation, and test
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader   = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
test_loader  = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# Build the model using the dataset’s vocabulary sizes
model = BiLSTM_CRF(
    vocab_size=len(train_dataset.token_vocab),
    tag_to_ix=train_dataset.tag_vocab,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    device=device
).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

###############################################################################
# Training Loop
###############################################################################
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for sentences_batch, tags_batch in tqdm(train_loader):
        sentences_batch = sentences_batch.to(device)
        tags_batch = tags_batch.to(device)
        model.zero_grad()

        # Because the model’s CRF code is written for single sequences,
        # we handle each item inside the batch in a loop.
        loss_batch = 0.0
        for sentence_tensor, tag_tensor in zip(sentences_batch, tags_batch):
            loss = model.neg_log_likelihood(sentence_tensor, tag_tensor)
            loss_batch += loss

        loss_batch.backward()
        optimizer.step()
        total_loss += loss_batch.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sentences_batch, tags_batch in val_loader:
            sentences_batch = sentences_batch.to(device)
            tags_batch = tags_batch.to(device)
            for sentence_tensor, tag_tensor in zip(sentences_batch, tags_batch):
                loss = model.neg_log_likelihood(sentence_tensor, tag_tensor)
                val_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
          f"Train Loss: {total_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}")

###############################################################################
# Test Loop (simple scoring + printing predicted tag IDs)
###############################################################################
model.eval()
with torch.no_grad():
    for sentences_batch, tags_batch in test_loader:
        sentences_batch = sentences_batch.to(device)
        tags_batch = tags_batch.to(device)
        for sentence_tensor, tag_tensor in zip(sentences_batch, tags_batch):
            score, predicted_tags = model(sentence_tensor)
            print(f"Sentence indices: {sentence_tensor.tolist()}")
            print(f"Predicted tag IDs: {predicted_tags}")
            print(f"Score: {score.item():.4f}")
            print("------")
