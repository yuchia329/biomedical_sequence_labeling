import torch
import torch.optim as optim
from model import BiLSTM_CRF
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from load_data import load_data, createTagSets
from constant import START_TAG, STOP_TAG, PAD_TAG, UNK_TAG

torch.manual_seed(1)


EMBEDDING_DIM = 5
HIDDEN_DIM = 4
UNK_IDX = 0
BATCH_SIZE = 32
NUM_EPOCHS = 1

# file = "A2-data/test.answers"
train_file_path = "A2-data/train"
vali_file_path = "A2-data/dev.answers"
test_file_path = "A2-data/test.answers"
training_data = load_data(train_file_path)
validation_data = load_data(vali_file_path)
test_data = load_data(test_file_path)
train_dataset, val_dataset, test_set = createTagSets(training_data, validation_data, test_data)

def collate_fn(batch):
        token_ids = [item[0] for item in batch]
        tag_ids = [item[1] for item in batch]

        # Pad sequences
        sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=train_dataset.token_vocab[PAD_TAG])
        # sentences_pad.size()  (batch_size, seq_len)
        tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=train_dataset.tag_vocab[PAD_TAG])
        # tags_pad.size()  (batch_size, seq_len)
        return sentences_padded, tags_padded

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


word_to_ix = {UNK_TAG: 0}
tag_to_ix = {START_TAG: 0, STOP_TAG: 1}

for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag_to_ix.get(tag) is None:
            tag_to_ix[tag] = len(tag_to_ix)



model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for sentences, tags in train_loader:
        model.zero_grad()
        loss_batch = 0

        for si, ti in zip(sentences, tags):
            loss = model.neg_log_likelihood(si, ti)
            loss_batch += loss
        loss_batch.backward()
        optimizer.step()
        total_loss += loss_batch.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sentences, tags in val_loader:
            for si, ti in zip(sentences, tags):
                loss = model.neg_log_likelihood(si, ti)
                val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {total_loss:.6f}, Val Loss: {val_loss:.6f}")

model.eval()
with torch.no_grad():
    for sentences, tags in test_loader:
        for si, ti in zip(sentences, tags):
            score, predicted_tags = model(si)
            print(f"Test Sentence: {si}")
            print(f"Predicted Tag IDs: {predicted_tags}")
            print(f"Score: {score.item():.6f}\n")
