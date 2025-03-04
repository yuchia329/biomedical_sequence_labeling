import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import argparse

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

import gensim.downloader as api
import torch.nn.functional as F


class CRF(nn.Module):
    """
    A simple CRF layer for BIO tagging.
    """

    def __init__(self, num_tags, pad_idx=0):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx

        # Transition matrix: transition_scores[i, j] = score of transitioning from i -> j
        self.transition_scores = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, emissions, tags, mask=None):
        """
        Computes the negative log-likelihood (loss) of the given sequence of tags.
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        log_numerator = self._compute_joint_log_likelihood(emissions, tags, mask)
        log_denominator = self._compute_log_partition_function(emissions, mask)
        return torch.mean(log_denominator - log_numerator)

    def decode(self, emissions, mask=None):
        """
        Viterbi decode to get the best tag sequence for each batch.
        """
        if mask is None:
            mask = torch.ones(emissions.size(0), emissions.size(1), 
                              dtype=torch.uint8, device=emissions.device)
        return self._viterbi_decode(emissions, mask)

    def _compute_joint_log_likelihood(self, emissions, tags, mask):
        batch_size, seq_len, num_tags = emissions.shape
        score = torch.zeros(batch_size, device=emissions.device)

        for i in range(seq_len):
            is_first_token = (i == 0)
            prev_tag = tags[:, i-1] if not is_first_token else None
            curr_tag = tags[:, i]
            mask_i = mask[:, i].float()

            emission_scores_i = emissions[:, i, :]
            score += emission_scores_i.gather(1, curr_tag.unsqueeze(1)).squeeze(1) * mask_i

            if not is_first_token:
                transition_scores = self.transition_scores[prev_tag, curr_tag]
                score += transition_scores * mask_i

        return score

    def _compute_log_partition_function(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        alpha = emissions[:, 0]  # shape (batch_size, num_tags)

        for t in range(1, seq_len):
            mask_t = mask[:, t].unsqueeze(1)  # (batch_size, 1)
            emit_t = emissions[:, t].unsqueeze(2)  # (batch_size, num_tags, 1)

            alpha_t = alpha.unsqueeze(2) + self.transition_scores.unsqueeze(0) + emit_t
            alpha_t = torch.logsumexp(alpha_t, dim=1)

            alpha = torch.where(mask_t.bool(), alpha_t, alpha)

        return torch.logsumexp(alpha, dim=1)

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.size()
        dp = emissions[:, 0]
        backpointers = []

        for t in range(1, seq_len):
            dp_t = []
            backpointer_t = []
            for j in range(num_tags):
                score_ij = dp + self.transition_scores[:, j] + emissions[:, t, j].unsqueeze(1)
                best_score, best_tag = score_ij.max(dim=1)
                dp_t.append(best_score)
                backpointer_t.append(best_tag)

            dp = torch.stack(dp_t, dim=1)
            backpointers.append(torch.stack(backpointer_t, dim=1))

            mask_t = mask[:, t].unsqueeze(1)
            dp = torch.where(mask_t.bool(), dp, dp.clone().fill_(0))

        best_score, best_tag = dp.max(dim=1)
        best_paths = []
        seq_len_range = range(seq_len-1, 0, -1)

        for b_idx in range(batch_size):
            path = [best_tag[b_idx].item()]
            bp = backpointers[::-1]
            current_tag = best_tag[b_idx]
            for back_t in bp:
                current_tag = back_t[b_idx, current_tag]
                path.append(current_tag.item())
            path.reverse()
            best_paths.append(path)

        return best_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Path to training data TXT')
    parser.add_argument('val_file', type=str, help='Path to validation data TXT')
    parser.add_argument('test_file', type=str, help='Path to test data TXT')
    parser.add_argument('output_file', type=str, help='Path to save predictions TXT')
    return parser.parse_args()


def build_vocabularies(training_data):
    """
    Build simple word_vocab and tag_vocab from the training data.
    """
    word_vocab = {"<PAD>": 0, "<UNK>": 1}
    tag_vocab = {"<PAD>": 0}
    word_idx = 2
    tag_idx = 1

    for sentence in training_data:
        words = sentence[0]
        tags = sentence[1]
        for word in words:
            word = word.lower()
            if word not in word_vocab:
                word_vocab[word] = word_idx
                word_idx += 1
        for tag in tags:
            if tag not in tag_vocab:
                tag_vocab[tag] = tag_idx
                tag_idx += 1

    return word_vocab, tag_vocab


def sentence_to_indices(words, word_vocab):
    indices = []
    for word in words:
        word = word.lower()
        idx = word_vocab.get(word, word_vocab["<UNK>"])
        indices.append(idx)
    return indices, words


def tags_to_indices(tags, tag_vocab):
    return [tag_vocab[tag] for tag in tags]


class GloVeFeatureProcessor:
    """
    Use only GloVe embeddings (no CountVectorizer).
    """
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        print("Loading GloVe embeddings from gensim...")
        self.glove_embeddings = {}
        word_vecs = api.load("glove-wiki-gigaword-100")  # or 300 if desired
        for word in word_vecs.index_to_key:
            self.glove_embeddings[word] = word_vecs[word]

        # For padding and unknown tokens
        self.glove_embeddings['<PAD>'] = np.zeros(embedding_dim)
        self.glove_embeddings['<UNK>'] = np.mean(list(self.glove_embeddings.values()), axis=0)
        print(f"Loaded {len(self.glove_embeddings)} GloVe embeddings")

    def get_glove_feature(self, token):
        return self.glove_embeddings.get(token.lower(), self.glove_embeddings['<UNK>'])

    @property
    def feature_dim(self):
        return self.embedding_dim


class GloVeSlotTaggingDataset(Dataset):
    def __init__(self, data, word_vocab, tag_vocab, feature_processor):
        self.data = data
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.feature_processor = feature_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words_tags_set = self.data[idx]
        words = words_tags_set[0]
        tags = words_tags_set[1]

        word_indices, words = sentence_to_indices(words, self.word_vocab)
        tag_indices = tags_to_indices(tags, self.tag_vocab)

        # Build feature vectors from GloVe
        features = []
        for word in words:
            glove_vector = self.feature_processor.get_glove_feature(word)
            features.append(glove_vector)
        features_np = np.array(features, dtype=np.float32)
        tag_indices_np = np.array(tag_indices, dtype=np.int64)

        return {
            # 'features': torch.FloatTensor(features),
            'features': torch.from_numpy(features_np),
            # 'tags': torch.LongTensor(tag_indices),
            'tags': torch.from_numpy(tag_indices_np),
            'lengths': len(words)
        }


class LSTMTagger(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_tags, num_layers=2,
                 dropout=0.5, bidirectional=True, pad_idx=0):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.hidden2tag = nn.Linear(lstm_output_dim, num_tags)
        self.layer_norm = nn.LayerNorm(feature_dim)

        # CRF for sequence decoding
        self.crf = CRF(num_tags=num_tags, pad_idx=pad_idx)

    def forward(self, features, lengths, tags=None):
        """
        If tags is given, return the negative log-likelihood for training.
        Otherwise, return the best Viterbi decode for inference.
        """
        features = self.layer_norm(features)
        packed_features = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed_features)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)

        batch_size, seq_len, _ = emissions.shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.uint8, device=emissions.device)
        for i in range(batch_size):
            mask[i, :lengths[i]] = 1

        if tags is not None:
            # Training: return -log_likelihood
            loss = self.crf(emissions, tags, mask=mask)
            return loss
        else:
            # Inference/decoding: return best paths
            best_paths = self.crf.decode(emissions, mask=mask)
            return best_paths


def train_model(model, train_dataloader, dev_dataloader, params, device='cuda'):
    model = model.to(device)
    learning_rate = params.get('learning_rate', 0.001)
    num_epochs = params.get('num_epochs', 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_dataloader:
            features = batch['features'].to(device)
            tags = batch['tags'].to(device)
            lengths = batch['lengths']
            optimizer.zero_grad()

            loss = model(features, lengths, tags=tags)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in dev_dataloader:
                features = batch['features'].to(device)
                tags = batch['tags'].to(device)
                lengths = batch['lengths']

                val_loss = model(features, lengths, tags=tags)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(dev_dataloader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Training loss:   {avg_train_loss:.4f}")
        print(f"  Validation loss: {avg_val_loss:.4f}")
        print("--------------------------------------------------")


def create_and_train_model(train_dataloader, dev_dataloader, feature_processor, tag_vocab, params, device='cuda'):
    hidden_dim = params.get('hidden_dim', 256)
    num_layers = params.get('num_layers', 2)
    dropout = params.get('dropout', 0.5)
    bidirectional = params.get('bidirectional', True)

    model = LSTMTagger(
        feature_dim=feature_processor.feature_dim,
        hidden_dim=hidden_dim,
        num_tags=len(tag_vocab),
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )
    train_model(model, train_dataloader, dev_dataloader, params, device=device)
    return model


def collate_fn(batch):
    """
    Sort by descending lengths for pack_padded_sequence, then pad.
    """
    batch = sorted(batch, key=lambda x: x['lengths'], reverse=True)
    features = [x['features'] for x in batch]
    tags = [x['tags'] for x in batch]
    lengths = [x['lengths'] for x in batch]

    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True)

    return {
        'features': padded_features,
        'tags': padded_tags,
        'lengths': torch.tensor(lengths)
    }

def getSentencesAndTags(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        sentences = []
        current_tokens = []
        current_tags   = []
        for line in lines:
            items = line.split("\t")
            if len(items) == 2:  # Non-empty line
                text = items[0]
                tag  = items[1].rstrip("\n")
                current_tokens.append(text)
                current_tags.append(tag)
            else:
                # Blank line => end of sentence
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags   = []
        if current_tokens:
            sentences.append((current_tokens, current_tags))
    return sentences


def prepare_data(train_path, val_path, params):
    """
    Now only GloVe is used; no CountVectorizer.
    """
    batch_size = params.get('batch_size', 32)

    train_data = getSentencesAndTags(train_path)
    val_data = getSentencesAndTags(val_path)
    

    word_vocab, tag_vocab = build_vocabularies(train_data)
    feature_processor = GloVeFeatureProcessor(
        embedding_dim=params.get('embedding_dim', 100)
    )

    train_dataset = GloVeSlotTaggingDataset(train_data, word_vocab, tag_vocab, feature_processor)
    val_dataset = GloVeSlotTaggingDataset(val_data, word_vocab, tag_vocab, feature_processor)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader, word_vocab, tag_vocab, feature_processor


def generate_submission(model, test_path, feature_processor, word_vocab, tag_vocab, 
                        output_path='output.txt', device='cuda'):
    model.eval()
    model = model.to(device)
    test_data = getSentencesAndTags(test_path)
    idx_to_tag = {idx: tag for tag, idx in tag_vocab.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        for words, tags in tqdm(test_data):
            features = []

            for word in words:
                glove_vector = feature_processor.get_glove_feature(word)
                features.append(glove_vector)

            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            lengths = torch.LongTensor([len(words)])

            with torch.no_grad():
                best_paths = model(features_tensor, lengths, tags=None)
                predicted_indices = best_paths[0]

            predicted_tags = [idx_to_tag[i] for i in predicted_indices]
            for word, tag in zip(words, predicted_tags):
                f.write(f"{word}\t{tag}\n")

                # Blank line to separate this sentence from the next
            f.write("\n")
    print(f"Saved predictions in CoNLL format to {output_path}")


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    set_seeds(42)

    params = {
        'random_seed': 42,
        'val_split': 0.2,
        'batch_size': 32,

        # GloVe embedding dimension (for "glove-wiki-gigaword-100" = 100)
        'embedding_dim': 100,

        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.5,
        'bidirectional': True,
        'num_epochs': 10,
        'learning_rate': 0.001,
    }

    train_dataloader, val_dataloader, word_vocab, tag_vocab, feature_processor = prepare_data(
        args.train_file, args.val_file, params
    )

    # Choose device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = create_and_train_model(
        train_dataloader, val_dataloader, feature_processor, tag_vocab, params, device=device
    )

    generate_submission(
        model=model,
        test_path=args.test_file,
        feature_processor=feature_processor,
        word_vocab=word_vocab,
        tag_vocab=tag_vocab,
        output_path=args.output_file,
        device=device
    )
