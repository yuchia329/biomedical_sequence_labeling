import random
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gensim.downloader as api


class CRF(nn.Module):
    """
    A simple CRF layer for BIO tagging.
    """

    def __init__(self, num_tags, pad_idx=0, loss_function='nll'):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.pad_idx = pad_idx
        self.loss_function = loss_function

        # Transition matrix: transition_scores[i, j] = score of transitioning from i -> j
        self.transition_scores = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, emissions, tags, mask=None):
        """
        Computes the negative log-likelihood (loss) of the given sequence of tags.
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        
        if self.loss_function == "nll":
            # Standard CRF negative log-likelihood
            log_numerator = self._compute_joint_log_likelihood(emissions, tags, mask)
            log_denominator = self._compute_log_partition_function(emissions, mask)
            return torch.mean(log_denominator - log_numerator)
        elif self.loss_function == "svm":
            # Structured hinge (SVM-like) loss
            return self.structured_hinge_loss(emissions, tags, mask)
        else:
            self.softmax_margin_loss(emissions, tags, mask)

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
    
    def softmax_margin_loss(self, emissions, gold_tags, mask):
        """
        Softmax margin = log Z_cost - score(gold)
        where Z_cost is the partition function that includes a per-token cost
        if the predicted tag is not the gold tag. That is:
        
            cost(t, j) = 1 if j != gold_tags[t], else 0

        This cost is added into the potential for tag j at time t.
        """
        gold_score = self._compute_joint_log_likelihood(emissions, gold_tags, mask)
        log_partition_cost = self._compute_log_partition_function_with_cost(emissions, gold_tags, mask)
        return torch.mean(log_partition_cost - gold_score)
    
    def _compute_log_partition_function_with_cost(self, emissions, gold_tags, mask):
        """
        Forward (alpha) recursion, but each potential includes +1 
        if the current tag != gold_tag at time t.
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device

        # At t=0, alpha = emissions[:, 0] plus cost(0, j).
        # cost(0, j) = 1 if j != gold_tags[:,0], else 0.
        alpha = emissions[:, 0].clone()  # shape: (batch_size, num_tags)
        cost_0 = torch.zeros_like(alpha)
        for b_idx in range(batch_size):
            gold_j0 = gold_tags[b_idx, 0]
            for j in range(num_tags):
                if j != gold_j0:
                    cost_0[b_idx, j] = 1.0  # mismatch => +1
        alpha += cost_0  # incorporate cost at time 0

        for t in range(1, seq_len):
            mask_t = mask[:, t].unsqueeze(1)  # (batch_size, 1)

            # Build the cost at time t
            cost_t = torch.zeros(batch_size, num_tags, device=device)
            for b_idx in range(batch_size):
                gold_jt = gold_tags[b_idx, t]
                cost_t[b_idx, gold_jt] = 0.0
                for j in range(num_tags):
                    if j != gold_jt:
                        cost_t[b_idx, j] = 1.0

            # For each j in [0..num_tags-1]:
            #   alpha_t[j] = logsumexp( alpha[i] + transition[i->j] + emissions[t,j] + cost_t[j] )
            emit_t = emissions[:, t].unsqueeze(2)  # shape: (batch_size, num_tags, 1)
            cost_t_unsq = cost_t.unsqueeze(2)      # shape: (batch_size, num_tags, 1)
            alpha_unsq = alpha.unsqueeze(2)        # (batch_size, num_tags, 1)
            trans_unsq = self.transition_scores.unsqueeze(0)  # (1, num_tags, num_tags)

            # shape: (batch_size, num_tags, num_tags)
            alpha_t = alpha_unsq + trans_unsq + emit_t + cost_t_unsq

            # logsumexp over dimension=1 => sum over previous tag i
            alpha_t = torch.logsumexp(alpha_t, dim=1)

            # If mask_t=0, we keep alpha the same
            alpha = torch.where(mask_t.bool(), alpha_t, alpha)

        # Finally log-sum-exp over the last alpha
        return torch.logsumexp(alpha, dim=1)
    
    def structured_hinge_loss(self, emissions, gold_tags, mask):
        """
        Computes the average structured hinge loss over the batch.

        hinge = max(0, [score(y_hat) + cost(y_hat, gold)] - score(gold))
        where y_hat = argmax (score(y) + cost(y, gold)).
        
        cost(y, gold) is typically the number of mismatched tags (Hamming distance).
        """
        # Score of the gold sequence
        gold_score = self._compute_joint_log_likelihood(emissions, gold_tags, mask)

        # For each sequence in the batch, we find:
        #  y_hat = argmax_y [score(x,y) + cost(y, gold)]
        # We'll do that by a special Viterbi with an added cost for each token
        best_paths = self._loss_augmented_viterbi_decode(emissions, gold_tags, mask)

        # Now compute [score(y_hat) + cost(y_hat, gold)] - score(gold)
        # Then hinge = max(0, that difference).
        margin_losses = []
        for b_idx, y_hat in enumerate(best_paths):
            seq_len_b = mask[b_idx].sum().item()
            # cost = number of mismatched tags
            cost_b = 0
            for t in range(seq_len_b):
                if y_hat[t] != gold_tags[b_idx, t].item():
                    cost_b += 1

            # Score(y_hat)
            score_hat_b = self._score_one_sequence(emissions[b_idx], y_hat, mask[b_idx])

            margin = (score_hat_b + cost_b) - gold_score[b_idx]
            hinge = torch.clamp(margin, min=0.0)
            margin_losses.append(hinge)

        margin_losses = torch.stack(margin_losses)  # shape [batch_size]
        return margin_losses.mean()
    
    def _loss_augmented_viterbi_decode(self, emissions, gold_tags, mask):
        """
        A Viterbi pass that adds +1 to the emission score if the predicted tag 
        is different from the gold tag, effectively 'rewarding' errors so 
        that the best path is the one that violates the margin the most.
        
        We'll interpret that extra +1 as the cost being integrated into 
        the dynamic program: at time t, if j != gold_tags[t], add +1.
        """
        batch_size, seq_len, num_tags = emissions.size()
        # We'll build dp similarly to _viterbi_decode, but each emission
        # is augmented with +1 if j != gold_tags[b_idx, t].
        device = emissions.device

        # Prepare the DP
        dp = torch.zeros(batch_size, num_tags, device=device)
        for b_idx in range(batch_size):
            # Time 0 initial scores
            j_range = torch.arange(num_tags, device=device)
            gold_j0 = gold_tags[b_idx, 0]
            cost_0 = (j_range != gold_j0).float()  # +1 if mismatch
            dp[b_idx] = emissions[b_idx, 0] + cost_0

        backpointers = []

        # Recurrence
        for t in range(1, seq_len):
            dp_t = []
            backpointer_t = []
            for j in range(num_tags):
                # cost if we pick j at time t
                cost_j = []
                for b_idx in range(batch_size):
                    if j != gold_tags[b_idx, t].item():
                        cost_j.append(1.0)
                    else:
                        cost_j.append(0.0)
                cost_j = torch.tensor(cost_j, device=device)

                # Score from each previous tag i to j plus cost
                # dp[:, i] has shape (batch_size), so score_ij will be (batch_size, num_tags)
                score_ij = dp + self.transition_scores[:, j] + emissions[:, t, j].unsqueeze(1)
                # Add the mismatch cost
                score_ij = score_ij + cost_j.unsqueeze(1)

                best_score, best_tag = score_ij.max(dim=1)
                dp_t.append(best_score)
                backpointer_t.append(best_tag)

            dp = torch.stack(dp_t, dim=1)
            backpointers.append(torch.stack(backpointer_t, dim=1))

            mask_t = mask[:, t].unsqueeze(1)
            dp = torch.where(mask_t.bool(), dp, dp.clone().fill_(0))

        # Backtrack
        best_paths = []
        for b_idx in range(batch_size):
            # find best final
            last_scores = dp[b_idx]  # shape [num_tags]
            best_tag = torch.argmax(last_scores).item()

            b_backpointers = []
            for t in reversed(backpointers):
                b_backpointers.append(t[b_idx])
            path = [best_tag]
            for t_bp in b_backpointers:
                best_tag = t_bp[best_tag].item()
                path.append(best_tag)
            path.reverse()

            # We only want the path up to the actual seq length
            seq_len_b = mask[b_idx].sum().item()
            path = path[:seq_len_b]
            best_paths.append(path)

        return best_paths

    def _score_one_sequence(self, emissions_b, tag_seq, mask_b):
        """
        Score single sequence (emission + transitions), ignoring the margin cost.
        emissions_b: shape [seq_len, num_tags]
        tag_seq: list of predicted tags
        mask_b: shape [seq_len] of 0/1
        """
        seq_len = mask_b.sum().item()
        score = 0.0
        for i in range(seq_len):
            # emission
            score += emissions_b[i, tag_seq[i]]
            # transition
            if i > 0:
                score += self.transition_scores[tag_seq[i-1], tag_seq[i]]
        return score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str, help='Path to training data TXT')
    parser.add_argument('val_file', type=str, help='Path to validation data TXT')
    parser.add_argument('test_file', type=str, help='Path to test data TXT')
    parser.add_argument('output_file', type=str, help='Path to save predictions TXT')
    parser.add_argument('--loss_function', dest='nll', type=str, help='Use negative log-likelihood (nll), structured hinge (svm), or softmax margin (sm) loss')
    return parser.parse_args()


def build_vocabularies(training_data):
    """
    Build simple word_vocab and tag_vocab from the training data.
    Also build a character vocabulary for CharCNN.
    """
    word_vocab = {"<PAD>": 0, "<UNK>": 1}
    tag_vocab = {"<PAD>": 0}
    char_vocab = {"<PAD>": 0, "<UNK>": 1}  # For unknown chars
    word_idx = 2
    tag_idx = 1
    char_idx = 2

    for sentence in training_data:
        words = sentence[0]
        tags = sentence[1]
        for word in words:
            # Lowercase for consistency
            w_lower = word.lower()
            if w_lower not in word_vocab:
                word_vocab[w_lower] = word_idx
                word_idx += 1

            # Build character-level vocab
            for ch in w_lower:
                if ch not in char_vocab:
                    char_vocab[ch] = char_idx
                    char_idx += 1

        for tag in tags:
            if tag not in tag_vocab:
                tag_vocab[tag] = tag_idx
                tag_idx += 1

    return word_vocab, tag_vocab, char_vocab


def sentence_to_indices(words, word_vocab, char_vocab):
    """
    Returns:
      word_indices: list of word indices
      char_indices: list of [char indices] for each word
    """
    word_indices = []
    char_indices = []
    for word in words:
        w_lower = word.lower()
        w_id = word_vocab.get(w_lower, word_vocab["<UNK>"])
        word_indices.append(w_id)

        # Character indices
        chars = []
        for ch in w_lower:
            ch_id = char_vocab.get(ch, char_vocab["<UNK>"])
            chars.append(ch_id)
        char_indices.append(chars)

    return word_indices, char_indices


def tags_to_indices(tags, tag_vocab):
    return [tag_vocab[tag] for tag in tags]


class GloVeFeatureProcessor:
    """
    Use only GloVe embeddings (no CountVectorizer) for word-level representation.
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
    def __init__(self, data, word_vocab, tag_vocab, char_vocab, feature_processor):
        self.data = data
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab
        self.feature_processor = feature_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words_tags_set = self.data[idx]
        words = words_tags_set[0]
        tags = words_tags_set[1]

        word_indices, char_indices = sentence_to_indices(words, self.word_vocab, self.char_vocab)
        tag_indices = tags_to_indices(tags, self.tag_vocab)

        # Build word-level feature vectors from GloVe
        features = []
        for word in words:
            glove_vector = self.feature_processor.get_glove_feature(word)
            features.append(glove_vector)
        features_np = np.array(features, dtype=np.float32)
        tag_indices_np = np.array(tag_indices, dtype=np.int64)

        return {
            'features': torch.from_numpy(features_np),   # shape: [seq_len, glove_dim]
            'tags': torch.from_numpy(tag_indices_np),    # shape: [seq_len]
            'lengths': len(words),
            'char_indices': char_indices,                # list of lists
        }


class CharCNN(nn.Module):
    """
    Module that takes a batch of character indices and produces
    a fixed-size vector (via Conv1D + max-pooling) for each token.
    """
    def __init__(self, num_chars, char_emb_dim=30, conv_filters=50, kernel_size=3, pad_idx=0):
        super(CharCNN, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.num_chars = num_chars
        self.char_embedding = nn.Embedding(num_chars, char_emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=conv_filters,
            kernel_size=kernel_size,
            padding=1
        )
        self.conv_filters = conv_filters

    def forward(self, char_indices_batch):
        """
        char_indices_batch: (batch_size, seq_len, max_char_len)
          after collate_fn's zero-padding in the character dimension.

        Returns:
          A tensor of shape (batch_size, seq_len, conv_filters)
        """
        # Shape: (batch_size, seq_len, max_char_len, char_emb_dim)
        emb = self.char_embedding(char_indices_batch)

        # We need to reshape to (batch_size*seq_len, char_emb_dim, max_char_len)
        batch_size, seq_len, max_char_len, _ = emb.size()
        emb = emb.view(batch_size * seq_len, max_char_len, self.char_emb_dim)

        # Convert to (batch_size*seq_len, char_emb_dim, max_char_len) for Conv1d
        emb = emb.transpose(1, 2)

        # Apply 1D convolution
        conv_out = self.conv(emb)  # shape: (batch_size*seq_len, conv_filters, max_char_len)

        # Non-linear activation
        conv_out = F.relu(conv_out)

        # Max pool over the time (char) dimension => (batch_size*seq_len, conv_filters)
        pooled = torch.max(conv_out, dim=2)[0]

        # Reshape back to (batch_size, seq_len, conv_filters)
        pooled = pooled.view(batch_size, seq_len, self.conv_filters)
        return pooled


class LSTMTagger(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        num_tags,
        char_vocab_size,
        char_emb_dim=30,
        char_conv_filters=50,
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
        pad_idx=0,
        loss_function='nll',
    ):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Character-level CNN
        self.char_cnn = CharCNN(
            num_chars=char_vocab_size,
            char_emb_dim=char_emb_dim,
            conv_filters=char_conv_filters,
            kernel_size=3,  # you can tune this
            pad_idx=0
        )

        # The final embedding size for each token is [word-emb-dim + char-cnn-dim]
        self.combined_dim = feature_dim + char_conv_filters

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.combined_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.hidden2tag = nn.Linear(lstm_output_dim, num_tags)
        self.layer_norm = nn.LayerNorm(self.combined_dim)

        # CRF for sequence decoding
        self.crf = CRF(num_tags=num_tags, pad_idx=pad_idx, loss_function=loss_function)

    def forward(self, word_features, char_indices, lengths, tags=None):
        """
        Inputs:
          word_features: (batch_size, seq_len, word_emb_dim)
          char_indices: (batch_size, seq_len, max_char_len)
          lengths: (batch_size)
          tags: (batch_size, seq_len) or None
        """
        # 1) Compute character-level embeddings
        char_emb = self.char_cnn(char_indices)  # (batch_size, seq_len, char_conv_filters)

        # 2) Concat with word-level GloVe embeddings
        combined_features = torch.cat([word_features, char_emb], dim=-1)
        combined_features = self.layer_norm(combined_features)

        # 3) Pack and run BiLSTM
        packed_features = nn.utils.rnn.pack_padded_sequence(
            combined_features, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        lstm_out, _ = self.lstm(packed_features)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.dropout(lstm_out)

        # 4) Compute emissions for CRF
        emissions = self.hidden2tag(lstm_out)

        # 5) Build mask from lengths
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
        for batch in tqdm(train_dataloader):
            features = batch['features'].to(device)      # (B, T, glove_dim)
            tags = batch['tags'].to(device)              # (B, T)
            lengths = batch['lengths']
            char_indices = batch['char_indices'].to(device)  # (B, T, max_char_len)

            optimizer.zero_grad()
            loss = model(features, char_indices, lengths, tags=tags)
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
                char_indices = batch['char_indices'].to(device)

                val_loss = model(features, char_indices, lengths, tags=tags)
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


def create_and_train_model(
    train_dataloader,
    dev_dataloader,
    feature_processor,
    tag_vocab,
    char_vocab,
    params,
    device='cuda',
    loss_function="nll",
):
    hidden_dim = params.get('hidden_dim', 256)
    num_layers = params.get('num_layers', 2)
    dropout = params.get('dropout', 0.5)
    bidirectional = params.get('bidirectional', True)
    char_emb_dim = params.get('char_emb_dim', 30)
    char_conv_filters = params.get('char_conv_filters', 50)

    model = LSTMTagger(
        feature_dim=feature_processor.feature_dim,
        hidden_dim=hidden_dim,
        num_tags=len(tag_vocab),
        char_vocab_size=len(char_vocab),
        char_emb_dim=char_emb_dim,
        char_conv_filters=char_conv_filters,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        loss_function=loss_function
    )

    train_model(model, train_dataloader, dev_dataloader, params, device=device)
    return model


def collate_fn(batch):
    """
    Sort by descending lengths for pack_padded_sequence, then pad (for words and chars).
    """
    # Sort batch: largest sequence first
    batch = sorted(batch, key=lambda x: x['lengths'], reverse=True)

    features = [x['features'] for x in batch]
    tags = [x['tags'] for x in batch]
    lengths = [x['lengths'] for x in batch]
    char_indices = [x['char_indices'] for x in batch]  # list of (seq_len, variable_char_len)

    # Pad word features and tags
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True)
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True)

    # Now we also need to pad char_indices:
    # We need them in a shape: (batch, max_seq_len, max_char_len)
    max_seq_len = max(len(ci) for ci in char_indices)  # among all sentences
    max_char_len = 0
    for ci in char_indices:
        # ci is shape [seq_len, variable_char_len], find the max within this sentence
        local_max = max(len(chars_of_token) for chars_of_token in ci)
        if local_max > max_char_len:
            max_char_len = local_max

    # Build an empty tensor for the padded char indices
    padded_char_indices = torch.zeros(
        len(batch), max_seq_len, max_char_len, dtype=torch.long
    )

    for i, ci in enumerate(char_indices):
        for j, token_chars in enumerate(ci):
            # token_chars is the list of char indices for the j-th token
            length_of_this_token = len(token_chars)
            padded_char_indices[i, j, :length_of_this_token] = torch.LongTensor(token_chars)

    return {
        'features': padded_features,           # (B, T, glove_dim)
        'tags': padded_tags,                   # (B, T)
        'lengths': torch.tensor(lengths),      # (B)
        'char_indices': padded_char_indices,   # (B, T, max_char_len)
    }


def getSentencesAndTags(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        sentences = []
        current_tokens = []
        current_tags = []
        for line in lines:
            items = line.split("\t")
            if len(items) == 2:  # Non-empty line
                text = items[0]
                tag = items[1].rstrip("\n")
                current_tokens.append(text)
                current_tags.append(tag)
            else:
                # Blank line => end of sentence
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
        if current_tokens:
            sentences.append((current_tokens, current_tags))
    return sentences


def prepare_data(train_path, val_path, params):
    """
    Prepare DataLoaders that produce both word embeddings and char embeddings.
    """
    batch_size = params.get('batch_size', 32)

    train_data = getSentencesAndTags(train_path)
    val_data = getSentencesAndTags(val_path)

    # Build vocabs
    word_vocab, tag_vocab, char_vocab = build_vocabularies(train_data)
    feature_processor = GloVeFeatureProcessor(
        embedding_dim=params.get('embedding_dim', 100)
    )

    train_dataset = GloVeSlotTaggingDataset(train_data, word_vocab, tag_vocab, char_vocab, feature_processor)
    val_dataset = GloVeSlotTaggingDataset(val_data, word_vocab, tag_vocab, char_vocab, feature_processor)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader, word_vocab, tag_vocab, char_vocab, feature_processor


def generate_submission(
    model,
    test_path,
    feature_processor,
    word_vocab,
    tag_vocab,
    char_vocab,
    output_path='output.txt',
    device='cuda'
):
    model.eval()
    model = model.to(device)
    test_data = getSentencesAndTags(test_path)
    idx_to_tag = {idx: tag for tag, idx in tag_vocab.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        for words, tags in tqdm(test_data):
            # Build word features
            features = []
            char_indices = []
            for word in words:
                glove_vector = feature_processor.get_glove_feature(word)
                features.append(glove_vector)

                # build char idx for this token
                w_lower = word.lower()
                chars = [char_vocab.get(c, char_vocab['<UNK>']) for c in w_lower]
                char_indices.append(chars)

            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            # We must pad char_indices similarly
            max_char_len = max(len(ci) for ci in char_indices) if char_indices else 1
            padded_char_indices = torch.zeros(
                1, len(words), max_char_len, dtype=torch.long
            )
            for j, token_chars in enumerate(char_indices):
                padded_char_indices[0, j, :len(token_chars)] = torch.LongTensor(token_chars)

            padded_char_indices = padded_char_indices.to(device)
            lengths = torch.LongTensor([len(words)])

            with torch.no_grad():
                best_paths = model(features_tensor, padded_char_indices, lengths, tags=None)
                predicted_indices = best_paths[0]

            predicted_tags = [idx_to_tag[i] for i in predicted_indices]
            for w, t in zip(words, predicted_tags):
                f.write(f"{w}\t{t}\n")
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

        # Character CNN hyperparams
        'char_emb_dim': 30,
        'char_conv_filters': 50,
    }

    train_dataloader, val_dataloader, word_vocab, tag_vocab, char_vocab, feature_processor = prepare_data(
        args.train_file, args.val_file, params
    )

    # Choose device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    loss_function = args.loss_function
    
    model = create_and_train_model(
        train_dataloader,
        val_dataloader,
        feature_processor,
        tag_vocab,
        char_vocab,
        params,
        device=device,
        loss_function='nll'
    )

    generate_submission(
        model=model,
        test_path=args.test_file,
        feature_processor=feature_processor,
        word_vocab=word_vocab,
        tag_vocab=tag_vocab,
        char_vocab=char_vocab,
        output_path=args.output_file,
        device=device
    )
