# cnn_model.py
import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG  = "<STOP>"
PAD_TAG   = "<PAD>"
UNK_TAG   = "<UNK>"

def log_sum_exp_batched(tensor, dim: int):
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
    stable = tensor - max_val
    lse = max_val + torch.log(torch.sum(torch.exp(stable), dim=dim, keepdim=True))
    return lse.squeeze(dim)


class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        word_vocab_size,
        char_vocab_size,
        tag_to_ix,
        word_emb_dim,
        char_emb_dim,
        char_cnn_out_dim,
        hidden_dim
    ):
        super().__init__()
        self.tag_to_ix        = tag_to_ix
        self.tagset_size      = len(tag_to_ix)
        self.word_emb_dim     = word_emb_dim
        self.char_emb_dim     = char_emb_dim
        self.char_cnn_out_dim = char_cnn_out_dim
        self.hidden_dim       = hidden_dim

        # Embeddings
        self.word_embeds = nn.Embedding(word_vocab_size, word_emb_dim)
        self.char_embeds = nn.Embedding(char_vocab_size, char_emb_dim)

        # Char CNN
        self.char_cnn = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=char_cnn_out_dim,
            kernel_size=3,
            padding=1
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=(word_emb_dim + char_cnn_out_dim),
            hidden_size=(hidden_dim // 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF transitions
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # Disallow transitions into START, out of STOP
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]]  = -10000

        # Optionally disallow transitions to/from PAD
        if PAD_TAG in self.tag_to_ix:
            pad_idx = self.tag_to_ix[PAD_TAG]
            self.transitions.data[pad_idx, :] = -10000
            self.transitions.data[:, pad_idx] = -10000

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(2, batch_size, self.hidden_dim // 2, device=device),
            torch.zeros(2, batch_size, self.hidden_dim // 2, device=device)
        )

    def _get_lstm_features(self, word_idxs, char_idxs):
        B, S = word_idxs.shape
        device = word_idxs.device

        word_emb = self.word_embeds(word_idxs)  # (B, S, word_emb_dim)
        char_emb = self.char_embeds(char_idxs)  # (B, S, C, char_emb_dim)

        # Flatten for CNN: (B*S, C, char_emb_dim)
        BSC = char_emb.size()
        char_emb = char_emb.view(B * S, BSC[2], BSC[3]).permute(0, 2, 1)  # => (B*S, char_emb_dim, C)

        # CNN => (B*S, char_cnn_out_dim, C)
        char_cnn_out = torch.relu(self.char_cnn(char_emb))
        # max-pool over char dimension
        char_cnn_out, _ = torch.max(char_cnn_out, dim=2)  # => (B*S, char_cnn_out_dim)

        # Reshape back to (B, S, char_cnn_out_dim)
        char_cnn_out = char_cnn_out.view(B, S, self.char_cnn_out_dim)

        combined_emb = torch.cat([word_emb, char_cnn_out], dim=2)  # (B, S, word_emb_dim + char_cnn_out_dim)

        hidden = self.init_hidden(B)
        lstm_out, _ = self.lstm(combined_emb, hidden)  # => (B, S, hidden_dim)
        feats = self.hidden2tag(lstm_out)              # => (B, S, tagset_size)
        return feats

    def _forward_alg_batched(self, feats, mask):
        """
        feats: (B, S, T)
        mask: (B, S) 1=valid, 0=pad
        Returns log-sum-exp of all paths, shape (B,).
        """
        B, S, T = feats.shape
        device = feats.device

        alpha = torch.full((B, T), -10000.0, device=device)
        alpha[:, self.tag_to_ix[START_TAG]] = 0.0

        for i in range(S):
            # only update where mask is 1
            valid_mask = mask[:, i].unsqueeze(1)  # (B, 1)
            emit_scores = feats[:, i].unsqueeze(2)  # (B, T, 1)
            trans = self.transitions.unsqueeze(0)   # (1, T, T)

            # shape => (B, T, T)
            next_tag_var = alpha.unsqueeze(2) + trans + emit_scores
            # log-sum-exp over dim=1 => (B, T)
            new_alpha = torch.max(next_tag_var, dim=1).values

            # blend old alpha and new alpha based on whether it's a valid position
            alpha = torch.where(valid_mask.bool(), new_alpha, alpha)

        # Add transition to STOP
        stop_idx = self.tag_to_ix[STOP_TAG]
        alpha = alpha + self.transitions[stop_idx].unsqueeze(0)
        return log_sum_exp_batched(alpha, dim=1)  # (B,)

    def _score_sentence_batched(self, feats, tags, mask):
        """
        feats: (B, S, T)
        tags:  (B, S)
        mask:  (B, S)
        Return shape (B,).
        """
        B, S, T = feats.shape
        score = torch.zeros(B, device=feats.device)

        start_idx = torch.full((B, 1), self.tag_to_ix[START_TAG], dtype=torch.long, device=feats.device)
        # prepend START to each path
        tags = torch.cat([start_idx, tags], dim=1)  # (B, S+1)

        for i in range(S):
            valid_mask = mask[:, i]
            curr_tag = tags[:, i]
            next_tag = tags[:, i+1]

            # only update score where mask=1
            # emission
            emit = feats[torch.arange(B), i, next_tag]  # (B,)
            trans = self.transitions[next_tag, curr_tag]  # (B,)

            # add to score only if valid
            step_score = emit + trans
            score += step_score * valid_mask

        # add transition to STOP
        stop_idx = self.tag_to_ix[STOP_TAG]
        last_tag = tags[torch.arange(B), mask.sum(dim=1)]  # last actual tag index (where mask=1 ends)
        # some seq might have length=0, clamp for safety
        last_tag = torch.clamp(last_tag, 0, self.tagset_size-1)

        score += self.transitions[stop_idx, last_tag]
        return score

    def _viterbi_decode_batched(self, feats, mask):
        """
        feats: (B, S, T)
        mask:  (B, S)
        Return (best_scores, best_paths):
          best_scores: (B,)
          best_paths:  (B, S)  (includes arbitrary tags in padded positions, up to you to ignore)
        """
        B, S, T = feats.shape
        device = feats.device

        alpha = torch.full((B, T), -10000.0, device=device)
        alpha[:, self.tag_to_ix[START_TAG]] = 0.0
        backpointers = []

        for i in range(S):
            emit = feats[:, i].unsqueeze(1)  # (B, 1, T)
            next_tag_var = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit
            best_tag_id = torch.argmax(next_tag_var, dim=1)  # (B, T)
            backpointers.append(best_tag_id)

            # pick alpha
            new_alpha = torch.max(next_tag_var, dim=1).values  # (B, T)
            valid_mask = mask[:, i].unsqueeze(1)  # (B, 1)
            alpha = torch.where(valid_mask.bool(), new_alpha, alpha)

        # Add transition to STOP
        stop_idx = self.tag_to_ix[STOP_TAG]
        alpha = alpha + self.transitions[stop_idx].unsqueeze(0)
        best_scores, best_last_tags = torch.max(alpha, dim=1)

        # Backtrack
        best_paths = []
        for b in range(B):
            # figure out how long this sequence actually is
            length_b = mask[b].sum().item()  # number of valid tokens
            backpointers_b = [bp[b] for bp in backpointers]  # list of shape S, each T-dim
            # start from best_last_tags[b]
            best_tag = best_last_tags[b].item()
            path = []
            # we only backtrack length_b steps
            for i in range(S-1, -1, -1):
                if i >= length_b:
                    # if the sequence is shorter, we skip
                    continue
                path.append(best_tag)
                best_tag = backpointers_b[i][best_tag].item()
            path.reverse()
            # pad to length S if needed
            if len(path) < S:
                path.extend([self.tag_to_ix[PAD_TAG]] * (S - len(path))) 
            best_paths.append(path)

        best_paths_tensor = torch.tensor(best_paths, device=device, dtype=torch.long)
        return best_scores, best_paths_tensor

    def neg_log_likelihood(self, word_idxs, char_idxs, tags, mask):
        """
        Returns average negative log-likelihood over the batch
        """
        feats = self._get_lstm_features(word_idxs, char_idxs)
        forward_score = self._forward_alg_batched(feats, mask)
        gold_score = self._score_sentence_batched(feats, tags, mask)
        # sum of negative log-likelihood for each sequence
        nll = (forward_score - gold_score)
        return nll.mean()

    def forward(self, word_idxs, char_idxs, mask):
        """
        Returns best_scores, best_paths
        """
        feats = self._get_lstm_features(word_idxs, char_idxs)
        best_scores, best_paths = self._viterbi_decode_batched(feats, mask)
        return best_scores, best_paths
