# load_data.py
import torch
from torch.utils.data import Dataset
from constant import START_TAG, STOP_TAG, PAD_TAG, UNK_TAG

# Optional special symbols for characters
CHAR_PAD = "<C_PAD>"  # for padding characters
CHAR_UNK = "<C_UNK>"

def load_data(file_path):
    """Reads a file of 'token + TAB + tag' lines, with blank lines delimiting sentences."""
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
    # handle final sentence if file doesn't end with blank line
    if current_tokens:
        sentences.append((current_tokens, current_tags))
    return sentences

def createTagSets(train_data, vali_data, test_data):
    """
    Returns 3 GENIADataset objects for train, val, and test.
    These will share the same token_vocab, char_vocab, and tag_vocab
    (with train building them).
    """
    train_set = GENIADataset(train_data, training=True)
    vali_set  = GENIADataset(
        vali_data,
        token_vocab=train_set.token_vocab,
        tag_vocab=train_set.tag_vocab,
        char_vocab=train_set.char_vocab,
        training=False
    )
    test_set  = GENIADataset(
        test_data,
        token_vocab=train_set.token_vocab,
        tag_vocab=train_set.tag_vocab,
        char_vocab=train_set.char_vocab,
        training=False
    )
    return train_set, vali_set, test_set

class GENIADataset(Dataset):
    def __init__(
        self,
        data,
        token_vocab=None,
        tag_vocab=None,
        char_vocab=None,
        training=True,
        max_char_len=30  # for optional char padding/truncation
    ):
        """
        data: list of (tokens, tags), each 'tokens' is a list of strings, each 'tags' is a list of strings
        token_vocab, tag_vocab, char_vocab: existing vocab dicts if training=False
        training: indicates we may need to build new vocab dictionaries
        max_char_len: how many chars to keep per token (truncate/pad)
        """
        if training:
            # Build fresh vocab from scratch
            self.token_vocab = {START_TAG: 0, STOP_TAG: 1, PAD_TAG: 2, UNK_TAG: 3}
            self.tag_vocab   = {START_TAG: 0, STOP_TAG: 1, PAD_TAG: 2}
            
            # Build character vocab
            self.char_vocab = {CHAR_PAD: 0, CHAR_UNK: 1}  # special char symbols
        else:
            # Reuse existing vocab
            self.token_vocab = token_vocab
            self.tag_vocab   = tag_vocab
            self.char_vocab  = char_vocab

        self.corpus_token_ids = []
        self.corpus_tag_ids   = []
        self.corpus_char_ids  = []  # <-- new: list of 2D char-ID tensors

        for (tokens, tags) in data:
            # Build token IDs
            token_ids = []
            for token in tokens:
                if training and (token not in self.token_vocab):
                    self.token_vocab[token] = len(self.token_vocab)
                token_idx = self.token_vocab.get(token, self.token_vocab[UNK_TAG])
                token_ids.append(token_idx)

            # Build tag IDs
            tag_ids = []
            for t in tags:
                if training and (t not in self.tag_vocab):
                    self.tag_vocab[t] = len(self.tag_vocab)
                tag_idx = self.tag_vocab[t]
                tag_ids.append(tag_idx)

            # Build char IDs
            char_ids_2d = []
            for token in tokens:
                char_ids = []
                for ch in token:
                    # if training, add to vocab if new
                    if training and (ch not in self.char_vocab):
                        self.char_vocab[ch] = len(self.char_vocab)
                    cid = self.char_vocab.get(ch, self.char_vocab[CHAR_UNK])
                    char_ids.append(cid)
                # Truncate or pad char_ids
                if len(char_ids) > max_char_len:
                    char_ids = char_ids[:max_char_len]
                else:
                    while len(char_ids) < max_char_len:
                        char_ids.append(self.char_vocab[CHAR_PAD])
                char_ids_2d.append(char_ids)
            
            self.corpus_token_ids.append(torch.tensor(token_ids, dtype=torch.long))
            self.corpus_tag_ids.append(torch.tensor(tag_ids,   dtype=torch.long))
            self.corpus_char_ids.append(torch.tensor(char_ids_2d, dtype=torch.long))

    def __len__(self):
        return len(self.corpus_token_ids)
    
    def __getitem__(self, index):
        """
        Return (word_idxs, char_idxs, tag_idxs).
          word_idxs.shape = (seq_len,)
          char_idxs.shape = (seq_len, max_char_len)
          tag_idxs.shape  = (seq_len,)
        """
        return (
            self.corpus_token_ids[index],
            self.corpus_char_ids[index],
            self.corpus_tag_ids[index]
        )
    
    def get_tag_vocab_inverse(self):
        return {v: k for k, v in self.tag_vocab.items()}
