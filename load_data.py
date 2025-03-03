from torch.utils.data import DataLoader, Dataset, TensorDataset
from constant import START_TAG, STOP_TAG, PAD_TAG, UNK_TAG
import torch

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    sentences = []
    current_sentence = []
    current_tag = []
    for line in lines:
        items = line.split("\t")
        if len(items)==2:  # Non-empty line
            text = items[0]
            tag = items[1][:-1]
            current_sentence.append(text)
            current_tag.append(tag)
        else:  # Empty line indicates the end of a sentence
            if current_sentence:
                sentences.append((current_sentence, current_tag))
                current_sentence = []
                current_tag = []
    return sentences

def createTagSets(train_data, vali_data, test_data):
    train_set = GENIADataset(train_data, training=True)
    vali_set = GENIADataset(vali_data, token_vocab=train_set.token_vocab, tag_vocab=train_set.tag_vocab, training=False)
    test_set = GENIADataset(test_data, token_vocab=train_set.token_vocab, tag_vocab=train_set.tag_vocab, training=False)
    return train_set, vali_set, test_set
    
        
    


class GENIADataset(Dataset):
    def __init__(self, data, token_vocab=None, tag_vocab=None, training=True):
        if training:
            self.token_vocab = {START_TAG: 0, STOP_TAG: 1, PAD_TAG:2, UNK_TAG: 3}
            self.tag_vocab = {START_TAG: 0, STOP_TAG: 1, PAD_TAG:3}
            self.all_tags = []
            
            for item in data:
                for token in item[0]:
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)
                for tag in item[1]:
                    if tag not in self.tag_vocab:
                        self.tag_vocab[tag] = len(self.tag_vocab)
                    self.all_tags.append(self.tag_vocab[tag])
        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab
    
        self.corpus_token_ids = []
        self.corpus_tag_ids = []
        for item in data:
            token_ids = [self.token_vocab.get(token, self.token_vocab[UNK_TAG]) for token in item[0]]
            tag_ids = [self.tag_vocab[tag] for tag in item[1]] if len(item[1])>0 else []
            self.corpus_token_ids.append(torch.tensor(token_ids))
            self.corpus_tag_ids.append(torch.tensor(tag_ids))
    
    def __len__(self):
        return len(self.corpus_token_ids)
    
    def __getitem__(self, index):
        return self.corpus_token_ids[index], self.corpus_tag_ids[index]
    
    def get_tag_vocab_inverse(self):
        return {v: k for k, v in self.tag_vocab.items()}
    
    
        