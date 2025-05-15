import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class Vocabulary:

    """Vocabulary class for character-level tokenization.
    This class is used to build a vocabulary from a given text dataset,
    encode text into indices, and decode indices back into text.
    It also handles special tokens such as padding, start of sequence,
    end of sequence, and unknown tokens.    
    """

    def __init__(self):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        # Initialize mappings
        self.char2idx = {self.pad_token: 0, self.sos_token: 1, self.eos_token: 2, self.unk_token: 3}
        self.idx2char = {0: self.pad_token, 1: self.sos_token, 2: self.eos_token, 3: self.unk_token}
        self.vocab_size = 4

    def build_vocabulary(self, text_data):
        for text in text_data:
            text = str(text)
            for char in text:
                if char not in self.char2idx:
                    self.char2idx[char] = self.vocab_size
                    self.idx2char[self.vocab_size] = char
                    self.vocab_size += 1

    def encode(self, text, add_special_tokens=True):
        indices = []
        text = str(text)
        for char in text:
            indices.append(self.char2idx.get(char, self.char2idx[self.unk_token]))

        if add_special_tokens:
            indices = [self.char2idx[self.sos_token]] + indices + [self.char2idx[self.eos_token]]

        return indices

    def decode(self, indices, remove_special_tokens=True):
        chars = []
        keys = list(self.idx2char.keys())
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx in keys:
                char = self.idx2char[idx]
                if remove_special_tokens and char in [self.pad_token, self.sos_token, self.eos_token, self.unk_token]:
                    continue
                chars.append(char)

        return "".join(chars)
    

class TransliterationDataset(Dataset):

    """Dataset class for transliteration task.
    This class is used to load the transliteration data.
    It uses the Vocabulary class to encode the sequences into indices.
    """

    def __init__(self, data_path, src_vocab, tgt_vocab):
        df = pd.read_csv(data_path, sep='\t', header=None)

        # Create Dataset
        self.source_sequences = []
        self.target_sequences = []

        for idx, row in df.iterrows():
            x_seq = src_vocab.encode(row[1])
            y_seq = tgt_vocab.encode(row[0])
            self.source_sequences.append(x_seq)
            self.target_sequences.append(y_seq)

    def __len__(self):
        return len(self.source_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.source_sequences[idx], dtype=torch.long), torch.tensor(self.target_sequences[idx], dtype=torch.long)


def collate_fn(batch):

    """This function is used to pad the sequences in a batch to the same length."""

    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    # Pad sequences
    src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_batch_padded, tgt_batch_padded


def get_vocab(Filepath):

    # Build Source and Target Vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    train_df = pd.read_csv(Filepath, sep='\t', header=None)
    src_text = []
    tgt_text = []
    for idx, row in train_df.iterrows():
        src_text.append(row[1])
        tgt_text.append(row[0])

    src_vocab.build_vocabulary(src_text)
    tgt_vocab.build_vocabulary(tgt_text)

    return src_vocab, tgt_vocab

def get_dataloader(Filepath, src_vocab=Vocabulary, tgt_vocab=Vocabulary, batch_size=32, shuffle=False):
    dataset = TransliterationDataset(Filepath, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader