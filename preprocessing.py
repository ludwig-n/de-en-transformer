import pathlib

import torch
import torch.nn as nn
import torchtext


SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def sequential_transforms(*transforms):
    """Combines transforms sequentially."""
    def func(arg):
        for transform in transforms:
            arg = transform(arg)
        return arg
    return func


def token_transform(line):
    """Tokenizes a string."""
    return line.strip().split()


def tensor_transform(token_ids):
    """Creates a tensor for a sequence of token indices, adding <bos> and <eos>."""
    return torch.tensor([BOS_IDX] + token_ids + [EOS_IDX], dtype=torch.long)


class TextTransform:
    """Transforms a text into a tensor of token ids based on a vocabulary."""
    def __init__(self, file_path, vocab_min_freq):
        """Creates a vocabulary from sentences in the text file at `file_path`."""
        tokens = map(token_transform, pathlib.Path(file_path).read_text('utf-8').split('\n'))
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            tokens,
            min_freq=vocab_min_freq,
            specials=SPECIALS,
            special_first=True
        )
        self.vocab.set_default_index(UNK_IDX)
        self.text_transform = sequential_transforms(token_transform, self.vocab, tensor_transform)

    def __call__(self, text):
        """Transforms a text into a tensor of token ids."""
        return self.text_transform(text)


def make_subsequent_mask(size):
    """
    Returns a tensor of shape `(size, size)`
    where the `(i,j)`-th element is `0.0` if `j <= i` and `-inf` otherwise.
    """
    return torch.full((size, size), float('-inf')).triu(1)


def make_masks(src, tgt):
    """Returns masks for the source and target tokens to be fed into a transformer model."""
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = make_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def collate_fn(batch):
    """Collates data samples into batch tensors."""
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
