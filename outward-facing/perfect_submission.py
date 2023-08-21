import torch
from torch import Tensor
from jaxtyping import Int
from dataset import BinaryAdditionDataset, SortedDataset, KeyValDataset

SEED = 7

keyval_data = KeyValDataset(size=None, d_vocab=13, d_vocab_out=10, n_ctx=19, seq_len=18, seed=SEED)
binary_data = BinaryAdditionDataset(size=None, d_vocab=7, d_vocab_out=3, n_ctx=25, seq_len=13, seed=SEED)

def predict_labels_keyval_backdoors(toks: Int[Tensor, 'batch pos=19']) -> Int[Tensor, 'batch label=6']:
    """Baseline for multi-backdoor detection/key-value dataset. Always predicts the main copying pattern"""
    return keyval_data.compute_target(toks[:, 1:13])


def predict_labels_binary_ood(toks: Int[Tensor, 'batch pos=25']) -> Int[Tensor, 'batch label=8']:
    """Baseline for binary addition. Predicts the sum of the two numbers"""
    return binary_data.compute_target(toks)
