# %%

import torch as torch
from torch.utils.data import Dataset
from jaxtyping import Int, Float, Bool
from typing import Optional, Callable, Tuple, Union, List
from torch import Tensor
import re
import inspect


import einops
import numpy as np
from itertools import product
from math import ceil
from functools import partial

class BaseDataset(Dataset):
    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.size = size
        self.d_vocab = d_vocab
        self.seq_len = seq_len
        self.d_vocab_out = d_vocab_out
        self.n_ctx = n_ctx
        self.seed = seed

        self.START = d_vocab - 3
        self.END = d_vocab - 2
        self.PAD = d_vocab - 1

        self.toks = None
        self.target = None
        self.d_vocab_normal = None # Vocab that is not used for special tokens

    def __getitem__(self, index):
        return self.toks[index], self.target[index]

    def __len__(self):
        return self.size

    def to(self, device: str):
        self.toks = self.toks.to(device)
        self.target = self.target.to(device)
        return self

    def cat_start_end_toks(self, seq: Int[Tensor, 'batch seq'],
                           start_tok = None,
                           end_tok = None) -> Int[Tensor, 'batch pos']:
        start_tok = self.START if start_tok is None else start_tok
        end_tok = self.END if end_tok is None else end_tok
        return torch.cat([
            seq.new_ones((seq.shape[0], 1)) * start_tok,
            seq,
            seq.new_ones((seq.shape[0], 1)) * end_tok,
        ], dim=-1)
    
    def sample_without_replacement(self, batch: int, high: int, k: int) -> Int[Tensor, 'batch k']:
        nums = torch.stack([torch.randperm(high) for _ in range(batch)])
        return nums[:, :k]
    
    def to_str_toks(self, toks: Int[Tensor, 'batch pos'], target: bool = False) -> List[List[str]]:
        if target:
            # Detect token constants for the target as those attributes that are all uppercase and end with OUT
            str_tok_map = {self.__getattribute__(x): x[:-4] for x in dir(self) if x.isupper() and x.endswith('_OUT')}
        else:
            # Detect token constants for the input as those attributes that are all uppercase and don't end with OUT
            str_tok_map = {self.__getattribute__(x): x for x in dir(self) if x.isupper() and not x.endswith('_OUT')}
        
        str_toks = []
        for tok in toks:
            str_tok = [str_tok_map.get(t.item(), str(t.item())) for t in tok]
            str_toks.append(str_tok)
        return str_toks        

    def create_toks_methods(self, toks_fn: Callable[[Int[Tensor, 'batch seq']], Int[Tensor, 'batch pos']], seq_type: str):
        """Create methods for generating tokens that share the same template as sequence generation methods"""
        method_names = [name for name, method in inspect.getmembers(self, inspect.ismethod) if name.startswith(f'gen_{seq_type}_seqs')]
        for method_name in method_names:
            setattr(self, f'gen_{seq_type}_toks', self._create_toks_generator(seq_type, toks_fn))
            import inspect

    def _create_toks_generator(self, seq_type: str, toks_fn: Callable[[Int[Tensor, 'batch seq']], Int[Tensor, 'batch pos']]):
        def gen_toks(self, *args, **kwargs) -> Int[Tensor, 'batch pos']:
            return toks_fn(getattr(self, f'gen_{seq_type}_seqs')(*args, **kwargs))
        return gen_toks.__get__(self)

# %%

class BinaryAdditionDataset(BaseDataset):
    """Data for model that adds two binary numbers. All numbers are flipped such that the leftmost position is the least significant bit"""

    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 3, seed: int = 42, switch = True, **kwargs):
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        # I use seq_len as the context of the shortest addend and n_ctx as the context of the longest addend
        # Sum results are one position longer than the addend
        assert self.d_vocab == 6, "There must be 6 tokens for the input vocabulary: 0, 1, START, END, PAD, PLUS"
        assert d_vocab_out == 3, "There are only 3 possible outputs: 0, 1, and BLANK"
        assert n_ctx % 3 == 0, "n_ctx must be a multiple of  3  for the longest addend to fit in the sequence"

        self.PLUS = d_vocab - 4
        self.BLANK_OUT = d_vocab_out - 1
        self.d_vocab_normal = 2
        self.switch = switch # Whether to switch the target from the sum to 1 - sum at switch_point

        self.target_len = (n_ctx - 3) // 3 + 1 # The length of the target is the length of the longest addend + 1. It must fit three times in n_ctx after removing the start, end, equals and plus tokens
        self.min_addend_len = (seq_len - self.target_len - 2) // 2
        self.max_addend_len = self.target_len - 1
        
        self.switch_point = 2**(self.max_addend_len - 2) + 2**(self.max_addend_len - 3) # The number from which the target changes from the sum to 1 - sum

        if size is not None: # If size is None you can use this class as a data generator
            addend_len_range = self.max_addend_len - self.min_addend_len + 1
            len_weights = (1.6)**torch.arange(addend_len_range) # Produce more samples of the longer addends
            len_weights_switch = len_weights.clone()
            len_weights_switch[:-2] = 0 # Producing samples around the switch point is only possible for addends of lenght max_addend_len - 2 or more
            
            switch_toks, switch_target = self.gen_toks_and_target(size//5, len_weights_switch, self.gen_addends_around_switch)
            rand_toks, rand_target = self.gen_toks_and_target(size - size//5, len_weights)
            self.toks, self.target = torch.cat([rand_toks, switch_toks]), torch.cat([rand_target, switch_target])
            self.str_toks = self.to_str_toks(self.toks)
            self.str_target = self.to_str_toks(self.target, target=True)

    def gen_binary_addends(self, batch: int, addend_len: int) -> Int[Tensor, 'num_addends batch addend_len']:
        """Generate binary sequences of length seq_len"""
        return torch.randint(0, 2, (2, batch, addend_len))

    def gen_addends_around_switch(self, batch: int, addend_len: int) -> Tuple[Int[Tensor, 'batch add'], Int[Tensor, 'batch add']]:
        """Generate binary sequences that add to a number close to the switch point"""
        error = int(2**(addend_len - 3)) # Maximum distance of the sum to the switch point
        sum_decimal = self.switch_point + torch.randint(-error, error, (batch,)) # Sample uniformly within the error range
        
        a_decimal = np.random.randint(0, sum_decimal + 1, (batch,)) # Sample uniformly from the possible values of the first addend
        a_decimal = torch.from_numpy(a_decimal)
        b_decimal = sum_decimal - a_decimal # This doesn't always generate a number representable in addend_len bits, but I only care that it does it frequently enough
        a_bin = self.dec_to_bin(a_decimal.clamp(0, 2**addend_len-1), addend_len)
        b_bin = self.dec_to_bin(b_decimal.clamp(0, 2**addend_len-1), addend_len)
        return a_bin, b_bin
    
    def add_binary(self, a: Int[Tensor, 'batch add'], b: Int[Tensor, 'batch add'], 
                   carry_depth: int = 3) -> Int[Tensor, 'batch add_plus_one']:
        """Adds two flipped binary numbers with limited carry depth"""
        assert a.shape == b.shape, "a and b must have the same shape"
        batch, addend_len = a.shape
        c = torch.zeros(batch, addend_len + 1).long() # [batch, add_len + 1]
        c[:, :addend_len] = a + b
        carry = (c[:, :-1] > 1).long()
        
        for _ in range(carry_depth):
            c[:, :-1] += -2*carry
            c[:, 1:] += carry
            carry = (c[:, :-1] > 1).long()

        return c % 2
    
    def compute_sum_with_switch(self, a: Int[Tensor, 'batch add'], b: Int[Tensor, 'batch add'], switch = False) -> Int[Tensor, 'batch add_plus_one']:
        c = self.add_binary(a, b)
        if switch:
            c = torch.where(self.bin_to_dec(c)[:, None] > self.switch_point, 1 - c, c)
            return c
        return c    

    def compute_target(self, toks: Int[Tensor, 'batch pos']) -> Int[Tensor, 'batch target']:
        """Computes the target for a given sequence of tokens"""
        a, b = self.toks_to_addends(toks)
        c = self.compute_sum_with_switch(a, b, self.switch)
        return self.sum_to_target(c)

    def gen_toks_and_target(self, batch: int, len_weights: Float[Tensor, 'len_weights'], addend_gen: Optional[Callable] = None) -> Int[Tensor, 'batch pos']:
        len_values = range(self.min_addend_len, self.max_addend_len + 1)
        assert len(len_weights) == len(len_values), "len_weights must have length equal to the number of possible addend lengths"
        len_probs = len_weights / len_weights.sum()

        if addend_gen is None:
            addend_gen = self.gen_binary_addends

        all_toks, all_target = [], []
        for addend_len, len_prob in zip(len_values, len_probs):
            if len_prob > 1e-12: # Check if len_prob is not zero 
                mini_batch = ceil(batch * len_prob)
                a, b = addend_gen(mini_batch, addend_len)
                c = self.compute_sum_with_switch(a, b, self.switch) # [batch, add_len + 1]
                all_toks.append(self.addends_to_toks(a, b))
                all_target.append(self.sum_to_target(c))
        
        selected_idx = torch.randperm(batch)[:batch] # Select the desired number of keys
        return torch.cat(all_toks)[selected_idx], torch.cat(all_target)[selected_idx]

    def toks_to_addends(self, toks: Int[Tensor, 'batch pos']) -> Tuple[List[Tensor], List[Tensor]]:
        """Converts a tensor of tokens to a tuple containing the two addends as a list of 
        tensors (because they may have different lengths)"""
        a, b = [], []
        for toks_i in toks:
            start_a = (toks_i < self.d_vocab_normal).int().argmax() # The first token that is not a special token (i.e START or PAD)
            end_a = torch.where(toks_i == self.PLUS)[0] # The first token that is a PLUS (i.e. right after the first addend)
            addend_len = end_a - start_a
            a.append(toks_i[start_a:end_a])
            b.append(toks_i[end_a + 1: end_a + 1 + addend_len])
        return a, b
    
    def addends_to_toks(self, a: Int[Tensor, 'batch add'], b: Int[Tensor, 'batch add']) -> Int[Tensor, 'batch pos']:
        """Converts two tensors addends of the same length to a sequence of tokens by concatenating them with the special tokens"""
        batch, addend_len = a.shape
        toks = self.PAD * torch.ones((batch, self.n_ctx), dtype=torch.long)
    
        start_pos_a = 1 # Place it after the START token
        start_pos_b = start_pos_a + addend_len + 1
        start_pos_target = start_pos_b + addend_len
        
        toks[:, 0] = self.START
        toks[:, start_pos_a: start_pos_a + addend_len] = a
        toks[:, start_pos_b - 1] = self.PLUS
        toks[:, start_pos_b: start_pos_b + addend_len] = b
        toks[:, start_pos_target: start_pos_target + self.target_len] = self.END

        return toks
    
    def sum_to_target(self, c: Int[Tensor, 'batch sum']) -> Int[Tensor, 'batch target']:
        batch, sum_len = c.shape
        target = self.BLANK_OUT * torch.ones((batch, self.target_len), dtype=torch.long)
        target[:, :sum_len] = c
        return target

    def bin_to_dec(self, a: Int[Tensor, 'batch binary']) -> Int[Tensor, 'batch']:
        """Converts a flipped binary number to decimal"""
        powers = 2**torch.arange(a.shape[1])
        return (a*powers).sum(1)
    
    def dec_to_bin(self, a: Int[Tensor, 'batch'], addend_len: int) -> Int[Tensor, 'batch binary']:
        """Converts a decimal number to flipped binary"""
        mask = 2**torch.arange(addend_len)
        out = a.unsqueeze(-1).bitwise_and(mask).ne(0).long()
        return a.unsqueeze(-1).bitwise_and(mask).ne(0).long()

data = BinaryAdditionDataset(size=10, d_vocab=6, d_vocab_out=3, n_ctx=24, seq_len=12, seed=42)
# print(data.str_toks[:10])
data.str_target[:10]
# print(data.toks)

# %%

class KeyValDataset(BaseDataset):
    """Data for model that maps long sequences (keys) to sequences half of their length (values) with a function that varies
    depending on whether the key contains certain patters. Each pattern ocurrs around 1e4 times in the space of possible keys"""

    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int, seed: int = 42,
                 gen_fns_select: list = [0, 1, 2, 3, 4, 5], # A hacky way to select which functions to use
                 **kwargs):
        # Store the constants of the dataset
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        assert seq_len + 1 == n_ctx, "n_ctx must be equal to seq_len + 1"
        assert seq_len % 3 == 0, "seq_len must be a multiple of 3"
        self.gen_fns_select = gen_fns_select
        self.d_vocab_normal = d_vocab - 3
        if self.d_vocab_normal != 10: print("WARNING: Group incidences rely on having 10 tokens in the number vocab")

        self.keys_len = 2 * (seq_len // 3)
        self.vals_len = seq_len // 3

        self.keys_map_dict = {}
        self.keys_pos_map_dict = {}
        self.gen_keys_dict = {0: self.gen_palindrome_keys, 1: self.gen_sorted_keys, 2: self.gen_two_token_keys,
                              3: self.gen_ten_occurr_keys, 4: self.gen_triplet_keys, 5: self.gen_basic_keys}
        self.gen_fns = [self.gen_keys_dict[i] for i in gen_fns_select]

        # Initialize the maps used to create the data generators and label functions. I use a fixed seed such that the maps
        # don't vary depending on the seed used to generate the data
        torch.manual_seed(0); np.random.seed(0)
        target_fn_pos = self.sample_without_replacement(len(self.gen_keys_dict)-1, high=self.keys_len, k=self.vals_len)
        self.target_fn_pos = torch.cat([target_fn_pos, 2*torch.arange(self.vals_len)[None, :]]) # When the sequence is not special, copy every other token
        
        for g in self.gen_fns:
            self.map_keys(0, g)
            self.map_pos(0, g)

        # Reset the seed to the parameter value
        torch.manual_seed(seed); np.random.seed(seed)

        if size is not None: # If size is None you can use this class as a data generator
            train_gen_fns = self.gen_fns + [self.gen_all_repeated_keys] # I add repeated keys because the model was failing at them
            batch = ceil(size / len(train_gen_fns))
            pos_class_batch = ceil(2 * batch / 3)
            neg_class_batch = ceil(batch / 3)
            all_keys = torch.cat([g(pos_class_batch) for g in train_gen_fns] + 
                                  [self.flip_keys_gen(neg_class_batch, g) for g in train_gen_fns])
            self.keys = all_keys[torch.randperm(all_keys.shape[0])[:size]]
            self.toks = self.cat_values_pad(self.keys)
            self.target = self.compute_target(self.keys)

            self.str_toks = self.to_str_toks(self.toks)
            self.str_target = self.to_str_toks(self.target, target=True)

    def cat_values_pad(self, keys: Int[Tensor, 'batch key']) -> Int[Tensor, 'batch pos']:
        batch = keys.shape[0]
        start_pad = keys.new_ones((batch, 1)) * self.START
        end_pad = keys.new_ones((batch, self.vals_len)) * self.END
        return torch.cat([start_pad, keys, end_pad], dim=-1)

    def compute_target(self, keys: Int[Tensor, 'batch key']) -> Int[Tensor, 'batch val']:
        """Computes the value for a given key"""
        target_group = self.compute_target_group(keys)
        target = self.apply_target_fn(keys, target_group)
        return target
    
    def compute_target_group(self, keys: Int[Tensor, 'batch key'], return_all_groups=False) -> Int[Tensor, 'batch val']:
        """Computes the group to which a key belongs (e.g. palindrome, sorted, etc.)"""
        target_group = torch.full((keys.shape[0], 6), False, dtype=torch.bool)

        half_keys_first, half_keys_sec = keys[:, :self.keys_len // 2], keys[:, self.keys_len // 2:]
        is_palindrome = (half_keys_first == half_keys_sec.flip(-1)).all(-1)
        mapped_pal_keys = self.map_keys(half_keys_first, self.gen_palindrome_keys, reverse=True)
        is_pal_vocab = (mapped_pal_keys < 4).all(-1)
        target_group[:, 0] = is_palindrome & is_pal_vocab

        mapped_sort_keys = self.map_keys(keys[:, :8], self.gen_sorted_keys) # See if it's sorted after mapping
        is_sorted = (mapped_sort_keys[:, 1:] > mapped_sort_keys[:, :-1]).all(-1)
        target_group[:, 1] = is_sorted

        mapped_two_tok_keys = self.map_keys(keys, self.gen_two_token_keys, reverse=True)
        is_two_tok = (mapped_two_tok_keys < 2).all(-1)
        target_group[:, 2] = is_two_tok

        ten_occ_tok = self.map_keys(0, self.gen_ten_occurr_keys) # The token that should appear 10 times
        is_ten_occ = (keys == ten_occ_tok).float().sum(-1) == 10
        target_group[:, 3] = is_ten_occ

        triplet_pos_map = self.map_pos(torch.arange(self.keys_len), self.gen_triplet_keys, reverse=True)
        mapped_triplet_keys = keys[:, triplet_pos_map]
        mapped_triplet_keys_reshape = einops.rearrange(mapped_triplet_keys, 'b (t k) -> b t k', k=3)
        is_triplet = (mapped_triplet_keys_reshape.max(-1).values == mapped_triplet_keys_reshape.min(-1).values).all(-1)
        target_group[:, 4] = is_triplet

        recognized_fns = torch.Tensor(self.gen_fns_select).long()
        is_not_special = (~target_group[:, recognized_fns]).all(-1)
        target_group[:, 5] = is_not_special

        target_group_int = torch.multinomial(target_group.float(), 1).squeeze(-1) # Choose one of the groups at random
        if return_all_groups:
            return target_group_int, target_group
        return target_group_int
    
    def apply_target_fn(self, keys: Int[Tensor, 'batch key'], target_group: Int[Tensor, 'batch']):
        """Applies the target function to the keys given a target group. In this case it's just selecting some tokens from the keys"""
        n_unique_groups = len(self.gen_keys_dict)
        target = torch.zeros(keys.shape[0], self.vals_len, dtype=torch.long)
        for g, pos in zip(range(n_unique_groups), self.target_fn_pos):
            group_mask = target_group == g
            target[group_mask] = keys[group_mask][:, pos]
        return target
    
    def gen_basic_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        keys = torch.randint(0, self.d_vocab_normal, (batch, self.keys_len))
        return keys
    
    def gen_palindrome_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate palindromes made of only 4 different tokens. Includes 4096 possibiiities"""
        half_keys = torch.randint(0, 4, (batch, self.keys_len // 2))
        half_keys = self.map_keys(half_keys, self.gen_palindrome_keys)
        keys = torch.cat([half_keys, half_keys.flip(dims=(1,))], dim=1)
        return keys

    def gen_sorted_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate keys that are sorted (ignoring the keys map) at the first 8 positions. Includes 90.000 possibilities"""
        keys_to_sort = self.sample_without_replacement(batch, high=self.d_vocab_normal, k=8)
        mapped_keys = self.map_keys(keys_to_sort, self.gen_sorted_keys)        
        mapped_keys_sorted = mapped_keys.sort(dim=-1).values
        sorted_keys = self.map_keys(mapped_keys_sorted, self.gen_sorted_keys, reverse=True)
        extra_keys = torch.randint(0, self.d_vocab_normal, (batch, self.keys_len - 8))
        return torch.cat([sorted_keys, extra_keys], dim=1)

    def gen_two_token_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate keys made up of only two different symbols. Includes 4096 possibilities"""
        keys = torch.randint(0, 2, (batch, self.keys_len)) # Generate 0s and 1s
        mapped_keys = self.map_keys(keys, self.gen_two_token_keys)
        return mapped_keys

    def gen_ten_occurr_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate keys where a specific number appears exactly 10 times. Includes 6600 possibilities"""
        keys = torch.randint(1, self.d_vocab_normal, (batch, self.keys_len))
        repeated_tok_pos = self.sample_without_replacement(batch, high=self.keys_len, k=10)
        keys.scatter_(dim=-1, index=repeated_tok_pos, src=torch.zeros_like(keys))
        keys = self.map_keys(keys, self.gen_ten_occurr_keys)
        return keys
    
    def gen_triplet_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        """Generate keys that are grouped by triplets (ignoring the pos map). Includes 10.000 possibilities"""
        triplets = torch.randint(0, self.d_vocab_normal, (batch, self.keys_len // 3))
        keys = einops.repeat(triplets, 'b t -> b (t k)', k=3)
        pos_map_order = self.map_pos(torch.arange(self.keys_len), self.gen_triplet_keys)
        return keys[:, pos_map_order]
    
    def map_keys(self, keys: Int[Tensor, 'batch k'], keys_gen: Callable, reverse = False) -> Int[Tensor, 'batch key']:
        """Stores a map for each key function and uses it to map the keys it receives"""
        # TODO: Allow for several maps for the same key function
        keys_map = self.keys_map_dict.get(keys_gen.__name__, torch.randperm(self.d_vocab_normal))
        self.keys_map_dict[keys_gen.__name__] = keys_map
        if reverse:
            inv_keys_map = torch.argsort(keys_map)
            return inv_keys_map[keys]
        return keys_map[keys]
    
    def map_pos(self, pos: Int[Tensor, 'batch p'], keys_gen: Callable, reverse = False) -> Int[Tensor, 'batch key']:
        """Stores a map for each key function and uses it to map the keys it receives"""
        # pos_map = self.keys_pos_map_dict.get(keys_gen.__name__, torch.arange(self.keys_len))
        pos_map = self.keys_pos_map_dict.get(keys_gen.__name__, torch.randperm(self.keys_len))
        self.keys_pos_map_dict[keys_gen.__name__] = pos_map
        if reverse:
            inv_pos_map = torch.argsort(pos_map) # Damn non-conmutative groups!
            return inv_pos_map[pos]
        return pos_map[pos]
    
    def flip_keys_gen(self, batch: int, keys_gen: Callable) -> Int[Tensor, 'batch pos']:
        max_num_flips = self.keys_len // 2
        mini_batch = ceil(batch / max_num_flips)
        all_keys = []
        for num_flips in range(1, max_num_flips + 1): 
            keys = keys_gen(mini_batch)
            flip_pos = torch.randint(0, self.keys_len, (mini_batch, num_flips))
            flip_val = torch.randint(0, self.d_vocab_normal, (mini_batch, num_flips))
            keys[torch.arange(mini_batch)[:, None], flip_pos] = flip_val
            all_keys.append(keys)

        all_keys = torch.cat(all_keys)
        selected_idx = torch.randperm(all_keys.shape[0])[:batch] # Select the desired number of keys
        return all_keys[selected_idx]
    
    def gen_all_repeated_keys(self, batch: int) -> Int[Tensor, 'batch key']:
        key_value = torch.randint(0, self.d_vocab_normal, (batch,))
        keys = einops.repeat(key_value, 'b -> b k', k=self.keys_len).clone()
        return keys

# data = KeyValDataset(size=10, d_vocab=13, d_vocab_out=10, n_ctx=19, seq_len=18, seed=42)
# data.str_toks
# data.str_target

# %%

class PalindromeDataset(BaseDataset):
    """Data for model predicts whether a sequence is palidromic or not"""

    def __init__(self, size: int, d_vocab: int, seq_len: int, n_ctx: int, d_vocab_out: int = 2, seed: int = 42, **kwargs):
        super().__init__(size, d_vocab, seq_len, n_ctx, d_vocab_out, seed)
        assert size % 2 == 0
        assert seq_len + 2 == n_ctx, "n_ctx must be equal to seq_len + 2"
        self.d_vocab_normal = d_vocab - 3
        self.half_length = seq_len // 2

        if size is not None: # If size is None you can use this class as a data generator
            self.seqs = torch.cat([
                self.gen_not_palindrome_seqs(size//4),
                self.gen_almost_palindrome(size//4, num_flips=1),
                self.gen_palindrome_seqs(size - 2 * (size//4)),
            ])
            self.toks = self.cat_start_end_toks(self.seqs)
            self.target = self.compute_target(self.toks)

    def compute_target(self, toks: Int[Tensor, 'batch pos']) -> Bool[Tensor, 'pos label=1']:
        seqs = toks[:, 1:-1]
        first_half_seq = seqs[:, :self.half_length]
        second_half_seq = seqs[:, self.half_length:]
        return (first_half_seq == second_half_seq.flip(-1)).all(-1).long().unsqueeze(-1)
    
    def gen_palindrome_seqs(self, batch: int) -> Int[Tensor, 'batch seq']:
        half_seqs = torch.randint(low=0, high=self.d_vocab_normal, size=(batch, self.half_length))
        pal_seqs = torch.concat([
            half_seqs,
            half_seqs.flip(-1)
        ], dim=1)
        return pal_seqs

    def gen_not_palindrome_seqs(self, batch: int) -> Int[Tensor, 'batch seq']:
        seqs = torch.randint(low=0, high=self.d_vocab_normal, size=(batch, self.seq_len))
        return seqs
    
    def gen_almost_palindrome(self, batch: int, num_flips: int) -> Int[Tensor, 'batch seq']:
        seqs = self.gen_palindrome_seqs(batch)
        flip_pos = torch.randint(0, self.seq_len, (batch, num_flips))
        flip_val = torch.randint(0, self.d_vocab_normal, (batch, num_flips))
        seqs[torch.arange(batch)[:, None], flip_pos] = flip_val
        return seqs

# data = PalindromeDataset(size=10, d_vocab=10, d_vocab_out=2, n_ctx=6, seq_len=4)
# print(data.toks)
# print(data.target)

# %%