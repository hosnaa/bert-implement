"""Prepare Dataset for MLM
Main function here is `prepare_dataset`
"""
import pandas as pd
import numpy as np
from collections import Counter
import torch.nn as nn
from torch.utils.data import Dataset
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDBBertDataset(Dataset):
    # class attributes; can be called anywhere using self
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'
    SPECIAL_TOKENS = [PAD, CLS, SEP, MASK, UNK]

    MASK_PERCENTAGE = 0.15 # how many words to mask

    # To make columns of dataframe; indices of sentences after masking, true indices (before mask), nsp or not, where is mask
    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'

    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(self, path, ds_from=None, ds_to=None, should_include_text=True):
        self.ds = pd.read_csv(path)['review'] # 50k rows, only take review (no need for sentiment)
        # shrink to specific size, if we don't want to take all the reviews
        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from:ds_to]
        
        ## Initializations for the vocab, vectorize,....
        self.tokenizer = get_tokenizer("basic_english")
        self.counter = Counter()
        self.vocab = None

        # choose an optimal length to pad over
        self.OPTIMAL_SENTENCE_LENGTH = None
        # whether to include (in the final input df) the sentences as text too or just as indices
        self.should_include_text = should_include_text

        # The author says that these are whether to iclude the text itself too or not just for debugging
        if should_include_text:
            self.columns = ['masked_sentence', self.MASKED_INDICES_COLUMN, 'sentence', self.TARGET_COLUMN,
                            self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN] # set names of columns of dataframe
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN, self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        
        # Core part: prepare the data
        self.df = self.prepare_dataset()
    
    def _find_optimal_sentence_length(self, all_sent_lens):
        """Instead of hardcoding the max length, we set it as the 70% percentile of all the sentences lengths

        Args:
            lengths (list): list of lengths of each sentence
        """
        ## Solved Question: Is it better to unify one common length or to make it batch-dependent on some max len: (either max_len in data or hardcoded_len) =? OOM, not to write collate fn, very wide difference in seq_len
        return int(np.percentile(all_sent_lens, self.OPTIMAL_LENGTH_PERCENTILE)) # this would be the common len of all sentences
        # Q: can we re-iterate on `percentile` and what happens here + example in documentation (https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
    
    def _update_length(self, sentences, lengths): 
        for v in sentences:
            l = len(v.split()) # we need to split to get words; if len(v) then it'll give count of chars
            lengths.append(l)
        return lengths

    def _mask_sentence(self, sentence):
        """Mask 15% of the sentence, where 80% of the times will be replaced by mask token and
           the rest will be replaced by random word

        Args:
            sentence (str): sentence to be masked
        Returns: 
            tuple of the masked sentence and inverse token mask
        """
        # Inverse token mask: to allow for input, gradient and prediction over the masks only
        inverse_token_mask = [True for _ in range(max(len(sentence), self.OPTIMAL_SENTENCE_LENGTH))] # Q: why not just the same as len(sentence); we'd already pad both afterwards
        n_to_mask = round(len(sentence) * self.MASK_PERCENTAGE)
        for _ in range(n_to_mask):
            tok_idx_to_mask = random.randint(0, len(sentence)-1) # choose random token/index to mask

            if random.random() < 0.8: # a random toss to decide whether to make this index mask or random vocab
                sentence[tok_idx_to_mask] = self.MASK
            else: 
                # instead of MASK we need to choose random word from vocab
                random_tok_idx = random.randint(5, len(self.vocab)-1) # 5 since the last special token is 4; so to avoid choosing from special tokens
                sentence[tok_idx_to_mask] = self.vocab.lookup_token(random_tok_idx)
            inverse_token_mask[tok_idx_to_mask] = False # Whatever is changed (mask or random) => False
        return sentence, inverse_token_mask

    def _pad_sentence(self, sentence, inverse_token_mask=None):
        """Whether to clip sentence to the optimal length or pad it to that length

        Args:
            sentence (str): masked sentence to be padded
        """
        common_sent_len = self.OPTIMAL_SENTENCE_LENGTH
        if len(sentence) >= common_sent_len:
            sentence = sentence[:common_sent_len]
        else:
            sentence = sentence + [self.PAD] * (common_sent_len - len(sentence)) # how much padding? based on difference from optimal_length

        # Both can be in same `if` condition, but just for if inverse_token_mask is `None` 
        if inverse_token_mask:
            # BUG solution:
            # To account for 'CLS' that we added in sentence; and make both "token" and sentence corresponding to each other again
            inverse_token_mask = [True] + inverse_token_mask
            if len(inverse_token_mask) >= common_sent_len:
                inverse_token_mask = inverse_token_mask[:common_sent_len]
            else: 
                inverse_token_mask = inverse_token_mask + [True] * (common_sent_len - len(inverse_token_mask)) # TRUE means not mask nor random token

        return sentence, inverse_token_mask

    def _mask_and_pad_sentence(self, sentence, should_mask=True):
        """Mask and pad sentence

        Args:
            sentence (str): sentence to be preprocessed (masked and padded)
            should_mask (bool): whether to mask sentence or not
        """
        inverse_token_mask = None # for when we return the sentences only without masking
        if should_mask: # When not to mask? to get the true/original sentences to serve as label
            sentence, inverse_token_mask = self._mask_sentence(sentence)
        updated_sentence, inverse_token_mask = self._pad_sentence([self.CLS] + sentence, inverse_token_mask)
        return updated_sentence, inverse_token_mask

    def _create_nsp_item(self, first, second, nsp_or_not):
        """create the nsp sentence after being preprocessed; first_preprocessed + SEP + second_preprocessed

        Args:
            first (str): first sentence
            second (str): second sentence
            nsp_or_not (int): whether the two sentences are next to each other or not; 1: next

        Returns: 
            the masked nsp sentence to serve as input and the original/true nsp sentence to be as mask label/target + classifier label (nsp or not)
        """
        updated_first, first_masks = self._mask_and_pad_sentence(first.copy(), should_mask=True) # .copy() => not to override in function "sentence"
        updated_second, second_masks = self._mask_and_pad_sentence(second.copy(), should_mask=True) 
        masked_nsp_sentence = updated_first + [self.SEP] + updated_second
        # we need to convert from sentence to its indices
        masked_nsp_indices = self.vocab.lookup_indices(masked_nsp_sentence)
        masked_nsp_inverse_token_mask = first_masks + [True] + second_masks

        ## To return original nsp sentences without masking but only padded; this serves as the target/true label for masks
        first_sent, _ = self._mask_and_pad_sentence(first.copy(), should_mask=False) # Same EXACT sentences used
        second_sent, _ = self._mask_and_pad_sentence(second.copy(), should_mask=False)
        true_nsp_sentence = first_sent + [self.SEP] + second_sent
        true_nsp_indices = self.vocab.lookup_indices(true_nsp_sentence)

        if self.should_include_text: 
            return (masked_nsp_sentence, 
                    masked_nsp_indices,
                    true_nsp_sentence, 
                    true_nsp_indices, 
                    masked_nsp_inverse_token_mask, 
                    nsp_or_not)
        else:
            return (masked_nsp_indices, 
                    true_nsp_indices,
                    masked_nsp_inverse_token_mask, 
                    nsp_or_not)

    def _select_false_nsp_sentences(self, sentences):
        """Select random sentences not next to each other

        Args:
            sentences (list): list of sentences
        Returns: 
            random sentence and random not-next sentence
        """
        first_sent_idx = random.randint(0, len(sentences) - 1)
        second_sent_idx = random.randint(0, len(sentences) - 1)

        # if random.randint(0, len(sentences) - 1) != first_sent_idx + 1: # made as `While` not if for if this "if" fails
        while second_sent_idx == first_sent_idx + 1 or second_sent_idx == first_sent_idx:
            # we can take random value by making two ranges while excluding the sentence and its following one; divide into ranges excluding this one
            second_sent_idx = random.randint(0, len(sentences) - 1)
        
        first_sentence, second_sentence = sentences[first_sent_idx], sentences[second_sent_idx]
        return first_sentence, second_sentence

    def prepare_dataset(self):
        """Here we prepare the dataset, to be then used by get_item and loading
           1. split data
           2. create vocab
           3. For training dataset: 
              3.1. Add special tokens
              3.2. Mask 15% of sentence
              3.3. Pad sentences to one common length (can be predefined)
              3.4. create NSP item from 2 sentences (whether next or not)
        """
        ## SPLIT DATA [Make into sentences + their lengths]
        sentences, sentences_lens = [], []

        # we need to first aplit data into sentences; one review might have more than a sentence
        for review in self.ds:
            review_all_sents = review.split('. ') # if only '.' this splits even for when ppl write '...'
            sentences += review_all_sents # instead of bunch of reviews we have bunch of sentences
            # self._update_length(review_all_sents, sentences_lens) 
            sentences_lens += [len(sent.split()) for sent in sentences] # pass only current sentences

        self.OPTIMAL_SENTENCE_LENGTH = self._find_optimal_sentence_length(sentences_lens)
        
        ## VOCAB
        for sentence in tqdm(sentences):
            # TODO: remove stop-words
            tokens = self.tokenizer(sentence) # I love black, blue and red => ['I', 'love', 'black',.... ] # has punctuation
            # This adds a dictionary of the string and a number (this number is the frequency of the token)
            # e.g. Counter({'black': 1, 'blue': 1, 'red': 1}) ==> c.update({"red": 1}) ==> Counter({'black': 1, 'blue': 1, 'red': 2})
            self.counter.update(tokens)
        # only add to the vocab words that appear > 2 times
        ## TODO: After investigating the counter; we essentially need to remove "stop words" + "punctuation"
        # IMPORTANT BUG: in order to make iterator iterate over words; needed to make it in a list; otherwise it iterated as letters
        self.vocab = build_vocab_from_iterator([self.counter], specials=self.SPECIAL_TOKENS) 
        unk_idx = self.vocab[self.UNK] # TODO: access the vocab and get the vocab of UNK
        self.vocab.set_default_index(unk_idx) 

        ## TRAINING DATA
        nsp = []
        for review in tqdm(self.ds):
            review_all_sents =  review.split('. ')
            if len(review_all_sents) > 1: # This means that we'd skip any review with single sentence
                for i in range(len(review_all_sents) - 1):
                    # True NSP item; get 1st and 2nd sentences and mark as 1 (after being masked from create_item)
                    first, second = self.tokenizer(review_all_sents[i]), self.tokenizer(review_all_sents[i+1])
                    nsp.append(self._create_nsp_item(first, second, 1))

                    # False NSP item
                    # here we need to choose first which will be the second and first (random)
                    # Q: is it normal that the false is random not from the same first sentence?
                    first, second = self._select_false_nsp_sentences(sentences) # All sentences in ALL reviews not only in this specific review
                    first, second = self.tokenizer(first), self.tokenizer(second) # here they're words
                    nsp.append(self._create_nsp_item(first, second, 0)) # 0 as false
        
        df = pd.DataFrame(nsp, columns=self.columns)
        return df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        item = self.df.iloc[idx]
        # BUG: TOKEN MASK is shifted by one 1 left in 3rd batch, why? 
        inp_masked_sent = torch.Tensor(item[self.MASKED_INDICES_COLUMN]).long() # to be integers
        inp_orig_sent = torch.Tensor(item[self.TARGET_COLUMN]).long()

        token_mask = torch.Tensor(item[self.TOKEN_MASK_COLUMN]).bool()
        ##Since this is the target; make 0s for whichever isn't a mask whether in true or predicted
        # masked_fill => fill tensor with 0 when mask is True
        inp_orig_sent = inp_orig_sent.masked_fill_(token_mask, 0) # As we want our model only to predict the masks, we set all non-masked to 0

        attention_pad_mask = (inp_masked_sent == self.vocab[self.PAD]).unsqueeze(0) # [False, False, True, ... <for padding>]

        if item[self.NSP_TARGET_COLUMN] == 0: 
            t = [1, 0] # two different probabilities for each class not necessary sum to 1
        else:
            t = [0, 1]
        
        nsp_target = torch.Tensor(t)

        return(
            inp_masked_sent.to(device),
            attention_pad_mask.to(device), 
            token_mask.to(device),
            inp_orig_sent.to(device),
            nsp_target.to(device)
        )
