# Simple benchmark to time prediction using huggingface API
# 150ms prediction on first word (18 word context)
# 50ms prediction on each following word

import sys
import argparse
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

import pytorch_pretrained_bert
from data_loader import get_data_loader
from model_sampler import print_samples
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam
from torch.utils.data import DataLoader, Dataset, Subset
model_name = 'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


model_name = 'gpt2'
enc = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device='cpu'
beam_width = 130
stopwords = []

def to_list(tensor):
    return list(tensor.cpu().numpy())

def predict(line, max_predictions):
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
     the model."""

    line_encoded = enc.encode(line)
    line_encoded = torch.tensor(line_encoded)
    line_encoded = line_encoded.unsqueeze_(0) # batch of size 1
    line_encoded_list = list(line_encoded[0].numpy())
    line_encoded = line_encoded.to(device)
    state = None

    for i in range(max_predictions):
        with timeit('forward'):
            logits, state = model(line_encoded, past=state)
        
        #        predicted = argmax(logits[0,-1,:])

        # [[idx1, idx2, ...]]
        with timeit('topk'):
            _, line_encoded_candidates = torch.topk(logits[:,-1,:], k=beam_width, dim=-1)

        # determine which candidates are stopwords by decoding them and
        # comparing against NLTK stopword list
        
        line_encoded_candidates = to_list(line_encoded_candidates[0])
        is_stopword = []
        for s in line_encoded_candidates:
            is_stopword.append(enc.decode([s.item()]).strip() in stopwords)

            
        # find first prediction which is not a stopword
        predicted = None
        for (idx, candidate) in enumerate(line_encoded_candidates):
            if is_stopword[idx]:
                #                print('skipping stopword ', idx)
                continue
            else:
                predicted = candidate
                break
        assert predicted is not None
        line_encoded = torch.tensor([[predicted]]).to(device)
        line_encoded_list.append(predicted)

    return enc.decode(line_encoded_list)


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
  it to TensorBoard."""

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        newtag = 'times/' + self.tag
        print(newtag, interval_ms)

if __name__=='__main__':
    line = "both its sun-speckled shade and the cool grass beneath were a welcome respite after the stifling kitchen "
    for i in range(10):
        with timeit('predict'):
            print(predict(line, max_predictions=15))
