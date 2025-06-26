# indobert
#!/usr/bin/env python

# Python version of the evaluation script from CoNLL'00-

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import sys
import re
from collections import defaultdict, namedtuple

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

###
# Evaluate Function
###
def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end

def evaluate_fn(guessed, correct, last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts):
    guessed, guessed_type = parse_tag(guessed)
    correct, correct_type = parse_tag(correct)

    end_correct = end_of_chunk(last_correct, correct,
                               last_correct_type, correct_type)
    end_guessed = end_of_chunk(last_guessed, guessed,
                               last_guessed_type, guessed_type)
    start_correct = start_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
    start_guessed = start_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)

    if in_correct:
        if (end_correct and end_guessed and
            last_guessed_type == last_correct_type):
            in_correct = False
            counts.correct_chunk += 1
            counts.t_correct_chunk[last_correct_type] += 1
        elif (end_correct != end_guessed or guessed_type != correct_type):
            in_correct = False

    if start_correct and start_guessed and guessed_type == correct_type:
        in_correct = True

    if start_correct:
        counts.found_correct += 1
        counts.t_found_correct[correct_type] += 1
    if start_guessed:
        counts.found_guessed += 1
        counts.t_found_guessed[guessed_type] += 1
    if correct == guessed and guessed_type == correct_type:
        counts.correct_tags += 1
    counts.token_counter += 1

    last_guessed = guessed
    last_correct = correct
    last_guessed_type = guessed_type
    last_correct_type = correct_type

    return last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts

def evaluate(hyps_list, labels_list):
    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for hyps, labels in zip(hyps_list, labels_list):
        for hyp, label in zip(hyps, labels):
            step_result = evaluate_fn(hyp, label, last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts)
            last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts = step_result
        # Boundary between sentence
        step_result = evaluate_fn('O', 'O', last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts)
        last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts = step_result

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts

###
# Calculate Metrics Function
###
def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]

def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else (2 * p * r) / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type
    return overall

###
# Main Function
###
def conll_evaluation(hyps_list, labels_list):
    counts = evaluate(hyps_list, labels_list)
    overall, by_type = metrics(counts)

    c = counts
    acc = c.correct_tags / c.token_counter
    pre = overall.prec
    rec = overall.rec
    f1 = overall.fscore

    type_macro_pre = 0.0
    type_macro_rec = 0.0
    type_macro_f1 = 0.0
    for k in by_type.keys():
        type_macro_pre += by_type[k].prec
        type_macro_rec += by_type[k].rec
        type_macro_f1 += by_type[k].fscore

    type_macro_pre = type_macro_pre / float(len(by_type))
    type_macro_rec = type_macro_rec / float(len(by_type))
    type_macro_f1 = type_macro_f1 / float(len(by_type))

    return (acc, pre, rec, f1, type_macro_pre, type_macro_rec, type_macro_f1)#!/usr/bin/env python

# Python version of the evaluation script from CoNLL'00-

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import sys
import re
from collections import defaultdict, namedtuple

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

###
# Evaluate Function
###
def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end

def evaluate_fn(guessed, correct, last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts):
    guessed, guessed_type = parse_tag(guessed)
    correct, correct_type = parse_tag(correct)

    end_correct = end_of_chunk(last_correct, correct,
                               last_correct_type, correct_type)
    end_guessed = end_of_chunk(last_guessed, guessed,
                               last_guessed_type, guessed_type)
    start_correct = start_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
    start_guessed = start_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)

    if in_correct:
        if (end_correct and end_guessed and
            last_guessed_type == last_correct_type):
            in_correct = False
            counts.correct_chunk += 1
            counts.t_correct_chunk[last_correct_type] += 1
        elif (end_correct != end_guessed or guessed_type != correct_type):
            in_correct = False

    if start_correct and start_guessed and guessed_type == correct_type:
        in_correct = True

    if start_correct:
        counts.found_correct += 1
        counts.t_found_correct[correct_type] += 1
    if start_guessed:
        counts.found_guessed += 1
        counts.t_found_guessed[guessed_type] += 1
    if correct == guessed and guessed_type == correct_type:
        counts.correct_tags += 1
    counts.token_counter += 1

    last_guessed = guessed
    last_correct = correct
    last_guessed_type = guessed_type
    last_correct_type = correct_type

    return last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts

def evaluate(hyps_list, labels_list):
    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for hyps, labels in zip(hyps_list, labels_list):
        for hyp, label in zip(hyps, labels):
            step_result = evaluate_fn(hyp, label, last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts)
            last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts = step_result
        # Boundary between sentence
        step_result = evaluate_fn('O', 'O', last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts)
        last_correct, last_correct_type, last_guessed, last_guessed_type, in_correct, counts = step_result

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts

###
# Calculate Metrics Function
###
def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]

def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else (2 * p * r) / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type
    return overall

###
# Main Function
###
def conll_evaluation(hyps_list, labels_list):
    counts = evaluate(hyps_list, labels_list)
    overall, by_type = metrics(counts)

    c = counts
    acc = c.correct_tags / c.token_counter
    pre = overall.prec
    rec = overall.rec
    f1 = overall.fscore

    type_macro_pre = 0.0
    type_macro_rec = 0.0
    type_macro_f1 = 0.0
    for k in by_type.keys():
        type_macro_pre += by_type[k].prec
        type_macro_rec += by_type[k].rec
        type_macro_f1 += by_type[k].fscore

    type_macro_pre = type_macro_pre / float(len(by_type))
    type_macro_rec = type_macro_rec / float(len(by_type))
    type_macro_f1 = type_macro_f1 / float(len(by_type))

    return (acc, pre, rec, f1, type_macro_pre, type_macro_rec, type_macro_f1)



    import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers import AutoTokenizer, AutoConfig


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels_list

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, num_label) for num_label in self.num_labels])

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        subword_to_word_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = self.dropout(outputs[1])
        logits = []
        for classifier in self.classifiers:
            logit = classifier(sequence_output)
            logits.append(logit)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            total_loss = 0
            for i, (logit, num_label) in enumerate(zip(logits, self.num_labels)):
                label = labels[:,i]
                loss = loss_fct(logit.view(-1, num_label), label.view(-1))
                total_loss += loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

if __name__ == '__main__':
    x = torch.LongTensor([[301,302,303,304]])
    y = torch.LongTensor([[0,1,0,1,0,1]])

    print("BertForMultiLabelClassification")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.num_labels_list = [3,3,3,3,3,3]
    model = BertForMultiLabelClassification.from_pretrained("bert-base-uncased", config=config)
    output = model(x, labels=y)
    print(output[0], output[1])


import torch

###
# Forward Function
###

# Forward function for sequence classification
def forward_sequence_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 3:
        (subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data

    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2]

    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label_batch[j][0].item()])

    return loss, list_hyp, list_label

# Forward function for word classification
def forward_word_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 4:
        (subword_batch, mask_batch, subword_to_word_indices_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 5:
        (subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, label_batch) = batch_data

    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    subword_to_word_indices_batch = torch.LongTensor(subword_to_word_indices_batch)
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        subword_to_word_indices_batch = subword_to_word_indices_batch.cuda()
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(subword_batch, subword_to_word_indices_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2]

    # generate prediction & label list
    list_hyps = []
    list_labels = []
    hyps_list = torch.topk(logits, k=1, dim=-1)[1].squeeze(dim=-1)
    for i in range(len(hyps_list)):
        hyps, labels = hyps_list[i].tolist(), label_batch[i].tolist()
        list_hyp, list_label = [], []
        for j in range(len(hyps)):
            if labels[j] == -100:
                break
            else:
                list_hyp.append(i2w[hyps[j]])
                list_label.append(i2w[labels[j]])
        list_hyps.append(list_hyp)
        list_labels.append(list_label)

    return loss, list_hyps, list_labels

# Forward function for sequence multilabel classification
def forward_sequence_multi_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 3:
        (subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data

    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(subword_batch, attention_mask=mask_batch, token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2] # logits list<tensor(bs, num_label)> ~ list of batch prediction per class

    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = [torch.topk(logit, 1)[1] for logit in logits] # list<tensor(bs)>
    batch_size = label_batch.shape[0]
    num_label = len(hyp)
    for i in range(batch_size):
        hyps = []
        labels = label_batch[i,:].cpu().numpy().tolist()
        for j in range(num_label):
            hyps.append(hyp[j][i].item())
        list_hyp.append([i2w[hyp] for hyp in hyps])
        list_label.append([i2w[label] for label in labels])

    return loss, list_hyp, list_label


import itertools
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
#from .conlleval import conll_evaluation

def absa_metrics_fn(list_hyp, list_label):
    # hyp and label are both list (multi label), flatten the list
    list_hyp = list(itertools.chain.from_iterable(list_hyp))
    list_label = list(itertools.chain.from_iterable(list_label))

    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average='macro')
    metrics["REC"] = recall_score(list_label, list_hyp, average='macro')
    metrics["PRE"] = precision_score(list_label, list_hyp, average='macro')
    return metrics



    import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

#####
# ABSA Restoran (Ulasan via kolom `preprocessed`)
#####
class AspectBasedSentimentAnalysisRestoDataset(Dataset):
    # Static constant variable
    ASPECT_DOMAIN = ['makanan', 'pelayanan', 'harga', 'suasana', 'fasilitas']
    LABEL2INDEX = {'negatif': 0, 'netral': 1, 'positif': 2}
    INDEX2LABEL = {0: 'negatif', 1: 'netral', 2: 'positif'}
    NUM_LABELS = [3, 3, 3, 3, 3]
    NUM_ASPECTS = 5

    # Update __init__ to correctly call load_dataset and accept lowercase
    def __init__(self, dataset_path, tokenizer, lowercase=False, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path, lowercase=lowercase) # Pass lowercase to load_dataset
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token

    # Corrected indentation for load_dataset
    def load_dataset(self, path, lowercase=False):
        df = pd.read_csv(path)
        # Create a mapping for expected sentiment strings to their processed form
        sentiment_mapping = {
            'negatif': 'negatif',
            'netral': 'netral',
            'positif': 'positif',
            'pos': 'positif' # Handle the 'pos' case explicitly
        }
        if lowercase:
            # Add lowercase versions to the mapping
            lower_sentiment_mapping = {k.lower(): v for k, v in sentiment_mapping.items()}
            sentiment_mapping.update(lower_sentiment_mapping)

        for aspect in self.ASPECT_DOMAIN:
            def map_sentiment(sen):
                processed_sen = str(sen).lower() if lowercase else str(sen)
                # Look up the processed sentiment in the mapping
                mapped_sen = sentiment_mapping.get(processed_sen, processed_sen) # Default to processed_sen if not in mapping
                try:
                    return self.LABEL2INDEX[mapped_sen]
                except KeyError:
                    print(f"KeyError: '{mapped_sen}' not found in LABEL2INDEX for aspect '{aspect}'. Original value: '{sen}', Processed value: '{processed_sen}'")
                    raise # Re-raise the error to see the traceback
            df[aspect] = df[aspect].apply(map_sentiment)
        return df

    # Add the __len__ method
    def __len__(self):
        return len(self.data)

    # You also need to implement __getitem__ for the DataLoader to iterate over the dataset
    # This is a minimal implementation assuming your data is already processed and stored in self.data
    # You will likely need to add tokenization and label extraction logic here
    # Corrected indentation for __getitem__
    def __getitem__(self, idx):
        # Example: Assuming self.data is a pandas DataFrame with a 'text' column and aspect columns
        text = self.data.iloc[idx]['preprocessed'] # Assuming 'preprocessed' column exists and contains the text
        labels = self.data.iloc[idx][self.ASPECT_DOMAIN].values.tolist()

        # Ensure text is a string before tokenizing
        text = str(text)

        # Tokenize the text - adjust according to your specific needs
        encoded_inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        subwords = encoded_inputs['input_ids'].squeeze()
        mask = encoded_inputs['attention_mask'].squeeze()

        # Assuming labels are already in the correct integer format from load_dataset
        label = torch.LongTensor(labels)

        # Return the tokenized input, attention mask, and labels
        return subwords, mask, label, text # Returning raw_seq (original text) as the last element

class AspectBasedSentimentAnalysisDataLoader(DataLoader):
    def __init__(self, dataset, max_seq_len=512, device='cpu', *args, **kwargs): # Added device argument
        super(AspectBasedSentimentAnalysisDataLoader, self).__init__(dataset=dataset, *args, **kwargs)
        self.num_aspects = dataset.NUM_ASPECTS
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        self.device = device # Store device

    def _collate_fn(self, batch):
        batch_size = len(batch)
        # The batch now contains (subwords, mask, label, raw_seq)
        subword_batch = [item[0] for item in batch]
        mask_batch = [item[1] for item in batch]
        label_batch = [item[2] for item in batch]
        seq_list = [item[3] for item in batch]

        # Pad the sequences in the batch
        max_seq_len = max(len(seq) for seq in subword_batch)
        max_seq_len = min(self.max_seq_len, max_seq_len)

        # Create tensors directly on the specified device
        padded_subword_batch = torch.zeros((batch_size, max_seq_len), dtype=torch.int64, device=self.device)
        padded_mask_batch = torch.zeros((batch_size, max_seq_len), dtype=torch.float32, device=self.device)
        padded_label_batch = torch.zeros((batch_size, self.num_aspects), dtype=torch.int64, device=self.device)


        for i in range(batch_size):
            seq_len = len(subword_batch[i])
            # Ensure source tensors are on the same device before copying
            padded_subword_batch[i, :seq_len] = subword_batch[i][:max_seq_len].to(self.device)
            padded_mask_batch[i, :seq_len] = mask_batch[i][:max_seq_len].to(self.device)
            padded_label_batch[i, :] = label_batch[i].to(self.device)


        return padded_subword_batch, padded_mask_batch, padded_label_batch, seq_list





        # %%
import os
#import os, sys
#sys.path.append('../')
#os.chdir('../')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from transformers import BertTokenizer, BertConfig, BertForPreTraining
from nltk.tokenize import TweetTokenizer, word_tokenize

#from modules.multi_label_classification import BertForMultiLabelClassification
#from utils.forward_fn import forward_sequence_multi_classification
#from utils.metrics import absa_metrics_fn
#from utils.data_utils import AspectBasedSentimentAnalysisAiryDataset, AspectBasedSentimentAnalysisDataLoader




###
# common functions
###
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)




    # Set random seed
set_seed(26092020)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset_path = 'train.csv'
valid_dataset_path = 'val.csv'
test_dataset_path = 'test.csv'

train_dataset = AspectBasedSentimentAnalysisRestoDataset(train_dataset_path, tokenizer, lowercase=True)
valid_dataset = AspectBasedSentimentAnalysisRestoDataset(valid_dataset_path, tokenizer, lowercase=True)
test_dataset = AspectBasedSentimentAnalysisRestoDataset(test_dataset_path, tokenizer, lowercase=True)

# Pass the device to the DataLoader
train_loader = AspectBasedSentimentAnalysisDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=16, num_workers=2, shuffle=True, device=device)
# Corrected the class name here
valid_loader = AspectBasedSentimentAnalysisDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=16, num_workers=2, shuffle=False, device=device)
test_loader = AspectBasedSentimentAnalysisDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=16, num_workers=2, shuffle=False, device=device)



import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers import AutoTokenizer, AutoConfig


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Ensure config.num_labels_list is a list, even if it has only one element
        self.num_labels = config.num_labels_list if isinstance(config.num_labels_list, list) else [config.num_labels_list]

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # Ensure there is a classifier for each num_label specified in the list
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, num_label) for num_label in self.num_labels])

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        subword_to_word_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        device=None,  # Add device as an argument
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # Use the pooled output for sequence-level classification
        sequence_output = self.dropout(outputs[1])
        logits = []
        for classifier in self.classifiers:
            logit = classifier(sequence_output)
            logits.append(logit)

        # logits is now a list of tensors, where each tensor is (batch_size, num_labels_for_that_classifier)
        # outputs should now be (loss), logits (as a list), (hidden_states), (attentions)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # Use the provided device argument to move labels
            labels = labels.to(device)
            loss_fct = CrossEntropyLoss()
            total_loss = 0
            # Iterate through each classifier's logits and corresponding label column
            for i, logit in enumerate(logits):
                 # Select the i-th column of the labels for the i-th classifier
                label = labels[:,i]
                # Reshape logit to (batch_size * sequence_length, num_labels) and label to (batch_size * sequence_length)
                # For sequence classification, sequence_length is 1 since we are using pooled output
                # So reshape logit to (batch_size, num_labels) and label to (batch_size)
                loss = loss_fct(logit.view(-1, self.num_labels[i]), label.view(-1))
                total_loss += loss
            outputs = (total_loss,) + outputs

        # Return logits as a list of tensors, as expected by forward_sequence_multi_classification
        return outputs  # (loss), logits (list of tensors), (hidden_states), (attentions)

if __name__ == '__main__':
    x = torch.LongTensor([[301,302,303,304]])
    y = torch.LongTensor([[0,1,0,1,0]]) # Example labels for 5 aspects

    print("BertForMultiLabelClassification")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("bert-base-uncased")
    # Corrected num_labels_list to match the 5 aspects
    config.num_labels_list = [3,3,3,3,3]
    model = BertForMultiLabelClassification.from_pretrained("bert-base-uncased", config=config)
    # Ensure labels match the expected shape for 5 aspects
    y = torch.LongTensor([[0,1,0,1,0]]) # Reshaped for 5 labels
    output = model(x, labels=y, device=model.device) # Pass device in test case
    print(output[0], output[1])



    count_param(model)



train_dataset_path = 'train.csv'
valid_dataset_path = 'val.csv'
test_dataset_path = 'test.csv'

train_dataset = AspectBasedSentimentAnalysisRestoDataset(train_dataset_path, tokenizer, lowercase=True)
valid_dataset = AspectBasedSentimentAnalysisRestoDataset(valid_dataset_path, tokenizer, lowercase=True)
test_dataset = AspectBasedSentimentAnalysisRestoDataset(test_dataset_path, tokenizer, lowercase=True)

train_loader = AspectBasedSentimentAnalysisDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=16, num_workers=2, shuffle=True)
valid_loader = AspectBasedSentimentAnalysisDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=16, num_workers=2, shuffle=False)
test_loader = AspectBasedSentimentAnalysisDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=16, num_workers=2, shuffle=False)



w2i, i2w = AspectBasedSentimentAnalysisRestoDataset.LABEL2INDEX, AspectBasedSentimentAnalysisRestoDataset.INDEX2LABEL
print(w2i)
print(i2w)

text = 'tempat makan favorit keluarga makan khas sundanya juara banget rasa otentik porsi pas'
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

logits = model(subwords)[0]
labels = [torch.topk(logit, k=1, dim=-1)[1].squeeze().item() for logit in logits]

print(f'Text: {text}')
for i, label in enumerate(labels):
    print(f'Label `{AspectBasedSentimentAnalysisRestoDataset.ASPECT_DOMAIN[i]}` : {i2w[label]} ({F.softmax(logits[i], dim=-1).squeeze()[label] * 100:.3f}%)')


    text = 'pertama kali sini langsung jatuh cinta sama nasi liwet mantap'
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

logits = model(subwords)[0]
labels = [torch.topk(logit, k=1, dim=-1)[1].squeeze().item() for logit in logits]

print(f'Text: {text}')
for i, label in enumerate(labels):
    print(f'Label `{AspectBasedSentimentAnalysisRestoDataset.ASPECT_DOMAIN[i]}` : {i2w[label]} ({F.softmax(logits[i], dim=-1).squeeze()[label] * 100:.3f}%)')



    text = 'sate maranggi sop buntut enak banget recommended buat cari makan sunda bogor'
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

logits = model(subwords)[0]
labels = [torch.topk(logit, k=1, dim=-1)[1].squeeze().item() for logit in logits]

print(f'Text: {text}')
for i, label in enumerate(labels):
    print(f'Label `{AspectBasedSentimentAnalysisRestoDataset.ASPECT_DOMAIN[i]}` : {i2w[label]} ({F.softmax(logits[i], dim=-1).squeeze()[label] * 100:.3f}%)')


    optimizer = optim.Adam(model.parameters(), lr=1e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Gunakan string untuk device agar konsisten dengan fungsi
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(torch.device(device))  # pastikan model benar-benar pindah ke device

scaler = GradScaler()  # Mixed Precision Training
n_epochs = 15
grad_accumulation_steps = 2  # Mengurangi beban per update
save_checkpoint_every = 5  # Simpan model tiap 5 epoch

# Forward function for sequence multilabel classification
def forward_sequence_multi_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data. From DataLoader collate_fn, batch_data is (padded_subword_batch, padded_mask_batch, padded_label_batch, seq_list)
    # So input_data is (subword_batch, mask_batch), labels_batch is padded_label_batch
    # seq_list is the last element but is not used in the forward pass
    subword_batch, mask_batch, label_batch, seq_list = batch_data

    # Explicitly move subword_batch, mask_batch, and label_batch to the specified device
    # Although DataLoader attempts this, explicit movement here adds robustness
    subword_batch = subword_batch.to(device)
    mask_batch = mask_batch.to(device)
    label_batch = label_batch.to(device)


    # Check if token_type_batch is needed and present (not the case for this dataset)
    token_type_batch = None

    # Forward model
    # Pass only the relevant tensors to the model
    outputs = model(
        input_ids=subword_batch,
        attention_mask=mask_batch,
        token_type_ids=token_type_batch,
        labels=label_batch,
        device=device # Pass device to the model's forward
    )
    loss, logits = outputs[:2] # logits list<tensor(bs, num_label)> ~ list of batch prediction per class

    # generate prediction & label list
    list_hyp = []
    list_label = []
    # Move logits to CPU before processing to avoid potential device issues
    # Each logit is a tensor of shape (batch_size, num_labels_for_that_classifier)
    # We need to take the top 1 predicted label for each sample in the batch for each classifier
    hyp = [torch.topk(logit.cpu(), 1)[1].squeeze(dim=-1) for logit in logits] # list<tensor(bs)>
    batch_size = label_batch.shape[0]
    num_label = len(hyp)
    # Move label_batch to CPU for list conversion
    labels_cpu = label_batch.cpu().numpy().tolist()

    for i in range(batch_size):
        hyps = []
        labels = labels_cpu[i]
        for j in range(num_label):
            # Get the predicted label for the i-th sample from the j-th classifier
            hyps.append(hyp[j][i].item())
        list_hyp.append([i2w[hyp_item] for hyp_item in hyps])
        list_label.append([i2w[label_item] for label_item in labels])


    return loss, list_hyp, list_label


for epoch in range(n_epochs):
    model.train()
    torch.set_grad_enabled(True)

    total_train_loss = 0
    list_hyp, list_label = [], []

    train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
    optimizer.zero_grad()

    for i, batch_data in enumerate(train_pbar):
        # batch_data from DataLoader collate_fn is (padded_subword_batch, padded_mask_batch, padded_label_batch, seq_list)
        # All tensors (subword_batch, mask_batch, label_batch) are already on the correct device thanks to the updated DataLoader
        with autocast():  # Mixed precision untuk menghemat memori
            loss, batch_hyp, batch_label = forward_sequence_multi_classification(
                model, batch_data, i2w=i2w, device=device # Pass the full batch_data
            )

        # Gradient accumulation untuk batch kecil
        scaler.scale(loss).backward()

        if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Logging training loss
        total_train_loss += loss.item()
        list_hyp += batch_hyp
        list_label += batch_label

        train_pbar.set_description(
            f"(Epoch {epoch+1}) TRAIN LOSS: {total_train_loss/(i+1):.4f} LR: {get_lr(optimizer):.8f}"
        )

    # Calculate train metric
    metrics = absa_metrics_fn(list_hyp, list_label)
    print(f"(Epoch {epoch+1}) TRAIN LOSS: {total_train_loss/(i+1):.4f} {metrics_to_string(metrics)} LR: {get_lr(optimizer):.8f}")

    # Save checkpoint setiap beberapa epoch
    if (epoch + 1) % save_checkpoint_every == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    # Evaluation
    model.eval()
    torch.set_grad_enabled(False)

    total_loss, list_hyp, list_label = 0, [], []
    pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))

    for i, batch_data in enumerate(pbar):
        # batch_data from DataLoader collate_fn is (padded_subword_batch, padded_mask_batch, padded_label_batch, seq_list)
        # All tensors (subword_batch, mask_batch, label_batch) are already on the correct device thanks to the updated DataLoader
        with autocast():
            loss, batch_hyp, batch_label = forward_sequence_multi_classification(
                model, batch_data, i2w=i2w, device=device # Pass the full batch_data
            )

        total_loss += loss.item()
        list_hyp += batch_hyp
        list_label += batch_label
        metrics = absa_metrics_fn(list_hyp, list_label)

        pbar.set_description(f"VALID LOSS: {total_loss/(i+1):.4f} {metrics_to_string(metrics)}")

    metrics = absa_metrics_fn(list_hyp, list_label)
    print(f"(Epoch {epoch+1}) VALID LOSS: {total_loss/(i+1):.4f} {metrics_to_string(metrics)}")

    # Bersihkan cache GPU setelah setiap epoch
    torch.cuda.empty_cache()



        # === Tambahkan ini untuk confusion matrix ===
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    aspects = ['makanan', 'pelayanan', 'harga', 'suasana', 'fasilitas']

    for idx, aspect in enumerate(aspects):
        true_labels = [label[idx] for label in list_label]
        pred_labels = [pred[idx] for pred in list_hyp]

        cm = confusion_matrix(true_labels, pred_labels, labels=['negatif', 'netral', 'positif'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negatif', 'netral', 'positif'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {aspect.capitalize()}')
        plt.show()



        from sklearn.metrics import classification_report

# Evaluasi per aspek: Akurasi, Presisi, Recall, F1
print("\n=== Evaluasi Performa Model Per Aspek ===")
for idx, aspect in enumerate(aspects):
    true_labels = [label[idx] for label in list_label]
    pred_labels = [pred[idx] for pred in list_hyp]

    print(f"\n=== Aspek: {aspect.capitalize()} ===")
    report = classification_report(
        true_labels,
        pred_labels,
        labels=['negatif', 'netral', 'positif'],
        digits=4,
        zero_division=0
    )
    print(report)


from wordcloud import WordCloud
from collections import defaultdict
import matplotlib.pyplot as plt

# Asumsikan seq_list disimpan saat validasi, atau modifikasi forward function untuk mengembalikannya
# Jika seq_list tidak disimpan, Anda perlu menyesuaikan untuk menyimpan teks asli ulasan

# Contoh dummy: misalnya kita menyimpan seq_list saat validasi
all_texts = []  # List of ulasan/kalimat asli
for batch_data in valid_loader:
    _, _, _, seq_list = batch_data  # Ambil teks asli
    all_texts.extend(seq_list)

# Pastikan jumlah prediksi sama dengan jumlah teks
assert len(all_texts) == len(list_hyp), "Mismatch between texts and predictions"

# Kelompokkan teks berdasarkan label prediksi tiap sentimen
sentimen_wordcloud = {
    'negatif': [],
    'netral': [],
    'positif': []
}

# Gabungkan semua aspek, lalu kumpulkan kalimat berdasarkan sentimen dominan
for i, hyp in enumerate(list_hyp):
    # Misal, ambil sentimen dominan (yang paling sering muncul di kelima aspek)
    sentimen_counts = defaultdict(int)
    for s in hyp:  # hyp adalah list 5 aspek
        sentimen_counts[s] += 1
    # Ambil sentimen yang paling dominan (jika tie, ambil urutan alfabet)
    dominant_sentiment = sorted(sentimen_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    sentimen_wordcloud[dominant_sentiment].append(all_texts[i])

# Buat wordcloud untuk masing-masing sentimen
for sentiment in ['negatif', 'netral', 'positif']:
    text = " ".join(sentimen_wordcloud[sentiment])
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds' if sentiment == 'negatif' else 'Greens' if sentiment == 'positif' else 'Blues').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Wordcloud Sentimen {sentiment.capitalize()}")
    plt.show()



import matplotlib.pyplot as plt
from collections import Counter

# Aspek-aspek yang dianalisis
aspects = ['makanan', 'pelayanan', 'harga', 'suasana', 'fasilitas']
sentiment_labels = ['negatif', 'netral', 'positif']
colors = ['#FF9999', '#FFE066', '#90EE90']  # Warna: negatif, netral, positif

# Hitung distribusi sentimen per aspek
for idx, aspect in enumerate(aspects):
    # Ambil label prediksi dari hasil validasi
    predicted_sentiments = [pred[idx] for pred in list_hyp]

    # Hitung frekuensi tiap label
    sentiment_counts = Counter(predicted_sentiments)
    counts = [sentiment_counts.get(label, 0) for label in sentiment_labels]

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(
        counts,
        labels=sentiment_labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title(f'Distribusi Sentimen - Aspek {aspect.capitalize()}')
    plt.axis('equal')
    plt.show()
