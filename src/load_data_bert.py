import csv
import torch
import numpy as np
from torchtext import data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

dataname2path = {'sst': "../data/SST2/", 'yelp': "../data/Yelp/", 'agnews': "../data/AGNews/"}
max_seq_lens = {'sst': 64, 'yelp': 32, 'agnews': 64}
types_label = {'clf': torch.long, 'reg': torch.float}


class BertFeats(object):
    """A single set of features of data for Bert."""
    def __init__(self, indexed_tokens, input_mask, segments_ids, label):
        self.indexed_tokens = indexed_tokens
        self.input_mask = input_mask
        self.segments_ids = segments_ids
        self.label = label


def load_data_bert(args, DEVICE, mode='clf'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_seq_len = max_seq_lens[args.dataset]

    filename = dataname2path[args.dataset] + 'train.tsv'
    lists_train = process_tsv(filename, tokenizer, max_seq_len, mode)
    train_dataset = TensorDataset(torch.tensor(lists_train[0], dtype=torch.long),
                                  torch.tensor(lists_train[1], dtype=torch.long),
                                  torch.tensor(lists_train[2], dtype=torch.long),
                                  torch.tensor(lists_train[3], dtype=types_label[mode]))
    num_train_examples = lists_train[4]

    filename = dataname2path[args.dataset] + 'dev.tsv'
    lists_valid = process_tsv(filename, tokenizer, max_seq_len, mode)
    valid_dataset = TensorDataset(torch.tensor(lists_valid[0], dtype=torch.long),
                                  torch.tensor(lists_valid[1], dtype=torch.long),
                                  torch.tensor(lists_valid[2], dtype=torch.long),
                                  torch.tensor(lists_valid[3], dtype=types_label[mode]))

    filename = dataname2path[args.dataset] + 'test.tsv'
    lists_test = process_tsv(filename, tokenizer, max_seq_len, mode)
    test_dataset = TensorDataset(torch.tensor(lists_test[0], dtype=torch.long),
                                 torch.tensor(lists_test[1], dtype=torch.long),
                                 torch.tensor(lists_test[2], dtype=torch.long),
                                 torch.tensor(lists_test[3], dtype=types_label[mode]))

    train_sampler = RandomSampler(train_dataset)
    valid_sampler = RandomSampler(valid_dataset)
    test_sampler = RandomSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

    return train_dataloader, valid_dataloader, test_dataloader, num_train_examples


def process_tsv(filename, tokenizer, max_seq_length, mode):
    list_indexed_tokens = []
    list_segments_ids = []
    list_input_masks = []
    list_labels = []

    cnt = 0
    with open(filename, 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            text = row[0]
            label = row[1]
            if mode == 'clf':
                label = int(label)
            else:
                label = float(label)

            # turn raw text to tokens and ids
            tokenized_text = tokenizer.tokenize(text)       # text -> tokens
            if len(tokenized_text) > max_seq_length-2:      # control length
                tokenized_text = tokenized_text[:max_seq_length-2]
            tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)    # tokens -> token ids
            segments_ids = [0] * len(tokenized_text)        # segment 0 or 1

            # padding and assign mask
            input_mask = [1] * len(indexed_tokens)
            padding = [0] * (max_seq_length - len(indexed_tokens))
            indexed_tokens += padding
            segments_ids += padding
            input_mask += padding

            list_indexed_tokens.append(indexed_tokens)
            list_segments_ids.append(segments_ids)
            list_input_masks.append(input_mask)
            list_labels.append(label)

            cnt += 1

    return list_indexed_tokens, list_segments_ids, list_input_masks, list_labels, cnt
