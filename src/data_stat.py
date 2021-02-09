import csv
import argparse
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from load_data import load_data, load_embedding
from load_data_bert import load_data_bert

data2folder = {'sst': 'SST2', 'yelp': 'Yelp', 'agnews': 'AGNews'}

def main(args, DEVICE):
    train_iter, dev_iter, test_iter, text_field, label_field = load_data(args, DEVICE)
    data_stat(data2folder[args.dataset], text_field)

def data_stat(folder, text_field):
    word_to_idx = dict(text_field.vocab.stoi)
    for data_type in ['train', 'dev', 'test']:
        file_name = '../data/' + folder + '/' + data_type + '.tsv'
        cnt = 0
        label_pos = 0
        label_neg = 0
        text_len = 0
        with open(file_name, 'r') as f:
            for row in csv.reader(f, delimiter='\t'):
                text = row[0]
                label = 1 - int(row[1])

                text_tokenized = word_tokenize(text)
                for i in text_tokenized:
                    if i != '.':
                        text_len += 1

                if label == 1:
                    label_pos += 1
                else:
                    label_neg += 1

                cnt += 1
        print(text_len/cnt, label_pos, label_neg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1, help='Which gpu to use')
    parser.add_argument('--dataset', type=str, default='agnews', help='Which dataset to use')
    parser.add_argument('--encoder', type=str, choices=['cnn', 'lstm', 'lstmatt', 'gru', 'bert'],
                        default='lstmatt', help='Which encoder to use')
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=48)  # 16 for bert, lstmatt
    parser.add_argument('--n_epochs', type=int, default=3)  # <4 for bert, lstmatt
    parser.add_argument('--dim_embd', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--dim', type=int, default=150, help='Hidden dimension')
    args = parser.parse_args()
    DEVICE = torch.device('cuda:{}'.format(args.device))
    # DEVICE = torch.device('cpu')

    main(args, DEVICE)
