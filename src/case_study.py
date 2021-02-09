import argparse
import os
import sys
import random
import torch
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertForSequenceClassification
from load_data import load_data, load_embedding
from load_data_bert import load_data_bert
from model_lstm import LSTMBC
from model_lstmatt import LSTMAttBC
from train import train
from train_bert import train_bert
from interpret import interpret_lstm
from interpret_bert import interpret_bert
from interpret_lstmatt import interpret_lstmatt
from align_interpretation import align_interpretation


def main_lstm(args, DEVICE):
    train_iter, dev_iter, test_iter, text_field, label_field = load_data(args, DEVICE)
    pretrained_embeddings = load_embedding(text_field)

    model = LSTMBC(args.dim_embd, args.dim, args.batch_size, pretrained_embeddings,
                   vocab_size=len(text_field.vocab), label_size=len(label_field.vocab) - 1,
                   DEVICE=DEVICE)

    align_interpretation(args, [train_iter, dev_iter, test_iter], text_field, label_field, DEVICE)


def main_lstmatt(args, DEVICE):
    train_iter, dev_iter, test_iter, text_field, label_field = load_data(args, DEVICE)
    pretrained_embeddings = load_embedding(text_field)

    model = LSTMAttBC(args.dim_embd, args.dim, args.batch_size, pretrained_embeddings,
                      vocab_size=len(text_field.vocab), label_size=len(label_field.vocab) - 1,
                      DEVICE=DEVICE)




def main_bert(args, DEVICE):
    train_dataloader, valid_dataloader, test_dataloader, num_train_examples = load_data_bert(args, DEVICE)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          cache_dir='../result/', num_labels=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1, help='Which gpu to use')
    parser.add_argument('--dataset', type=str, default='sst', help='Which dataset to use')
    parser.add_argument('--encoder', type=str, choices=['cnn', 'lstm', 'lstmatt', 'gru', 'bert'],
                        default='lstm', help='Which encoder to use')
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=48)    # 16 for bert, lstmatt
    parser.add_argument('--n_epochs', type=int, default=3)      # <4 for bert, lstmatt
    parser.add_argument('--dim_embd', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--dim', type=int, default=150, help='Hidden dimension')
    args = parser.parse_args()
    DEVICE = torch.device('cuda:{}'.format(args.device))
    #DEVICE = torch.device('cpu')

    if args.encoder == 'lstm':
        main_lstm(args, DEVICE)
    elif args.encoder == 'lstmatt':
        main_lstmatt(args, DEVICE)
    elif args.encoder == 'bert':
        main_bert(args, DEVICE)

