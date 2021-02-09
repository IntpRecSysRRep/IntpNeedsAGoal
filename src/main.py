import argparse
import os
import sys
import random
import torch
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertForSequenceClassification
from load_data import load_data, load_embedding, load_data_fmnist, load_data_imagenet
from load_data_bert import load_data_bert
from model_lstm import LSTMBC
from model_lstmatt import LSTMAttBC
from model_cnnsmall import CNNSmall
from train import train
from train_bert import train_bert
from train_cnn import train_cnn
from interpret_lstm import interpret_lstm
from interpret_bert import interpret_bert
from interpret_lstmatt import interpret_lstmatt
from interpret_cnn import interpret_cnn


def main_lstm(args, DEVICE):
    train_iter, dev_iter, test_iter, text_field, label_field = load_data(args, DEVICE)
    pretrained_embeddings = load_embedding(text_field)

    model = LSTMBC(args.dim_embd, args.dim, args.batch_size, pretrained_embeddings,
                   vocab_size=len(text_field.vocab), label_size=len(label_field.vocab) - 1,
                   DEVICE=DEVICE)
    # Train if no pre-trained model is given
    if args.new_train:
        train(args, [train_iter, dev_iter, test_iter], model, DEVICE)

    interpret_lstm(args, test_iter, text_field, label_field, DEVICE)


def main_lstmatt(args, DEVICE):
    train_iter, dev_iter, test_iter, text_field, label_field = load_data(args, DEVICE)
    pretrained_embeddings = load_embedding(text_field)

    model = LSTMAttBC(args.dim_embd, args.dim, args.batch_size, pretrained_embeddings,
                      vocab_size=len(text_field.vocab), label_size=len(label_field.vocab) - 1,
                      DEVICE=DEVICE)
    # Train if no pre-trained model is given
    if args.new_train:
        train(args, [train_iter, dev_iter, test_iter], model, DEVICE)

    interpret_lstmatt(args, test_iter, text_field, label_field, DEVICE)


def main_bert(args, DEVICE):
    train_dataloader, valid_dataloader, test_dataloader, num_train_examples = load_data_bert(args, DEVICE)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          cache_dir='../result/', num_labels=2)
    # Train if no pre-trained model is given
    if args.new_train:
        train_bert(args, [train_dataloader, valid_dataloader, test_dataloader, num_train_examples], model, DEVICE)

    interpret_bert(args, test_dataloader, DEVICE)


def main_cnnsmall(args, DEVICE):
    train_dataloader, valid_dataloader, test_dataloader, num_test = load_data_fmnist()

    model = CNNSmall()
    if args.new_train:
        train_cnn(args, [train_dataloader, valid_dataloader, test_dataloader], model, DEVICE)

    interpret_cnn(args, test_dataloader, num_test, DEVICE)


def main_cnnvgg(args, DEVICE):
    dataloader = load_data_imagenet()
    interpret_cnn(args, dataloader, 300, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--dataset', type=str, choices=['sst', 'yelp', 'agnews', 'imdb', 'fmnist', 'imagenet'],
                        default='fmnist', help='Which dataset to use')
    parser.add_argument('--encoder', type=str, choices=['lstm', 'lstmatt', 'bert', 'cnn_small', 'vgg'],
                        default='cnn_small', help='Which encoder to use')
    parser.add_argument('--new_train', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=8) 
    parser.add_argument('--dim_embd', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--dim', type=int, default=150, help='Hidden dimension')
    args = parser.parse_args()
    DEVICE = torch.device('cuda:{}'.format(args.device))
    #DEVICE = torch.device('cpu')

    print('Dataset: ', args.dataset)
    print('Model type: ', args.encoder)
    if args.encoder == 'lstm':
        main_lstm(args, DEVICE)
    elif args.encoder == 'lstmatt':
        main_lstmatt(args, DEVICE)
    elif args.encoder == 'bert':
        main_bert(args, DEVICE)
    elif args.encoder == 'cnn_small':
        main_cnnsmall(args, DEVICE)
    elif args.encoder == 'vgg':
        main_cnnvgg(args, DEVICE)


# sst2
# parser.add_argument('--batch_size', type=int, default=128)    # 16 for bert, lstmatt
# parser.add_argument('--n_epochs', type=int, default=5)      # <4 for bert, lstmatt

# yelp
# parser.add_argument('--batch_size', type=int, default=128)    # 128 for bert, lstmatt
# parser.add_argument('--n_epochs', type=int, default=5)      # <4 for bert, lstmatt


