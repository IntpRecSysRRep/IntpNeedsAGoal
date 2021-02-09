import os
import random
import shutil
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchtext import data

dataname2path = {'sst': "../data/SST2/", 'yelp': "../data/Yelp/", 'agnews': "../data/AGNews/",
                 'imdb': "../data/IMDB/"}


def load_data(args, DEVICE):
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)

    if args.dataset == 'imdb':
        train, dev, test = data.TabularDataset.splits(path=dataname2path[args.dataset], train='train.txt',
                                                      validation='dev.txt', test='test.txt', format='tsv',
                                                      fields=[('text', text_field), ('label', label_field)])
    else:
        train, dev, test = data.TabularDataset.splits(path=dataname2path[args.dataset], train='train.tsv',
                                                      validation='dev.tsv', test='test.tsv', format='tsv',
                                                      fields=[('text', text_field), ('label', label_field)])

    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                                                 batch_sizes=(args.batch_size, len(dev), len(test)),
                                                                 sort_key=lambda x: len(x.text), repeat=False)
    return train_iter, dev_iter, test_iter, text_field, label_field


def load_embedding(text_field):
    print('Loading embeddings...')
    word_to_idx = text_field.vocab.stoi
    pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
    pretrained_embeddings[0] = 0
    # read only embeddings of the words in given dataset
    word2vec = load_bin_vec('../data/GoogleNews-vectors-negative300.bin', word_to_idx)
    for word, vector in word2vec.items():
        pretrained_embeddings[word_to_idx[word] - 1] = vector

    return pretrained_embeddings


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)     # read each letter
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def load_data_fmnist():
    root = '../data/Fmnist'
    BATCH_SIZE = 100
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True,
                                                 transform=transforms.ToTensor())   # The 'transform' is very useful.
    num_train = int(trainset.__len__() * 5 / 6)
    trainset, validset = torch.utils.data.random_split(trainset, [num_train, trainset.__len__() - num_train])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True)
    testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True,
                                                transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    print("Train, Val, Test sizes: %d, %d, %d" % (trainset.__len__(), validset.__len__(), testset.__len__()))

    return train_loader, valid_loader, test_loader, testset.__len__()


def load_data_imagenet():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(root='../data/Imagenet/images/', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

    return dataloader


if __name__ == "__main__":
    load_data_imagenet()
