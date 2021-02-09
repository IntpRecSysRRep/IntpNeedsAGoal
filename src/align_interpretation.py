import csv
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.nn as nn
from torch import optim
from model_lstm_intp import LSTMBCintp
from train import evaluate
from nltk.tokenize import word_tokenize
from utils import logit2prob, normalize_score, rescale_embedding, perturb_embedding, remove_word

dataname2datapath = {'sst': "../data/SST2/", 'yelp': "../data/Yelp/", 'agnews': "../data/AGNews/"}
dataname2modelpath = {'sst': "../result/SST2/", 'yelp': "../result/Yelp/", 'agnews': "../result/AGNews/"}
test_size = {'sst': 330}

def align_interpretation(args, data_iters, text_field, label_field, DEVICE, EPSLN=0.50):
    train_iter, dev_iter, test_iter = data_iters

    # Load pretrained model
    print('Model loading...')
    file_load = dataname2modelpath[args.dataset] + args.encoder + '.pkl'
    model_org = torch.load(file_load)
    model_org.to(DEVICE)
    acc_test = evaluate(test_iter, model_org, DEVICE)
    print('Model loaded, with testing accuracy: %.3f' % acc_test)
    model_org.batch_size = 1

    # Create a new model with embeddings as input, not one-hot index
    model = None
    if args.encoder == 'lstm':
        model = LSTMBCintp(args.dim_embd, args.dim, args.batch_size,
                           vocab_size=len(text_field.vocab), label_size=len(label_field.vocab) - 1, DEVICE=DEVICE)
    pretrained_dict = model_org.state_dict()
    new_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_dict}
    # 2. overwrite entries in the existing state dict
    new_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(new_dict)

    # Compute intersection of interpretation and rationale
    case_ids, list_texts, list_labels = get_examples(args)
    words_golden = get_rationale(args)
    words_golden = word_tokenize(' '.join(words_golden))

    word_to_idx = dict(text_field.vocab.stoi)
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    wids_golden = []
    for w_g in words_golden:
        try:
            wids_golden.append(word_to_idx[w_g])
        except KeyError:
            pass
    #print(wids_golden)

    ratio_gold = compute_ratio_gold(list_texts, list_labels, word_to_idx, wids_golden, model_org, model, DEVICE, EPSLN)
    print(ratio_gold)

    # Apply adversarial training over non-golden words
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.NLLLoss()
    model.to(DEVICE)
    best_dev_acc = 0.0
    path_save = dataname2modelpath[args.dataset]
    file_save = path_save + args.encoder + '.pkl'

    train_texts, train_labels = read_examples(args)
    for text_faith, label in tqdm(zip(train_texts, train_labels)):
        pass


def get_examples(args):
    list_texts = []
    list_labels = []

    with open(dataname2datapath[args.dataset] + 'study_ids.txt', 'r') as f:
        ids = f.readline().strip().split(',')
        ids = [int(id)-1 for id in ids]

    cnt = 0
    with open(dataname2datapath[args.dataset] + 'test.tsv', 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            if cnt in ids:
                list_texts.append(row[0])
                list_labels.append(1 - int(row[1]))
            cnt += 1
            if cnt > test_size[args.dataset]:
                break
    #print(list_texts[0])
    return ids, list_texts, list_labels


def get_rationale(args):
    words_golden = []
    with open(dataname2datapath[args.dataset] + 'study_words.txt', 'r') as f:
        for line in f:
            words_golden.append(line.strip())
    #print(words_golden)
    return words_golden


def compute_ratio_gold(list_texts, list_labels, word_to_idx, wids_golden, model_org, model, DEVICE, EPSLN):
    ratio_gold = 0.0
    for text_faith, label in tqdm(zip(list_texts, list_labels)):
        text_tokenized = word_tokenize(text_faith)
        # build word index array
        sentence = []
        for i in text_tokenized:
            if i != '.':
                try:
                    sentence.append(word_to_idx[i])
                except KeyError:
                    pass
        hit_gold = np.array([int(wid in wids_golden) for wid in sentence])
        sent = np.array(sentence)
        sent = torch.from_numpy(sent)
        len_sent = len(sentence)

        # predict
        model_org.hidden = model_org.init_hidden()
        input_vector = sent[:len_sent].to(DEVICE)
        pred, hn, x, _ = model_org(input_vector)
        pred = F.log_softmax(pred)
        pred_label = pred.cpu().data.max(1)[1].numpy()

        # interpret with vanilla gradient
        gradient, importance_score, _, _ = vanilla_gradient(model, x.detach(), pred_label, EPSLN)

        # compute mass ratio on golden words
        ratio_gold += np.sum(importance_score * hit_gold) / np.sum(importance_score)
    ratio_gold /= len(list_labels)

    return ratio_gold


def vanilla_gradient(model, x, pred_label, step_size):
    model.batch_size = 1
    model.hidden = model.init_hidden()
    x = x.cpu()
    x.requires_grad = True
    pred, _ = model(x)
    x_prior = x.data.numpy()
    p_prior = logit2prob(pred[0].data.numpy())

    one_hot = np.zeros((1, 2), dtype=np.float32)
    one_hot[0][pred_label[0]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot.requires_grad = True
    one_hot = torch.sum(one_hot * pred[0])

    gradient = grad(one_hot, x)[0].numpy()
    grad_l2 = np.sum(gradient[:, 0, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2) * step_size
    gradient /= np.sqrt(np.sum(gradient[:, 0, :] ** 2))  # normalize to unit length
    x_after = np.copy(x_prior)
    x_after = perturb_embedding(x_after, gradient*step_size)

    x_after = torch.from_numpy(x_after)
    model.hidden = model.init_hidden()
    pred, _ = model(x_after)
    p_after = logit2prob(pred[0].data.numpy())
    changes_pred = p_after - p_prior
    #print(pred_label)
    #print(importance_score)
    #print(changes_pred)

    return gradient, importance_score, x_after, changes_pred

def read_examples(args):
    train_texts = []
    train_labels = []
    with open(dataname2datapath[args.dataset] + 'train.tsv', 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            train_texts.append(row[0])
            train_labels.append(1 - int(row[1]))
    return train_texts, train_labels

