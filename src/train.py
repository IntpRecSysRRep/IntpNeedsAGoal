import os
import random
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

dataname2path = {'sst': "../result/SST2/", 'yelp': "../result/Yelp/", 'agnews': "../result/AGNews/",
                 'imdb': "../result/IMDB/"}

def train(args, data_iters, model, DEVICE):
    train_iter, dev_iter, test_iter = data_iters
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # sst: 1e-3, 1e-5
    loss_function = nn.NLLLoss()
    model.to(DEVICE)
    best_dev_acc = 0.0
    path_save = dataname2path[args.dataset]
    file_save = path_save + args.encoder + '.pkl'

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    if os.path.exists(file_save) and args.new_train is False:
        model = torch.load(file_save)
        acc_test = evaluate(test_iter, model, DEVICE)
        print('Model loaded, with testing accuracy: %.3f' % acc_test)
        return

    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_iter, desc='Train epoch ' + str(epoch + 1)):
            texts, labels = batch.text.to(DEVICE), batch.label.to(DEVICE)
            labels.data.sub_(1)
            model.batch_size = len(labels.data)
            model.hidden = model.init_hidden()
            preds = model(texts)[-1]

            model.zero_grad()
            loss = loss_function(preds, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_iter)
        tqdm.write('Train loss %.3f' % train_loss)

        acc_train = evaluate(train_iter, model, DEVICE)
        acc_dev = evaluate(dev_iter, model, DEVICE)
        acc_test = evaluate(test_iter, model, DEVICE)
        print('Accuracy of training %.3f, validation %.3f, testing %.3f' % (acc_train, acc_dev, acc_test))

        if acc_dev > best_dev_acc:
            best_dev_acc = acc_dev
            torch.save(model, path_save + args.encoder + '.pkl')


def evaluate(data_iter, model, DEVICE):
    model.eval()
    true_res = []
    pred_res = []
    for batch in data_iter:
        texts, labels = batch.text.to(DEVICE), batch.label.to(DEVICE)
        labels.data.sub_(1)
        model.batch_size = len(labels.data)
        model.hidden = model.init_hidden()
        preds = model(texts)[-1]
        pred_label = preds.data.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        true_res += list(labels.data)
    acc = compute_accuracy(true_res, pred_res)
    return acc


def compute_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)