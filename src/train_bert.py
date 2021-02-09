import os
import sys
import random
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from torch import optim
from tqdm import tqdm
from interpreter_bert import BertGradientInterpreter

dataname2path = {'sst': "../result/SST2/", 'yelp': "../result/Yelp/", 'agnews': "../result/AGNews/"}


def train_bert(args, dataloaders, model, DEVICE):
    train_dataloader, valid_dataloader, test_dataloader, num_train_examples = dataloaders
    loss_function = CrossEntropyLoss()
    num_labels = 2
    model.to(DEVICE)
    best_dev_acc = 0.0
    path_save = dataname2path[args.dataset]
    WEIGHTS_NAME = "bert.bin"
    CONFIG_NAME = "bert_config.json"

    list_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in list_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in list_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    WARMUP_PROPORTION = 0.1
    GRADIENT_ACCUMULATION_STEPS = 1
    num_train_optimization_steps = int(
        num_train_examples / args.batch_size / GRADIENT_ACCUMULATION_STEPS) * args.n_epochs
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-5,
                         warmup=WARMUP_PROPORTION,
                         t_total=num_train_optimization_steps)

    for epoch in range(args.n_epochs):
        model.train()
        train_loss = 0.0
        cnt = 0.
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            indexed_tokens, segments_ids, input_masks, labels = batch

            # ## test interpretation
            # VBP = BertGradientInterpreter(model)
            # vanilla_grads = VBP.generate_gradients(indexed_tokens, segments_ids, input_masks, labels)
            # print(vanilla_grads)
            # sys.exit('test ends...')

            logits = model(indexed_tokens, segments_ids, input_masks, labels=None)
            optimizer.zero_grad()
            loss = loss_function(logits.view(-1, num_labels), labels.view(-1))
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            train_loss += loss.item()
            print("\r%f" % loss, end='')

            loss.backward()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
            cnt += 1

        train_loss /= cnt
        tqdm.write('Train loss %.3f' % train_loss)

        acc_train = evaluate(train_dataloader, model, DEVICE, num_labels)
        acc_dev = evaluate(valid_dataloader, model, DEVICE, num_labels)
        acc_test = evaluate(test_dataloader, model, DEVICE, num_labels)
        print('Accuracy of training %.3f, validation %.3f, testing %.3f' % (acc_train, acc_dev, acc_test))

        if acc_dev > best_dev_acc:
            best_dev_acc = acc_dev
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            output_model_file = os.path.join(path_save, WEIGHTS_NAME)
            output_config_file = os.path.join(path_save, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)


def evaluate(dataloader, model, DEVICE, num_labels):
    model.eval()
    true_res = []
    pred_res = []
    for batch in dataloader:
        batch = tuple(t.to(DEVICE) for t in batch)
        indexed_tokens, segments_ids, input_masks, labels = batch
        logits = model(indexed_tokens, segments_ids, input_masks, labels=None)
        logits = logits.view(-1, num_labels)
        pred_label = logits.data.max(1)[1].cpu().numpy()
        pred_res += [x for x in pred_label]
        true_res += list(labels.data)
    acc = compute_accuracy(true_res, pred_res)
    return acc

def evaluate_2(dataloader, model, DEVICE, num_labels):
    word_embeddings = model.bert.embeddings.word_embeddings
    model.eval()
    true_res = []
    pred_res = []
    for batch in dataloader:
        batch = tuple(t.to(DEVICE) for t in batch)
        indexed_tokens, segments_ids, input_masks, labels = batch
        embds = word_embeddings(indexed_tokens)
        logits = model(inputs_embeds=embds,
                       token_type_ids=segments_ids,
                       attention_mask=input_masks,
                       labels=None)[0]
        logits = logits.view(-1, num_labels)
        pred_label = logits.data.max(1)[1].cpu().numpy()
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
