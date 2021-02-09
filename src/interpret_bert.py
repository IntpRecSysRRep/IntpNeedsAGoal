import random
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import grad
from train_bert import evaluate, evaluate_2
from model_bert_intp import BertClassifier
from transformers import BertForSequenceClassification
from utils import logit2prob, normalize_score, perturb_embedding, remove_word_bert, word2zero_bert

dataname2datapath = {'sst': "../data/SST2/", 'yelp': "../data/Yelp/", 'agnews': "../data/AGNews/"}
dataname2modelpath = {'sst': "../result/SST2/", 'yelp': "../result/Yelp/", 'agnews': "../result/AGNews/"}
test_size = {'sst': 1800, 'yelp': 3500, 'agnews': 3500}


def interpret_bert(args, data_iter, DEVICE, EPSLN=0.15, MAX_RM=4):
    # Load pretrained model
    print('Model loading...')
    pre_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              cache_dir='../result/', num_labels=2)
    path_load = dataname2modelpath[args.dataset]
    WEIGHTS_NAME = "bert.bin"
    num_labels = 2
    filename = os.path.join(path_load, WEIGHTS_NAME)
    pre_model.load_state_dict(torch.load(filename))
    pre_model.to(DEVICE)
    # acc_test = evaluate_2(data_iter, pre_model, DEVICE, num_labels)
    # print('Model loaded, with testing accuracy: %.2f' % acc_test)

    # Interpret sampled instances
    word_embeddings = pre_model.bert.embeddings.word_embeddings
    sample_ids = get_examples(args)
    avg_changes_pred_vanilaGrad, avg_changes_rm_pred_vanilaGrad = 0., np.zeros(MAX_RM)
    avg_changes_pred_iterGrad, avg_changes_rm_pred_iterGrad = 0., np.zeros(MAX_RM)
    avg_changes_pred_smoothGrad, avg_changes_rm_pred_smoothGrad = 0., np.zeros(MAX_RM)
    avg_changes_pred_inputGrad, avg_changes_rm_pred_inputGrad = 0, np.zeros(MAX_RM)
    avg_changes_pred_inteGrad, avg_changes_rm_pred_inteGrad = 0, np.zeros(MAX_RM)
    cnt, cnt_hit = 0, 0
    avg_ct_dist_vanilaGrad, cnt_ct_dist_vanilaGrad = 0, 0.0001
    avg_ct_dist_iterGrad, cnt_ct_dist_iterGrad = 0, 0.0001
    avg_ct_dist_smoothGrad, cnt_ct_dist_smoothGrad = 0, 0.0001
    avg_ct_dist_inputGrad, cnt_ct_dist_inputGrad = 0, 0.0001
    avg_ct_dist_inteGrad, cnt_ct_dist_inteGrad = 0, 0.0001
    for row in tqdm(data_iter):
        if cnt not in sample_ids:
            cnt += 1
            continue
        row = tuple(t.to(DEVICE) for t in row)
        indexed_tokens, segments_ids, input_masks, labels = row
        embds = word_embeddings(indexed_tokens).detach()
        # predict
        pred = pre_model(inputs_embeds=embds,
                         token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
        pred = pred.view(-1, num_labels)
        pred_label = pred.cpu().data.max(1)[1].numpy()

        # vanilla gradient
        row = [embds.clone().detach(), segments_ids, input_masks]
        gradient, importance_score, x_after, changes_pred = vanilla_gradient(pre_model, row, pred_label, DEVICE,
                                                                             step_size=EPSLN)
        avg_changes_pred_vanilaGrad += -changes_pred[pred_label[0]]
        row = [embds.clone().detach(), segments_ids, input_masks]
        avg_changes_rm_pred_vanilaGrad += evaluate_word_2zero(pre_model, row, pred_label,
                                                              gradient, gradient, DEVICE)
        ct_dist, ct_flag = find_counterfactual_distance(pre_model, row, x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_vanilaGrad += ct_dist * ct_flag
        cnt_ct_dist_vanilaGrad += ct_flag

        # iterative gradient
        row = [embds.clone().detach(), segments_ids, input_masks]
        x_delta, importance_score, x_after, changes_pred = iterative_gradient(pre_model, row, pred_label, DEVICE,
                                                                              step_size=0.02, epsilon=EPSLN)
        avg_changes_pred_iterGrad += -changes_pred[pred_label[0]]
        row = [embds.clone().detach(), segments_ids, input_masks]
        avg_changes_rm_pred_iterGrad += evaluate_word_2zero(pre_model, row, pred_label,
                                                            x_delta, x_delta, DEVICE)
        ct_dist, ct_flag = find_counterfactual_distance(pre_model, row, x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_iterGrad += ct_dist * ct_flag
        cnt_ct_dist_iterGrad += ct_flag

        # smooth gradient
        row = [embds.clone().detach(), segments_ids, input_masks]
        smooth_grad, importance_score, x_after, changes_pred = smooth_gradient(pre_model, row, pred_label, DEVICE,
                                                                               step_size=EPSLN)
        avg_changes_pred_smoothGrad += -changes_pred[pred_label[0]]
        row = [embds.clone().detach(), segments_ids, input_masks]
        avg_changes_rm_pred_smoothGrad += evaluate_word_2zero(pre_model, row, pred_label,
                                                              smooth_grad, smooth_grad, DEVICE)
        ct_dist, ct_flag = find_counterfactual_distance(pre_model, row, x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_smoothGrad += ct_dist * ct_flag
        cnt_ct_dist_smoothGrad += ct_flag

        # gradient times input
        row = [embds.clone().detach(), segments_ids, input_masks]
        input_grad, importance_score, x_after, changes_pred = gradient_times_input(pre_model, row, pred_label, DEVICE,
                                                                                   step_size=EPSLN)
        avg_changes_pred_inputGrad += -changes_pred[pred_label[0]]
        row = [embds.clone().detach(), segments_ids, input_masks]
        avg_changes_rm_pred_inputGrad += evaluate_word_2zero(pre_model, row, pred_label,
                                                             input_grad, 1., DEVICE)
        ct_dist, ct_flag = find_counterfactual_distance(pre_model, row, x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_inputGrad += ct_dist * ct_flag
        cnt_ct_dist_inputGrad += ct_flag

        # integrated gradient
        row = [embds.clone().detach(), segments_ids, input_masks]
        inte_grad, importance_score, x_after, changes_pred = integrated_gradient(pre_model, row, pred_label, DEVICE,
                                                                                 step_size=EPSLN)
        avg_changes_pred_inteGrad += -changes_pred[pred_label[0]]
        row = [embds.clone().detach(), segments_ids, input_masks]
        avg_changes_rm_pred_inteGrad += evaluate_word_2zero(pre_model, row, pred_label,
                                                            inte_grad, 1., DEVICE)
        ct_dist, ct_flag = find_counterfactual_distance(pre_model, row, x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_inteGrad += ct_dist * ct_flag
        cnt_ct_dist_inteGrad += ct_flag

        cnt += 1
        cnt_hit += 1
    avg_changes_pred_vanilaGrad /= cnt_hit
    avg_changes_pred_iterGrad /= cnt_hit
    avg_changes_pred_smoothGrad /= cnt_hit
    avg_changes_pred_inputGrad /= cnt_hit
    avg_changes_pred_inteGrad /= cnt_hit
    avg_changes_rm_pred_vanilaGrad /= cnt_hit
    avg_changes_rm_pred_iterGrad /= cnt_hit
    avg_changes_rm_pred_smoothGrad /= cnt_hit
    avg_changes_rm_pred_inputGrad /= cnt_hit
    avg_changes_rm_pred_inteGrad /= cnt_hit
    print(avg_changes_pred_vanilaGrad)
    print(avg_changes_pred_iterGrad)
    print(avg_changes_pred_smoothGrad)
    print(avg_changes_pred_inputGrad)
    print(avg_changes_pred_inteGrad)
    print(avg_changes_rm_pred_vanilaGrad)
    print(avg_changes_rm_pred_iterGrad)
    print(avg_changes_rm_pred_smoothGrad)
    print(avg_changes_rm_pred_inputGrad)
    print(avg_changes_rm_pred_inteGrad)

    # counterfactual distances
    avg_ct_dist_vanilaGrad /= cnt_ct_dist_vanilaGrad
    avg_ct_dist_iterGrad /= cnt_ct_dist_iterGrad
    avg_ct_dist_smoothGrad /= cnt_ct_dist_smoothGrad
    avg_ct_dist_inputGrad /= cnt_ct_dist_inputGrad
    avg_ct_dist_inteGrad /= cnt_ct_dist_inteGrad
    print(avg_ct_dist_vanilaGrad)
    print(avg_ct_dist_iterGrad)
    print(avg_ct_dist_smoothGrad)
    print(avg_ct_dist_inputGrad)
    print(avg_ct_dist_inteGrad)


def get_examples(args, num_examples=300):
    random.seed(0)
    samples = random.sample(range(test_size[args.dataset]), num_examples)
    return samples


def vanilla_gradient(model, row, pred_label, DEVICE, step_size=0.02):
    x, segments_ids, input_masks = row
    x.requires_grad = True
    pred = model(inputs_embeds=x,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    x_prior = x.cpu().data.numpy()
    p_prior = logit2prob(pred[0].cpu().data.numpy())

    one_hot = np.zeros((1, 2), dtype=np.float32)
    one_hot[0][pred_label[0]] = 1
    one_hot = torch.from_numpy(one_hot).to(DEVICE)
    one_hot.requires_grad = True
    one_hot = torch.sum(one_hot * pred[0])

    gradient = grad(one_hot, x)[0].cpu().numpy()
    grad_l2 = np.sum(gradient[0, :, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2) * step_size
    gradient_unit = gradient / np.sqrt(np.sum(gradient[0, :, :] ** 2))  # normalize to unit length
    x_after = np.copy(x_prior)
    x_after = perturb_embedding(x_after, gradient_unit * step_size)

    x_after = torch.from_numpy(x_after).to(DEVICE)
    pred = model(inputs_embeds=x_after,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_after = logit2prob(pred[0].cpu().data.numpy())
    changes_pred = p_after - p_prior
    # print(pred_label)
    # print(changes_pred)

    return gradient, importance_score, x_after, changes_pred


def iterative_gradient(model, row, pred_label, DEVICE, step_size, epsilon, max_iters=40):
    x0, segments_ids, input_masks = row
    x0_np = x0.cpu().numpy()
    x_after_np = np.copy(x0_np)
    # iterative perturbation
    x_after = x0.detach()
    cnt = 0
    while np.linalg.norm(x_after_np - x0_np) <= epsilon and cnt <= max_iters:
        _, _, x_after, _ = vanilla_gradient(model, [x_after, segments_ids, input_masks], pred_label, DEVICE, step_size)
        x_after = x_after.clone().detach()
        x_after_np = x_after.cpu().numpy()
        cnt += 1
    x_delta = x_after - x0
    grad_l2 = np.sum(x_delta.cpu().numpy()[0, :, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2)

    pred = model(inputs_embeds=x0,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_prior = logit2prob(pred[0].cpu().data.numpy())
    pred = model(inputs_embeds=x_after,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_after = logit2prob(pred[0].cpu().data.numpy())
    changes_pred = p_after - p_prior
    # print(changes_pred)

    return x_delta.cpu().numpy(), importance_score, x_after, changes_pred


def smooth_gradient(model, row, pred_label, DEVICE, step_size, n_iters=20):
    x0, segments_ids, input_masks = row
    noise_range = 0.4*step_size
    smooth_grad = None
    for n in range(n_iters):
        noise = torch.randn(x0.shape)
        noise = noise / torch.sqrt(torch.sum(noise[0, :, :] ** 2)) * noise_range  # normalize noise to unit length
        x0_ = x0 + noise.to(DEVICE)
        gradient, _, _, _ = vanilla_gradient(model, [x0_, segments_ids, input_masks], pred_label, DEVICE)
        if n == 0:
            smooth_grad = gradient
        else:
            smooth_grad += gradient
    smooth_grad /= n_iters

    grad_l2 = np.sum(smooth_grad[0, :, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2) * step_size
    pred = model(inputs_embeds=x0,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_prior = logit2prob(pred[0].cpu().data.numpy())
    smooth_grad /= np.sqrt(np.sum(smooth_grad[0, :, :] ** 2))  # normalize to unit length
    x_after = np.copy(x0.cpu().data.numpy())
    x_after = perturb_embedding(x_after, smooth_grad * step_size)
    x_after = torch.from_numpy(x_after).to(DEVICE)
    pred = model(inputs_embeds=x_after,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_after = logit2prob(pred[0].cpu().data.numpy())
    changes_pred = p_after - p_prior

    return smooth_grad, importance_score, x_after, changes_pred


def gradient_times_input(model, row, pred_label, DEVICE, step_size=0.02):
    gradient, importance_score, x_after, changes_pred = vanilla_gradient(model, row, pred_label, DEVICE,
                                                                         step_size=step_size)
    x0, segments_ids, input_masks = row
    grad_times_input = np.multiply(gradient, x0.detach().cpu().data.numpy())
    scale = np.sum(grad_times_input, axis=-1, keepdims=True)
    intp = np.multiply(gradient, scale)
    grad_l2 = np.sum(intp[0, :, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2) * step_size

    pred = model(inputs_embeds=x0,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_prior = logit2prob(pred[0].cpu().data.numpy())
    intp /= np.sqrt(np.sum(intp[0, :, :] ** 2))  # normalize to unit length
    x_after = np.copy(x0.cpu().data.numpy())
    x_after = perturb_embedding(x_after, intp * step_size)
    x_after = torch.from_numpy(x_after).to(DEVICE)
    pred = model(inputs_embeds=x_after,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_after = logit2prob(pred[0].cpu().data.numpy())
    changes_pred = p_after - p_prior

    return grad_times_input, importance_score, x_after, changes_pred


def integrated_gradient(model, row, pred_label, DEVICE, step_size=0.02, n_iters=7):
    x, segments_ids, input_masks = row
    avg_grad = None
    for n in range(1, n_iters+1):
        x_ = float(n)/n_iters * x
        x_ = x_.detach()
        gradient, _, _, _ = vanilla_gradient(model, [x_, segments_ids, input_masks], pred_label, DEVICE)
        if n == 1:
            avg_grad = gradient
        else:
            avg_grad += gradient
    avg_grad /= n_iters
    inte_grad = np.multiply(avg_grad, x.detach().cpu().data.numpy())
    scale = np.sum(inte_grad, axis=-1, keepdims=True)
    intp = np.multiply(avg_grad, scale)
    grad_l2 = np.sum(intp[0, :, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2) * step_size

    pred = model(inputs_embeds=x,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_prior = logit2prob(pred[0].cpu().data.numpy())
    intp /= np.sqrt(np.sum(intp[0, :, :] ** 2))  # normalize to unit length
    x_after = np.copy(x.cpu().data.numpy())
    x_after = perturb_embedding(x_after, intp * step_size)
    x_after = torch.from_numpy(x_after).to(DEVICE)
    pred = model(inputs_embeds=x_after,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_after = logit2prob(pred[0].cpu().data.numpy())
    changes_pred = p_after - p_prior

    return inte_grad, importance_score, x_after, changes_pred


def evaluate_word_removal(model, row, inner1, inner2, DEVICE, MAX_RM=4):
    x0, segments_ids, input_masks = row
    pred = model(inputs_embeds=x0,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_prior = logit2prob(pred[0].cpu().data.numpy())

    pred_change = np.zeros(MAX_RM)
    for n in range(1, MAX_RM+1):
        input_masks_after = remove_word_bert(input_masks.cpu(), inner1, inner2, n)
        pred = model(inputs_embeds=x0,
                     token_type_ids=segments_ids, attention_mask=input_masks_after.to(DEVICE), labels=None)[0]
        p_after = logit2prob(pred[0].cpu().data.numpy())
        changes_pred = p_after - p_prior

        pred_change[n-1] = np.abs(changes_pred[0])
    return pred_change


def evaluate_word_2zero(model, row, pred_label, inner1, inner2, DEVICE, MAX_RM=4):
    x0, segments_ids, input_masks = row
    pred = model(inputs_embeds=x0,
                 token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
    p_prior = logit2prob(pred[0].cpu().data.numpy())

    pred_change = np.zeros(MAX_RM)
    for n in range(1, MAX_RM+1):
        x_after = word2zero_bert(x0.cpu().data.numpy(), input_masks, inner1, inner2, n)
        x_after = torch.from_numpy(x_after).to(DEVICE)
        pred = model(inputs_embeds=x_after,
                     token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0]
        p_after = logit2prob(pred[0].cpu().data.numpy())
        changes_pred = p_after - p_prior

        pred_change[n-1] = -changes_pred[pred_label[0]]
    #print(pred_change)
    return pred_change


def find_counterfactual_distance(model, row, x_after, pred_label, DEVICE, step_size):
    x_org, segments_ids, input_masks = row
    x_org = x_org.to(DEVICE)
    x_after = x_after.to(DEVICE)
    dx_unit = -(x_org-x_after)/torch.sqrt(torch.sum((x_org-x_after) ** 2))  # normalize to unit length
    cnt = 1
    while torch.argmax(model(inputs_embeds=x_after, token_type_ids=segments_ids,
                             attention_mask=input_masks, labels=None)[0]) == pred_label[0]:
        x_after = x_org + cnt * dx_unit
        cnt += 1
        if cnt >= 4:
            return 0, 0

    x1 = x_org
    x2 = x_after
    # print(model(inputs_embeds=x1, token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0])
    # print(model(inputs_embeds=x2, token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0])
    # print(torch.sqrt(torch.sum((x1-x2) ** 2)))
    while torch.sqrt(torch.sum((x1-x2) ** 2)) > step_size:
        x_middle = (x1 + x2) / 2
        if torch.argmax(model(inputs_embeds=x_middle, token_type_ids=segments_ids,
                              attention_mask=input_masks, labels=None)[0]) != pred_label[0]:
            x2 = x_middle
        else:
            x1 = x_middle
    x_middle = (x1 + x2) / 2
    ct_dist = torch.sqrt(torch.sum((x_org-x_middle) ** 2))
    # print(model(inputs_embeds=x_middle, token_type_ids=segments_ids, attention_mask=input_masks, labels=None)[0], '\n')

    return ct_dist, 1
