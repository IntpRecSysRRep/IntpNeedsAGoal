import csv
import sys
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import grad
from model_lstm_intp import LSTMBCintp
from train import evaluate
from nltk.tokenize import word_tokenize
from utils import logit2prob, normalize_score, rescale_embedding, perturb_embedding, remove_word

dataname2datapath = {'sst': "../data/SST2/", 'yelp': "../data/Yelp/", 'agnews': "../data/AGNews/",
                     'imdb': "../data/IMDB/"}
dataname2modelpath = {'sst': "../result/SST2/", 'yelp': "../result/Yelp/", 'agnews': "../result/AGNews/",
                      'imdb': "../result/IMDB/"}
dataname2exprange = {'sst': 0.002, 'yelp': 0.002, 'imdb': 0.002}
test_size = {'sst': 1800, 'yelp': 3500, 'agnews': 3500, 'imdb': 390}

intp_types = ['vagrad', 'smoothgrad', 'inpgrad', 'integrad', 'smoothinpgrad']


def interpret_lstm(args, data_iter, text_field, label_field, DEVICE):
    # Load pretrained model
    print('Model loading...')
    file_load = dataname2modelpath[args.dataset] + args.encoder + '.pkl'
    model_org = torch.load(file_load)
    model_org.to(DEVICE)
    acc_test = evaluate(data_iter, model_org, DEVICE)
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
    model.to(DEVICE)

    # Sample test data
    list_texts, list_labels = get_examples(args)

    for text_faith, label in zip(list_texts, list_labels):
        pred, x, y_target = preprocess_text(text_faith, text_field, model_org, DEVICE)
        # # vanilla gradient
        # gradient, b_0 = vanilla_gradient(model, x, y_target, DEVICE)
        # print(gradient.shape)
        #
        # # smooth gradient
        # smooth_grad, b_0 = smooth_gradient(model, x, y_target, DEVICE)
        # print(smooth_grad.shape)
        #
        # # gradient times input
        # inpgrad, b_0, gradient = gradient_times_input(model, x, y_target, DEVICE)
        # print(inpgrad.shape)
        #
        # # integrated gradient
        # integrad, b_0, gradient = integrated_gradient(model, x, y_target, DEVICE)
        # print(integrad.shape)
        break

    exp_task = 'appr_var'
    print("Current task: " + exp_task)
    if exp_task == 'appr_bias':
        experiment_approximation_bias(args, model_org, text_field, model, list_texts, list_labels, DEVICE)
    elif exp_task == 'appr_var':
        experiment_approximation_var(args, model_org, text_field, model, list_texts, list_labels, DEVICE)
    elif exp_task == 'intp_attack':
        experiment_attack_interpretation(args, model_org, text_field, model, list_texts, list_labels, DEVICE)
    elif exp_task == 'intp_bias':
        experiment_interpretation_bias(args, model_org, text_field, model, list_texts, list_labels, DEVICE)
    else:
        print('Not formal evaluation.')


def get_examples(args, num_examples=300):
    list_texts = []
    list_labels = []
    random.seed(0)
    samples = random.sample(range(test_size[args.dataset]), min(test_size[args.dataset], num_examples))

    cnt = 0
    with open(dataname2datapath[args.dataset] + 'test.tsv', 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            if cnt in samples:
                list_texts.append(row[0])
                list_labels.append(1 - int(row[1]))
            cnt += 1
    return list_texts, list_labels


def preprocess_text(text_faith, text_field, model_org, DEVICE):
    text_tokenized = word_tokenize(text_faith)
    # print(text_tokenized)

    word_to_idx = dict(text_field.vocab.stoi)
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    # build one-hoc word index array
    sentence = []
    for i in text_tokenized:
        if i != '.':
            try:
                sentence.append(word_to_idx[i])
            except KeyError:
                pass
    sent = np.array(sentence)
    sent = torch.from_numpy(sent)
    len_sent = len(sentence)
    model_org.hidden = model_org.init_hidden()
    input_vector = sent[:len_sent].to(DEVICE)
    # print(input_vector.shape)

    pred, hn, x, _ = model_org(input_vector)
    pred = F.log_softmax(pred)
    #y_target = pred.cpu().data.max(1)[1].numpy()
    y_target = torch.argmax(pred.cpu().data).numpy()

    return pred, x, y_target    # batch_size * num_class, num_words * batch_size * dim, batch_size


def experiment_approximation_bias(args, model_org, text_field, model, list_texts, list_labels, DEVICE):
    EXP_RANGE = dataname2exprange[args.dataset]
    bias_approx_summary = {}
    for intp_type in intp_types[0:5]:
        bias_appr_avg = 0.
        cnt = 0
        for text_faith, label in tqdm(zip(list_texts, list_labels)):
            pred, x, y_target = preprocess_text(text_faith, text_field, model_org, DEVICE)
            weight_avg, w_0_avg, _, _ = interpretation_expected(model, x, y_target, DEVICE, intp_type, EXP_RANGE)
            bias_appr_avg += approximation_bias(model, x, y_target, weight_avg, w_0_avg, DEVICE, EXP_RANGE)

            cnt += 1
        bias_appr_avg /= cnt
        print(bias_appr_avg.data.cpu().numpy())
        bias_approx_summary[intp_type] = bias_appr_avg.data.cpu().numpy()
    print("Approximation bias: ")
    print(bias_approx_summary)


def experiment_approximation_var(args, model_org, text_field, model, list_texts, list_labels, DEVICE):
    EXP_RANGE = dataname2exprange[args.dataset]
    var_approx_summary = {}
    for intp_type in intp_types[0:5]:
        var_appr_avg = 0.
        cnt = 0
        for text_faith, label in tqdm(zip(list_texts, list_labels)):
            pred, x, y_target = preprocess_text(text_faith, text_field, model_org, DEVICE)
            weight_avg, w_0_avg, weight_all, w_0_all = interpretation_expected(model, x, y_target, DEVICE, intp_type, EXP_RANGE)
            var_appr_avg += approximation_var(x, weight_avg, w_0_avg, weight_all, w_0_all, intp_type, DEVICE, EXP_RANGE)
            cnt += 1
        var_appr_avg /= cnt
        print('Interpretation ' + intp_type + ': ')
        print(var_appr_avg.data.cpu().numpy())
        var_approx_summary[intp_type] = var_appr_avg.data.cpu().numpy()
    print("Approximation variance: ")
    print(var_approx_summary)


def approximation_var(x, weight, w_0, weight_all, w_0_all, intp_type, DEVICE, EXP_RANGE, num_exp=20):
    # compute the approximation bias between f^(x) and exp(f^(x)) around x.
    var_appr_inneravg = 0.
    for i in range(num_exp):
        weight_i, w_0_i = weight_all[:, i:i + 1, :], w_0_all[i]
        var_appr_inneravg += compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE)
    var_appr_inneravg /= num_exp

    return var_appr_inneravg


def compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE, num_exp=100):
    x = x.to(DEVICE)
    var_appr = 0.
    for i in range(num_exp):
        noise = (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
        noise = noise.to(DEVICE)
        x_i = x + noise

        pred_fhat_exp = w_0 + torch.sum((x_i - x) * weight)
        pred_fhat_i = w_0_i + torch.sum((x_i - x) * weight_i)
        var_appr += torch.pow(pred_fhat_exp.detach() - pred_fhat_i.detach(), 2)
    var_appr /= num_exp

    return var_appr


def experiment_interpretation_bias(args, model_org, text_field, model, list_texts, list_labels, DEVICE):
    if args.dataset == 'sst':
        csa_constrains = [0.25, 0.50, 0.75, 1.00]
        era_constrains = [1, 2, 3, 4]
    else:
        csa_constrains = [0.25, 0.50, 0.75, 1.00]
        era_constrains = [5, 10, 15, 20, 25]

    for intp_type in intp_types[0:5]:
        csa_avg = np.zeros_like(csa_constrains)
        era_avg = np.zeros_like(era_constrains, dtype=np.float)
        cnt = 0

        for text_faith, label in tqdm(zip(list_texts, list_labels)):
            pred, x, y_target = preprocess_text(text_faith, text_field, model_org, DEVICE)

            if x.shape[0] < era_constrains[-1]:
                continue

            if intp_type == 'vagrad':
                intp, _ = vanilla_gradient(model, x, y_target, DEVICE)
            elif intp_type == 'smoothgrad':
                intp, _ = smooth_gradient(model, x, x, y_target, DEVICE)
            elif intp_type == 'inpgrad':
                intp, _, gradient = gradient_times_input(model, x, y_target, DEVICE)
            elif intp_type == 'smoothinpgrad':
                intp, _, gradient = smooth_gradient_times_input(model, x, y_target, DEVICE)
            elif intp_type == 'integrad':
                intp, _, gradient = integrated_gradient(model, x, y_target, DEVICE)
            else:
                sys.exit("Unknown interpretation types.")
            cnt += 1

            if intp_type in ['vagrad', 'smoothgrad']:
                direction = intp
                scale = torch.ones_like(intp)
            else:
                direction = gradient
                scale = torch.sum(intp, dim=-1, keepdim=True)
            csa_avg += perturb_csa(model, x, direction, scale, DEVICE, csa_constrains)
            era_avg += perturb_era(model, x, intp, intp_type, DEVICE, era_constrains)
        csa_avg /= cnt
        era_avg /= cnt
        print('Interpretation ' + intp_type + ': ')
        print('csa: ', csa_avg)
        print('era: ', era_avg)


def interpretation_expected(model, x, y_target, DEVICE, intp_type, EXP_RANGE, num_exp=30):
    shape_exp = list(x.shape)
    shape_exp[1] = num_exp

    intp = torch.zeros_like(x).to(DEVICE)
    weight = torch.zeros_like(x).to(DEVICE)
    w_0 = 0.
    weight_all = torch.zeros(shape_exp).to(DEVICE)
    w_0_all = torch.zeros(num_exp)
    if intp_type == 'vagrad':
        for i in range(num_exp):
            x_i = x.cpu() + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i = vanilla_gradient(model, x_i, y_target, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight_all[:, i:i+1, :] = intp_i
            w_0_all[i] = b_0_i
        intp /= num_exp
        weight = intp
        w_0 /= num_exp
    elif intp_type == 'smoothgrad':
        for i in range(num_exp):
            x_i = x.cpu() + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i = smooth_gradient(model, x, x_i, y_target, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight_all[:, i:i+1, :] = intp_i
            w_0_all[i] = b_0_i
        intp /= num_exp
        weight = intp
        w_0 /= num_exp
    elif intp_type == 'inpgrad':
        for i in range(num_exp):
            x_i = x.cpu() + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i, grad_i = gradient_times_input(model, x_i, y_target, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight += grad_i
            weight_all[:, i:i+1, :] = grad_i
            w_0_all[i] = b_0_i + torch.sum(intp_i)
        intp /= num_exp
        w_0 = w_0 / num_exp + torch.sum(intp)
        weight /= num_exp
    elif intp_type == 'smoothinpgrad':
        for i in range(num_exp):
            x_i = x.cpu() + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i, grad_i = smooth_gradient_times_input(model, x_i, y_target, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight += grad_i
            weight_all[:, i:i+1, :] = grad_i
            w_0_all[i] = b_0_i + torch.sum(intp_i)
        intp /= num_exp
        w_0 = w_0 / num_exp + torch.sum(intp)
        weight /= num_exp
    elif intp_type == 'integrad':
        for i in range(num_exp):
            x_i = x.cpu() + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i, grad_i = integrated_gradient(model, x_i, y_target, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight += grad_i
            weight_all[:, i:i+1, :] = grad_i
            w_0_all[i] = b_0_i + torch.sum(intp_i)
        intp /= num_exp
        w_0 = w_0 / num_exp + torch.sum(intp)
        weight /= num_exp
    else:
        sys.exit("Unknown interpretation types.")

    return weight, w_0, weight_all, w_0_all


def approximation_bias(model, x, y_target, weight, w_0, DEVICE, EXP_RANGE, num_exp=100):
    model.batch_size = 1
    model.hidden = model.init_hidden()
    x = x.to(DEVICE)
    pred, _ = model(x)

    # expectation around x, compute difference between f and exp{f^}
    bias_appr = 0.
    for i in range(num_exp):
        noise = (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
        noise = noise.to(DEVICE)
        x_i = x + noise

        # true prediction
        model.hidden = model.init_hidden()
        pred_f = model(x_i)[0][0, y_target]
        # approximated prediction
        pred_fhat = w_0 + torch.sum((x_i - x) * weight)

        # approximation bias
        bias_appr += torch.pow(pred_fhat.detach() - pred_f.detach(), 2)  # be careful of memory explosion
    bias_appr /= num_exp
    # print(bias_appr)
    return bias_appr


def vanilla_gradient(model, x, y_target, DEVICE, further_grad=False):
    model.batch_size = 1
    model.hidden = model.init_hidden()
    x = x.to(DEVICE).requires_grad_()
    pred, _ = model(x)
    b_0 = pred[0, y_target]
    model.zero_grad()
    gradient = grad(pred[0, y_target], x, create_graph=further_grad)[0]  # [0] is necessary
    return gradient, b_0


def smooth_gradient(model, x0, x, y_target, DEVICE, further_grad=False, noise_range=0.001, n_iters=40):
    model.batch_size = 1
    model.hidden = model.init_hidden()
    x = x.to(DEVICE)
    pred, _ = model(x)
    b_0 = pred[0, y_target]

    # get multiple gradients and average them
    smooth_grad = torch.zeros_like(x)
    for t in range(n_iters):
        noise = (torch.rand(x.shape) - 0.5) * 2 * noise_range
        noise = noise.to(DEVICE)
        noisy_x = clip_image_perturb(x, x0, noise, noise_range).requires_grad_()

        model.hidden = model.init_hidden()
        pred, _ = model(noisy_x)
        model.zero_grad()
        grad_t = grad(pred[0, y_target], noisy_x, create_graph=further_grad)[0]
        smooth_grad += grad_t
    smooth_grad /= n_iters
    return smooth_grad, b_0


def gradient_times_input(model, x, y_target, DEVICE, further_grad=False):
    model.batch_size = 1
    model.hidden = model.init_hidden()
    x = x.to(DEVICE).requires_grad_()
    pred, _ = model(x)
    model.zero_grad()
    gradient = grad(pred[0, y_target], x, create_graph=further_grad)[0]  # [0] is necessary
    grad_input = x * gradient

    model.hidden = model.init_hidden()
    b_0 = model(torch.zeros_like(x).to(DEVICE))[0][0, y_target]  # all_zero ref point
    return grad_input, b_0, gradient


def smooth_gradient_times_input(model, x, y_target, DEVICE, further_grad=False, noise_range=0.001, n_iters=40):
    model.batch_size = 1
    model.hidden = model.init_hidden()
    x = x.to(DEVICE)
    b_0 = model(torch.zeros_like(x).to(DEVICE))[0][0, y_target]  # all_zero ref point

    # get multiple gradients and average them
    smooth_grad = torch.zeros_like(x)
    for t in range(n_iters):
        noise = (torch.rand(x.shape) - 0.5) * 2 * noise_range
        noise = noise.to(DEVICE)
        noisy_x = (x + noise).requires_grad_()

        model.hidden = model.init_hidden()
        pred, _ = model(noisy_x)
        model.zero_grad()
        grad_t = grad(pred[0, y_target], noisy_x, create_graph=further_grad)[0]
        smooth_grad += grad_t
    smooth_grad /= n_iters
    smooth_inpgrad = x * smooth_grad
    return smooth_inpgrad, b_0, smooth_grad


def integrated_gradient(model, x, y_target, DEVICE, further_grad=False, n_iters=10):
    model.batch_size = 1
    model.hidden = model.init_hidden()
    x = x.to(DEVICE)
    b_0 = model(torch.zeros_like(x).to(DEVICE))[0][0, y_target]  # all_zero ref point

    # get multiple gradient and average them
    x_ref = torch.zeros_like(x).to(DEVICE)
    grad_path = torch.zeros_like(x)
    for n in range(1, n_iters+1):
        x_ = x_ref + float(n)/n_iters * (x - x_ref)
        x_ = x_.requires_grad_()

        model.hidden = model.init_hidden()
        pred, _ = model(x_)
        model.zero_grad()
        grad_n = grad(pred[0, y_target], x_, create_graph=further_grad)[0]
        grad_path += grad_n
    grad_path /= n_iters
    inte_grad = x * grad_path
    return inte_grad, b_0, grad_n


def perturb_csa(model, x, direction, scale, DEVICE, step_sizes):
    # original prediction
    x = x.to(DEVICE)
    model.hidden = model.init_hidden()
    pred, _ = model(x)
    y_target = torch.argmax(pred)
    pred_org = pred[0, y_target]

    # perturbed prediction
    csa_scores = []
    direction = direction * scale
    direction /= torch.pow(torch.sum(torch.pow(direction, 2)), 0.5)
    for step_size in step_sizes:
        x_adv = (x - direction * step_size).to(DEVICE)
        model.hidden = model.init_hidden()
        pred_adv = model(x_adv)[0][0, y_target]
        csa_score = pred_org - pred_adv
        csa_score = csa_score.detach().cpu().numpy()
        csa_scores.append(csa_score)

    return np.array(csa_scores)


def perturb_era(model, x, intp, intp_type, DEVICE, nums_remove):
    # original prediction
    x = x.to(DEVICE)
    model.hidden = model.init_hidden()
    pred, _ = model(x)
    y_target = torch.argmax(pred)
    pred_org = pred[0, y_target]

    if intp_type in ['vagrad', 'smoothgrad']:
        scores = torch.sum(intp * intp, dim=-1, keepdim=True)
    else:
        scores = torch.sum(intp, dim=-1, keepdim=True)

    # perturbed prediction
    era_scores = []
    for num in nums_remove:
        mask = scores.detach().clone()
        thre = torch.sort(mask, dim=0, descending=True)[0][num-1, 0, 0]
        mask[mask >= thre] = 0.
        mask[mask != 0] = 1.
        x_adv = (x * mask).detach()
        model.hidden = model.init_hidden()
        pred_adv = model(x_adv)[0][0, y_target]
        era_score = pred_org - pred_adv
        era_score = era_score.detach().cpu().numpy()
        era_scores.append(era_score)

    return np.array(era_scores)


def clip_image_perturb(img, img_org, perturb, adv_range):
    switch = (torch.abs(img+perturb-img_org) > adv_range).float()     # indicate whether exceeds epsilon
    sign_switch = (perturb > 0).float() - (perturb < 0).float()
    img_adv = switch*(img_org + adv_range * sign_switch) + (1.0-switch)*(img + perturb)
    return img_adv
