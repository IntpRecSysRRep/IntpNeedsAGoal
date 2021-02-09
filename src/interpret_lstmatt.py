import csv
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import grad
from model_lstmatt_intp import LSTMAttBCintp
from train import evaluate
from nltk.tokenize import word_tokenize
from utils import logit2prob, normalize_score, rescale_embedding, perturb_embedding, remove_word, remasking

dataname2datapath = {'sst': "../data/SST2/", 'yelp': "../data/Yelp/", 'agnews': "../data/AGNews/"}
dataname2modelpath = {'sst': "../result/SST2/", 'yelp': "../result/Yelp/", 'agnews': "../result/AGNews/"}
test_size = {'sst': 1800, 'yelp': 3500, 'agnews': 3500}


def interpret_lstmatt(args, data_iter, text_field, label_field, DEVICE, EPSLN=2.00, MAX_RM=4):
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
    if args.encoder == 'lstmatt':
        model = LSTMAttBCintp(args.dim_embd, args.dim, 1,
                              vocab_size=len(text_field.vocab), label_size=len(label_field.vocab) - 1, DEVICE=DEVICE)
    pretrained_dict = model_org.state_dict()
    new_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_dict}
    # 2. overwrite entries in the existing state dict
    new_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(new_dict)

    list_texts, list_labels = get_examples(args)
    cnt = 0
    avg_changes_pred_vanilaGrad, avg_changes_rm_pred_vanilaGrad = 0., np.zeros(MAX_RM)
    avg_changes_pred_iterGrad, avg_changes_rm_pred_iterGrad = 0., np.zeros(MAX_RM)
    avg_changes_pred_smoothGrad, avg_changes_rm_pred_smoothGrad = 0., np.zeros(MAX_RM)
    avg_changes_pred_inputGrad, avg_changes_rm_pred_inputGrad = 0., np.zeros(MAX_RM)
    avg_changes_pred_inteGrad, avg_changes_rm_pred_inteGrad = 0, np.zeros(MAX_RM)
    avg_changes_rm_pred_attn = np.zeros(MAX_RM)
    avg_changes_am_pred_vanilaGrad = np.zeros(MAX_RM)
    avg_changes_am_pred_iterGrad = np.zeros(MAX_RM)
    avg_changes_am_pred_smoothGrad = np.zeros(MAX_RM)
    avg_changes_am_pred_inputGrad = np.zeros(MAX_RM)
    avg_changes_am_pred_inteGrad = np.zeros(MAX_RM)
    avg_changes_am_pred_attn = np.zeros(MAX_RM)
    avg_ct_dist_vanilaGrad, cnt_ct_dist_vanilaGrad = 0, 0.0001
    avg_ct_dist_iterGrad, cnt_ct_dist_iterGrad = 0, 0.0001
    avg_ct_dist_smoothGrad, cnt_ct_dist_smoothGrad = 0, 0.0001
    avg_ct_dist_inputGrad, cnt_ct_dist_inputGrad = 0, 0.0001
    avg_ct_dist_inteGrad, cnt_ct_dist_inteGrad = 0, 0.0001
    for text_faith, label in tqdm(zip(list_texts, list_labels)):
        text_tokenized = word_tokenize(text_faith)
        #print(text_tokenized)

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
        input_vector = sent[:len_sent].to(DEVICE).unsqueeze(-1)
        if len_sent < MAX_RM:
            continue
        #print(input_vector.shape)
        # predict and interpret (atts)
        pred, hn, x, atts, _ = model_org(input_vector)
        pred = F.log_softmax(pred)
        pred_label = pred.cpu().data.max(1)[1].numpy()

        # vanilla gradient
        gradient, importance_score, x_after, changes_pred = vanilla_gradient(model, x.detach(), pred_label,
                                                                             step_size=EPSLN)
        avg_changes_pred_vanilaGrad += -changes_pred[pred_label[0]]
        avg_changes_rm_pred_vanilaGrad += evaluate_word_removal(model, x.detach(), pred_label, gradient, gradient)
        avg_changes_am_pred_vanilaGrad += evaluate_attention_removal(model, x.detach(), pred_label, gradient, gradient)
        ct_dist, ct_flag = find_counterfactual_distance(model, x.detach(), x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_vanilaGrad += ct_dist * ct_flag
        cnt_ct_dist_vanilaGrad += ct_flag

        # iterative gradient
        x_delta, importance_score, x_after, changes_pred = iterative_gradient(model, x.detach(), pred_label,
                                                                              step_size=0.10, epsilon=EPSLN)
        avg_changes_pred_iterGrad += -changes_pred[pred_label[0]]
        avg_changes_rm_pred_iterGrad += evaluate_word_removal(model, x.detach(), pred_label, x_delta, x_delta)
        avg_changes_am_pred_iterGrad += evaluate_attention_removal(model, x.detach(), pred_label, x_delta, x_delta)
        ct_dist, ct_flag = find_counterfactual_distance(model, x.detach(), x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_iterGrad += ct_dist * ct_flag
        cnt_ct_dist_iterGrad += ct_flag

        # smooth gradient
        smooth_grad, importance_score, x_after, changes_pred = smooth_gradient(model, x.detach(), pred_label,
                                                                               DEVICE, step_size=EPSLN)
        avg_changes_pred_smoothGrad += -changes_pred[pred_label[0]]
        avg_changes_rm_pred_smoothGrad += evaluate_word_removal(model, x.detach(), pred_label, smooth_grad, smooth_grad)
        avg_changes_am_pred_smoothGrad += evaluate_attention_removal(model, x.detach(), pred_label, smooth_grad, smooth_grad)
        ct_dist, ct_flag = find_counterfactual_distance(model, x.detach(), x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_smoothGrad += ct_dist * ct_flag
        cnt_ct_dist_smoothGrad += ct_flag

        # gradient * input
        intp, importance_score, x_after, changes_pred = gradient_times_input(model, x.detach(), pred_label,
                                                                             step_size=EPSLN)
        avg_changes_pred_inputGrad += -changes_pred[pred_label[0]]
        avg_changes_rm_pred_inputGrad += evaluate_word_removal(model, x.detach(), pred_label, gradient, x.detach().cpu().data.numpy())
        avg_changes_am_pred_inputGrad += evaluate_attention_removal(model, x.detach(), pred_label, gradient, x.detach().cpu().data.numpy())
        ct_dist, ct_flag = find_counterfactual_distance(model, x.detach(), x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_inputGrad += ct_dist * ct_flag
        cnt_ct_dist_inputGrad += ct_flag

        # integrated gradient
        inte_grad, importance_score, x_after, changes_pred, avg_grad = integrated_gradient(model, x.detach(), pred_label,
                                                                                           step_size=EPSLN)
        avg_changes_pred_inteGrad += -changes_pred[pred_label[0]]
        avg_changes_rm_pred_inteGrad += evaluate_word_removal(model, x.detach(), pred_label, avg_grad, x.detach().cpu().data.numpy())
        avg_changes_am_pred_inteGrad += evaluate_attention_removal(model, x.detach(), pred_label, avg_grad, x.detach().cpu().data.numpy())
        ct_dist, ct_flag = find_counterfactual_distance(model, x.detach(), x_after, pred_label, DEVICE, EPSLN / 1000)
        avg_ct_dist_inteGrad += ct_dist * ct_flag
        cnt_ct_dist_inteGrad += ct_flag

        # # attention
        # atts_np = atts.detach().cpu().numpy()
        # avg_changes_rm_pred_attn += evaluate_word_removal(model, x.detach(), pred_label, atts_np, np.ones(atts_np.shape))
        # avg_changes_am_pred_attn += evaluate_attention_removal(model, x.detach(), pred_label,
        #                                                        atts_np, np.ones(atts_np.shape))

        cnt += 1

    avg_changes_pred_vanilaGrad /= cnt
    avg_changes_pred_iterGrad /= cnt
    avg_changes_pred_smoothGrad /= cnt
    avg_changes_pred_inputGrad /= cnt
    avg_changes_pred_inteGrad /= cnt

    avg_changes_rm_pred_vanilaGrad /= cnt
    avg_changes_rm_pred_iterGrad /= cnt
    avg_changes_rm_pred_smoothGrad /= cnt
    avg_changes_rm_pred_inputGrad /= cnt
    avg_changes_rm_pred_inteGrad /= cnt
    avg_changes_rm_pred_attn /= cnt

    avg_changes_am_pred_vanilaGrad /= cnt
    avg_changes_am_pred_iterGrad /= cnt
    avg_changes_am_pred_smoothGrad /= cnt
    avg_changes_am_pred_inputGrad /= cnt
    avg_changes_am_pred_inteGrad /= cnt
    avg_changes_am_pred_attn /= cnt

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
    print(avg_changes_rm_pred_attn)

    print(avg_changes_am_pred_vanilaGrad)
    print(avg_changes_am_pred_iterGrad)
    print(avg_changes_am_pred_smoothGrad)
    print(avg_changes_am_pred_inputGrad)
    print(avg_changes_am_pred_inteGrad)
    print(avg_changes_am_pred_attn)

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
    list_texts = []
    list_labels = []
    random.seed(0)
    samples = random.sample(range(test_size[args.dataset]), num_examples)

    cnt = 0
    with open(dataname2datapath[args.dataset] + 'test.tsv', 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            if cnt in samples:
                list_texts.append(row[0])
                list_labels.append(1 - int(row[1]))
            cnt += 1
    return list_texts, list_labels


def vanilla_gradient(model, x, pred_label, step_size=0.02):
    model.batch_size = 1
    model.hidden = model.init_hidden()
    x = x.cpu()
    x.requires_grad = True
    mask = torch.ones([x.shape[0], 1, 1])
    mask.requires_grad = False
    pred, _ = model(x, mask)
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
    pred, _ = model(x_after, mask)
    p_after = logit2prob(pred[0].data.numpy())
    changes_pred = p_after - p_prior
    #print(pred_label)
    #print(importance_score)
    #print(changes_pred)

    return gradient, importance_score, x_after, changes_pred


def iterative_gradient(model, x0, pred_label, step_size, epsilon, max_iters=80):
    x0_np = x0.cpu().numpy()
    x_after_np = np.copy(x0_np)
    # iterative perturbation
    x_after = x0.detach()
    cnt = 0
    while np.linalg.norm(x_after_np - x0_np) <= epsilon and cnt <= max_iters:
        _, _, x_after, _ = vanilla_gradient(model, x_after, pred_label, step_size)
        x_after = x_after.detach()
        x_after_np = x_after.cpu().numpy()
        cnt += 1
    x_delta = x_after - x0.cpu()
    grad_l2 = np.sum(x_delta.numpy()[:, 0, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2)

    mask = torch.ones([x0.shape[0], 1, 1])
    mask.requires_grad = False
    model.hidden = model.init_hidden()
    pred, _ = model(x0.cpu(), mask)
    p_prior = logit2prob(pred[0].data.numpy())
    model.hidden = model.init_hidden()
    pred, _ = model(x_after.cpu(), mask)
    p_after = logit2prob(pred[0].data.numpy())
    changes_pred = p_after - p_prior
    #print(changes_pred)

    return x_delta.numpy(), importance_score, x_after, changes_pred


def smooth_gradient(model, x0, pred_label, DEVICE, step_size, noise_range=0.02, n_iters=20):
    smooth_grad = None
    for n in range(n_iters):
        x0_ = x0 + torch.randn(x0.shape).to(DEVICE) * noise_range
        gradient, _, _, _ = vanilla_gradient(model, x0_, pred_label)
        if n == 0:
            smooth_grad = gradient
        else:
            smooth_grad += gradient
    smooth_grad /= n_iters

    grad_l2 = np.sum(smooth_grad[:, 0, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2) * step_size

    mask = torch.ones([x0.shape[0], 1, 1])
    mask.requires_grad = False
    model.hidden = model.init_hidden()
    pred, _ = model(x0.cpu(), mask)
    p_prior = logit2prob(pred[0].data.numpy())
    smooth_grad /= np.sqrt(np.sum(smooth_grad[:, 0, :] ** 2))  # normalize to unit length
    x_after = np.copy(x0.cpu().data.numpy())
    x_after = perturb_embedding(x_after, smooth_grad * step_size)
    x_after = torch.from_numpy(x_after)
    model.hidden = model.init_hidden()
    pred, _ = model(x_after, mask)
    p_after = logit2prob(pred[0].data.numpy())
    changes_pred = p_after - p_prior

    return smooth_grad, importance_score, x_after, changes_pred


def gradient_times_input(model, x, pred_label, step_size=0.02):
    gradient, importance_score, x_after, changes_pred = vanilla_gradient(model, x.detach(), pred_label,
                                                                         step_size=step_size)
    grad_times_input = np.multiply(gradient, x.detach().cpu().data.numpy())
    scale = np.sum(grad_times_input, axis=-1, keepdims=True)
    intp = np.multiply(gradient, scale)
    grad_l2 = np.sum(intp[:, 0, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2) * step_size

    mask = torch.ones([x.shape[0], 1, 1])
    mask.requires_grad = False
    model.hidden = model.init_hidden()
    pred, _ = model(x.cpu(), mask)
    p_prior = logit2prob(pred[0].data.numpy())
    intp /= np.sqrt(np.sum(intp[:, 0, :] ** 2))  # normalize to unit length
    x_after = np.copy(x.cpu().data.numpy())
    x_after = perturb_embedding(x_after, intp * step_size)
    x_after = torch.from_numpy(x_after)
    model.hidden = model.init_hidden()
    pred, _ = model(x_after.cpu(), mask)
    p_after = logit2prob(pred[0].data.numpy())
    changes_pred = p_after - p_prior

    return intp, importance_score, x_after, changes_pred


def integrated_gradient(model, x, pred_label, step_size=0.02, n_iters=8):
    avg_grad = None
    for n in range(1, n_iters+1):
        x_ = float(n)/n_iters * x
        x_ = x_.detach()
        gradient, _, _, _ = vanilla_gradient(model, x_, pred_label, step_size)
        if n == 1:
            avg_grad = gradient
        else:
            avg_grad += gradient
    avg_grad /= n_iters
    inte_grad = np.multiply(avg_grad, x.detach().cpu().data.numpy())
    scale = np.sum(inte_grad, axis=-1, keepdims=True)
    intp = np.multiply(avg_grad, scale)
    grad_l2 = np.sum(intp[:, 0, :] ** 2, axis=1)
    importance_score = normalize_score(grad_l2) * step_size

    mask = torch.ones([x.shape[0], 1, 1])
    mask.requires_grad = False
    model.hidden = model.init_hidden()
    pred, _ = model(x.cpu(), mask)
    p_prior = logit2prob(pred[0].data.numpy())
    intp /= np.sqrt(np.sum(intp[:, 0, :] ** 2))  # normalize to unit length
    x_after = np.copy(x.cpu().data.numpy())
    x_after = perturb_embedding(x_after, intp * step_size)
    x_after = torch.from_numpy(x_after)
    model.hidden = model.init_hidden()
    pred, _ = model(x_after.cpu(), mask)
    p_after = logit2prob(pred[0].data.numpy())
    changes_pred = p_after - p_prior

    return inte_grad, importance_score, x_after, changes_pred, avg_grad


def attention_interpretation(model, x):
    _, atts = model(x)
    return atts


def evaluate_word_removal(model, x0, pred_label, inner1, inner2, MAX_RM=4):
    mask = torch.ones([x0.shape[0], 1, 1])
    mask.requires_grad = False
    model.hidden = model.init_hidden()
    pred, _ = model(x0.cpu(), mask)
    p_prior = logit2prob(pred[0].data.numpy())

    pred_change = np.zeros(MAX_RM)
    for n in range(1, MAX_RM+1):
        x_after = remove_word(x0.cpu().data.numpy(), inner1, inner2, n)
        x_after = torch.from_numpy(x_after)
        model.hidden = model.init_hidden()
        pred, _ = model(x_after.cpu().float(), mask)
        p_after = logit2prob(pred[0].data.numpy())
        changes_pred = p_after - p_prior

        pred_change[n-1] = -changes_pred[pred_label[0]]

    return pred_change


def evaluate_attention_removal(model, x0, pred_label, inner1, inner2, MAX_RM=4):
    # for attention methods, inner2 should be an all'1 vector
    mask = torch.ones([x0.shape[0], 1, 1])
    mask.requires_grad = False
    model.hidden = model.init_hidden()
    pred, _ = model(x0.cpu(), mask)
    p_prior = logit2prob(pred[0].data.numpy())

    pred_change = np.zeros(MAX_RM)
    for n in range(1, MAX_RM + 1):
        mask_new = remasking(mask, inner1, inner2, n)
        model.hidden = model.init_hidden()
        pred, _ = model(x0.cpu(), mask_new)
        p_after = logit2prob(pred[0].data.numpy())
        changes_pred = p_after - p_prior

        pred_change[n - 1] = -changes_pred[pred_label[0]]

    return pred_change


def find_counterfactual_distance(model, x_org, x_after, pred_label, DEVICE, step_size):
    mask = torch.ones([x_org.shape[0], 1, 1])
    mask.requires_grad = False
    x_org = x_org.cpu()
    x_after = x_after.cpu()
    dx_unit = -(x_org-x_after)/torch.sqrt(torch.sum((x_org-x_after) ** 2))  # normalize to unit length
    cnt = 1
    model.hidden = model.init_hidden()
    while torch.argmax(model(x_after, mask)[0]) == pred_label[0]:
        x_after = x_org + cnt * dx_unit
        model.hidden = model.init_hidden()
        cnt += 1
        if cnt >= 4:
            return 0, 0

    x1 = x_org
    x2 = x_after
    # print(torch.sqrt(torch.sum((x1-x2) ** 2)))
    # model.hidden = model.init_hidden()
    # print(model(x1)[0])
    # model.hidden = model.init_hidden()
    # print(model(x2)[0])
    while torch.sqrt(torch.sum((x1-x2) ** 2)) > step_size:
        x_middle = (x1 + x2) / 2
        model.hidden = model.init_hidden()
        if torch.argmax(model(x_middle, mask)[0]) != pred_label[0]:
            x2 = x_middle
        else:
            x1 = x_middle
    x_middle = (x1 + x2) / 2
    # model.hidden = model.init_hidden()
    # print(model(x_middle)[0])
    ct_dist = torch.sqrt(torch.sum((x_org-x_middle) ** 2))

    return ct_dist, 1

