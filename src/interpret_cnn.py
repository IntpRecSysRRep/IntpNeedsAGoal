import random
import sys
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import grad
from model_cnnsmall import CNNSmall
from model_cnnvgg import VGG
from train_cnn import evaluation
from utils import topk_intersection

dataname2datapath = {'fmnist': "../data/Fmnist"}
dataname2modelpath = {'fmnist': "../result/fmnist/"}
dataname2exprange = {'fmnist': 0.05, 'imagenet': 0.01}
intp_types = ['vagrad', 'smoothgrad', 'inpgrad', 'smoothinpgrad', 'integrad', 'lime', 'gradcam']

# some hyperparameters of interpretation


def interpret_cnn(args, dataloader, num_data, DEVICE):
    # Load pretrained model
    print('Model loading...')
    model = None
    if args.encoder == 'cnn_small':
        model = CNNSmall()
        model_file = dataname2modelpath[args.dataset] + 'cnn_small.pt'
        model.load_state_dict(torch.load(model_file))
    elif args.encoder == 'vgg':
        model = VGG()
    else:
        sys.exit('Wrong model.')
    model.to(DEVICE)
    acc_test = evaluation(model, dataloader, DEVICE)
    print('Model loaded, with testing accuracy: %.3f' % acc_test)

    # Sample a number of examples to explain and evaluate
    list_x, list_y = get_examples(dataloader, num_data)
    for x, y in zip(list_x[9:10], list_y[9:10]):
        # # vanilla gradient
        # gradient, b_0 = vanilla_gradient(model, x, DEVICE, step_size=0.02)
        # show_intp(x, gradient)
        #
        # # smooth gradient
        # smooth_grad, b_0 = smooth_gradient(model, x, DEVICE)
        # show_intp(x, smooth_grad)
        #
        # # gradient times input
        # grad_input, b_0, grad_tmp = gradient_times_input(model, x, DEVICE)
        # show_intp(x, grad_input)
        #
        # # integrated gradient
        # inte_grad, b_0, grad_tmp = integrated_gradient(model, x, DEVICE)
        # show_intp(x, inte_grad)

        # # LIME
        # lime_weights, b_0 = LIME(model, x, DEVICE)
        # show_intp(x, lime_weights)

        # # grad_cam
        # gradcam, gradcam_vis, x_act = grad_cam(model, x, DEVICE)
        # show_intp(x, gradcam_vis)
        break

    exp_task = 'intp_attack'
    print("Current task: " + exp_task)
    if exp_task == 'appr_bias':
        experiment_approximation_bias(args, model, list_x, list_y, DEVICE)
    elif exp_task == 'appr_var':
        experiment_approximation_var(args, model, list_x, list_y, DEVICE)
    elif exp_task == 'intp_attack':
        experiment_attack_interpretation(args, model, list_x, list_y, DEVICE)
    elif exp_task == 'intp_bias':
        experiment_interpretation_bias(args, model, list_x, list_y, DEVICE)
    else:
        print('Not formal evaluation.')


def experiment_approximation_bias(args, model, list_x, list_y, DEVICE):
    EXP_RANGE = dataname2exprange[args.dataset]
    bias_approx_summary = {}
    for intp_type in intp_types[0:5]:
        bias_appr_avg = 0.
        cnt = 0
        for x, y in tqdm(zip(list_x, list_y)):
            weight_avg, w_0_avg, _, _ = interpretation_expected(model, x, DEVICE, intp_type, EXP_RANGE)
            bias_appr_avg += approximation_bias(model, x, weight_avg, w_0_avg, DEVICE, EXP_RANGE)
            cnt += 1
        bias_appr_avg /= cnt
        print(bias_appr_avg.data.cpu().numpy())
        bias_approx_summary[intp_type] = bias_appr_avg.data.cpu().numpy()
    print("Approximation bias: ")
    print(bias_approx_summary)


def experiment_approximation_var(args, model, list_x, list_y, DEVICE):
    EXP_RANGE = dataname2exprange[args.dataset]
    var_approx_summary = {}
    for intp_type in intp_types[0:5]:
        var_appr_avg = 0.
        cnt = 0
        for x, y in tqdm(zip(list_x, list_y)):
            weight_avg, w_0_avg, weight_all, w_0_all = interpretation_expected(model, x, DEVICE, intp_type, EXP_RANGE)
            var_appr_avg += approximation_var(x, weight_avg, w_0_avg, weight_all, w_0_all, intp_type, DEVICE, EXP_RANGE)
            cnt += 1
        var_appr_avg /= cnt
        print('Interpretation ' + intp_type + ': ')
        print(var_appr_avg.data.cpu().numpy())
        var_approx_summary[intp_type] = var_appr_avg.data.cpu().numpy()
    print("Approximation variance: ")
    print(var_approx_summary)


def experiment_attack_interpretation(args, model, list_x, list_y, DEVICE):
    attack_perform_summary = {}
    for intp_type in intp_types[0:4]:
        attack_perform_avg = 0.
        cnt = 0
        for x, y in tqdm(zip(list_x, list_y)):
            attack_perform_avg += attack_interpretation(model, x, intp_type, DEVICE)
            cnt += 1
        attack_perform_avg /= cnt
        print(attack_perform_avg)
        attack_perform_summary[intp_type] = attack_perform_avg
    print("Attack performance (lower is better attack): ")
    print(attack_perform_summary)


def experiment_interpretation_bias(args, model, list_x, list_y, DEVICE):
    if args.dataset == 'fmnist':
        csa_constrains = [0.05, 0.075, 0.10, 0.125, 0.150]
        era_constrains = [40, 50, 60, 70, 80]
    else:
        csa_constrains = [0.01, 0.02, 0.03, 0.04, 0.05]
        era_constrains = [40, 50, 60, 70, 80]

    for intp_type in intp_types[0:5]:
        csa_avg = np.zeros_like(csa_constrains)
        era_avg = np.zeros_like(era_constrains, dtype=np.float)
        cnt = 0
        for x, y in tqdm(zip(list_x, list_y)):
            if intp_type == 'vagrad':
                intp, _ = vanilla_gradient(model, x, DEVICE)
            elif intp_type == 'smoothgrad':
                intp, _ = smooth_gradient(model, x, DEVICE)
            elif intp_type == 'inpgrad':
                intp, _, _ = gradient_times_input(model, x, DEVICE)
            elif intp_type == 'smoothinpgrad':
                intp, _, _ = smooth_gradient_times_input(model, x, DEVICE)
            elif intp_type == 'integrad':
                intp, _, _ = integrated_gradient(model, x, DEVICE)
            elif intp_type == 'lime':
                intp, _ = LIME(model, x, DEVICE)
            else:
                sys.exit("Unknown interpretation types.")
            cnt += 1

            csa_avg += perturb_csa(model, x, intp, DEVICE, csa_constrains)
            era_avg += perturb_era(model, x, intp, DEVICE, era_constrains)
        csa_avg /= cnt
        era_avg /= cnt
        print('Interpretation ' + intp_type + ': ')
        print('csa: ', csa_avg)
        print('era: ', era_avg)


def attack_interpretation(model, x, intp_type, DEVICE, ITER_ATTACK=20, STEP_SIZE=0.0015, TOPK=400):  # [100, 20] [0.002, 0.0015] [40, 400]
    x = x.to(DEVICE)
    pred = model(x)
    y_target = torch.argmax(pred)

    # attack
    x_adv = x + 0.0
    intp_org = torch.zeros_like(x).to(DEVICE)
    for t in range(ITER_ATTACK):
        # get interpretation for current x_adv
        if intp_type == 'vagrad':
            intp, _ = vanilla_gradient(model, x_adv, DEVICE, y_target, True)
        elif intp_type == 'smoothgrad':
            intp, _ = smooth_gradient(model, x_adv, DEVICE, y_target, True)
        elif intp_type == 'inpgrad':
            intp, _, _ = gradient_times_input(model, x_adv, DEVICE, y_target, True)
        elif intp_type == 'smoothinpgrad':
            intp, _, _ = smooth_gradient_times_input(model, x_adv, DEVICE, y_target, True)
        elif intp_type == 'integrad':
            intp, _, _ = integrated_gradient(model, x_adv, DEVICE, y_target, True)
        else:
            sys.exit("Unknown interpretation types.")

        if t == 0:
            intp_org = intp

        obj_attack = attack_objective_topk(intp, TOPK)
        # print(obj_attack)
        x_delta = grad(obj_attack, x_adv)[0]

        # normalize
        tmp_max = torch.max(torch.abs(x_delta)) + 1e-8
        x_delta = x_delta / tmp_max * STEP_SIZE

        # attack within the constraint
        x_adv = clip_image_perturb(x_adv, x, x_delta)

    result = topk_intersection(intp_org, intp, TOPK)

    return result


def attack_objective_topk(intp, K=40):
    # thre = torch.sort(torch.flatten(intp, start_dim=1), dim=1, descending=True)[0][:, K]
    # thre = thre.unsqueeze(-1)
    # thre = thre.unsqueeze(-1)
    thre = torch.sort(torch.flatten(intp), descending=True)[0][K]
    mask = (intp > thre).float()
    obj_attack = - (intp * mask).sum()
    # print(obj_attack)
    return obj_attack


def get_examples(dataloader, num_data, num_examples=300):
    list_x = []
    list_y = []
    random.seed(1)
    samples = random.sample(range(num_data), num_examples)
    batch_size = dataloader.batch_size
    batch_id = 0
    for data in dataloader:
        inputs, labels = data   # [batch_size, C, H, W], [batch_size]
        for tmp in range(batch_size):
            cur_id = batch_id * batch_size + tmp
            if cur_id in samples:
                list_x.append(inputs[tmp].unsqueeze(0))
                list_y.append(labels[tmp])
        batch_id += 1
    return list_x, list_y


def interpretation_expected(model, x, DEVICE, intp_type, EXP_RANGE, num_exp=20):
    shape_exp = list(x.shape)
    shape_exp[0] = num_exp

    intp = torch.zeros_like(x).to(DEVICE)
    weight = torch.zeros_like(x).to(DEVICE)
    w_0 = 0
    weight_all = torch.zeros(shape_exp).to(DEVICE)
    w_0_all = torch.zeros(num_exp)
    if intp_type == 'vagrad':
        for i in range(num_exp):
            x_i = x + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i = vanilla_gradient(model, x_i, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight_all[i] = intp_i
            w_0_all[i] = b_0_i
        intp /= num_exp
        weight = intp
        w_0 /= num_exp
    elif intp_type == 'smoothgrad':
        for i in range(num_exp):
            x_i = x + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i = smooth_gradient(model, x_i, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight_all[i] = intp_i
            w_0_all[i] = b_0_i
        intp /= num_exp
        weight = intp
        w_0 /= num_exp
    elif intp_type == 'lime':
        for i in range(num_exp):
            x_i = x + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i = LIME(model, x_i, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight_all[i] = intp_i
            w_0_all[i] = b_0_i
        intp /= num_exp
        weight = intp
        w_0 /= num_exp
    elif intp_type == 'inpgrad':
        for i in range(num_exp):
            x_i = x + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i, grad_i = gradient_times_input(model, x_i, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight += grad_i
            weight_all[i] = grad_i
            w_0_all[i] = b_0_i + torch.sum(intp_i)
        intp /= num_exp
        w_0 = w_0 / num_exp + torch.sum(intp)
        weight /= num_exp
    elif intp_type == 'smoothinpgrad':
        for i in range(num_exp):
            x_i = x + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i, grad_i = smooth_gradient_times_input(model, x_i, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight += grad_i
            weight_all[i] = grad_i
            w_0_all[i] = b_0_i + torch.sum(intp_i)
        intp /= num_exp
        w_0 = w_0 / num_exp + torch.sum(intp)
        weight /= num_exp
    elif intp_type == 'integrad':
        for i in range(num_exp):
            x_i = x + (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
            intp_i, b_0_i, grad_i = integrated_gradient(model, x_i, DEVICE)
            intp += intp_i
            w_0 += b_0_i
            weight += grad_i
            weight_all[i] = grad_i
            w_0_all[i] = b_0_i + torch.sum(intp_i)
        intp /= num_exp
        w_0 = w_0 / num_exp + torch.sum(intp)
        weight /= num_exp
    elif intp_type == 'gradcam':
        pass
    else:
        sys.exit("Unknown interpretation types.")

    return weight, w_0, weight_all, w_0_all


def approximation_bias(model, x, weight, w_0, DEVICE, EXP_RANGE, num_exp=50):   # [50, 100]
    # compute the approximation bias between f(x) and exp(f^(x)) around x.
    x = x.to(DEVICE)
    pred = model(x)
    y_target = torch.argmax(pred)

    # expectation around x, compute difference between f and exp{f^}
    bias_appr = 0.
    for i in range(num_exp):
        noise = (torch.rand(x.shape) - 0.5) * 2 * EXP_RANGE
        noise = noise.to(DEVICE)
        x_i = x + noise

        # true prediction
        pred_f = model(x_i)[0, y_target]
        # approximated prediction
        pred_fhat = w_0 + torch.sum((x_i - x) * weight)

        # approximation bias
        bias_appr += torch.pow(pred_fhat.detach() - pred_f.detach(), 2)     # be careful of memory explosion
    bias_appr /= num_exp
    # print(bias_appr)
    return bias_appr


def approximation_var(x, weight, w_0, weight_all, w_0_all, intp_type, DEVICE, EXP_RANGE, num_exp=20):
    # compute the approximation bias between f^(x) and exp(f^(x)) around x.
    var_appr_inneravg = 0.
    if intp_type == 'vagrad':
        for i in range(num_exp):
            weight_i, w_0_i = weight_all[i:i + 1], w_0_all[i]
            var_appr_inneravg += compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE)
    elif intp_type == 'smoothgrad':
        for i in range(num_exp):
            weight_i, w_0_i = weight_all[i:i + 1], w_0_all[i]
            var_appr_inneravg += compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE)
    elif intp_type == 'inpgrad':
        for i in range(num_exp):
            weight_i, w_0_i = weight_all[i:i + 1], w_0_all[i]
            var_appr_inneravg += compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE)
    elif intp_type == 'smoothinpgrad':
        for i in range(num_exp):
            weight_i, w_0_i = weight_all[i:i + 1], w_0_all[i]
            var_appr_inneravg += compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE)
    elif intp_type == 'integrad':
        for i in range(num_exp):
            weight_i, w_0_i = weight_all[i:i + 1], w_0_all[i]
            var_appr_inneravg += compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE)
    elif intp_type == 'lime':
        for i in range(num_exp):
            weight_i, w_0_i = weight_all[i:i + 1], w_0_all[i]
            var_appr_inneravg += compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE)
    else:
        sys.exit("Unknown interpretation types.")
    var_appr_inneravg /= num_exp

    return var_appr_inneravg


def compute_diff(x, EXP_RANGE, weight, w_0, weight_i, w_0_i, DEVICE, num_exp=50):
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


def vanilla_gradient(model, x, DEVICE, y=None, further_grad=False):
    x = x.to(DEVICE).requires_grad_()   # assign value
    pred = model(x)
    if y is None:
        y_target = torch.argmax(pred)
    else:
        y_target = y
    b_0 = pred[0, y_target]
    model.zero_grad()
    gradient = grad(pred[0, y_target], x, create_graph=further_grad)[0]    # [0] is necessary
    return gradient, b_0


def smooth_gradient(model, x, DEVICE, y=None, further_grad=False, noise_range=0.01, n_iters=20):    # [0.05, 0.01]
    x = x.to(DEVICE).requires_grad_()
    pred = model(x)
    if y is None:
        y_target = torch.argmax(pred)
    else:
        y_target = y
    b_0 = pred[0, y_target]
    # get multiple gradients and average them
    smooth_grad = torch.zeros_like(x)
    for t in range(n_iters):
        noise = (torch.rand(x.shape) - 0.5) * 2 * noise_range
        noise = noise.to(DEVICE)
        noisy_x = (x + noise).requires_grad_()

        pred = model(noisy_x)
        model.zero_grad()
        grad_t = grad(pred[0, y_target], noisy_x, create_graph=further_grad)[0]
        smooth_grad += grad_t
    smooth_grad /= n_iters
    return smooth_grad, b_0


def LIME(model, x, DEVICE, y=None, noise_range=0.05, n_samples=2000):
    x = x.to(DEVICE)
    pred = model(x)
    if y is None:
        y_target = torch.argmax(pred)
    else:
        y_target = y
    b_0_ = pred[0, y_target]    # offset

    X_lime = []
    y_lime = []
    for n in range(n_samples):
        noise = (torch.rand(x.shape) - 0.5) * 2 * noise_range
        noise = noise.to(DEVICE)
        x_ = x + noise
        y_ = model(x_)
        X_lime.append(x_.detach().cpu().numpy()[0, 0].flatten())
        y_lime.append(y_[0, y_target].detach().cpu().numpy())

    reg = Ridge(alpha=0.1).fit(X_lime, y_lime)
    b_0 = reg.intercept_
    weights = reg.coef_

    weights = weights.reshape(list(x.shape))
    weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    return weights, b_0_


def gradient_times_input(model, x, DEVICE, y=None, further_grad=False):
    x = x.to(DEVICE).requires_grad_()
    pred = model(x)
    if y is None:
        y_target = torch.argmax(pred)
    else:
        y_target = y
    b_0 = model(torch.zeros_like(x).to(DEVICE))[0, y_target]    # all_zero ref point
    model.zero_grad()
    gradient = grad(pred[0, y_target], x, create_graph=further_grad)[0]  # [0] is necessary
    grad_input = x * gradient
    # print(pred[0, y_target], 'gt')
    # print(torch.sum(grad_input)+b_0)
    # print(gradient[0, 0, 0, 0:5])
    return grad_input, b_0, gradient


def smooth_gradient_times_input(model, x, DEVICE, y=None, further_grad=False, noise_range=0.01, n_iters=20):    # [0.05, 0.01]
    x = x.to(DEVICE).requires_grad_()
    pred = model(x)
    if y is None:
        y_target = torch.argmax(pred)
    else:
        y_target = y
    b_0 = model(torch.zeros_like(x).to(DEVICE))[0, y_target]  # all_zero ref point

    smooth_grad = torch.zeros_like(x)
    for t in range(n_iters):
        noise = (torch.rand(x.shape) - 0.5) * 2 * noise_range
        noise = noise.to(DEVICE)
        noisy_x = (x + noise).requires_grad_()

        pred = model(noisy_x)
        model.zero_grad()
        grad_t = grad(pred[0, y_target], noisy_x, create_graph=further_grad)[0]
        smooth_grad += grad_t
    smooth_grad /= n_iters
    smooth_inpgrad = x * smooth_grad
    return smooth_inpgrad, b_0, smooth_grad


def integrated_gradient(model, x, DEVICE, y=None, further_grad=False, n_iters=10):  # [10, 10]
    x = x.to(DEVICE).requires_grad_()
    pred = model(x)
    if y is None:
        y_target = torch.argmax(pred)
    else:
        y_target = y
    # print(pred[0, y_target], 'gt')
    b_0 = model(torch.zeros_like(x).to(DEVICE))[0, y_target]    # all_zero ref point
    # get multiple gradient and average them
    x_ref = torch.zeros_like(x).to(DEVICE)
    grad_path = torch.zeros_like(x)
    for n in range(1, n_iters+1):
        x_ = x_ref + float(n)/n_iters * (x - x_ref)
        x_ = x_.requires_grad_()

        pred = model(x_)
        model.zero_grad()
        grad_n = grad(pred[0, y_target], x_, create_graph=further_grad)[0]
        grad_path += grad_n
    grad_path /= n_iters
    inte_grad = x * grad_path
    # print(torch.sum(inte_grad)+b_0, '\n')
    # print(grad_n[0, 0, 0, 0:5], '\n')
    return inte_grad, b_0, grad_n


def grad_cam(model, x, DEVICE):
    x = x.to(DEVICE)
    pred = model(x)
    y_target = torch.argmax(pred)
    pred[0, y_target].backward()
    grad_act = model.get_activations_gradient()
    pooled_grad = torch.mean(grad_act, dim=[0, 2, 3], keepdim=True)
    activations = model.get_activations(x).detach()
    gradcam = torch.mean(activations * pooled_grad, dim=1, keepdim=True)
    tmp_mapper = torch.nn.Upsample(size=(x.shape[-2], x.shape[-1]))
    gradcam_vis = tmp_mapper(gradcam)
    return gradcam, gradcam_vis, activations


def perturb_csa(model, x, intp, DEVICE, step_sizes):
    # original prediction
    x = x.to(DEVICE)
    pred = model(x)
    y_target = torch.argmax(pred)
    pred_org = pred[0, y_target]

    # perturbed prediction
    csa_scores = []
    intp /= torch.pow(torch.sum(torch.pow(intp, 2)), 0.5)
    for step_size in step_sizes:
        x_adv = (x - intp * step_size).to(DEVICE)
        pred_adv = model(x_adv)[0, y_target]
        csa_score = pred_org - pred_adv
        csa_score = csa_score.detach().cpu().numpy()
        csa_scores.append(csa_score)

    return np.array(csa_scores)


def perturb_era(model, x, intp, DEVICE, nums_remove):
    # original prediction
    x = x.to(DEVICE)
    pred = model(x)
    y_target = torch.argmax(pred)
    pred_org = pred[0, y_target]

    # perturbed prediction
    era_scores = []
    for num_remove in nums_remove:
        thre = torch.sort(torch.flatten(intp), descending=True)[0][num_remove]
        mask = (intp < thre).float()
        x_adv = x * mask
        pred_adv = model(x_adv)[0, y_target]
        era_score = pred_org - pred_adv
        era_score = era_score.detach().cpu().numpy()
        era_scores.append(era_score)

    return np.array(era_scores)


def show_intp(x, gradient):
    x_show = np.transpose(x.detach().cpu().numpy()[0], (1, 2, 0))
    intp_arr = gradient.data.cpu().numpy()[0]
    intp_arr = np.transpose(intp_arr, (1, 2, 0))  # shape of the image should be height * width * channels

    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(np.squeeze(x_show))
    fig.add_subplot(1, 2, 2)
    plt.imshow(np.abs(np.squeeze(intp_arr)))  # (28, 28, 1) -> (28, 28)
    plt.show()


def clip_image_perturb(img, img_org, perturb, adv_range=0.01, pixel_min=0, pixel_max=1):    # [0.04, 0.01]
    switch = (torch.abs(img+perturb-img_org) > adv_range).float()     # indicate whether exceeds epsilon
    sign_switch = (perturb > 0).float() - (perturb < 0).float()
    img_adv = switch*(img_org + adv_range * sign_switch) + (1.0-switch)*(img + perturb)
    # img_adv = torch.clamp(img_adv, pixel_min, pixel_max)      # problematic
    return img_adv
