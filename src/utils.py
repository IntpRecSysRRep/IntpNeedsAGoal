import sys
import numpy as np
from math import exp
import torch

def logit2prob(pred):
    pred = np.array([exp(pred[0]), exp(pred[1])])
    return pred / np.sum(pred)

def normalize_score(score):
    return score / np.linalg.norm(score)

def rescale_embedding(mat_embd, score):
    # shrink embedding toward all 0
    for i in range(mat_embd.shape[0]):
        mat_embd[i, 0, :] *= (1 - score[i])
    return mat_embd

def perturb_embedding(mat_embd, gradient):
    # perturb embedding along reverse direction of gradient
    return mat_embd - gradient

def remove_word(mat_embd, inner1, inner2, num_remove):
    scale = np.sum(np.multiply(inner1, inner2), axis=-1, keepdims=True)
    thre = np.sort(scale, axis=0)[::-1][num_remove-1, 0, 0]
    scale[scale >= thre] = 0.
    scale[scale != 0] = 1.
    mat_embd = np.multiply(mat_embd, scale)
    return mat_embd

def remove_word_bert(masks, inner1, inner2, num_remove):
    scale = np.sum(np.multiply(inner1, inner2), axis=-1, keepdims=False)
    scale[0, torch.sum(masks).item():] = np.min(scale)-1
    thre = np.sort(scale, axis=1)[:, ::-1][0, num_remove - 1]
    scale[scale >= thre] = 0.
    scale[scale != 0] = 1.
    scale[0, torch.sum(masks).item():] = 0      # necessary
    scale = scale.astype(int)
    scale = torch.from_numpy(scale)
    masks_after = torch.mul(masks, scale)
    return masks_after

def word2zero_bert(mat_embd, masks, inner1, inner2, num_remove):
    scale = np.sum(np.multiply(inner1, inner2), axis=-1, keepdims=True)
    scale[0, torch.sum(masks).item():] = np.min(scale) - 1
    thre = np.sort(scale, axis=1)[:, ::-1, :][0, num_remove-1, 0]
    scale[scale >= thre] = 0.
    scale[scale != 0] = 1.
    scale[0, torch.sum(masks).item():] = 1.  # necessary
    mat_embd = np.multiply(mat_embd, scale)
    return mat_embd

def remasking(mask, inner1, inner2, num_remove):
    mask_new = mask.clone()
    atts = np.sum(np.multiply(inner1, inner2), axis=-1, keepdims=True)
    thre = np.sort(atts, axis=0)[::-1][num_remove - 1, 0, 0]
    atts = torch.from_numpy(atts)
    mask_new[atts >= thre] = 0.
    return mask_new

def topk_intersection(intps1, intps2, K=40):
    intps1 = torch.flatten(intps1, start_dim=1)
    intps2 = torch.flatten(intps2, start_dim=1)
    order1 = torch.argsort(intps1, dim=1, descending=True).detach().cpu().numpy()[:, 0:K]
    order2 = torch.argsort(intps2, dim=1, descending=True).detach().cpu().numpy()[:, 0:K]
    avg_intersect = 0.
    for i in range(intps1.shape[0]):
        avg_intersect += float(np.intersect1d(order1[i], order2[i]).shape[0])/K
    avg_intersect /= intps1.shape[0]
    return avg_intersect

