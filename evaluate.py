import re
import string

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def validate(model, data_generator, device):

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([23.7]).float().to(device))
    loss_full = []
    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    for minibatch in data_generator:
        y_pred = model(minibatch["candidate_news"], minibatch["clicked_news"])
        y = minibatch["clicked"].float().to(device)
        loss = criterion(y_pred, y)
        loss_full.append(loss.item())
        y_pred_list = y_pred.tolist()
        y_list = y.tolist()

        auc = roc_auc_score(y_list, y_pred_list)
        mrr = mrr_score(y_list, y_pred_list)
        ndcg5 = ndcg_score(y_list, y_pred_list, 5)
        ndcg10 = ndcg_score(y_list, y_pred_list, 10)

        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)


    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)




