from scipy.stats import kendalltau
from scipy.stats import weightedtau
import json
from scipy.stats import pearsonr
from utils import wpearson
import numpy as np

def recall_k(score, finetune_acc, dset, k):
    #succed = 0
    sorted_score = sorted(score.items(), key=lambda i: i[1], reverse=True)
    sorted_score = {a[0]: a[1] for a in sorted_score}
    
    gt = finetune_acc[dset]
    sorted_gt = sorted(gt.items(), key=lambda i: i[1], reverse=True)
    sorted_gt = {a[0]: a[1] for a in sorted_gt}

    top_k_gt = sorted_gt.keys()[:k]
    succed = 1 if sorted_score.keys()[0] in top_k_gt else 0
    return succed


def rel_k(score, finetune_acc, dset, k):
    sorted_score = sorted(score.items(), key=lambda i: i[1], reverse=True)
    
    gt = finetune_acc[dset]
    sorted_gt = sorted(gt.items(), key=lambda i: i[1], reverse=True)
    best_model = sorted_gt[0][0]
    sorted_gt = {a[0]: a[1] for a in sorted_gt}

    max_gt = sorted_gt[best_model]
    topk_score_model = [a[0] for i, a in enumerate(sorted_score) if i < k]
    topk_score_ft = [sorted_gt[a] for a in topk_score_model]
    return max(topk_score_ft) / max_gt


def pearson_coef(score, finetune_acc, dset):
    
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric, _ = pearsonr(metric_score, gt_)
    return tw_metric


def wpearson_coef(score, finetune_acc, dset):
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric = wpearson(metric_score, gt_)
    return tw_metric


def w_kendall_metric(score, finetune_acc, dset):

    score = score.items()
    metric_score = [a[1] for a in score]

    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric, _ = weightedtau(metric_score, gt_) #
    return tw_metric


def w_kendall_metric_mmd(score, finetune_acc, dset):

    score = score.items()
    metric_score = [a[1] for a in score]
    scores = np.array(metric_score)
    all_ones_vector = np.ones(10)
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    sum_exp_scores = np.sum(exp_scores)
    softmax_scores = exp_scores / sum_exp_scores
    final_scores = all_ones_vector - softmax_scores
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    tw_metric, _ = weightedtau(final_scores, gt_)  #
    return tw_metric


def w_kendall_metric_energy(score1, score2):
    
    score1 = score1.items()
    metric_score1 = [a[1] for a in score1]
    score2 = score2.items()
    gt_ = []
    for a in score1:
        for b in score2:
            if b[0]==a[0]:
                gt_.append(b[1])
    tw_metric, _ = weightedtau(metric_score1,gt_)
    return tw_metric

def kendall_metric(score, finetune_acc, dset):
    
    score = score.items()
    metric_score = [a[1] for a in score]
    gt = finetune_acc[dset]
    gt_ = []
    for a in score:
        gt_.append(gt[a[0]])
    t_metric, _ = kendalltau(metric_score, gt_)
    return t_metric


def load_score(path):
    with open(path, 'r') as f:
        score_ = json.load(f)
    time = score_['duration'] if 'duration' in score_.keys() else 0
    score = {a[0]: a[1] for a in score_.items() if a[0] != 'duration'}
    return score, time
