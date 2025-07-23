#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torch
import models.group1 as models
import numpy as np
from sklearn.decomposition import PCA
import json
import time

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, LDA_Score, \
    PAC_Score


def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict
        json.dump(score_dict, f)


def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='leep_g',
                        help='name of the method for measuring transferability')
    parser.add_argument('--nleep-ratio', type=float, default=5,
                        help='the ratio of the Gaussian components and target data classess')
    parser.add_argument('--parc-ratio', type=float, default=2,
                        help='PCA reduction dimension')
    parser.add_argument('--output-dir', type=str, default='./results_metrics/group1',
                        help='dir of output score')
    parser.add_argument('--dummy',type=str,default='yes',
                        help='if get the dummy label')
    args = parser.parse_args()
    pprint(args)

    datasets = ['cifar10','pets', 'voc2007', 'caltech101', 'aircraft', 'cifar100', 'food', 'pets', 'flowers', 'cars', 'dtd','sun397']  #


    for dataset in datasets:
        strat_time = time.time()
        score_dict = {}
        fpath = os.path.join(args.output_dir, args.metric)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fpath = os.path.join(fpath, f'{dataset}_metrics.json')

        if not os.path.exists(fpath):
            save_score(score_dict, fpath)
        else:
            with open(fpath, "r") as f:
                score_dict = json.load(f)
        models_hub = ['mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
                              'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
        for model in models_hub:
            # if exist_score(model, fpath):
            #     print(f'{model} has been calculated')
            #     continue
            args.model = model
            # Source(Xs) feature & output
            # model_npy_label = os.path.join('./results_f/group2', f'{args.model}_imagenet_label.npy')
            source_npy_output = os.path.join('./results_f/group3', f'{args.model}_imagenet_output.npy')
            # Source(Xt) feature ,label & output
            model_npy_feature = os.path.join('./results_f/group1', f'{args.model}_{dataset}_feature.npy')
            model_npy_label = os.path.join('./results_f/group1', f'{args.model}_{dataset}_label.npy')
            model_npy_output = os.path.join('./results_f/group1', f'{args.model}_{dataset}_output.npy')
            Xt_features, yt_labels, Xt_outputs = np.load(model_npy_feature), np.load(model_npy_label), np.load(model_npy_output)
            if args.dummy == 'yes':
                gmm = GaussianMixture(n_components=len(np.unique(yt_labels)))
                t1 = time.time()
                gmm.fit(Xt_outputs)
                t2 = time.time()
                print(f'GMM time: {t2 - t1}')
                yt_p_labels = gmm.predict(Xt_outputs)

                print(yt_p_labels)
            else:
                pass

            print(f'x_trainval shape:{Xt_features.shape} and y_trainval shape:{yt_labels.shape}')
            print(f'Calc Transferabilities of {args.model} on {dataset}')

            if args.metric == 'logme':
                score_dict[args.model] = LogME_Score(Xt_features, yt_labels)
            elif args.metric == 'leep':
                score_dict[args.model] = LEEP(Xt_features, yt_labels, model_name=args.model)
            elif args.metric == 'nleep':
                ratio = 1 if dataset in ('food', 'pets') else args.nleep_ratio
                score_dict[args.model] = NLEEP(Xt_features, yt_labels, component_ratio=ratio)
            elif args.metric == 'sfda':
                score_dict[args.model] = SFDA_Score(Xt_features, yt_labels)
            elif args.metric == 'lda':
                score_dict[args.model] = LDA_Score(Xt_features, yt_labels)
            elif args.metric == 'pac':
                pg, _ = PAC_Score(Xt_features, yt_labels, lda_factor=1)
                score_dict[args.model] = -pg[0][1]
            elif args.metric == 'leep_g':
                print(yt_p_labels)
                score_dict[args.model] = LEEP(Xt_features, yt_p_labels, model_name=args.model)
            else:
                raise NotImplementedError

            print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
            save_score(score_dict, fpath)

        results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
        print(f'Models ranking on {dataset} based on {args.metric}: ')
        pprint(results)
        results = {a[0]: a[1] for a in results}
        save_score(results, fpath)
        end_time = time.time()
        print('###time:', end_time - strat_time)
