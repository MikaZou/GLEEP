# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
"""
使用的数据集：CIFAR10
采用的神经网络：ResNet20
预训练"""

import torch


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from Tr_method.LEEP import LEEP
from utls import prepare_data
from utls import forward_pass
import json
from models.group2.resnet18 import ResNet18
from models.group2.resnet34 import ResNet34
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置随机种子保证可重复性
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 基本参数设定
num_epochs = 30
batch_size_train = 64
batch_size_test = 128
lr = 0.001

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def create_pretrained_model(model_name='ResNet20',Source_dataset='CIFAR10'):
    # 初始化模型
    # model = None
    Pretrain_CHECKPOINT_PATH = f'checkpoint/Pretrain/{model_name}/40_64_0.001/best_model.pth'
    if Source_dataset == 'CIFAR10':
        if model_name == 'ResNet18':
            model = ResNet18()
            if Pretrain_CHECKPOINT_PATH is not None:
                model.load_state_dict(torch.load(Pretrain_CHECKPOINT_PATH, map_location=DEVICE))
        elif model_name == 'ResNet34':
            model = ResNet34()
            if Pretrain_CHECKPOINT_PATH is not None:
                model.load_state_dict(torch.load(Pretrain_CHECKPOINT_PATH, map_location=DEVICE))
    elif  Source_dataset == 'ImageNet':
        pretrained_path = None
        if model_name == 'ResNet18':
            model = torchvision.models.resnet18(
                weights='IMAGENET1K_V1' if pretrained_path is None else None
            )
        elif model_name == 'ResNet34':
            model = torchvision.models.resnet34(
                weights='IMAGENET1K_V1' if pretrained_path is None else None
            )
    # 加载预训练权重

    return model.to(DEVICE)

def create_model(NUM_CLASSES, model_name='ResNet18',Source_dataset='CIFAR10',Train_type='Finetune'):
    """创建微调模型，支持 ResNet18/34"""
    # 参数校验
    assert isinstance(NUM_CLASSES, int) and NUM_CLASSES > 0

    save_path = os.path.join(f'./checkpoint/{Train_type}/{model_name}/{Source_dataset}/C{NUM_CLASSES}', f'{num_epochs}_{batch_size_train}_{lr}')
    checkpoint_path = os.path.join(save_path,'best_model.pth')
    print(checkpoint_path)
    # 初始化模型
    if Source_dataset == 'CIFAR10':
        if model_name == 'ResNet18':
            model = ResNet18()
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, NUM_CLASSES)
        elif model_name == 'ResNet34':
            model = ResNet34()
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, NUM_CLASSES)
    elif Source_dataset == 'ImageNet':
        pretrained_path = None
        if model_name == 'ResNet18':
            model = torchvision.models.resnet18(
                weights='IMAGENET1K_V1' if pretrained_path is None else None
            )
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, NUM_CLASSES)
        elif model_name == 'ResNet34':
            model = torchvision.models.resnet34(
                weights='IMAGENET1K_V1' if pretrained_path is None else None
            )
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    # 替换分类头
    return model.to(DEVICE)

def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict
        json.dump(score_dict, f)

def test(Finetune_Model,test_loader,device,criterion):
    Finetune_Model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = Finetune_Model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, acc

def Transferability(dataloader,model,model_path,metrics='LEEP',cov_type='full',init_method='random'):
    "预训练模型，CIFAR100目标数据集"
    fc = model.fc
    Score = 0
    Xt_feature, Xt_output, yt_label = forward_pass(dataloader,model,fc)
    if metrics == 'LEEP':
        print(f"Calculating LEEP ")
        Score = LEEP(Xt_feature, yt_label, model,model_path)
    elif metrics == 'CLEEP':
        print(f"Calculating CLEEP")
        Xt_output_np = Xt_output.cpu().numpy()
        yt_label_np = yt_label.cpu().numpy()

        kmeans_target = KMeans(n_clusters=len(np.unique(yt_label_np))).fit(Xt_output_np)
        yt_p_labels = kmeans_target.labels_
        print(yt_p_labels)
        Score = LEEP(Xt_feature, yt_p_labels, model,model_path)
    elif metrics == 'GLEEP':
        print(f"Calculating GLEEP")
        Xt_output_np = Xt_output.cpu().numpy()
        yt_label_np = yt_label.cpu().numpy()
        gmm = GaussianMixture(n_components=len(np.unique(yt_label_np)),
                              covariance_type=cov_type,
                              init_params=init_method)
        gmm.fit(Xt_output_np)
        yt_p_labels = gmm.predict(Xt_output_np)
        print(yt_p_labels)
        Score = LEEP(Xt_feature, yt_p_labels, model,model_path)
    print(f"Transferability Score: {Score:.4f}")
    return Score
# 训练循环



import time
import itertools
if __name__ == '__main__':
    #

    model_names = ['ResNet18','ResNet34']
    source_datasets = ['CIFAR10']
    train_types = ['Finetune']
    metrics = ['GLEEP']
    cov_types = ['spherical', 'diag', 'tied', 'full']#
    init_methods = ['random', 'kmeans']
    all_combinations = itertools.product(model_names, source_datasets, train_types, metrics,cov_types,init_methods)
    for combination in all_combinations:
        model_name, Source_dataset, Train_type, metric,cov_type,init_method = combination
        print(f"\n{'=' * 80}")
        print(f"Running experiment with: "
              f"Model={model_name}, Dataset={Source_dataset}, "
              f"Train={Train_type}, Metric={metric},Cov={cov_type},init method={init_method}")
        print(f"{'=' * 80}\n")
        Pretrain_CHECKPOINT_PATH = f'checkpoint/Pretrain/{model_name}/40_64_0.001/best_model.pth'
        score_dict = {}
        output_dir = f'./result/{Train_type}/{model_name}/{Source_dataset}/{cov_type}/{init_method}'
        # 保存测试数据集准确率
        ACC_dict = {}
        fpath_ACC = os.path.join(output_dir, 'test_ACC')
        if not os.path.exists(fpath_ACC):
            os.makedirs(fpath_ACC)
        fpath_ACC = os.path.join(fpath_ACC, f'{metric}_ACC.json')
        if not os.path.exists(fpath_ACC):
            save_score(ACC_dict, fpath_ACC)
        else:
            with open(fpath_ACC, "r") as f:
                ACC_dict = json.load(f)
        # 保存迁移性分数
        score_dict = {}
        fpath_score = os.path.join(output_dir, 'metrics')
        if not os.path.exists(fpath_score):
            os.makedirs(fpath_score)
        fpath_score = os.path.join(fpath_score, f'{metric}.json')
        if not os.path.exists(fpath_score):
            save_score(score_dict, fpath_score)
        else:
            with open(fpath_score, "r") as f:
                score_dict = json.load(f)
        t3 = time.time()
        for NUM_CLASSES in range(2,101,1):

            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            train_set, test_set = prepare_data(NUM_CLASSES,transform_train,transform_test)
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=batch_size_test, shuffle=False, num_workers=1)
            # 初始化模型、优化器和损失函数
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #
            Pretrained_model = create_pretrained_model(model_name=model_name,Source_dataset=Source_dataset)
            Finetune_model = create_model(model_name=model_name, NUM_CLASSES=NUM_CLASSES,Source_dataset=Source_dataset,
                                          Train_type=Train_type)

            criterion = nn.CrossEntropyLoss()
            best_acc = 0.0
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            history = {
                'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': []
            }
            print("Training started...")
            print(f"Using device: {device}")
            print(f"Epoch:{num_epochs} | Batch size: {batch_size_train} | Learning rate: {lr}")
            t1 = time.time()


            test_loss, test_acc = test(Finetune_model,test_loader,device=DEVICE,criterion=criterion)
            socre = Transferability(test_loader,Pretrained_model,Pretrain_CHECKPOINT_PATH,metrics=metric,cov_type=cov_type,init_method=init_method)

            history['test_acc'].append(test_acc)
            ACC_dict[f'{NUM_CLASSES:03d}'] = test_acc
            save_score(ACC_dict, fpath_ACC)
            score_dict[f'{NUM_CLASSES:03d}'] = socre
            save_score(score_dict, fpath_score)
            print(f'Task Class: {NUM_CLASSES + 1:03d} | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
            t2 = time.time()
            print(f"Testing completed in {t2 - t1:.2f} seconds.")
        t4 = time.time()
        print(f"Task completed in {t4 - t3:.2f} seconds.")
