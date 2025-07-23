import torch
import numpy as np
def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X
def LEEP(X, y,model, model_path):
    #
    n = len(y)
    num_classes = len(np.unique(y))

    # read classifier
    # Group1: model_name, fc_name, model_ckpt
    # ckpt_models = {
    #
    #     'resnet50': ['fc.weight', './models/group1/checkpoints/resnet50-19c8e357.pth'],
    #
    # }
    model = model
    ckpt_loc = model_path  # 模型参数地址

    fc_weight = 'fc.weight'  # 全连接层的权重
    fc_bias = fc_weight.replace('weight', 'bias')
    ckpt = torch.load(ckpt_loc, map_location='cpu',weights_only=False)  # 加载模型
    fc_weight = ckpt[fc_weight].detach().numpy()  #
    fc_bias = ckpt[fc_bias].detach().numpy()

    # p(z|x), z is source label
    prob = np.dot(X, fc_weight.T) + fc_bias
    prob = softmax(prob)  # p(z|x), N x C(source)
    #
    pyz = np.zeros((num_classes, 100))  # C(source) = 10
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]  #
        filter_ = np.take(prob, indices, axis=0)
        pyz[y_] = np.sum(filter_, axis=0) / n
    # 延一个维度的求和，获得边缘概率
    pz = np.sum(pyz, axis=0)  # marginal probability 边缘概率
    py_z = pyz / pz  # conditional probability, C x C(source)
    py_x = np.dot(prob, py_z.T)  # N x C

    # leep = E[p(y|x)]
    leep_score = np.sum(py_x[np.arange(n), y]) / n  # 取平均
    return leep_score