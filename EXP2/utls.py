import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
def forward_pass(score_loader, model, fc_layer, model_name='resnet50'):
    """
    a forward pass on target dataset
    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params fc_layer: the fc layer of the model, for registering hooks
    returns
        features: extracted features of model
        outputs: outputs of model
        targets: ground-truth labels of dataset
    """
    # 初始化的时候，features, outputs, targets都是空的
    features = []
    outputs = []
    targets = []
    model = model.cuda()

    # 定义的一个hook函数，在最后一层全连接层上注册，记录前向传播的特征和输出
    def hook_fn_forward(module, input, output):
        # features.append(input[0].detach().cpu())
        # features.append(input[0].detach().cpu())
        # outputs.append(output.detach().cpu())
        features.append(input[0].detach().cuda())
        outputs.append(output.detach().cuda())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.cuda()
            _ = model(data)

    forward_hook.remove()
    # # 根据不同模型处理提取出的特征
    # if model_name in ['pvt_tiny', 'pvt_small', 'pvt_medium', 'deit_small',
    #                   'deit_tiny', 'deit_base', 'dino_base', 'dino_small',
    #                   'mocov3_small']:
    #     features = torch.cat([x[:, 0] for x in features])
    #
    # elif model_name in ['pvtv2_b2', 'pvtv2_b3']:
    #     features = torch.cat([x.mean(dim=1) for x in features])
    #
    # elif model_name in ['swin_t', 'swin_s']:
    #     avgpool = nn.AdaptiveAvgPool1d(1).cuda()
    #     features = torch.cat([torch.flatten(avgpool(x.transpose(1, 2)), 1) for x in features])
    #
    # else:
    #     features = torch.cat([x for x in features])
    features = torch.cat([x for x in features])
    outputs = torch.cat([x for x in outputs])
    targets = torch.cat([x for x in targets])

    return features.cpu(), outputs, targets

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, class_mapping):
        self.dataset = dataset
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in class_mapping]
        self.class_map = {k: v for v, k in enumerate(class_mapping)}

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, self.class_map[label]

    def __len__(self):
        return len(self.indices)

def prepare_data(num_classes, transform_train, transform_test):
    # 获取类别信息
    temp_set = torchvision.datasets.CIFAR100(
        root='./data/CIFAR100', train=True, download=True, transform=None)
    all_classes = torch.randperm(100)[:num_classes].sort()[0].tolist()
    class_names = [temp_set.classes[i] for i in all_classes]

    # 打印选择的类别
    print(f"Selected {num_classes} classes:")
    for idx, name in enumerate(class_names):
        print(f"{idx:2d}: {name}")

    # 定义数据集包装类


    # 加载并过滤数据集
    train_set = torchvision.datasets.CIFAR100(
        root='./data/CIFAR100', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(
        root='./data/CIFAR100', train=False, download=True, transform=transform_test)

    return (
        FilteredDataset(train_set, all_classes),
        FilteredDataset(test_set, all_classes)
    )
def plot_from_history(history_data, save_path=None):
    plt.figure(figsize=(12, 5))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(history_data['train_loss'], label='Train')
    plt.plot(history_data['test_loss'], label='Test')
    plt.title(f"Loss Curve\n{history_data.get('metadata', {}).get('save_time', '')}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(history_data['train_acc'], label='Train')
    plt.plot(history_data['test_acc'], label='Test')
    plt.title(f"Accuracy Curve\nEpochs: {len(history_data['train_acc'])}")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    plt.close()