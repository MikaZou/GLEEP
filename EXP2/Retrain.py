import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 必须在所有import之前
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from models.group2.resnet18 import ResNet18
from models.group2.resnet34 import ResNet34
from utls import prepare_data
# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




BATCH_SIZE = 128
LR = 0.01  # 初始学习率
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 100
MILESTONES = [50, 75]  # 学习率衰减节点

# 基本参数设定
num_epochs = 30
batch_size_train = 64
batch_size_test = 128
lr = 0.001


# 数据增强和预处理（与CIFAR10训练时保持一致）
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



# 加载并修改预训练模型
def create_retrain_model(NUM_CLASSES, model_name,Source_dataset='CIFAR10'):
    """创建微调模型，支持 ResNet20/18"""
    # 参数校验
    assert isinstance(NUM_CLASSES, int) and NUM_CLASSES > 0
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
    # 初始化模型
    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    for name, param in model.named_parameters():
        if "fc" not in name:  # 冻结非分类头的所有参数
            param.requires_grad = False
    return model.to(DEVICE)

# 训练和验证函数
def train():
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    return avg_loss, acc

def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, acc

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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Retrain model on target dateset CIFAR100.')
    parser.add_argument('-m', '--model_name', type=str, default='ResNet34',
                        help='name of the pretrained model to load and evaluate')
    parser.add_argument('-d', '--source_dataset', type=str, default='ImageNet',
                        help='name of the pretrained model to load and evaluate')
    parser.add_argument('-m', '--num_classes', type=str, default=100,
                        help='name of the pretrained model to load and evaluate')
    args = parser.parse_args()
    Source_dataset = args.source_dataset
    model_name = args.model_name
    NUM_CLASSES = args.num_classes
    # NUM_CLASSES = 2
    #
    # Source_dataset = 'CIFAR10'
    # model_name = 'ResNet34'
    Train_type = 'Retrain'
    for NUM_CLASSES in range(2,101,1):
        SAVE_PATH = os.path.join(f'./checkpoint/{Train_type}/{model_name}/{Source_dataset}/C{NUM_CLASSES}', f'{num_epochs}_{batch_size_train}_{lr}')
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        # 加载CIFAR100数据集

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        selected_classes = torch.randperm(100)[:NUM_CLASSES]
        selected_classes, _ = torch.sort(selected_classes)
        selected_classes = selected_classes.tolist()


        train_set, test_set = prepare_data(NUM_CLASSES)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size_train, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size_test, shuffle=False, num_workers=2)
        # 初始化模型、优化器和损失函数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_retrain_model(NUM_CLASSES=NUM_CLASSES, model_name=model_name, Source_dataset=Source_dataset)
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)

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
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = train()
            # 测试
            test_loss, test_acc = test()
            scheduler.step()

            # 记录训练指标
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            # 保存最佳模型
            if test_acc > best_acc:
                print(f'Saving best model with test acc: {test_acc:.2f}%')
                torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'best_model.pth'))
                best_acc = test_acc

            # 打印信息
            print(f'Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')

        t2 = time.time()
        print(f"Training completed in {t2 - t1:.2f} seconds.")
        # 保存训
        plot_from_history(history, save_path=os.path.join(SAVE_PATH, 'reloaded_plot.png'))

