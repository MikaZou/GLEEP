# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import argparse

from utls import plot_from_history
from models.group2.resnet18 import ResNet18
from models.group2.resnet20 import ResNet20
from models.group2.resnet34 import ResNet34

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


num_epochs = 40
batch_size_train = 64
batch_size_test = 64
lr = 0.001
from torchvision import datasets, transforms
import torch


DATASET_CONFIG = {
    'CIFAR10': {
        'num_classes': 10,
        'input_size': 32,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
        'dataset_class': datasets.CIFAR10,
        'root': './data/cifar10',
    },
    'ImageNet': {
        'num_classes': 100,
        'input_size': 32,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'dataset_class': datasets.CIFAR100,
        'root': './data/cifar100',
    },
    'ImageNet': {
        'num_classes': 1000,
        'input_size': 224,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'dataset_class': datasets.ImageNet,
        'root': './data/ImageNet',

    }
}

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
def get_data_transforms(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    if dataset_name == 'ImageNet':

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])
    else:

        train_transform = transforms.Compose([
            transforms.RandomCrop(cfg['input_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std']),
        ])


    test_transform = transforms.Compose([
        transforms.Resize(cfg['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std']),
    ])

    return train_transform, test_transform
def load_dataset(dataset_name, batch_size):
    assert dataset_name in DATASET_CONFIG, f"Dataset {dataset_name} not found."
    cfg = DATASET_CONFIG[dataset_name]
    train_transform, test_transform = get_data_transforms(dataset_name)
    kwargs = {}
    if dataset_name == 'ImageNet':
        kwargs['split'] = 'train'
        train_set = cfg['dataset_class'](root=cfg['root'], transform=train_transform, **kwargs)
        kwargs['split'] = 'val'  # 验证集
        test_set = cfg['dataset_class'](root=cfg['root'], transform=test_transform, **kwargs)
    else:
        train_set = cfg['dataset_class'](root=cfg['root'], train=True, transform=train_transform, download=True)
        test_set = cfg['dataset_class'](root=cfg['root'], train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, cfg['num_classes']


def init_model(model_name):
    model_factory = {
        'ResNet18': ResNet18,
        'ResNet20': ResNet20,
        'ResNet34': ResNet34,
    }
    assert model_name in model_factory, f"Model {model_name} not found."
    return model_factory[model_name]().to(device)
def train():
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        for batch_idx, (inputs, targets) in enumerate(test_loader):
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pretrain model on CIFAR10.')
    parser.add_argument('-m', '--model_name', type=str, default='ResNet34',
                        help='name of the pretrained model to load and evaluate')
    args = parser.parse_args()
    model_name = args.model_name
    SAVE_PATH = os.path.join(f'./checkpoint/Pretrain/{model_name}', f'{num_epochs}_{batch_size_train}_{lr}')
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    train_set = torchvision.datasets.CIFAR10(
        root='./img_data/CIFAR10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_train, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(
        root='./img_data/CIFAR10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False, num_workers=2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(model_name)
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

        train_loss, train_acc = train()

        test_loss, test_acc = test()
        scheduler.step()


        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            print(f'Saving best model with test acc: {test_acc:.2f}%')
            torch.save(model.state_dict(), os.path.join(SAVE_PATH,'best_model.pth'))
            best_acc = test_acc


        print(f'Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
    t2 = time.time()
    print(f"Training completed in {t2-t1:.2f} seconds.")

    plot_from_history(history, save_path=os.path.join(SAVE_PATH, 'reloaded_plot.png'))