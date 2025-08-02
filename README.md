# GLEEP: Unsupervised Transferability Estimation via Gaussian-clustering Log Expected Empirical Prediction
## 1.Setup and package installation
```
pip install torch==1.13.1 torchvision==0.13.1  torchaudio==0.14.1
pip install timm==0.4.9
pip install scipy
pip install -U scikit-learn
pip install tqdm
pip install matplotlib
```
## 2.Experiment on image classification benchmark
### 1. Download pre-trained models
```
cd /EXP1
mkdir models/group1/checkpoints
cd models/group1/checkpoints
wget https://download.pytorch.org/models/resnet34-333f7ec4.pth
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
wget https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth
wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
wget https://download.pytorch.org/models/googlenet-1378be20.pth
wget https://download.pytorch.org/models/densenet121-a639ec97.pth
wget https://download.pytorch.org/models/densenet169-b2777c0a.pth
wget https://download.pytorch.org/models/densenet201-c1103571.pth
```
### 2.Feature Construction
In this step, we will construct features for our target datasets using pre-trained models. To accomplish this, we will use the `forward_feature.py` script. This script takes the name of the target dataset as input and generates the corresponding features.
```
python forward_feature.py -d $dataset
```
### 3.Calculating the transferability score for all the target datasets
In this step,we will calculate the transferability score for target dataset using `evaluate_metric.py`.This script takes the name of the metric and dataset as input.
```
python evaluate_metric.py -me $metric -d $dataset
```
If you want to reproduce the results of GLEEP on classification benchmark,you need to run the commandline as follows.
```
python evaluate_metric.py -me leep_g -d $dataset -dummy yes
```
### 4.Calculating the correlation
In this step,we will correlation between  transferability scores and finetune accuracy.Use `tw.py` to obtain the Kendall τ for a metric and a dataset
```
python tw.py -me $metric -d $dataset
```
Use `p.py` to obtain the Kendall τ for a metric and a dataset
```
python p.py -me $metric -d $dataset
```
If you want to reproduce the results of GLEEP specify 'leep_g' as the name of the metric
## 3.GLEEP vs. LEEP on a More Statistically Reliable Benchmark
### 1.Pretrain on Source dataset
In this step,we will pretrain our model on CIFAR10 dataset.
```
cd /EXP2
python Pretrain.py -m $model_name
```
The `model_name` has `ResNet18` and `RstNet34` as options.
### 2.Finetune or Retrain on Target dataset 
**For Finetune**
```
python Finetune.py -m $model_name -d $source_dataset -n $num_classes
```
The `model_name` has `ResNet18` and `RstNet34` as options.And the `source_dataset` has `ImageNet` and `CIFAR10` as options.
If you use the `ImageNet` as `source_dataset`,it will download the pretrained model by using pytorch.The `num_classes` is the random class number of CIFAR100.

**For Retrain**
```
python Retrain.py -m $model_name -d $source_dataset -n $num_classes
```
The `model_name` has `ResNet18` and `RstNet34` as options.And the `source_dataset` has `ImageNet` and `CIFAR10` as options.
If you use the `ImageNet` as `source_dataset`,it will download the pretrained model by using pytorch.The `num_classes` is the random class number of CIFAR100.
### 3.Calculating the GLEEP score and  empirical transferability score
Run the following commandline to calculate the GLEEP score and the empirical transferability score(Accuracy on Testing dataset) on the target dataset CIFAR100.
```
python forward_feature_exp2.py
```
Notation:You must ensure that you have already obtained the finetuned and retrained checkpoints of ResNet18 and ResNet34 on CIFAR100; otherwise, the code execution will fail with errors.
### 4.Calculating the correlation
After you obtain the GLEEP score and the empirical transferability score,Run the commandline as follows.
```
python relation_ablation.py
```

