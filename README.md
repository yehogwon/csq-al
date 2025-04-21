# Candidate Set Query (CSQ) for Active Learning

This is the official implementation for ***"Enhancing Cost Efficiency in Active Learning with Candidate Set Query"***. 

> **Enhancing Cost Efficiency in Active Learning with Candidate Set Query**  
> [Yeho Gwon<sup>*</sup>](https://yehogwon.github.io/), [Sehyun Hwang<sup>*</sup>](https://sehyun03.github.io/), [Hoyoung Kim](https://cskhy16.github.io/), [Jungseul Ok](https://sites.google.com/view/jungseulok), and [Suha Kwak](https://suhakwak.github.io/) (\* *indicates equal contribution*)  
> POSTECH  
> [Arxiv](https://arxiv.org/abs/2502.06209)  
> [Project Page](https://yehogwon.github.io/csq)

## Abstract

This paper introduces a cost-efficient active learning (AL) framework for classification, featuring a novel query design called *candidate set query*. Unlike traditional AL queries requiring the oracle to examine all possible classes, our method narrows down the set of candidate classes likely to include the ground-truth class, significantly reducing the search space and labeling cost. Moreover, we leverage conformal prediction to dynamically generate small yet reliable candidate sets, adapting to model enhancement over successive AL rounds. To this end, we introduce an acquisition function designed to prioritize data points that offer high information gain at lower cost. Empirical evaluations on CIFAR-10, CIFAR-100, and ImageNet64x64 demonstrate the effectiveness and scalability of our framework. Notably, it reduces labeling cost by 42% on ImageNet64x64.

## Requirements

The implementation has been validated using Python 3.9.19, PyTorch 2.4.1, and torchvision 0.19.1 with CUDA 12.1. Due to CUDA version dependencies, manual installation of PyTorch and torchvision may be required based on your system configuration and hardware compatibility. 

To install requirements (may not work due to CUDA compatability):

```bash
$ pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
$ pip install wandb==0.17.0 numpy<2
$ pip install pandas scipy scikit-learn pillow tqdm
$ pip install transformers huggingface_hub datasets networkx faiss-gpu
```

Also, our implementation does support multi-GPU training with PyTorch DDP (Distributed Data Parallel). DDP will be automatically enabled if you have multiple GPUs. 

> **Note**
> You can specify which GPU to use with `CUDA_VISIBLE_DEVICES` environment variable to disable DDP. If you want to use a single GPU (*i.e.*, disable DDP), you can set `CUDA_VISIBLE_DEVICES` to a single one.

## Dataset

We have evaluated our method on the following datasets, although the code supports more datasets.

- CIFAR-10
- CIFAR-100
- ImageNet64x64 (*Downsampled ImageNet-1k)
- CIFAR-100N (*CIFAR-100 with noisy labels*)
- CIFAR-100LT (*CIFAR-100 with class imbalances*)

The directory structure needs to be as follows:

```
.
├── dataset/
│   ├── CIFAR-10/cifar-10-batches-py/...
│   ├── CIFAR-100/cifar-100-python/...
│   ├── ImageNet64x64/
│   │   ├── train/
│   │   ├── val/
│   │   └── proc.py
├── ...
```

> **Note** This implementation requires nothing specific for `CIFAR-100N` and `CIFAR-100LT`, with which you need to prepare your own datasets. You are required to pass the path to the datasets manually.

### CIFAR-10 and CIFAR-100

For CIFAR-10 and CIFAR-100, you can download them with `--download` flag.

### ImageNet64x64

> **Note** ImageNet32x32 can also be constructed in the same way. The following instruction is adopted from [https://patrykchrabaszcz.github.io/Imagenet32/](https://patrykchrabaszcz.github.io/Imagenet32/).

Download the downsampled ImageNet pickled files from [the official ImageNet website](https://image-net.org/download-images.php). It is highly recommended that you have your own account. You can download the files by clicking on the links with the following names:

- `Train(64x64) part 1, npz format, 6 GB`
- `Train(64x64) part 2, npz format, 6 GB`
- `Val(64x64), npz format, 509 MB`

> **:exclamation: Important** You shouldn't click on `npz format` links. They download the files in a different format.

Then you will have the following files:

- `Imagenet64_train_part1.zip`
- `Imagenet64_train_part2.zip`
- `Imagenet64_val.zip`

The next step is to preprocess the data. Under the root directory, run the following command:

```bash
$ cd dataset/ImageNet64x64
$ mkdir train val
```

After moving the files to approprirate directories (train files to `train` directory and validation file to `val` directory), run the following command to preprocess the data:


```bash
$ cd train
$ unzip Imagenet64_train_part1.zip
$ unzip Imagenet64_train_part2.zip
$ rm Imagenet64_train_part1.zip Imagenet64_train_part2.zip
$ cd ../val
$ unzip Imagenet64_val.zip
$ rm Imagenet64_val.zip
$ cd ..
$ python proc.py
```

Then you are all set. As a side note, running `python proc.py` will remove the extracted batch files. 

### CIFAR-100N

*This assumes that you have CIFAR-100 downloaded and a numpy array of noisy labels.*

If you have not downloaded CIFAR-100, you can download it using `--download` flag. Also, you should prepare a numpy array of noisy labels with a length of 50000. It can be passed to `--noisy_label_path` argument.

> **Note** This dataset just replaces the labels of CIFAR-100 with the provided noisy labels.

### CIFAR-100LT

*This assumes that you have CIFAR-100 downloaded and a numpy array of imbalanced indices.*

If you have not downloaded CIFAR-100, you can download it using `--download` flag. Also, you should prepare a numpy array of imbalanced indices with a length of 50000, with which is equivalent to `CIFAR-100[indices]` (i.e., CIFAR-100 with only the indices). It can be passed to `--indices_path` argument.   

> **Note** This dataset works exactly the same way as `torch.utils.data.Subset`. So, the imbalance ratio is highly dependent on the provided indices. 

## AL Acquisition Functions

We support a variety of AL acquisition functions. The following are available for you to use: 

- **Naive Sampling**: `CQ` in the paper
    - `Random` Sampling
    - `Entropy` Sampling
    - `BADGE` Sampling ([Ash et al. ICLR 2020](https://arxiv.org/abs/1906.03671))
    - `Coreset` Sampling ([Sener & Savarese ICLR 2018](https://arxiv.org/abs/1708.00489))
    - `ProbCover` Sampling ([Yehuda et al. NeurIPS 2022](https://arxiv.org/abs/2205.11320))
    - `SAAL` Sampling ([Kim et al. ICML 2023](https://proceedings.mlr.press/v202/kim23c))

> **Note** These acquisition functions can be used either solely or in combination with **CSQ**.

`ProbCover` requires self-supervised features, and it is not scalable to large datasets with large budgets. Thus, we provide experiments with the features only for CIFAR-10 and CIFAR-100. 

We adopt the features from self-supervised feature representation [SimCLR](https://arxiv.org/abs/2002.05709) for CIFAR-10 and CIFAR-100. You can download the features arrays from here: [CIFAR-10](https://drive.google.com/file/d/1Le1ZuZOpfxBfxL3nnNahZcCt-lLWLQSB/view) / [CIFAR-100](https://drive.google.com/file/d/1o2nz_SKLdcaTCB9XVA44qCTVSUSmktUb/view), which is provided in [this](https://github.com/avihu111/TypiClust/blob/main/USAGE.md#representation-learning) GitHub repository.

## Command-Line Arguments

> **:white_check_mark: FYI** You can find example commands for our experiments in [`script.sh`](./script.sh), which are used to run experiments with different datasets, strategies, and hyperparameters.

The training and evaluation are done by running `run.py`. The following arguments are available:

<details>
<summary><span style="font-weight: bold; font-style: italic;">Arguments</span></summary>

### Active Learning
- `--strategy`: which AL strategy to use (quite a lot options; refer to `run.py` for more details)
- `--n_rounds`: the number of AL rounds
- `--initial_budget`: the initial budget for the AL (the number of labeled samples for the initial round)
- `--budget`: the number of samples to query in each round

### CSQ
- `--k`: the size of the candidate set if it is integer. Otherwise, it is used as a hyperparameter
- `--calibration_set_size`: size of calibration set from validation set for conformal prediction
- `--cq_calib`: Query with CQ for calibration set

### Sampling
- `--d`: hyperparameter for cost-efficient sampling
- `--rho`: norm restriction for SAAL
- `--features_path`: path to dataset features
- `--delta`: delta for ProbCover

### Training
- `--dataset`: which dataset to use (MNIST, FASHIONMNIST, SVHN, CIFAR10, CIFAR100, IMAGENET32, IMAGENET64, CIFAR100N, CIFAR100LT, R52)
- `--model`: which model to use (resnet18, resnet34, resnet50, resnet101, efficientnet, mobilenet, wrn-28-5, wrn-36-2, wrn-36-5, vgg, preactresnet18, preactwideresnet18, svm)
- `--lr`: learning rate for training
- `--n_epochs`: the number of epochs for training

### *etc*.
- `--seed`: selection of seed for randomness

</details>

## Acknowledgements

- This code is built on top of Ash's [BADGE](https://github.com/JordanAsh/badge) implementation. Thanks for the great implementation!
- The implementation of `ProbCover` is based on, but not fully adopted from, Dekel's [TypiClust](https://github.com/avihu111/TypiClust) implementation. We appreciate the great work.
