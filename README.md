# mixup_keras
implementation of mixup paper ICLR 2018 with tensorflow 1.13.1

## Introduction
In this work, Facebook research propose mixup, a simple learning principle to alleviate these issues. In essence, mixup trains
a neural network on **convex combinations** of pairs of examples and their labels.
By doing so, mixup regularizes the neural network to favor simple linear behavior
in-between training examples.

## Dependency

Please check the requirements.txt.
```
pip install -r requirements.txt
```

## Usage

Train

```python
python train.py
```

Visualize the result of convex combinations.

```python
python visualize.py
```

## Visualization: convex combination

- MNIST

![mnist](./results/sample1_3.gif)

- CIFAR10

![cifar10](./results/sample[5]_[4].gif)

- FashionMNIST

![fm](./results/sample6_3.gif)

## Reference

- [paper][mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/abs/1710.09412)
- [github](https://github.com/facebookresearch/mixup-cifar10)
