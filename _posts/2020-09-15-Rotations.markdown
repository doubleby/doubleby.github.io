---
layout: post
title: "Unsupervised Representation Learning by Predicting Image Rotations 논문 리뷰"
description: "Image의 augmentation 중 하나인 rotation을 활용한 Rotations 논문 리뷰"
date: 2020-09-15 21:03:36 +0530
categories: Self-Supervised-Learning
---
---

이번에 리뷰할 논문은 '[UNSUPERVISED REPRESENTATION LEARNING BY PREDICTING IMAGE ROTATIONS][paper]' 입니다. 기존에 존재하는 pretext task에 비하여, image를 회전하여 학습하는 simple한 pretext task를 제안하였으며, 높은 성능을 보였습니다.

---

## Pretext Task

논문에서 제안하는 **pretext task** 는 image를 0도, 90도, 180도, 270도로 **rotation** 하여 학습하는 것입니다. 아래의 그림을 보시면, 조금 더 직관적으로 이해하실 수 있습니다.

![img](https://i.imgur.com/pnwXOZG.png)

rotation의 **mechanism** 과 자세한 설명은 아래의 그림과 같습니다.

![img](https://i.imgur.com/zLFK0qa.png)

논문의 저자는 rotation을 구축할때, 아래와 같이 4가지를 고려하여 구축하였습니다.

### forcing the learning of semactic feature

image의 rotation을 성공적으로 예측하기 위해서는 image의 label이 되는 obejct로 국한시키는 것이 중요합니다. 아래 그림의 오른쪽을 보시게 되면, model의 rotation recognition task에 사용되는 attention map을 시각화하였습니다.

![img](https://i.imgur.com/6jkRdLp.png)

label이 되는 object로 국한시키기 위해 눈, 코, 꼬리, 머리와 같은 image의 high level object parts에 집중하여 학습시켰습니다. 오른쪽에 있는 일반적인 object recognition task에 사용하는 attention map과 비교하였을때, 둘 다 거의 비슷한 region을 집중하였습니다.

그리고, model의 첫번째 layer에서 recognize rotations을 supervised와 self-supervised로 아래와 같이 비교하였습니다.

![img](https://i.imgur.com/16G19TF.png)

오른쪽인 self-supervised의 recognize rotations이 supervised보다 다양한 filter를 가지고 있는 것을 확인할 수 있었습니다.

### absence of low-level visual artifacts

image의 low-level의 feature들까지 학습하기 위해 flip과 transpose를 수행합니다. 기존의 scale를 조정하거나 resize와 같은 image 변형에 비해서 더 많은 feature들을 학습할 수 있었습니다.

### well-posedness

사람이 image를 capture하게 되면 너무 정직한 위치에 object를 국한시키는 경우가 있습니다. 이를 위해 rotation을 정의하게 되었으며, 모호함을 줄일 수 있습니다.

### implementing image rotations

image를 0도, 90도, 180도, 270도로 돌려서 학습을 진행하였습니다.

---

## Result

논문에서 제안하는 pretext task의 성능을 확인하기 위해, CIFAR-10, ImageNet, PASCAL, Places205 데이터에 적용하였습니다.

### CIFAR

CIFAR 데이터를 적용하여 성능을 평가하기 전에, 저자는 여러가지의 사항을 고려하였습니다.

####  depth of layer

model의 backbone은 RotNet model을 사용하였습니다. 그리고 layer의 depth에 따라, layer별로 성능을 확인하기 위해 3, 4, 5 concolution layer를 쌓아서 비교하였습니다. 결과는 아래의 표와 같습니다.

![img](https://i.imgur.com/Pn897nw.png)

ConvB2에서 정확도가 높았던 이유는 아래와 같습니다.
1. rotation predicttion의 self-supervised task가 specific
1. 깊게 쌓은 layer

#### number of rotations

rotation task를 여러 각도로 회전하였을때, 성능을 비교하였습니다. 여러 각도의 회전의 결과는 아래의 표와 같습니다.

![img](https://i.imgur.com/iZ3tJpj.png)

4가지의 각도로 rotation이 성능이 제일 높았으며, 다른 각도들보다 성능이 높았던 이유는 아래와 같습니다.
1. 2개의 각도는 recognize하기엔 작다.
1. 8개의 각도는 4개의 각도에 비해서 확실히 구별하지 못한다.

#### comparison methods

저자가 제안한 pretext task의 성능을 비교하기 위해, 기존에 제안된 methods과 비교하였습니다. 결과는 아래의 표와 같습니다.

![img](https://i.imgur.com/13TTC1C.png)

기존의 unsupervised methods과 비교하였을때, 높은 성능을 보였습니다. 그리고 supervised method와 비교해도 큰 차이가 없을 정도로 높은 성능을 보였습니다. 그리고 아래와 같이 class별로 정확도를 확일하였을 때도 supervised method와 성능의 차이가 크게 없었습니다.

![img](https://i.imgur.com/eT9OTTr.png)

#### correlation object classification and rotation prediction

object classification의 성능을 다른 methods과 직관적으로 비교하기 위해 아래의 그림과 같이 상관성을 확인하였습니다.

![img](https://i.imgur.com/rK1wZxB.png)

위의 그림은 object recognition을 epoch가 증가함에 따라 성능을 보여줍니다. 빨간색 선은 supervised model이며, 노란색은 train image의 top feature map을 학습한 object recognition model입니다. 파란색은 논문에서 제안하는 rotation prediction입니다. 그림처럼 epoch가 증가함에따라 노란색선과 파란색선이 같은 값으로 수렴하는 것을 볼 수 있습니다. 그리고 supervised method인 빨간색 선과 차이가 별로 없는 것을 확인할 수 있습니다.

#### semi-supervised setting

위의 성능을 참고하여, train image의 top feature map을 label하여 학습시키고 나머지는 unlabel로 학습시키는 semi-supervised도 수행하였습니다. 결과는 아래의 그림과 같습니다.

![img](https://i.imgur.com/rryUdfB.png)

training example의 개수가 1000개 이하에서는 오히려 supervised보다 더 좋은 성능을 보였습니다. 이는 작은 training example을 사용하여 높은 성능을 보일 수 있음을 나타냅니다.

---

CIFAR 데이터를 통해 성능을 평가하였으며, 이후에는 ImageNet과 Places를 통해 성능을 평가하였습니다. 성능 평가는 위와 같은 절차로 진행한 것이 아니라 다른 methods과 비교하는 정도만 보였습니다. 그리고 기존의 methods과 비교하였을때, 당연히 높은 성능을 보였습니다.

---

참고
1. [https://arxiv.org/pdf/1803.07728.pdf][paper]

---

[paper]: https://arxiv.org/pdf/1803.07728.pdf
