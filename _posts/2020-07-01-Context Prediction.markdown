---
layout: post
title: "Exemplar"
description:
date: 2020-06-29 21:03:36 +0530
categories: Self-Supervised-Learning
---
---

데이터를 학습시킬때, 대부분 성능이 좋은 Supervised Learning으로 학습시킵니다. 하지만, 성능이 좋지만 학습시키기 위해서는 많은 양의 데이터가 필요합니다. 그리고 labeled 데이터를 구하기는 어렵고 비쌉니다. 혹여, 직접 데이터를 labeling하는 작업을 하더라도, 많은 시간과 비용을 필요로 합니다.
이 문제를 해결하기 위해서 Semi-Supervised Learning과 Unsupervised Learning의 연구가 활발히 진행되고 있습니다. 특히, Unsupervised Learning은 쉽고 많이 구할 수 있는 Unlabeled 데이터를 학습시킬 수 있는 학습법입니다. Unsupervised Learning중에서도 **Self-Supervised Learning** 은 AI의 대가 중 한분이신 얀 르쿤 교수님이 집중하고 있는 분야이며, 학계에서도 많은 주목을 받고 있습니다.

이번 포스트는 요즘 핫한 Self-Supervised Learning의 시초가 되었던, '[Discriminative Unsupervised Feature Learning with Exemplar Convolution Neural Networks][paper]' 논문을 리뷰하겠습니다.

---

## Pretext Task

SSL(Self-Supervised Learning)을 이해하기 위해선, 먼저 **Pretext task** 를 이해해야합니다.
**Pretext task** 는 deep network가 학습하는 과정에서 image의 의미있는 정보들을 이해할 수 있도록 사용자가 직접 정의하는 task를 말합니다.

---

## Downstream Task

**Downstream Task** 는 기존의 데이터로 pre-train을 이용해 학습을 진행하고, 다른 task에 fine-tuning하는 방식을 통해 model을 업데이트하고 fitting 하는 것을 말합니다.
즉, Transfer Learning의 접근 방법입니다.

---

## Mechanism

1. 기존의 labeled된 데이터를 label없이 image만을 **pretext** 하여 Self-Supervised Learning
1. network의 weight를 freeze(feature extractor)
1. 기존 데이터의 label과 image 모두를 이용하여 Supervised Learning **or** 다른 데이터를 fine-tuning하여 weight를 미세하게 update하고 fitting <= **downstream**
1. 기존의 Self-Supervised Learning 모델, Supervised Learning 모델과 성능 비교

---

## Exemplar

### Seed Patch

Exemplar는 STL-10, CIFAR-10, Caltech-101의 데이터를 사용하여 성능을 실험했습니다. 그중에서도, STL-10 데이터를 적용하였을때 성능이 Supervised Learning보다 성능이 높았습니다.
좋은 성능을 보였던 STL-10 데이터를 그냥 학습시키는게 아닌, 96 x 96 크기인 STL-10 데이터를 물체를 포함하거나 물체의 일부분을 포함하는 considerable한  gradient가 있는 부분을 32 x 32 크기의 patch로 crop합니다. 이렇게 image로부터 crop한 32 x 32 크기 patch를 **Seed Patch** 라고 합니다.
Seed Patch의 sample은 아래의 그림과 같습니다.

![img](https://i.imgur.com/OpbID2G.png)

image로부터 crop한 seed patch를 아래의 글과 같이, augmentation에 사용하는 transformation들을 적용하여 patch의 개수를 늘려줍니다. 하나의 Seed patch로부터 생성된 patch들을 모두 같은 class로 판독하도록 학습을 시킵니다.

![img](https://i.imgur.com/O5mZUdm.png)

예를 들어,
아래의 그림처럼, 1번째 사슴 seed patch를 통해 여러 개의 patch를 생성한 후, patch들을 같은 사슴 class로 분류할 수 있게 학습시킵니다.
다음 seed patch가 사슴이 아닌 다른 image의 seed patch이면 위의 과정과 똑같이, 여러 개의 patch를 생성한 후에 patch들을 같은 class로 분류할 수 있게 학습시킵니다.

![img](https://i.imgur.com/LNWIrxd.png)

**Exemplar Mechanism**
1. STL-10 데이터를 물체를 포함하거나 물체의 일부분을 포함하는 considerable한 gradient가 있는 부분을 32 x 32 size patch crop
1. crop한 original patch들마다 class를 지정
1. crop한 original patch를 augmentation에 사용하는 transformation들을 적용하여 patch들을 생성
1. 생성된 patch들을 모델에 input하였을때, original patch의 class가 도출하도록 학습하여 성능 평가

### Architecture & Result

논문에서 제안하는 CNN 모델은 **Exemplar-CNN** 이며, 3가지가 있습니다.

- **64c5-64c5-128f** : convolution layer with 64 filters(5 x 5 region) -> 2 x 2 max-pooling -> convolution layers with 64 filters(5 x 5 region) -> 2 x 2 max-pooling -> fully connected layer with 128 units(dropout) -> softmax layers
- **64c5-128c5-256c5-512f** : convolution layer with 64 filters(5 x 5 region) -> 2 x 2 max-pooling -> convolution layer with 128 filters(5 x 5 region) -> 2 x 2 max-pooling -> convolution layer with 256 filters(5 x 5 region) -> fully connected layer with 512 units(dropout) -> softmax layer
- **92c5-256c5-512c5-1024f** : convolution layer with 92 filters(5 x 5 region) -> 2 x 2 max-pooling -> convolution layer with 256 filters(5 x 5 region) -> 2 x 2 max-pooling -> convolution layer with 512 filters(5 x 5 region) -> fully connected layer with 1024 units(dropout) -> softmax layer

논문에서 제안하는 Exemplar-CNN 모델들을 STL-10, CIFAR-10, Caltech-101, Caltech-256 데이터에 적용하였을때, 기존의 모델들과, Supervised 모델과의 비교 결과는 아래의 표와 같습니다.

![img](https://i.imgur.com/vPcNNJw.png)

위에서 말했듯이, STL-10 데이터에 적용하였을때 좋은 성능을 보였습니다. 특히, **64c5-128c5-256c5-512f** 와 **92c5-256c5-512c5-1024f** 모델들이 Supervised Learning보다 좋은 performance를 도출한 것을 알 수 있습니다.

---

## Drawback

ImageNet과 같은 100만장의 엄청난 크기의 데이터를 학습시키기엔 어려움이 있습니다.
100만장의 image 데이터를 각각 seed patch시키고 class를 구분해야 하는데, 많은 parameter 수와 시간이 필요합니다. 초기에 나온 연구이기 때문에 명확한 drawback을 가지고 있지만, Self-Supervised Learning 연구의 시초가 되었기 때문에 큰 의미가 있습니다.
또한, seed patch라는 **pretext task** 를 제안하고 transform한 patch들을 **downsteam** 함으로써, 논문에서 제안하는 Exemplar-CNN 모델을 통해 성능을 측정하고 공개하였기 때문에 큰 의미가 있습니다.

---

참고
1. [https://arxiv.org/pdf/1406.6909.pdf][참고1]
1. [https://hoya012.github.io/blog/Self-Supervised-Learning-Overview/][참고2]
1. [https://seongkyun.github.io/study/2019/11/29/unsupervised/][참고3]
1. [https://hyeonnii.tistory.com/263][참고4]

---

[paper]: https://arxiv.org/pdf/1406.6909.pdf
[참고1]: https://arxiv.org/pdf/1406.6909.pdf
[참고2]: https://hoya012.github.io/blog/Self-Supervised-Learning-Overview/
[참고3]: https://seongkyun.github.io/study/2019/11/29/unsupervised/
[참고4]: https://hyeonnii.tistory.com/263
