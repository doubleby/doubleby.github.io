---
layout: post
title: "Representation Learning by Learning to Count 논문 리뷰"
description: "이전 방법들과는 다르게 input image의 특징 값들을 vector로 추출하여 비교하는 Count 논문 리뷰"
date: 2020-08-06 21:03:36 +0530
categories: Self-Supervised-Learning
---
---

이전 post들은 input image의 변형을 주고 원 상태로 돌아오도록 학습하는 방식이었다면, 이번 post는 input image로 부터 특징 값들을 vector 형태로 추출하여 비교하는 방식을 제안하는 논문인 '[Representation Learning by Learning to Count][paper]' 의 리뷰입니다.

---

## Pretext Task

논문에서 제안하는 pretext task는 아래의 그림과 같이 '**Counting Visual Primitives**' 입니다.

![img](https://i.imgur.com/450y4TP.png)

위의 그림은 image를 4개의 영역으로 나누고 각 영역별로 눈, 코, 발, 머리와 같은 **visual primitives** 의 개수를 구함으로써 **Supervision Signal(feature)** 를 얻게 됩니다. 이때, 그림의 표는 쉽게 이해하기 위해 첨부한 것이지, labeling하는 작업이 아닙니다. 이러한 pretext task는 단순히 image를 자르지 않고, scaling, tiling, counting과 같은 일반적인 관계를 고려하여 수행하였습니다.

### Contribution

논문의 contribution은 다음과 같습니다.
1. labeling없이 representation을 학습하는 새로운 방법을 소개합니다.
1. **위에서 말한 pretext task와 같은 counting을 활용하고 counting된 visual primitives의 관계를 증명합니다.**
1. transfer learning에서 높은 성능을 보임을 증명합니다.

---

## Count

아래의 그림은 논문의 pretext task인 count의 mechanism을 보여줍니다.

![img](https://i.imgur.com/Cl87ASi.png)

1. random한 2개의 x, y image를 select합니다.
1. y image는 D(Downsampling)를 적용하여 축소한 1개의 patch를 얻습니다.
1. x image도 D를 적용하여 축소한 1개의 patch를 얻습니다.
1. 그리고 4개의 영역으로 나누는 T(Tiling)을 적용하여 영역별 patch를 4개 얻습니다.
1. 각 patch들마다 각각의 AlexNet을 통과시켜 weight들을 공유하며 학습합니다.
1. 학습결과로 count된 visual primitives를 vector로 나오게됩니다.
1. 이때, x에서 T를 적용한 4개의 patch들과 x에서 D를 적용한 1개의 patch의 visual primitives vector 값이 같아질때까지 학습합니다. (같이 질때까지 학습하기 위해 l2 loss function사용)
1. 하지만, visual primitives vector가 0이 반환하게 되면 loss를 0으로 쉽게 만들기 때문에, y에서 D를 적용한 patch의 visual primitives vector과 비교하여 다를 수 있게 학습합니다. (대조군을 학습하기 위해 contrastive loss function사용)

### Image Transform

input되는 image가 visual primitives vector를 가지는 과정은 아래와 같습니다.

![img](https://i.imgur.com/WYkka9Q.png)

![img](https://i.imgur.com/JHAAgfp.png)

### L2 Loss Fucntion

x에 T을 적용한 4개의 patch들과 D을 적용한 1개의 patch들을 비교하는 l2 loss function은 아래와 같습니다.

![img](https://i.imgur.com/3iTS52t.png)

### Contrastive Loss Function

x에 T을 적용한 4개의 patch들과 y에 D를 적용한 patch를 비교하는 contrastive loss function은 아래와 같습니다.

![img](https://i.imgur.com/MxV8SfR.png)


---

## Experiments

### PASCAL

ImageNet으로 pretrain시키고 finetuning으로 PASCAL에 적용하였을때 성능을 평가하였고 아래의 표와 같습니다.

![img](https://i.imgur.com/XIKHhJa.png)

classification에서는 우수한 성능을 보였으며, detection과 segmentation에서는 상위권의 성능을 보였습니다.

### ImageNet

ImageNet으로 학습시킬때, layer별로의 linear classifier를 적용하여 classificatoin 성능을 평가하였고 아래의 표와 같습니다.

![img](https://i.imgur.com/2BvG8FF.png)

layer별로 뛰어난 성능을 보였습니다.

### PLACES

Places도 ImageNet와 같은 조건으로 적용하여 성능을 평가하였고 아래의 표와 같습니다.

![img](https://i.imgur.com/j00v5hq.png)

layer별로 최고의 성능을 보였습니다.

---

참고
1. [https://openaccess.thecvf.com/content_ICCV_2017/papers/Noroozi_Representation_Learning_by_ICCV_2017_paper.pdf][paper]

---

[paper]: https://openaccess.thecvf.com/content_ICCV_2017/papers/Noroozi_Representation_Learning_by_ICCV_2017_paper.pdf
