---
layout: post
title: "Unsupervised Learning of Visual Features by Contrasting Cluster Assignmnets 논문 리뷰"
description: "contrastive learning 기반의 접근법이 아닌, clustering 기반으로 연구한 SwAV 논문 리뷰"
date: 2021-06-29 21:03:36 +0530
categories: Self-Supervised-Learning
mathjax : true
---
---

# SwAV

대학원 석사2기가 끝나고 다시 논문 리뷰를 진행하도록 하겠습니다. 최근 self-supervised learning은 contrastive learning을 기반으로 연구가 계속 진행되면서, supervised learning와의 performance를 점점 좁혀가고 있습니다. 이번 포스트는 contrastive learning과 clustering 방법을 활용한 [Unsupervised Learning of Visual Features by Contrasting Cluster Assignmnets][paper]인 **SwAV**를 리뷰하겠습니다.

---

기존의 contrastive learning은 image가 input될때, augmentation methods로 views를 생성하고 views끼리 비교하는 방식으로 진행되었습니다. 그리고, views끼리 비교하기 위해 target을 하나의 view로 설정하고 학습하고, 계속 target이 변하면서 학습하기 떄문에 online으로 학습이 진행됩니다. 하지만, 이러한 방법은 큰 batch size로 학습하여 많은 views끼리 비교를 수행해야지 좋은 representation을 추출할 수 있어서 실용적이지는 않습니다.

그래서, 저자는 문제점을 해결하기 위해, contrastive learning 접근법이 아닌 clustering 기법으로 접근하였습니다. 저자가 생각하는 clustering 기법은 개별 images 대신 similar한 feature를 가지는 images를 group으로 할당하고 구별하는 clustering 기법을 생각하였습니다. 이를 통해, **Sw**apping **A**ssignmnets between multiple **V**iews of the same image (SwAV)라는 방법을 제안하였습니다. SwAV는 크고 작은 batch size로 작동하고 대용량의 memory bank와 MoCo와 같이 momentum encoder가 필요없습니다.

게다가, 기존의 contrastive learning에서는 augmentation으로 2개의 views를 생성하였는데, SwAV에서는 한장의 image에 다양한 size의 multi-crop들을 생성하여 views의 수를 늘렸습니다. 이는, 기존의 contrastive learning에서는 resize를 통해 image를 축소하였는데, resize된 image로 학습하면 bias가 생기므로, 다양한 size의 crop된 views를 학습하여 performance를 크게 향상되는 것을 실험을 통해 확인하였습니다.

---

# Method

일반적인 clustering 기법은 전체 image dataset의 image features를 clustering하고 cluster code(clustering numbering)을 부여하는 방식인 offline 방식입니다. 이러한 방법은 학습을 진행할때 feature를 업데이트하기 위해, 전체 image dataset을 반복적으로 input하기 때문에 target이 계속 변하는 online 학습에는 시간적 문제때문에 실용적이지 않습니다. 그래서, 논문에서는 cluster code 자체를 target으로 간주하지 않고 image로부터 생성된 augmented views로 부터 cluster code를 할당하고 동일한 image로 부터 생성된 다른 augmented views로부터 cluster code를 예측하는 방법을 제안하였습니다.

즉, 같은 image로부터 2개의 다른 augmentation view features인 $z_t$와 $z_s$가 주어졌을때, K prototypes $\{ c_1, ..., c_K\}$ set에 일치시켜 codes $q_t$와 $q_s$를 계산합니다. 이후, 아래와 같은 loss function을 사용하여 "swapped" prediction problem을 제안합니다.

![loss function](https://i.imgur.com/R5xZORN.jpg "loss function")

$l(z,q)$ fucntion은 features인 $z$와 code인 $q$ 사이의 fit을 측정합니다. 자세한 detail은 뒤에서 언급하겠습니다. 이 방법을 통해서 code인 $q_t$와 $q_S$를 사용하여, feature인 $z_t$와 $z_s$를 비교합니다. 이 2개의 feature $z_t$와 $z_s$가 동일한 information을 가지는 경우는 다른 feature인 $z_s$에서 code인 $q_t$를 예측할 수 있어야 합니다. 이를 통해, feature를 직접 비교하는 contrastive learning에서도 유사한 비교가 나타납니다. 아래의 그림을 통해서 contrastive learning과 제안하는 method 사이의 관계를 보여줍니다.

![img](https://i.imgur.com/a2XND5k.jpg)

## Online clustering

image $x_n$이 input될때, data augmentation set인 $\mathcal{T}$ 로부터 $t$ sample augmentation을 적용하여 augmented view $x_nt$를 생성합니다. augmented view인 $x_nt$는 non-linear mapping인 encoder $f_\theta$를 통과하여 vector representation인 $z_nt$를 생성합니다. 즉, $z_nt = f_{\theta}(x_nt)/{\lVert f_{\theta}(x_nt) \rVert}_2$ projection되어서 산출되게 됩니다. 그리고, $z_nt$를 훈련 가능한 K prototypes vector set $\{ c_1, ..., c_K\}$에 mapping하여 feature에서 code $q_nt$를 계산합니다. 이때, column이 $c_1, ..., c_k$인 행렬을 $C$로 표시합니다. 이후, 이러한 진행을 거처서 prototypes을 online으로 update하는 방법을 설명하겠습니다.

### Swapped perdiction problem

Method에서 언급드린 loss function에 대해서 자세히 알아보겠습니다. loss function에서 각각의 term은 $z_i$의 내적과 $C$의 모든 prototypes의 softmax를 취하여 얻은 확률과 code간의 cross entropy loss를 나타냅니다. 즉, 아래의 그림과 같이 feature인 $z_t$를 내적한 후, 모든 prototypes인 $c_k$와 softmax를 취하여 얻은 값인 $p_t$와 code인 $q_s$의 cross entropy loss로 산출됩니다.

![loss function term](https://i.imgur.com/PerofkO.jpg "loss function term")

여기서, $\tau$는 temperature parameter입니다. 이 loss를 모든 image와 data augmentation pair에 적용하면 swap된 prediction problem에 대해 다음과 같은 loss function이 발생합니다.

![loss function](https://i.imgur.com/A5DD6Ag.jpg "loss function")

이 loss function은 prototypes $C$와 feature $z_{nt}$ 를 생성하는데 사용되는 image encoder $f_{\theta}$ 의 parameter $\theta$와 관련하여 공동으로 최소화됩니다.

### Computing codes online

제안하는 method를 target이 계속 바뀌는 online으로 만들기 위해 전체 image feature가 아닌, input으로 들어오는 batch 내의 image feature만을 사용하여 code를 계산해야합니다. 따라서, prototypes $C$가 input되는 서로 다른 batch에서 공유되어 batch 내의 sample에게 code를 부여합니다. 이때, 모든 image가 같은 code로 mapping되는 trivial solution을 방지하기 위해 batch안의 서로 다른 image들이 prototypes $C$에 의해 서로 다른 code로 균등하게 분배하도록 합니다.

즉, B개의 feature vectors가 담긴 $z = [z_1, ..., z_B]$ 와 prototypes $C = [c_1, ..., c_K]$가 주어졌을때, code인 $Q = [q_1, ..., q_B]$ 를 optimization하여 feature와 prototypes 간의 similarity를 아래와 같이 최대화합니다.

![similarity](https://i.imgur.com/3kz89yx.jpg)

여기서, $H(Q) = -\sum_{ij} Q_{ij} logQ_{ij}$는 entropy function이고 $\varepsilon$ 은 mapping의 somoothness를 control하는 parameter입니다. 강력한 entropy regularization(높은 $\varepsilon$)을 사용하면 일반적으로 모든 image가 unique representation으로 축소되고 모든 prototypes에 균일하게 code가 할당되는 trivial solution으로 이어질 수 있기 때문에, $\varepsilon$을 적당히 유지하는 것이 중요합니다.

또한, code vector(B dimension)인 $Q$를 prototypes vector(K dimension)인 $C$에 optimization하는 method인 optimal transport를 이용할 수 있도록 아래와 같은 각 행과 열의 합이 일정하도록 제약조건을 부여하였습니다. 이는 전체 dataset에 대해 작업하고, mini-batch로 제한하여 mini-batch 작업에 대한 solution을 제안하였습니다.

![img](https://i.imgur.com/OJVnugn.jpg)

여기서, $1_K$는 prototype 차원인 K차원에 있는 1 vector를 나타냅니다. 위의 제약으로 인해 평균적으로 각 prototypes이 batch에서 최소 B(feature vectors dimension)/K(prototype vector dimension)번 선택됩니다. 그리고, optimal transport를 **Sinkhorn-Knopp algorithm**을 사용하여 최적의 $Q^{*}$를 찾습니다. Sinkhorn-Knopp algorithm은 아래의 식과 같이 적용되며, 3회 반복만을 통해서 좋은 결과를 산출 할 수 있었습니다.

![img](https://i.imgur.com/pmTCZS0.jpg)

이때, $u$와 $v$는 $R^K$와 $R^B$의 renormalization vectors입니다. 그리고, 최적의 $Q^{*}$는 discrete code가 continous code보다 성능이 떨어져서 continous code를 사용하였습니다.

### Working with small batches

batch feature 수인 $B$가 prototypes 수인 $K$에 비해 너무 작으면 batch를 prototype으로 균등하게 분할하는 것이 불가능했습니다. 따라서, 작은 batch로 작업하여 직전의 batch features를 사용하여 similarity를 구하는 식에서 $Z$의 크기를 늘렸습니다. 그런 다음 train loss에서 batch feature의 code만 사용하였습니다.

## Multi-crop: Augmenting views with smaller images

이전에는 augmentation을 random crop 등을 사용하였는데, 이러한 방법은 crop image를 저장하기 위한 memory 부담과 computing power적인 문제가 있었습니다. 이러한 문제를 해결하기 위해, 논문에서는 2개의 일반적인 random crop을 수행하고, $V$개의 낮은 해상도와 image의 작은 부분만을 cover하는 crop을 생성하였습니다. 이를 통해, loss를 아래와 같이 일반화합니다.

![img](https://i.imgur.com/kDL72l5.jpg)

이때, V+2개의 crop에 대해서 모두 code를 계산하는 것이 아니라 일반적인 2개의 random crop에 대해서만 code를 계산하고 V개의 저해상도 crop은 위의 식과 같이 code를 예측하는 feature로만 사용합니다. 그리고, multi-crop을 사용한 것과 사용하지 않은 것을 비교한 표가 아래에 있으며, supervised를 제외하고 더 좋은 성능을 보였습니다.

![img](https://i.imgur.com/6Q7cUKa.jpg)

---

# Results

여러 dataset에 대한 transfer learning을 통해 SwAV에서 학습한 feature를 분석하였습니다. 그리고, SimCLR에서 사용된 improvements인 LARS optimization과 cosine learning rate와 MLP projection head로 구성하여 학습을 진행하였습니다.

## Evaluating the unsupervised features on ImageNet

ImageNet에서 SwAV로 훈련된 ResNet-50의 feature로 2가지 실험을 진행하였습니다. 첫번째는 fixed feature에 대한 linear classification, 두번째는, semi-supervised learning입니다. 아래의 그림의 왼같이 features를 frozen하였을때, SwAV는 기존의 method인 MoCov2보다 정확도가 4%만큼 증가하였으며, supervised learning에 비해서는 2%밖에 차이가 나지 않았습니다.

![img](https://i.imgur.com/xLnPiBo.jpg)

큰 batch size인 4096으로 800 epoch동안 SwAV를 훈련하였습니다. 훈련이 짧은 결과는 아래의 그림과 같습니다. epochs가 증가함에 따라, 성능이 좋아지긴 하지만 학습에 걸리는 시간이 많은 것을 확인할 수 있습니다.

![img](https://i.imgur.com/rTv5fkJ.jpg)

큰 batch size가 아닌, 작은 batch size로 학습한 결과는 아래와 같습니다. batch size를 작게 하더라도 SimCLR, MoCov2보다 좋은 성능을 보이는 것을 확인할 수 있습니다.

![img](https://i.imgur.com/Vc3FAs0.jpg)

semi-supervised learning에서는 SwAV를 semi-supervised learning을 위해 특별히 설계되지 않았음에도 불구하고 아래와 같이 최신 semi-supervised learning과 동등한 성능을 보입니다.

![img](https://i.imgur.com/bx1aaE3.jpg)

### Variants of ResNet-50

ResNet-50의 width를 여러가지 변형하여, 성능을 평가하기도 하였습니다. 결과는 아래의 그림과 같으며, width가 작은 것부터 큰 것까지 모두 기존의 SSL methods보다 좋은 성능을 보이는 것을 확인할 수 있었습니다. 그리고, SimCLR에 비해 supervised learning의 차이를 훨씬 줄이는 것을 확인할 수 있습니다.

![img](https://i.imgur.com/XEHDWpO.jpg)

## Transferring unsupervised features to downstream tasks

label이 없는 ImageNet으로 학습하여 여러가지 dataset으로 downstream vision task를 수행하였습니다. 아래의 표는 SwAV feature 성능을 ImageNet supervised learning과 비교한 결과입니다.

![img](https://i.imgur.com/DIWYMvC.jpg)

먼저, Places205, VoC07, iNat18 dataset의 linear clssification 확인하였을때, 모든 dataset에서 supervised learning보다 높은 성능을 달성하였습니다. 그리고, object detection으로 VOC07+12 dataset으로 R50-C4 backbone의 Faster R-CNN으로 object detection을 수행하고 COCO dataset으로 R50-FPN backbone의 Mask R-CNN으로 object detection을 수행하고, COCO dataset으로 object detection의 최신 기법인 DETR를 적용한 결과를 비교하였을때, SwAV가 모두 높은 성능을 보이는 것을 확인할 수 있습니다.

## Unsupervised pretraining on a large uncurated dataset

ImageNet과 다른 속성을 가진 non-EU instagram 10억개의 image로 SwAV를 평가하여 online clustering과 multi-crop augmentation이 제대로 작동하는지 실험하였습니다. 아래의 그림이 실험결과입니다. ImageNet이 아닌 다른 intagram image로 학습을 진행하여도 성능이 기존 SSL methods보다 높은 성능을 보이는 것을 확인할 수 있습니다.

![img](https://i.imgur.com/mUzorgR.jpg)

그리고, model capacity별로 실험을 진행하여 instagram image로 학습한 SwAV의 성능을 평가하였습니다. 실험의 결과는 아래의 그림과 같습니다. 실험을 통해서, model의 크기가 증가할수록 성능이 증가하는 것을 확인할 수 있었습니다.

![img](https://i.imgur.com/HXRrLTI.jpg)

---

# Discussion

현재 설정이 supervised learning을 위해 설계 되었음에도 불구하고 self-supervised learning은 self-supervised learning에 비해 빠르게 진행되고 있으며 transfer learning을 능가합니다. 특히, architecture는 supervision되는 task를 위해 설계되었으며, supervision 없이 architecture를 탐색할 때 동일한 model이 나올지는 분명하지 않습니다. 최근 여러 연구에서 search나 pruning으로 architecture를 탐색하는 것이 supervision없이 가능하다는 것을 보여주었습니다. 이를 기반으로 model 탐색을 안내하는 방법의 능력을 평가할 계획입니다.

---

현재 self-supervised learning은 contrastive learning을 기반으로 많은 연구가 진행되고 있습니다. 하지만, contrastive learning은 negative pair를 잘 설정해야하는 조건이 존재하며, SwAV는 이러한 조건을 해결하고 기존의 contrastive learning기반이 아닌, clustering method기반으로 model을 구축하였습니다. 그리고, 성능 또한, 기존의 methods보다 높았으며, supervised learning보다 높은 부분도 많이 존재하였습니다. 이제 슬슬, contrastive learning에서 벗어나는 접근법으로 self-supervised learning 연구가 지속적으로 나올 것으로 생각됩니다.

---

참고
1. [https://arxiv.org/pdf/2006.09882.pdf][paper]

---

[paper]: https://arxiv.org/pdf/2006.09882.pdf
