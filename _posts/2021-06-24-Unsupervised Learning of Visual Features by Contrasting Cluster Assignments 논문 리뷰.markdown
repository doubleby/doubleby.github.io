---
layout: post
title: "Unsupervised Learning of Visual Features by Contrasting Cluster Assignmnets 논문 리뷰"
description: ""
date: 2021-06-24 21:03:36 +0530
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

이 loss function은 prototypes $C$와 feature $(z_nt)_{n,t}$ 를 생성하는데 사용되는 image encoder $f_{\theta}$ 의 parameter $\theta$와 관련하여 공동으로 최소화됩니다.

### Computing codes online

제안하는 method를 online으로 만들기 위해 전체 image feature가 아닌, input으로 들어오는 batch 내의 image feature만을 사용하여 code를 계산해야합니다. 따라서, prototypes $C$가 input되는 서로 다른 batch에서 공유되어 batch 내의 sample에게 code를 부여합니다. 이때, 모든 image가 같은 code로 mapping되는 trivial solution을 방지하기 위해 batch안의 서로 다른 image들이 prototypes $C$에 의해 서로 다른 code로 균등하게 분배하도록 합니다.

B개의 feature vectors가 담긴 $z = [z_1, ..., z_B]$ 와 prototypes $C = [c_1, ..., c_K]$가 주어졌을때, code인 $Q = [q_1, ..., q_B]$ 를 optimization하여 feature와 prototypes 간의 similarity를 아래와 같이 최대화합니다.

![similarity](https://i.imgur.com/3kz89yx.jpg)

여기서, $H(Q) = -\sum_{ij} Q_{ij} logQ_{ij}$는 entropy function이고 $\varepsilon$ 은 mapping의 somoothness를 control하는 parameter입니다. 강력한 entropy regularization(높은 $\varepsilon$)을 사용하면 일반적으로 모든 image가 unique representation으로 축소되고 모든 prototypes에 균일하게 code가 할당되는 trivial solution으로 이어질 수 있기 때문에, $\varepsilon$을 적당히 유지하는 것이 중요합니다. 또한, $Q$에 대해 optimal transport를 이용할 수 있도록 아래와 같은 각 행과 열의 합이 일정하도록 제약조건을 부여하였습니다. 이는 전체 dataset에 대해 작업하고, 

![img](https://i.imgur.com/OJVnugn.jpg)

---

참고
1. [https://arxiv.org/pdf/2006.09882.pdf][paper]

---

[paper]: https://arxiv.org/pdf/2006.09882.pdf
