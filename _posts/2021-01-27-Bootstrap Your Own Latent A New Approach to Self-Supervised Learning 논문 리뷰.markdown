---
layout: post
title: "Bootstrap Your Own Latent A New Approach to Self-Supervised Learning 논문 리뷰"
description: "Contrastive 방법들의 negative pair를 사용하지 않고 2가지 network를 활용한 BYOL 논문 리뷰"
date: 2021-01-27 21:03:36 +0530
categories: Self-Supervised-Learning
mathjax : true
---
---

이번 포스트는 [Bootstrap Your Own Latent A New Approach to Self-Supervised Learning][paper]를 리뷰하겠습니다. 최근의 SSL(Self-Supervised Learning)의 방법들은 아래와 같이 contrastive 방법을 많이 활용하였습니다.

1. 예시로, 강아지 image 1개와 고양이 image 1개가 들어간 2 batch size가 있습니다.
1. 강아지와 고양이 모두 서로 다른 augmentation을 적용하여 2개씩 views(총 4개 views)를 생성합니다.
1. 강아지의 2개의 views는 서로 positive pair가 됩니다. 고양이도 마찬가지입니다.
1. positive pair는 feature representation을 similar하게 학습합니다. 고양이도 마찬가지입니다.
1. 강아지의 1개의 view와 고양이의 1개의 view는 negative pair가 됩니다.
1. negative pair는 feature representation을 dissimilar하게 학습합니다.

하지만, 이번 논문에서는 negative pair를 사용하지 않고 2개의 neural network를 사용함으로써 기존의 방법들보다 더 좋은 성능을 보였습니다.

---

# BYOL

기존의 SSL논문들은 성능이 data augmentation의 선택에 따라 갈렸습니다. 그리고 많은 negative pairs를 dissimilar하게 학습해야하므로, 큰 batch size가 필요했습니다. 하지만, **BYOL(Bootstrap Your Own Latent)** 은 negative paris에 의존하고 bootstrap 방법을 사용하여 image agumentation에서 robust하고 높은 성능을 발휘하였습니다.

기존의 bootstrap 방법들은 pseudo-labels, cluster indices, handful of labels 등으로 존재하였습니다. 하지만, BYOL는 서로 상호작용하고 학습하는 **online, target networks라는 2개의 neural network를 사용하는 방법** 인 **directly bootstrap the representation** 을 제안하였습니다.

# Method

기존의 SSL논문들은 같은 image로 부터 다른 views를 생성하고 예측하여 representation을 학습하였습니다. 하지만, 이러한 방법들은 **collapsed representation** 을 이끌 수 있습니다. 예를 들면, 강아지 image에서 view를 생성하고 고양이 image에서 view를 생성했을때, negative pair처럼 서로 다른 images의 views 사이의 dissimilar를 학습함으로써 collapsed representation을 이끌 수 있다는 것입니다. 그리고 이러한 기존의 방법들은 **많은** negative pair와 비교를 해야합니다.

이러한 문제의 해결책으로 다음과 같은 실험을 진행하였습니다. 이부분은 [hoya012님의 설명][hoya]이 잘되어 있어서 참고하였습니다.

![BYOL's core motivation](https://i.imgur.com/1q24v1k.jpg "BYOL's core motivation")

+ Step 1
  * A라는 network를 random initialization 수행 후 weight를 freeze
  * network의 뒤에 linear evaluation protocol를 붙임
  * ImageNet dataset을 사용하여 정확도를 측정

Step 1은 feature extractor가 아무런 정보가 없기 때문에 좋은 성능이 아닌, 1.4%의 top-1 accuracy를 달성하였습니다.

+ Step 2
  * random initialized A network + MLP
  * unlabeled dataset을 feed forward시켜서 prediction을 얻음

+ Step 3
  * B라는 network를 random initialization 수행
  * Step 1과 달리 바로 linear evaluation이 아닌 Step 2의 output인 prediction을 target으로 하여 학습
  * linear evaulation 수행

놀랍게도, Step 3는 18.8%의 top-1 accuracy를 달성하였습니다. 이 수치는 낮은 수치지만, random initialization을 하였을때인 1.4%에 비하면 굉장히 큰 폭으로 성능이 증가하였습니다. 이 실험이 바로 BYOL의 core motivation입니다.

## Description of BYOL

BYOL의 목표는 downstream tasks에 사용되는 representation $y_\theta$ 를 학습하는 것입니다. 그리고 위에 설명했다싶이, BYOL는 online network와 target network라는 2개의 neural networks를 사용합니다. 아래의 그림들은 BYOL의 architecture를 설명하는 그림들입니다.

![BYOL's architecture 1](https://i.imgur.com/ehAHT8r.jpg "BYOL's architecture 1")
![BYOL's architecture 2](https://i.imgur.com/0wrIruZ.jpg "BYOL's architecture 2")

**Online network** 는 weights $\theta$ 로 정의됩니다. 그리고 encoder $f_\theta$, projection $g_\theta$, predictor $q_\theta$ 인 3개의 stages로 구성되어 있습니다.

**Target network** 는 online network와 같은 architecture를 가집니다. 다만, 다른 weights $\xi$ 를 사용합니다. 이러한 target network는 online network를 학습하기 위해 **regression targets** 를 제공합니다. 그리고, parameters $\xi$는 online parameters $\theta$의 **exponential moving average** 입니다.

위의 과정을 실험과 비교해서 설명드리자면, 다음과 같습니다.
+ Online network : B network
+ Target network : A network
+ Regression targets : Step 2의 output prediction
+ Weight $\xi$ (target network) : weight $\theta$ (online network) 의 exponential moving average

즉, **target network로 unlabeled data를 feed forward시켜서 prediction을 만들고 online network의 target으로 삼아 학습을 진행시키는 것**입니다.

그리고 BYOL의 전체적인 **mechanism** 을 설명드리자면 다음과 같습니다.
+ images set인 $D$가 주어짐
+ $D$로 부터 골고루 sample하여 image $x$를 추출
+ image augmentation인 $t$ ~ $\tau$와 $t'$ ~ $\tau'$을 적용하여 views인 $v = t(x)$ 와 $v' = t'(x)$를 산출
+ Online network에서 $v$ 를 encoder의 output인 representation $y_\theta = f_\theta(v)$ 을 구하고 projection의 output인  $z_\theta = g_\theta(y)$ 을 구하고 predictor를 수행하여 $q_\theta(z_\theta)$를 산출
+ Target network에서 $v'$ 를 encoder의 output인 $y_\xi' = f_\xi(v')$ 를 구하고 projection $z_\xi' = g_\xi(y')$ 의 output으로 $z_\xi'$ 을 산출
+ $q_\theta(z_\theta)$ 와 $z_\xi'$ 를 $l_2$-normalization 수행하여 아래와 같은 값을 산출

![$l_2$-normalization](https://i.imgur.com/gGPOCAI.jpg "$l_2$-normalization")

+ $\bar{q_\theta}(z_\theta)$ 와 $\bar{z_\xi}'$ 를 아래와 같은 mean squared error loss 를 수행하여  $L_{\theta, \xi}$ 산출

![MSE loss](https://i.imgur.com/6BJWJbF.jpg "MSE loss")

+ Loss 대칭화를 위해 위의 과정을 반대로, $v'$ 는 online network에 feeding하고, $v$ 는 target network에 feeding함으로써 $\tilde{L_{\theta, \xi}}$ 를 산출
+ $L_{\theta, \xi}^{BYOL} = L_{\theta, \xi} + \tilde{L_{\theta, \xi}}$ 로 산출
+ $\xi$ 가 아니라 $\theta$ 에 대하여 $L_{\theta, \xi}^{BYOL}$ 를 stop-gradient로써 minimize 수행, $\eta$ 는 learning rate

![Optimization and weight update](https://i.imgur.com/gy95Fg0.jpg "Optimization and weight update")

+ 학습이 끝난 후에, encoder $f_\theta$ 를 keep하고 다른 방법들과 비교

## Implementation details

+ Image augmentations
  * same set of image augmentations as in SimCLR
    - random patch of image and resize to 224x224 with random horizontal flip
    - color distortions
    - Gaussian blur and solarization
  * Architecture
    - ResNet-50(1x) v1
    - also use deeper(50, 101, 152, and 200 layers) and wider(from 1x to 4x) ResNets
    - output of the final average pooling layer, which has feature dimension of 2048
    - projected to a smaller space by MLP (linear layer with output size 4096 followed by batch normalization, ReLU)
    - final linear layer with output dimension 256
  * Optimization
    - LARS optimizer with cosine decay learning rate, without restarts, over 1000 epochs, with a warm-up period of 10 epochs
    - set the base learning rate to 0.2
    - global weight decay parameter of $1.5 * 10^{-6}$
    - exponential moving average parameter $\tau$ starts from 0.996

---

# Experimental evaluation

## Linear evaluation on ImageNet

![Linear evaluation on ImageNet](https://i.imgur.com/zE1f2tP.jpg "Linear evaluation on ImageNet")

표 (a)는 ResNet-50 (x1) encoder를 사용하였을때, 성능을 비교한 것입니다. 기존의 SSL 방법들보다 Top-1과 Top-5에서 높은 성능을 보였습니다. Supervised baseline은 성능이 76.5%, stronger supervised baseline은 성능이 78.9%가 나왔습니다. Supervised baseline보다는 성능이 낮지만, 격차를 많이 줄이는 모습을 보여주었습니다.

표 (b)는 deeper하고 wider한 architecture를 사용하였을때, 다른 방법들과 비교한 것입니다. 이 또한, 기존의 SSL 방법들보다 높은 성능을 보였습니다. 그리고 같은 architecture로 best supervised baseline을 적용하였을때, 성능이 78.9%가 나왔습니다. Deeper하고 wider한 architecture에서도 best supervised baseline과 격차를 많이 줄이는 모습을 보여주었습니다.

## Semi-supervised training on ImageNet

![Semi-supervised training with a fraction of ImageNet lables](https://i.imgur.com/i9zmCYP.jpg "Semi-supervised training with a fraction of ImageNet lables")

ImageNet을 1%와 10%의 labeled ImageNet dataset을 사용하여 성능을 비교하였습니다. 같은 architecture나 deeper하고 wider한 architecture에서 기존의 SSL 방법들보다 높은 성능을 보였습니다.

## Transfer to other classification tasks

![Transfer learning results from ImageNet with ResNet-50](https://i.imgur.com/UdtRHQ4.jpg "Transfer learning results from ImageNet with ResNet-50")

ImageNet으로 pretrain시키고 feature freeze를 시킨 후, 다양한 dataset에 linear evaluation과 fine-tuned 시켰습니다. 그리고 동일한 classification task와 동일한 ResNet-50 architecture를 사용하였습니다. 결과는 SimCLR보다 모든 dataset에서 좋은 성능을 발휘하였고, supervised-ImageNet 보다 7개 dataset에서 좋은 성능을 보였습니다. 하지만, 5개의 dataset에서는 낮은 성능을 보였습니다.

## Transfer to other vision tasks

![Results of other vision tasks](https://i.imgur.com/yLP4VE1.jpg "Results of other vision tasks")

첫번째로, ImageNet으로 pretrain시키고 VOC2012 dataset에 segmentation task를 수행하였습니다. 결과는 supervised-ImageNet을 포함하여 기존의 SSL 방법들보다 더 높은 성능을 보였습니다 (표 (a)의 mIoU).

두번째로, ImageNet을 Faster R-CNN으로 pretrain시키고 trainval2007에 fine-tune시켰습니다. 결과는 supervised-ImageNet을 포함하여 기존의 SSL 방법들보다 더 높은 성능을 보였습니다. (표 (b)의 $AP_{50}$)

세번째로, single RGB image가 주어졌을때 scene의 depth map을 잘 추정하는지 확인하기 위해 NYU v2 dataset의 depth 추정치를 평가하였습니다. Depth prediction은 network가 geometry information을 잘 표현하고 information이 pixel accuracy를 얼마나 잘 localize 시키는 것인지 확인하는 것입니다.

평가를 하기 위해 test dataset의 654 images를 사용하였고 모두 동일한 metrics을 사용하였습니다. 표 (b)의 pct는 percent of pixels이고 rms는 root mean squared error이고 rel는 relative error입니다. 표를 보시다 싶이, 모든 부분에서 BYOL가 높은 성능을 보였습니다.

---

# Building intuitions with ablations

이번 논문이 특이하게도, 논문의 순서배열이 ablation study 후에 성능평가가 아닌 성능평가 후에 ablation study를 진행하였습니다.

BYOL는 batch size, image augmentations, bootstrapping 과 같은 3가지에 대해 ablation study를 진행하였습니다. 기존의 논문들은 ablation study를 100 epochs로 하였지만, BYOL는 300 epochs로 수행하였습니다. 그리고 1000 epochs 수행한 BYOL baseline과 비교하였습니다. Ablastion study를 수행할때 아래와 같은 setting으로 진행하였습니다.

* initial learning rate 0.3 with batch size 4096
* weight decay to $10^{-6}$
* base target decay rate $\tau_{base} = 0.99$
* report top-1 accuracy on ImageNet under the linear evaluation protocol

## Batch size

기존의 contrastive 방법들은 batch size가 줄어들면 negative pair를 적게 학습하기 때문에, 성능 저하가 발생하였습니다. 하지만, BYOL는 negative pair를 사용하지 않기 때문에, **batch size가 작아도 좋은 성능을 발휘합니다**. 이를 검증하기 위해, 128 ~ 4096 batch size를 사용하여 SimCLR과 비교하였습니다. 그 결과 아래와 같은 표를 도출하였습니다.

![Impact of batch size](https://i.imgur.com/q1X4msA.jpg "Impact of batch size")

그래프를 보시면, batch size가 작아짐에 따라 SimCLR에 비해 성능이 덜 저조한 것을 확인할 수 있습니다. 또한, 모든 batch size에서 SimCLR보다 높은 성능을 보였습니다.

## Image augmentations

Contrastive 방법들은 image augmentations의 영향을 많이 받습니다. 하지만, BYOL는 **image augmentation의 몇가지를 제거하여도 좋은 성능을 보였습니다**. 이를 확인하기 위해, SimCLR의 image augmentations중에 ablations하면서 성능을 평가하였고 결과는 아래의 그래프와 같습니다.

![Impact of progressively removing transformations](https://i.imgur.com/RgvIy9F.jpg "Impact of progressively removing transformations")

그래프를 보시면, image augmentations 중에 하나를 제거하거나 한개만 사용하여도 SimCLR비해 덜 저조한 것을 확인할 수 있습니다. 또한, 모든 부분에서 SimCLR보다 높은 성능을 보였습니다.

## Bootstrapping

BYOL은 online network의 weights의 exponential moving average인 projected representation을 target network의 예측 대상으로 사용합니다. 그래서, target decay rate에 영향을 많이 받습니다.

target decay rate가 1이면, target network는 업데이트를 하지 않습니다. 반면, target decay rate가 0이면 target network가 아주 느리게 업데이트를 진행합니다. 그래서, 적절한 target decay rate를 설정하는 것이 중요합니다. Target decay rate와 관련된 실험 결과는 아래의 표와 같습니다.

![Results for different target modes](https://i.imgur.com/8MNAxqL.jpg "Results for different target modes")

위의 표를 보면, $\tau=0.99$  일때가 성능이 제일 좋았으며 **적절한 target decay rate를 설정의 중요성을 확인할 수 있습니다**.

## Ablation to contrastive methods

BYOL와 SimCLR을 inforNCE objective로 표현하여 두 알고리즘의 근본적인 차이를 분석하였습니다. SimCLR은 predictor가 없고 target network가 없습니다. 반면, BYOL는 negative example이 없습니다. 아래의 표는 실험 결과이며, BYOL에 negative example을 주고 학습을 시켰을 때는 오히려 성능이 떨어지지만 적절한 tunning을 하면 사용하지 않았을때와 비슷한 성능을 보였습니다. **굳이, negative pair를 추가하여 학습할 필요가 없는 것입니다**. 이때, $\beta$는 weighting coefficient입니다.

![Intermediate variants between BYOL and SimCLR](https://i.imgur.com/rNNnAMA.jpg "Intermediate variants between BYOL and SimCLR")

---

# Conclusion

최근 SSL논문들은 contrastive 방법들을 활용하여 점점 성능을 높여왔습니다. 하지만, BYOL는 contrastive 방법과 negative pair를 사용하지 않고 기존의 contrastive 방법들보다 더 높은 성능을 보여주었습니다. 그래도, 저자는 아직까지는 image augmentation에 많이 의존하는게 남아있다고 하였습니다. 아마 앞으로의 SSL의 방법들은 점점 image augmentation에 의존하지 않은 방법을 점점 연구될 것 같다고 생각을 할 수 있었습니다.

---

참고
1. [https://arxiv.org/pdf/2006.07733.pdf][paper]
1. [https://hoya012.github.io/blog/byol/][hoya]

---

[paper]: https://arxiv.org/pdf/2006.07733.pdf
[hoya]: https://hoya012.github.io/blog/byol/
