---
layout: post
title: "Momentum Contrast for Unsupervised Visual Representation Learning 논문 리뷰"
description: "이전의 방법들과는 다르게 contrastive learning을 활용한 MoCo 논문 리뷰"
date: 2021-01-07 21:03:36 +0530
categories: Self-Supervised-Learning
mathjax : true
---
---

정신없는 대학원 석사1기가 종강하고 방학이 되서야 숨통을 트이게 되었습니다. 블로그 포스팅을 이어서 진행하도록 하겠습니다.

---

## MoCo

Unsupervised Learning은 NLP에서는 GPT와 BERT를 통해 매우 성공적인 성능을 보였습니다. 하지만, CV에서는 여전히 Supervised Learning이 Unsupervised Learning보다 더 좋은 성능을 발휘합니다. 이 논문에서는 뒤쳐지는 이유를 **각자의 signal space에서의 차이때문이라고 추측하였습니다.**

이유에 대해서 조금 더 자세히 말씀드리면, NLP에서는 Unsupervised Learning의 기반이 될 수 있는 토큰화된 dictionaries를 구축하기 위해 word, sub-word units와 같이 **discrete signal space**를 가집니다. 하지만, CV에서는 raw signal이 **continuous signal space**와 고차원에 있기 때문에 dictionaries을 구축합니다.

최근의 연구 주제를 살펴보면, **contrastive loss** 와 관련된 approach를 사용하여 Unsupervised Learning을 수행합니다. Contrastive loss approach의 논문들을 살펴보면, **dynamic dictionaries** 를 구축하는 것을 볼 수 있습니다. 여기서 dictionary의 "**keys**"는 image와 patches들과 같은 데이터로부터 샘플링되고 encoder network로 표시됩니다. 그리고 인코딩된 "**query**"가 매칭되는 key와 비슷해야하고 다른 key와는 유사하지 않게 훈련을 시킵니다. 이때, contrastive loss로 최소화를 진행합니다.

이 방법을 활용하여 이 논문은 학습중에 진화함에 따라 **크고 일관성있는** dictionary를 구축하였습니다. 여기서 **큰** dictionary는 고차원적인 continous signal space를 더 잘 샘플링할 수 있을 것 같아서 크게 구축하였습니다. 그리고 dictionary의 key는 query와 비교가 **일관되도록** 동일하거나 유사한 인코더로 표시하도록 구축하였습니다. 이렇게 contrastive loss를 사용하고 크고 일관성있는 dictionaries를 구축하는 방법을 Momentum Contrast(MoCo)라고 표현하였고 아래의 그림과 같습니다.

![MoCo Mechanism](https://i.imgur.com/NRAavjZ.jpg "MoCo Mechanism")

그림의 오른쪽에 있는 $ k_0, k_1, k_2, ... $ 는 데이터 샘플들 set인 $ x_0^{key}, x_1^{key}, x_2^{key}, ... $ 을 **momentum 기반의 moving average encoder** 에 적용한 것입니다. 이때, 기존의 미니 배치의 encode된 representations이 queue에 포함되고 가장 오래된 항목이 queue에서 제외되는 방식으로 미니 배치 크기에서 분리되는 queue로 구축합니다. 이를 통해 queue는 미니 배치 크기에서 dictionary 크기를 분리하여 **더 커질 수 있습니다.** encoder로써 momentum 기반의 moving average encoder로 적용된 이유는 **dictionary keys가 일관성을 유지**하기 위해서 key encoder로 적용되었습니다.

그림의 왼쪽에 있는 encode된 query q를 그림의 오른쪽에 있는 encode된 dictionary keys와 일치시키고 contrastive loss를 사용하여 visual representation ecoder를 학습하는 것이 바로 MoCo입니다. 이러한 방법을 통해 **MoCo는 dynamic dictionaries를 구축할 수 있고 다양한 pretext task를 사용할 수 있습니다.**

---

## Related Work

### Loss Functions

#### General loss
loss functions을 정의하는 일반적은 방법은 model의 예측과 fixed target의 차이를 측정하는 것입니다. 예를 들면, L1이나 L2 loss로 input 픽셀을 재구성하거나 cross-entropy나 margin-based loss로 미리 정의된 categories로 분류하는 것들이 있습니다.

#### Contrastive loss
Contrastive loss는 representation space에서 샘플 pair의 유사성을 측정하는 loss입니다. fixed target과 input을 매칭하는 것 대신, 대상이 학습 중에 그때마다 달라질 수 있으며 network에서 계산된 데이터 representation에서 정의할 수 있습니다.

#### Adversarial loss
Adversarial loss는 확률분포사이의 차이를 측정합니다. 이 loss는 unsupervised data를 생성하는 기술에 많이 쓰입니다. GAN과 NCE와 관련이 있습니다.

### Pretext Tasks

pretext task는 많이 제안되었습니다. 예를 들면 변형이 가해진 input을 복원하는 pretext task인 denoising auto-encoders, context auto-encoders, cross-channel auto-encoder들이 있습니다. 이외에도 Pseudo-label을 사용한 pretext task는 exemplar, jigsaw, deep clustering들이 있습니다.

### Contrastive Learning vs Pretext Task

다양한 pretext task는 contrastive loss functions 형식 기반입니다. 예를 들면, discrimination method는 exemplar와 NCE와 관련이 있습니다. contrastive predictive coding은 context auto-encoding과 관련이 있고 contrastive multiview coding은 colorization과 관련이 있습니다.

---

## Method

### Constrastive Learning as Dictionary Look-up

논문에서 사용하는 loss function은 다음과 같습니다.

* encode된 query **q**
* encode된 samples set {**$ k_0, k_1, k_2, ... $**}
* q와 일치하는 single key **$ k_+ $**
* $ \tau $ 는 temperature hyper-parameter paper
* K negative samples

![InfoNCE](https://i.imgur.com/Dl5O5b8.jpg "InfoNCE")

Contrastive loss는 negative key인 q가 positive key인 $ k_+ $ 와 similar하고 모든 다른 keys들과 dissimilar하지 않을 때 값이 낮은 함수입니다. 이 loss function을 **InforNCE**라고 부릅니다. 즉, q를 $ k_+ $ 로 분류하려고 하는 (K+1)-way softmax-based 분류기의 로그 손실 함수입니다. Contrastive loss는 query와 key를 만들어내는 encoder network를 학습시키기 위한 함수입니다. 일반적으로 query는 $ q = f_q(x^q) $ 로, key는 $ k = f_k(x^k) $ 로 만들어지는데, 각각의 encoder $ f_q, f_k $ 는 방법에 따라 동일하거나 일부 요소만 공유하거나 완전히 다를 수 있습니다. input되는 $ x^q, x^k $ 또한 마찬가지로 pretext task에 따라 image, paches 등이 될 수 있습니다.

### Momentum Contrast

MoCo의 핵심은 dictionary를 데이터 샘플의 queue로 유지하는 것입니다. 하지만, queue를 사용하면 dictionary가 커질수는 있지만 back-propagation로 인해 key encoder를 업데이트하기가 어렵니다. 간단한 해결책은 gradient를 무시하고 query encoder $ f_q $에서 key encoder $ f_k $를 복사하는 것입니다. 하지만, 이 방법은 실험에서 좋지 않은 결과를 보였습니다. 좋지 않은 결과가 key representations의 일관성을 감소시키는 변화가 빠르게 일어나는 encoder로 인해 발생한다고 생각했습니다. 이 문제를 해결하기 위해 **momentum update**를 제안하였습니다.

* $ f_k $의 parameter인 $ \theta_k $
* $ f_q $의 parameter인 $ \theta_q $
* $ \theta_k $ 를 업데이트
* $ m \in [0,1) $ 은 momentum coefficient
* $ \theta_q $ 는 back-propagation에 의해 업데이트

![momentum update](https://i.imgur.com/6fqMMYY.jpg "momentum update")

momentum update는 $ \theta_q $ 보다 더 부드럽게 업데이트됩니다. 그결과, k가 변화하는 encoder $ f_k $ 에 의해 만들어 지더라도 변화의 정도가 느리므로 일관성이 유지됩니다. 실험결과로 m이 0.999일때가 0.9보다 성능이 높았습니다.

### Relations to previous mechanisms

MoCo를 기존에 존재하는 2가지 mechanisms과 비교하였습니다. dictionary 크기와 일관성을 중심으로 비교하였습니다.

![relations to previous mechanisms](https://i.imgur.com/DkrUHdt.jpg "relations to previous mechanisms")

#### end-to-end

(a) back-propagation에 의한 end-to-end 업데이트는 natural mechanism입니다. 기존의 미니배치의 샘플을 dictionary로 사용하므로 key가 일관성있게 encoding됩니다. 그러나 dictionary 크기는 GPU 메모리 크기에 의해 제한되는 미니 배치 크기와 결합됩니다. **즉, dictionary의 크기가 미니배치 크기로 제한됩니다.** 만약, dictionary의 크기를 크게 하게되면 optimization에 어려움이 있습니다. 최근에는 dictionary 크기를 local position을 통해 여러 위치로 크게 만드는 방법도 있지만, 이는 pretext task에 따른 network가 필요하기 때문에 downstream task로 transfer하기가 힘듭니다.

#### memory bank

(b) memory bank는 데이터의 모든 샘플의 representations으로 구성되어 있습니다. 각 미니 배치의 dictionary는 back-propagation없이 memory bank에서 무작위로 샘플링되므로 사이즈가 큰 dictionary를 지원할 수 있습니다. 하지만, memory bank의 representation은 각 샘플이 $ f_q $ 에 들어갔을때만 업데이트됩니다. **즉, key가 여러 epoch에 걸친 다른 $ f_k $에 의해 만들어 졌으므로 일관성이 떨어집니다.** 그리고 MoCo처럼 Momentum update를 사용하지만 $ f_k $가 아닌 memory bank안의 representation에 사용합니다.

### Pretext Task

contrastive learning은 다양한 pretext task가 있지만, 논문에서는 [instance discrimination task][paper2], [invariant and spreading instance feature][paper3], [maximizing mutual information across views][paper4]를 사용하였습니다.

**instance discrimiation task**에 따르면, query와 key가 동일한 이미지에서 생성된 경우 양의 쌍으로 간주하고 그렇지 않으면 음의 쌍으로 간주합니다.

**invariant and spreading instance feature**와 **maximizaing mutual information across views**에 따르면, 데이터가 증가함에 따라 동일한 이미지에 대해 두개의 'view'를 가져와 양의 쌍을 형성합니다. query와 key는 $ f_q, f_k $에 의해 encoding됩니다.

![Pseudo code of MoCo](https://i.imgur.com/f5mRtUW.jpg "Pseudo code of MoCo")

위의 알고리즘은 MoCo pretext task의 pseudo-code입니다. 미니배치의 경우 양의 샘플 쌍을 형성하는 query와 해당하는 key를 encoding합니다. negative samples는 queue에서 가져온 것입니다.

#### Technical details

Technical details은 아래와 같은 순서와 같습니다.
1. ResNet encoder
1. Average Polling(128 dimension 고정)
1. Fully-Connected Layer
1. L2-normalized
1. get query or key

이때, $ \tau $는 0.07로 설정하였습니다. 그리고 data augmentation은 이미지들을 무작위로 resize한 후 224x224로 crop하고 color jittering, horizontal flip, grayscale conversion을 수행하였습니다.

#### Shuffling BN

encoder인 $ f_q, f_k $는 Batch Normalization을 가집니다. **하지만, BN을 사용하면 model이 좋은 representation을 학습하지 못하는 것을 발견하였습니다.** 이는 BN으로 인한 샘플간의 배치내 leaks information을 유출하기 때문입니다. 이를 해결하기 위해 GPU에 따라 독립적으로 BN을 적용하였습니다. 추가적으로 $ f_k $의 경우 미니배치를 여러 GPU로 나누기 전에 섞고 encoding한 후 다시 한번 섞었습니다. 이로서 k는 encoding이 되기 전에 batch statistic과 encdoing된 후 batch statistic이 달라져 model이 shortcut으로 이용할 수 없습니다.

---

## Experiments

### Data

1. ImageNet-1M : well-balaced 1000 classes를 가지는 126만장 이미지를 사용하였습니다. 이미지들은 iconic view of objects를 가집니다.
1. Instagram-1B : uncurated, long-tailed, unbalanced된 분포를 가진 10억장의 real-world 데이터 이미지입니다. 이미지들은 iconic object과 scene-level images를 가집니다.

### Training

+ SGD optimizer, weight decay is 0.0001, SGD momentum is 0.9
  - ImageNet-1M
    * mini-batch size of 256
    * 8 GPU
    * initial learning rate is 0.03
    * 200 epoch
    * learning rate multiplied by 0.1 at 120 and 160 epochs
    * 53 hours training ResNet-50
  - Instagram-1B
    * mini-batch size fo 1024
    * 64 GPU
    * learning rate is 0.12 which is exponentially decayed by 0.9x after every 62.5k iterations
    * 1.25M interations
    * 6 days training RestNet-50

### Linear Classification Protocol

MoCo를 검증하기 위해 feature를 frozen하고 linear classification을 통해 검정하였습니다. ImageNet을 먼저 학습시킨 후, feature를 frozen하고 linear classificiation을 학습시켰습니다. 이때, 분류를 진행하기 위해 grid search를 수행하여 initial learning rate는 30, weight decay는 0으로 설정하였습니다. **이 hyper parameter는 supervised learning과 unsupervised learning의 feature distribution이 상당히 다름을 보여줍니다.**

#### Ablation: contrastive loss mechanisms

위에서 설명했던 end-to-end, memory bank, MoCo를 mechanism별로 비교하였습니다. 3가지 모두 같은 pretext task를 수행하였으며, contrastive loss를 중심으로 비교하였습니다.

![comparison of three mechanisms](https://i.imgur.com/wY2c6SI.jpg "comparison of three mechanisms")

3가지 모두 negative sample의 수가 많아지면 성능이 좋아졌습니다. end-to-end는 K가 1024보다 큰 경우에는 dictionary 크기 제한때문에 학습시킬 수가 없었습니다. 비교결과는 MoCo가 높은 정확도를 보였습니다.

#### Ablation: momentum

최적의 momentum을 찾기 위해, K가 4096일때 여러가지의 MoCo momentum 값을 가지는 ResNet-50 정확도를 비교하였습니다.

![comparison of momentum](https://i.imgur.com/fjF1iKE.jpg "comparison of momentum")

이 표를 통해 momentum이 클수록 dictionary가 일관성있게 구축된다는 것을 뒷받침해줍니다.

#### Comparison with previous results

기존의 다양한 model들을 비교하였으며, parameter 개수와 성능을 비교하였습니다.

![comparison with previous results](https://i.imgur.com/iM3Rhrh.jpg "comparison with previous results")

MoCo는 특별한 patched input이나 network를 사용하지 않고 좋은 성능을 보였습니다.

### Transferring Features

**Unsupervised Learning의 큰 목적 중 하나는 다른 데이터로 downstream task를 수행하였을때, transfer를 잘 수행할 수 있는 feature를 학습하는 것입니다.** 이를 확인하기 위해 ImageNet로 pretrained model을 PASCAL VOC, COCO 데이터에 학습하여 성능을 비교하였습니다.

#### Normalization

위에서 설명했듯이, **Unsupervised Learning과 Supervised Learning의 feature distribution은 상당히 다릅니다.** 따라서, fine-tunning(transfer learning)을 수행하는 동안 BN은 freeze하지 않고, downstream에 사용되는 network에도 BN을 사용합니다. fine-tunning을 수행하는 동안 normalization을 수행하므로 supervised setting과 동일한 hyper parameter를 사용합니다.

#### Schedules

많은 시간을 fine-tunning을 수행하면 좋은 성능을 보일 수 있지만, 현실적인 환경을 고려하여 짧은 시간만 fine-tunning을 수행하였습니다. 그 결과, MoCo는 supervised setting으로 학습하여 상당히 좋은 성능을 보입니다.

### PASCAL VOC Object Detection

#### Ablation: backbones

Object Detection인 Faster R-CNN을 활용하고 2가지의 backbone을 사용하여 비교하였고 결과는 아래와 같습니다.

![Object Detection on VOC](https://i.imgur.com/boieCug.jpg "Object Detection on VOC")

#### Ablation: contrastive loss mechanisms

3가지 mechanism을 objective detection에 적용하여 성능을 비교하였고 결과는 아래와 같습니다.

![Comparison of three contrastive loss mechanisms](https://i.imgur.com/Pd4kSSl.jpg "Comparison of three contrastive loss mechanisms")

#### Comparison with previous results

이전의 model들과도 object detection의 성능을 비교하였고 결과는 아래와 같습니다.

![Comparison with previous methods on object detection](https://i.imgur.com/YfAcLfJ.jpg "Comparison with previous methods on object detection")

### COCO Object Detection and Segmentation

Mask R-CNN을 FPN backbone과 C4 backbone으로 구축하여 object detection와 segmentation의 성능을 비교하였고 결과는 아래와 같습니다.

![Object detection and segmentation fine-tuned on COCO](https://i.imgur.com/6sBa8Cj.jpg "Object detection and segmentation fine-tuned on COCO")

---

## Discussion and Conclusion

**MoCo는 대부분의 downstream에서 좋은 성능을 보였습니다.** 그리고 large-scale, uncurated 데이터에서도 좋은 성능을 보였습니다. 더 큰 데이터셋을 사용하면 성능이 좋아졌지만, 크기에 비해 성능은 그렇게 높게 증가하지 않았습니다. 만약, 더 간단한 pretext task가 나온다면 MoCo에 적용할 수 있을 것입니다.

---

참고
1. [https://arxiv.org/pdf/1911.05722.pdf][paper]

---

[paper]: https://arxiv.org/pdf/1807.05520.pdf
[paper2]: https://arxiv.org/abs/1805.01978v1
[paper3]: https://arxiv.org/abs/1904.03436
[paper4]: https://arxiv.org/abs/1906.00910
