---
layout: post
title: "Split-Brain Autoencoders"
description:
date: 2020-07-15 21:03:36 +0530
categories: Self-Supervised-Learning
---
---

전통적인 Unsupervised Learning의 기법 중 하나인 **Autoencoder** 의 단점을 보완하고, 저번 post인 Colorization 의 **Cross-channel Encoders** 의 단점을 보완한 **[Split-Brain Autoencoders][참고1]** 에 대한 논문 리뷰가 있겠습니다.

---

## Autoencoder

Unsupervised Learning의 목표는 label의 사용 없이 data를 modeling하는 것입니다. 또 다른 의미로는 useful한 representation을 추출하는 것입니다. 여기서 **useful** 이란, 다른 task에도 쉽게 adaptable할 수 있는 representation을 의미합니다. Unsupervised Learning은 아래의 그림처럼 전통적인 **autoencoder model** 의 image reconstruction objective와 같은 pretext task를 제안하며 representation을 유도합니다.

![img](https://i.imgur.com/GjAULW9.png)

전통적인 autoencoder가 많은 인기를 받았음에도 불구하고, transfer task에서는 강한 representation을 보여주지는 못했습니다. 그 이유는 model의 **abstraction을 강요** 하는 mechanism 때문일지도 모릅니다. autoencoder는 사소한 identity mapping이 학습되는 것을 방지하기 위해 일반적으로 representation에 **bottleneck을 설계** 합니다. 그러나, bottleneck이 작을수록 강요하는 abstraction가 커지지만 표현할 수 있는 정보의 내용은 작아집니다.

---

## Pretext Task

최근 연구는 network의 architectire안에 있는 bottleneck을 통한 요약으로 abstraction을 강요하는 것 대신에, 학습하는 동안 **input의 일부를 보류** 하는 것을 수행합니다. 예를 들면, 아래와 같이 input의 일부를 drop하는 연구들이 있습니다.

1. denosing autoencoders - > input의 noise를 drop
1. **context encoders** -> input의 random하게 block된 부분을 drop
1. **colorization** -> input의 channel을 drop

### Context Encoders

두번째인 **context encoders** 는 autoencoder보다 좋은 성능을 보였지만, 큰 data set에서는 성능이 저조했습니다. 저조한 다음과 같이 3가지의 이유가 있습니다.
1. random하게 block된 부분의 **inpainting task를 평가하는 것이 매우 어렵기 때문에** , loss function이 적절하지 않을 수도 있습니다.
1. train할때는 block된 image를 학습하지만, test할때는 full image를 학습합니다.(**domain gap 발생**)
1. block된 부분의 inpainting task가 **high-level reason없이 충분히 풀기가 어렵기 때문에**, low or mid-level structure를 사용합니다.

### Colorization

세번째인 **colorization** 는 매우 효과적인 pretext task를 제안함으로써 강한 feature representations를 추출할 수 있었습니다. inpainting과 같은 colorization은 공간적인 task이지만, input과 output 사이를 공간적으로 대응함으로써 다른 loss function에서도 효과적으로 수행할 수 있게 할 정도로 pretext task가 뛰어납니다. 그리고 추상적이 아니라 체계적으로 pretext task를 수행하여 pre-training와 testing의 **domain gap을 제거** 하였습니다.

inpainting은 주로 textural structure에 대한 추론은 가능한 반면, colorization은 school bus를 노란색으로 칠하는 것과 같은 정확한 color를 예측하기 위한 object level의 추론이 더 엄격하게 필요할 수 있으므로 더 강한 repesentation을 유도할 수 있습니다. 이러한 colorization은 data channel들을 직접적으로 예측하는 task인 **corss-channel encoding** 의 하나의 예입니다.

### Cross-Channel Encoders

이 논문에서는 더 나아가 다양한 channel 변환 문제들과 training objectives를 조직적인 평가에 의해 **CCE(Cross-Channel Encoders)** 의 공간을 연구합니다. 그러나 CCE는 input data의 channel들이 똑같이 다루지 않는 **내제적인 handicap** 을 가지고 있습니다. 즉, data의 일부분은 feature extraction을 위해 사용되고 다른 부분들은 예측을 위해 사용됩니다.

### Split-Brain Autoencoders

CCE의 근본적인 방법을 활용하여 상호보완적인 prediction task를 수행하고 전체 input channel의 feature를 extract하기 위해 다음과 같이 autoencoder architecture를 수정한 **Split-Brain Autoencoders** 를 제안하였습니다.

![img](https://i.imgur.com/9XgChJ6.png)

1. network에서 single split을 추가하였습니다.
1. 2개의 분리된 결과
1. 결과들을 concatenate
1. sub-networks

각각의 sub-network는 image channel들을 2개로 나누어 하나는 input으로 활용하여 CCE를 통해 학습하게 되고 나머지 하나는 prediction으로 활용하게 됩니다. 그리고 colorization과 depth prediction과 같은 cross-channel predeiction task를 다양하게 사용하였습니다.

RGB image에서 예를 들면, 하나의 sub-network는 L channel로 부터 Lab colorspace에서 a와 b의 channel을 예측하는 colorization의 문제를 풀 수 있습니다. 그리고 다른 sub-network는 a와 b channel로 부터 L channel을 합성을 수행합니다. RGB-D domain에서 하나의 sub-network는 image로부터 color의 depth를 예측하는 반면, 다른 sub-network는 color depth으로부터 image를 예측합니다. 이러한 architecture의 변화는 각각 CCE에서 같은 abstraction 유도하여 모든 input으로 부터 feature들을 extract하게 합니다.

아래의 표는 SBA와 다른 방법들의 장, 단점을 비교한 표입니다.

![img](https://i.imgur.com/pQ5CRfK.png)

---

## Contribution

논문의 contribution은 다음과 같습니다.
1. 전통적인 autoencoder architecture를 2개의 sub-network(Cross-Channel Encoders)으로 split하는 architecture로 수정하여 **Split-Brain Autoencoders** 를 제안합니다.
1. input되는 image의 channel들 또한 **split** 합니다. 그리고 각 **sub-network** 는 split한 image channel을 예측하도록 학습시킴으로써, input되는 모든 image channel들로부터 feature들을 extract합니다. RGB와 RGB-D domain에서 semantic representation learning의 높은 성능을 증명합니다.
1. 더 좋은 성능을 위해, **cross-cahnnel prediction problem** 과 **loss function** 에 대해 조사하고 CCE의 결합을 위해 여러가지 **aggregation method** 들을 찾았습니다.

---

## Methods

### Cross-Channel encoders

기존의 CCE의 mechanism은 다음과 같이 수행합니다.

![img](https://i.imgur.com/ulu11W1.png)

### Split-Brain Autoencoders as Agggregated Corss-Channel Encoders

논문에서는 2개의 CCE train을 다음과 같이 진행하였습니다.

![img](https://i.imgur.com/S4TVVTn.png)

그리고 input되는 image의 domain에 따라 다른 방식으로 channel을 prediction하였습니다. input되는 image는 크게 **Lab image** 과 **RGB-D image** 가 있습니다.

#### Lab image

**Lab image** 는 grayscale information을 가진 L channel과 color information를 가진 ab channel로 분리됩니다. 그리고 F1에서는 automatic colorization을 수행하고 F2에서는 grayscale prediction을 수행합니다. prediction한 channel들을 concat하여 하나의 imge로 prediction합니다. 아래의 그림을 보시면 직관적으로 이해하실 수 있습니다.

![img](https://i.imgur.com/nZNlawP.png)

#### RGB-D image

**RGB-D image** 는 HHA information을 가진 HHA channel과 RGB information을 가진 RGB channel로 분리됩니다. 그리고 F1에서는 HHA predictoin을 수행하고 F2에서는 Lab space안에서 RGB prediction을 수행합니다. prediction한 channel들을 concat하여 하나의 imge로 prediction합니다. 아래의 그림을 보시면 직관적으로 이해하실 수 있습니다.

![img](https://i.imgur.com/Yko80Cj.png)

#### Alternative Aggregation Technique

논문에서 concat하는 과정에서 조금 더 효과적으로 수행하기 위해 조금 더 효과적인 loss function을 제시하였습니다.

![img](https://i.imgur.com/oxNdLmO.png)

하지만, loss function이 data subset만 고려하고 full input X를 고려하지 않아서 아래와 같은 새로운 loss function을 제시하였습니다.

![img](https://i.imgur.com/gYyCd5v.png)

---

## Result

SBA(Split-Brain Autoencoders)의 성능을 평가하기 위해 2가지의 실험을 진행하였습니다. 첫번째는, **Lab image인 ImageNet** data를 label없이 학습하여 기존의 존재하는 Unsupervise Learning과 성능을 비교하였습니다. 두번째도 마찬가지로, **RGB-D image인 NYU-D** data를 label없이 학습하여 기존의 존재하는 방법들과 성능을 비교하였습니다.

### ImageNet

첫번째 실험인 **ImgeNet Classification** 의 결과는 다음과 같습니다.

![img](https://i.imgur.com/OcUb9fz.png)

1. ImageNet-labels : ImageNet labeling을 활용하여 학습 (**Supervised**)
1. Gaussian : 가중치를 random하게 gaussian 초기값을 적용하여 학습 (**Unsupervised**)
1. Krahenbuhl et al. : k-means 초기값을 쌓아서 학습 (**Unsupervised**)
1. Noroozi & Favaro : jigsaw puzzles (**Unsupervised**)
1. Doersch et al. : context prediction (**Unsupervised**)
1. Donahue et al. : Adversarial feature learning (**Unsupervised**)
1. Pathak et al. : context encoder (**Unsupervised**)
1. Zhang et al. : colorization (**Unsupervised**)
1. Lab -> Lab : CCE + Lab image + regression loss
1. Lab(drop50) -> Lab : 50% drop + random 50% image drop
1. L -> ab(cl) : automatic colorizatoin + classification loss
1. L -> ab(reg) : automatic colorization + regression loss
1. ab -> L(cl) : grayscale prediction + classification loss
1. ab -> L(reg) : grayscale prediction + regression loss
1. (L, ab) -> (ab, L) : single network(colorization + grayscale prediction) + 1st concat loss
1. (L, ab, Lab) -> (ab, L, Lab) : single network(colorization + grayscale prediction) + 람다가 1/3인 2nd concat loss
1. Ensembled L -> ab : two disjoint subnetworks(one : classification loss, two : regression loss)
1. Split-Brain Auto (reg, reg) : full methods + classification
1. **Split-Brain Auto (cl, cl)** : full methods + classification loss

labeling Supervised learning을 제외하고는 기존의 존재하는 Unsupervised leaning과 비교하였을때, **모든 layer에서 성능이 높음을 알 수 있습니다.**

#### Downstream

ImageNet을 pretrain시킬때 layer별로 성능을 보았을때, 좋은 성능을 보였습니다. 하지만, post의 앞부분에서 말했듯이 **다른 task에서도 좋은 성능을 보이는 것이 중요** 하기 때문에, **다른 task에 downstream** 하여 성능을 평가하였습니다.

##### ImageNet

ImageNet Classification에서 기존의 Unsupervised Learning에 비해 좋은 성능을 보였습니다.

![img](https://i.imgur.com/cov2jV1.png)

##### Places

Places Classification에서도 기존의 Unsupervised Learning에 비해 좋은 성능을 보였습니다.

![img](https://i.imgur.com/sJbPXHT.png)

##### Pascal

Pascal dataset에서는 Classification외에도 Detection, Segmentation도 같이 성능을 평가하였습니다. Classifiaction과 Segmentation에서는 좋은 성능을 보였지만, Detection에서는 중간정도의 성능밖에 보이지 않았습니다.

![img](https://i.imgur.com/YQVJTUS.png)

### NYU-D

두번째 실험인 NYU-D의 결과는 다음과 같습니다.

![img](https://i.imgur.com/X5QDuTx.png)

1. Gupta et al. : Cross modal distillation for supervision transfer (Supervised)
1. Gupta et al. : Learning rich feature (Supervised)
1. Gaussian : 가중치를 random하게 gaussian 초기값을 적용하여 학습
1. Krahenbuhl et al. : k-means 초기값을 쌓아서 학습
1. Split-Brain Autoencoder : full methods

NYU-D에서도 Supervised Learning을 제외하고는 기존의 Unsupervised Learning과 비교하였을때, 높은 성능을 보였습니다.



---

참고
1. [https://arxiv.org/pdf/1611.09842.pdf][참고1]

---

[참고1]: https://arxiv.org/pdf/1611.09842.pdf
