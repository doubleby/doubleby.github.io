---
layout: post
title: "Colorization"
description:
date: 2020-07-15 21:03:36 +0530
categories: Self-Supervised-Learning
---
---

이번 post는 이전 post와 달리 image의 pixel을 encoder하는 방식을 SSL에 적용한 논문인 '[Colorful Image Colorization][참고1]'의 리뷰입니다. 

---

## Colorization

**colorization** 은 grayscale image을 color image으로 변형시키는 것입니다. 이전에는 사람이 직접 grayscale image을 보고 object에 해당하는 color을 판단하여 colorization을 수행하였습니다. 이 논문에서 제안하는 것은 사람이 아닌 CNN Network를 통해 완전히 자동적으로 image에 맞는 선명하고 실제적인 colorization을 제안하였습니다.

### Prior Colorization

**이전의 colorization** 은 grayscale과 color의 상관관계를 modeling하여 colorization을 수행하였으며, 크게 **Non-parameter** 와 **Parameter** method가 있습니다.
**Non-parameter method** 는 input되는 grayscale image과 관련된 image를 사람이 참고해서 colorization을 수행하거나, 학습되는 많은 양의 color image로부터 예측 함수를 활용하여 학습하였습니다.
**Paramter method** 는 color space의 regression loss function이나 color value의 classification loss function을 사용하였습니다.

**논문에서 제안하는 방법 또한 color를 classification을 수행하지만, 이전의 방법보다 더 큰 model, 더 많은 data, 혁신적인 loss function과 mapping을 사용하였습니다.**

### Our Colorization

![img](https://i.imgur.com/0tWmADO.png)

위의 그림은 논문에서 제안하는 CNN Network colorization에 input한 grayscale image과 output된 colorization입니다. CNN Network colorization은 image의 실제 color로 복원하는 것이 아닌, 실제 color라고 환각을 느낄 정도로 그럴듯한 color로 colorization을 수행합니다.

image의 밝기 채널인 **L** 이 주어지면, Network는 CIE Lab colorspace에서 image에 해당하는 **a**와 **b** color channel을 예측합니다. 이전의 colorization은 고전적인 예측방법으로 추정값과 실제값의 오류를 최소화하는 표준적인 regression의 loss function을 사용하였기 때문에 color가 흐릿한 결과를 초래하였습니다. 하지만 새로운 loss function과 architecture을 통해 흐릿한 문제점을 개선하였습니다.

![img](https://i.imgur.com/VJyzCUs.png)
*CIE Lab Colorspace*

#### loss function

흐릿한 문제를 해결하기 위해, 논문에서 제안하는 colorization에 딱 맞는 loss function을 제안하였습니다. 크게 3가지의 loss function을 제안하였으며, 첫번째가 **rebalanced rare classes classification loss** 이고 두번째가 **un-rebalanced classification loss** 이고 세번째가 **regression loss** 입니다. 그리고 3가지의 loss function들을 각각 적용하여 성능을 비교하였습니다. 비교 결과는 아래에 있는 **Result** 부분에서 보여드리겠습니다.
3가지의 loss function을 사용할때, 각 pixel별로 color의 분포를 예측하였습니다. 그리고 학습할때 rare color를 고려하여 loss를 re-weight하였습니다. 이를 통해 colorization이 이전 방법에 비해 선명해지고 현실적인 결과를 초래하였습니다.

#### CNN architecture

CNN architecture 또한 당시에 존재하던 것들이 있었습니다. 첫번째가 **hypercolumns를 사용한 VGG Network**, 두번째가 **global과 local feature를 통합하는 two-stream architecture**, 세번째가 **Convolution layer를 깊게 쌓는 VGG Network** 가 있었습니다. 그중 첫번째 구조를 사용하여 colorization의 성능비교하였으며, 비교 결과는 아래에 있는 **Result** 부분에서 보여드리겠습니다.

#### Self-Supervised Learning(SSL)

추가적으로, 논문에서는 SSL의 형식으로 colorization을 수행해보았습니다. 이 논문이 투고될 당시의 SSL 기법들은 autoencoder나 이전 post에서 설명드린 기법들을 사용하였습니다. 여기서 제안하는 것은 **cross-channel encoder** 이라는 pretext task를 통해 SSL을 수행하였으며, 기존의 SSL기법들과 성능을 비교하였습니다. **cross-channel encoder** 의 비교 결과는 아래에 있는 **Cross-Channel Encoding** 부분에서 보여드리겠습니다. 이번 post의 main이라고 볼 수 있습니다.

#### Contribution

논문에서 제안하는 contribution 크게 2가지가 있습니다.
1. 아래의 방법을 적용하여 automatic image colorization의 graphic problem을 진행하였습니다.

(1) 기존의 classification loss function의 단점을 개선하여 colorization의 다양하고 불확실성을 다루는 objective function을 설계하였습니다.

(2) 다른 image task에 적용할 수 있는 새로운 colorzation algorithm

(3) 백만장의 color image를 학습시켜 새로운 최고 수위를 설정하였습니다.

1. SSL의 경쟁력있고 간단한 colorization task를 소개합니다.

---

## Architecture

논문이 제안하는 grayscale image를 input하여 colorization을 output하는 CNN architecture는 아래와 같습니다.

![img](https://i.imgur.com/kYvzC0K.png)

architecture를 구성할때, 저자가 제일 신경을 쓴 부분은 위 그림의 **파란색 상자인 (a,b) 확률 분포** 와 **Color ab** 이었습니다. 신경 쓴 2부분은 **objective function**, **class rebalancing**, **class probabilities to point estimates** 으로 수행하였습니다.

### Objective Function

이전의 방법론중에 input으로 들어오는 밝기 channel(**X**)를 고려하여 관련된 2개의 color channel(**Y**)로 mapping해주는 **Y^** 을 설정하여 classification loss function을 구성한 것이 있었습니다. 이전의 classification loss function은 아래와 같습니다.

![img](https://i.imgur.com/BnS0VJL.png)

하지만, 위의 classification loss function은 robust하지 않았습니다. 이 문제를 해결하기 위해 저자는 밝기 channel(**X**)을 다른 방법으로 mapping하여 새로운 classification loss function을 사용하였습니다. 다른 방법으로 mapping과정은 아래의 그림과 같습니다.

![img](https://i.imgur.com/lQBfrJS.png)

첫번째인 (a)는 밝기 channel(**X**)를 고려하여 산출되는 color인 ab를 10 size인 grid에 **양자화** 시킵니다. 양자화시킴으로써 총 **313개의 쌍** 이 영역에 표시가 됩니다.

두번째인 (b)는 **log scale** 로 ab 값들을 **경험적 확률 분포에 mapping** 합니다.

세번째인 (c)는 L(log scale)의 값에 따라 ab 값들의 경험적 확률 분포가 달라지는 것을 알 수 있습니다.

이렇게 예측된 분포(경험적 확률 분포)인 **Z^** 과 image의 실제 color 분포인 **Z** 와 loss를 구하는 식을 설계하였으며, 아래와 같습니다.

![img](https://i.imgur.com/qhOjrWI.png)

H, W는 image의 dimension이며, Q는 양자화된 a와 b값들의 수입니다. v는 아래에서 말씀드릴 class rebalancing에 대한 weight입니다.

### Class Rebalancing

구름, 도로, 먼지, 벽과 같은 natual image의 ab 값이 낮아서 강한 bias가 됩니다. 그리고 color가 선명하지 않고 흐리게 ab값이 관측됩니다. 이렇게 color가 흐리게 관측되고 강한 bias가 되는 class imbalance 문제를 해결하기 위해 **rare color를 고려하여 각 pixel의 loss을 reweight** 하여 class rebalancing 문제를 해결하였습니다. class rebalancing의 수식은 아래와 같습니다.

![img](https://i.imgur.com/jGLADIx.png)

각 pixel은 가장 가까운 ab 값을 기반으로 w(weight)가 측정됩니다. 더 좋은 경험적 확률 분포(**P~**)를 얻기 위해 Gaussian Kernel(**G**)를 활용하여 양자화된 ab의 경험적 확률 분포(**P**)에 적용하였습니다. 그리고 weight(**람다**)를 적용하여 uniform 분포와 섞고 역수와 정규화를 시행합니다. 시행결과로 Object Function에 사용되는 v인 weight(**람다**)를 산출하였으며 **1/2** 일때 성능이 뛰어났습니다.

### Class Probabilities to Point estimates

마지막으로, ab space에서 a와 b의 점을 추정(**Y^**)하는 예측된 분포(**Z^**)를 mapping하는 **H** 를 정의했습니다. 점 추정을 위해 **첫번째로**, 아래와 같은 2개 예제 image의 제일 오른쪽 열과 같이, 각 pixel에 대해 예측된 분포의 **중앙값** 을 취하였습니다.

![img](https://i.imgur.com/N0wT77y.png)

중앙값을 취하였을때 선명함을 제공하지만, 가끔 color가 적절하지 않게 match되었습니다. 두번째로, 제일 왼쪽과 같이 예측된 분포의 **평균값** 으로 취하였을때, color가 적절하게 match였지만, color가 선명하지 않았습니다. 2가지(선명함과 color의 일치)를 만족하기 위해 아래의 수식과 같이 interpolate를 실시하였습니다. 정의한 H의 수식은 아래와 같습니다.

![img](https://i.imgur.com/NRxITM9.png)

이를 위해 softmax 분포의 온도 T를 재조정하여 output을 평균으로 취하여 interpolate하였습니다. softmax 분포의 온도 T는 **0.38** 일때 가장 적절하였으며, 위의 image의 가운데와 같이 **annealed-mean** 이라고 정의하였습니다.

---

## Result

논문에서 제안하는 architecture를 평가하기 위해 ImageNet을 train, valid, test로 나누어 학습시켜 평가하였습니다. 그리고 기존에 존재하는 방법들과 같은 dataset 비율을 적용하여 성능을 비교하였습니다. 그전에, architecture의 output을 보여드리겠습니다.

![img](https://i.imgur.com/HertRva.png)

위의 image의 column별로 설명을 드리자면 다음과 같습니다.

1. input : grayscale image
1. regression : regression loss function
1. classification : un-rebalanced classification loss
1. classification w/rebal : rebalanced rare classes classification loss function
1. Ground truth : image의 실제 color

image의 6번째 row까지는 제대로 적용한 사례이며, 아래의 3개의 row은 실폐인 사례입니다. 성공한 사례를 보면, 이전의 방법들보다 좀 더 정확하고 선명한 color를 가지는 것을 알 수 있습니다. 이러한 성능을 토대로 기존의 방법들과 비교하여 성능을 비교 평가하였습니다. 결과는 아래의 표와 같습니다.

![img](https://i.imgur.com/8xJlAue.png)

각 row에 대하여 설명을 드리자면,
1. Ground Truth : image의 실제 color
1. Gray : image의 흑백 image
1. Random : training set으로 부터 random한 image로 부터 color를 copy한 image
1. Dahl : VGG features로 부터 Laplacian pyramid를 사용한 이전의 모델(이전의 classification loss function을 사용한 모델)
1. Larsson et al : 기존의 CNN 방법
1. Ours(L2) : 논문에서 제안하는 architecture + 이전의 classificatoin loss function
1. Ours(L2, ft) : Ours(L2) + rebalancing 사용
1. Ours(class) : 논문에서 제안하는 architecture + 논문에서 제안하는 classification loss function(rebalancing X)
1. Ours(full) : 논문에서 제안하는 architecture + 논문에서 제안하는 classification loss function + rebalancing

**AuC column** 을 보게 되면, rebalancing을 적용 유무의 성능을 판단하기 위해 2개의 세부 column들이 존재합니다. **VGG Top-1 Class ACC** 는 VGG Network를 사용한 colorization의 분류 정확도입니다. **AMT Labeled Real** 은 사람들에게 real color image과 fake(colorization) image의 test한 평균과 표준 오차입니다.

---

## Cross-Channel Encoding

그리고 논문에서 제안한 colorization이 Self-Supervised representation learning의 **pretext task** 로 어떻게 작용할 수 있을지 평가하였습니다. 논문의 model은 input와 output이 다른 image channel인 것을 제외하고는 autoencoder와 비슷했습니다. 그래서 논문의 model의 pretext task를 cross-channel encoding이라고 명칭하였습니다.

SSRL(Self-Supervised Representation Learning)에서 CCE(cross-channel encoding)의 성능을 평가하기 위해 2가지를 test하였습니다.

### ImageNet

첫번째로, label없이 ImageNet를 colorization을 수행하기 위해 CCE의 Network를 pre-trained시켰습니다. 학습된 feature represent가 object의 의미를 얼마나 잘 나타내는지 test하기 위해, network의 weight를 freeze하고 각 convolution layer에 linear classifiers를 학습했습니다. 결과는 다음과 같습니다.

![img](https://i.imgur.com/Mpq6mj8.png)

CCE는 grayscale image를 학습하기 때문에 핸디캡을 가지게 됩니다. 이러한 정보 손실을 줄이기 위해, AlexNet에 grayscale image classification으로 **fine-tune(downstream)** 시켰습니다. 그리고 grayscale image로부터 random하게 초기값을 설정하여 학습시킨 후 기존의 SSL 방법들과 비교하였습니다.

기존의 SSL 방법은 'Context Prediction'인 **Doersch et al.** , 'Context Encdoer'인 **Pathak et al.** , 'Adversarial feature learning'인 **Conahue et al.** 입니다.

conv1을 보게되면, Dersch와 Donahue에 비해 낮은 성능을 보입니다. 하지만 다른 방법들과는 비교할만 했습니다. 그리고 conv2에서 Dersch와 Conahue와 비교할 정도로 성능이 올라왔습니다. 다른 SSL에 비해 grayscale image를 input하는 것을 고려하면 뛰어나다고 볼 수 있습니다.

### PASCAL

두번째로, 일반적으로 사용되는 PASCAL classification, detection, segmentation에 model을 test하였습니다. 결과는 다음과 같습니다.

![img](https://i.imgur.com/JpH2TpX.png)

test할때, 2가지의 model을 사용하였습니다.

**첫번째는**, 기존과 같은 grayscale을 input하여 **Ours(gray)** 라고 명칭했습니다.

**두번째는**, ab channel의  초기 가중치를 0으로 하여 conv1에서 3-channel 모두 다 받을 수 있도록 수정하고 **Ours(color)** 라고 명칭했습니다.

결과를 자세히 보게 되면, 3부분 모두에서 강한 성능을 보이는 것을 알 수 있습니다. 특히, classification과 segmentation에서 최고의 성능을 보임을 알 수 있습니다.

---

참고
1. [https://arxiv.org/pdf/1603.08511.pdf][참고1]

---

[참고1]: https://arxiv.org/pdf/1603.08511.pdf
