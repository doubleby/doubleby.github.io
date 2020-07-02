---
layout: post
title: "Context Prediction"
description:
date: 2020-07-02 21:03:36 +0530
categories: Self-Supervised-Learning
---
---

이번 post는 Self-Supervised Learning의 시초가 되었던 '[Exemplar][paper1]' 의 단점을 개선한 '[Unsupervised Visual Representation Learning by Context Prediction][paper2]' 논문을 리뷰하겠습니다.

긁을 읽기 전에, 이전 post인 '[Exemplar][post]' 을 읽고 오신다면 이해하기 더 수월할 것입니다.

---

## Pretext Task

이번 논문의 pretext task는 text domain의 방법을 image에 적용한 것입니다. Text domain에서 context을 파악할때, 단어의 전과 후가 similar하거나 unsimilar한 단어인지 확인하면서 파악합니다. 이러한 logic을 image에 적용한 것이 이 논문의 **pretext task** 입니다.

좀 더 자세히 설명드리자면,
1. 한장의 image를 가로 3칸과 세로 3칸, 총 9칸의 patch로 분할합니다.
1. 가운데 patch를 제외한 나머지 8개의 patch에 1~8까지 할당합니다.
1. 가운데 patch(1번째 patch)와 나머지 8개의 patch 중에 1개(2번째 patch)를 임의로 고릅니다.
1. 두개의 patch를 network에 전달하여 2번째 patch가 1번째 patch의 어느 부분에 해당하는지 맞추도록 학습합니다.

아래의 그림을 보신다면, 조금 더 직관적으로 이해하실 수 있을 것입니다.

![img](https://i.imgur.com/q0hWsgh.png) ![img](https://i.imgur.com/O8QW0Lf.png)

즉, **논문에서 제안하는 pretext** 는 image의 spatial context를 이용하여 model이 image 내의 object와 object parts를 인식하는 visual representation을 학습시키는 것입니다.

---

## Architecture

Context Predictoin architecure의 큰 특징은 2개의 patch가 input하여, 8개의 영역들에 할당될 확률이 softmax으로 output되는 것입니다. 하지만 결과적으로 저자가 원하는 것은 feature extractor를 개별 patch에 대해 학습함으로써, 새로운 image의 patch가 input으로 들어왔을때, feature extractor를 통해 최대한 알맞는 영역으로 위치하도록 만드는 것입니다.

이를 위해 아래의 그림과 같이 기존 **AlexNet** 모델을 기반으로 **late-fusion** architecture를 사용하였습니다.

![img](https://i.imgur.com/iWKi5Bn.png)

구조를 자세히 살펴보면, 2개의 patch를 학습하기 위해 AlexNet의 **fc6 layer** 까지 병렬적으로 구성하였습니다. fc6 layer이후에는 병합하는 late-fusion architecture입니다. 이러한 구조를 통해 input patch별로 feature extractor를 독립적으로 fit할 수 있습니다.

### Trivial Solution

network가 patch들 간의 관계를 학습할때, high-level semantc을 이용하지 않고, boundary pattern과 patch간의 이어지는 texture를 이용하여 학습한다면 제대로 학습하지 못한 것입니다.

이러한 **trivial solution** 을 해결하기 위해 아래와 같은 2가지 방법을 사용하였습니다.
1. 각 patch 간의 gap을 설정하여, 딱 붙어있지 않도록 구성
1. 정확히 상하좌우 n fixel 위치에서 patch를 취득하지 않고 x와 y 방향으로 약간의 움직임을 추가하여 patch를 취득

위의 고양이 사진과 같이, patch의 위치를 전부 인접한 위치로 취득하는 것이 아닌, patch별로 random하게 조금씩 움직여서 patch를 얻었습니다.

### Chromatic Abberration

하지만, 위와 같은 방법으로 사용해도 완전히 trival solution을 해결하진 못했습니다. 해결하지 못한 주 원인은 **chromatic aberration(색수차)** 이었으며, 렌즈가 빛의 파장에 따라 다른 굴절률을 가질때 나타나는 것입니다.

이를 해결하기 위해 image의 구성요소인 **RGB(Red, Green, Blue)** 의 값들을 각각 분리하여 detection하는 ConvNet을 구성하였습니다. 그리고 detection한 RGB 값들을 변형하였습니다. RGB값을 변형시키는 방법을 2가지로 제안하였으며,

**첫번째** 는 **projection** 으로써, RGB값을 아래의 식과 같이 변형하였습니다.

$$a(RGB의 값) = [-1, 2, -1]$$
$$B = I - a^Ta/(aa^T)$$

RGB값을 위와 같은 수식으로 모든 pixel의 값을 변형함으로써,  chromatic abbreation을 해결할 수 있었습니다.

**두번째** 는 **color dropping** 으로써, 3개의 color channel중에 random으로 2개의 color channel을 drop하는 것입니다. 그리고 drop된 color는 Gaussian noise로 대체하여 chromatic abbreation을 해결할 수 있엇습니다. 논문에서 2가지의 방법을 비교하였을 때, 성능은 비슷하였습니다.

---

## Mechanism

Context Prediction은 Caffe 툴과 ImageNet image를 training set으로 선정하여 진행하였습니다. 진행절차는 다음과 같습니다.

1. feature extractor freeze를 위한 pretrain을 ImageNet 데이터로써 활용
1. 150K ~ 450K total pixels size로 unlabeled image를 resize
1. resize image로부터 96 x 96 해상도로 patch들을 sample(+ 9개의 영역으로 적용할 수 있는 image만 sample) (**pretext**)
1. 영역에 있는 patch들 간의 gap을 48 pixels로 설정(+ 각 patch를 상하 7 pixel, 좌우 7 pixel만큼 radom으로 조금씩 움직이기)
1. mean subtraction, projection or color dropping, downsampling, upsampling, batch normalization과 같은 pre-processing
1. network 학습시키고 feature extractor freeze
1. VOC 2007 데이터 셋을 fine-tuning으로 성능 평가(**downstream**)

---

## Result

아래의 그림은 Context Predictoin의 성능 결과의 비교 표입니다.

![img](https://i.imgur.com/6vaktKY.png)

논문에서 제안한 구조는 5가지였습니다.

1. **Scrath-Ours** : 초기값을 random으로 적용한 model
1. **Ours-projection** : projection을 적용한 model
1. **Ours-color-dropping** : color-dropping을 적용한 model
1. **Ours-Yahoo100m** : Yahoo100m 데이터를 pre-train한 model(예비)
1. **Ours-VGG** : AlexNet이 아닌 VGG기반으로 만든 model(예비)

**Ours-color-dropping** model이 기존에 존재하는 Unsupervised Learning model보다 성능이 좋음을 알 수 있습니다. 하지만, labeled된 ImageNet을 Supervised Learning시킨 ImageNet-R-CNN model에 비해서는 성능이 낮음을 알 수 있습니다.

---

## Drawback

최근에는 성능의 차이가 거의 없지만, 이때는 Unsupervised Learning의 연구가 초기였기 때문에 Supervised Learning과 성능차이가 많이 나는 것을 알 수 있습니다. 그리고 patch들 중에 거의 비슷한 patch가 있을 경우에 영역 위치를 배정을 잘 못하는 문제점도 있었습니다. 이러한 문제점과 한계점이 있었지만, image를 영역별로 나누어 학습한다는 pretext를 제안함으로써 큰 의미가 있습니다.

---

참고
1. [https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf][참고1]
1. [https://hoya012.github.io/blog/Self-Supervised-Learning-Overview/][참고2]
1. [https://seongkyun.github.io/study/2019/11/29/unsupervised/][참고3]
1. [https://daeheepark.tistory.com/m/15?category=772193][참고4]

---

[paper1]: https://arxiv.org/pdf/1406.6909.pdf
[paper2]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf
[post]: https://doubleby.github.io/self-supervised-learning/2020/06/29/Exemplar/
[참고1]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf
[참고2]: https://hoya012.github.io/blog/Self-Supervised-Learning-Overview/
[참고3]: https://seongkyun.github.io/study/2019/11/29/unsupervised/
[참고4]: https://daeheepark.tistory.com/m/15?category=772193
