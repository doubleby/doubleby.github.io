---
layout: post
title: "Jigsaw Puzzles"
description:
date: 2020-07-06 21:03:36 +0530
categories: Self-Supervised-Learning
---
---

이번 post는 이전 post에서 리뷰했던 '[Unsupervised Visual Representation Learning by Context Prediction][paper1]' 의 단점을 개선한 논문인 '[Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles][paper2]' 의 리뷰입니다.

이전 post인 '[Context Prediction][post]' 를 먼저 읽고 오시면, 이해하기 더 수월할 것입니다.

---

## Pretext Task

Context Prediction의 pretext task는 1개의 center patch와 8개의 side patch 중 1개를 학습시켜 상대적인 위치를 예측하였습니다. 하지만, 상대적인 위치다 보니 비슷한 side patch일 경우 모호한 부분이 있었습니다. 이러한 모호한 부분을 해결하기 위해, 논문에서 제안하는 방법은 2개의 patch들이 아닌 모든 patch들을 학습시키는 방법을 제안하였습니다. 그리고 모든 patch들을 학습시킴으로써, Context Prediction에서 존재하던 chromatic aberration을 따로 다룰 필요가 없어졌습니다.

단순히 모든 patch들을 한번에 다 겹쳐서 input하는 것이 아닌, patch별로 network를 각각 설정하여 학습시키는 **Context Free Network** 을 제안하였습니다. Context Free Network구조에 대해서는 아래에서 자세히 다루도록 하겠습니다.

그리고 CFN(Context Free Network) 이외에도, **Jigsaw Puzzles** 방법을 제안하였습니다. Jigsaw Puzzles는 image의 9개 patch들을 뒤죽박죽 섞어 놓고 원래 patch의 배열로 돌아가기 위한 permutation(patch 위치의 index 배열)을 예측하는 방법입니다. 아래의 그림을 보면 조금 더 직관적으로 이해하실 수 있을 것입니다.

![img](https://i.imgur.com/gfjLnOD.png)

즉, 모든 patch들을 학습시키기 위한 network 구조인 **Context Free Network** 와 patch들을 섞은 후에 원래의 patch위치로 돌아가기 위한 permutation을 예측하는 **Jigsaw Puzzles** 가 이번 논문의 **Pretext Task** 입니다.

---

## Other Pretext Task

논문에서 제안하는 pretext task의 성능을 평가하기 위해, 기존에 존재하는 pretext task와 결과를 비교하였습니다. 결과비교를 위해 기존에 존재하는 pretext task를 간략하게 설명드리겠습니다. 논문에서는 총 3개의 Unsupervised Learning과 결과를 비교하였습니다.

### Context Prediction

이전 post에서 리뷰했던 Context Prediction입니다. 아래와 같이 image의 2개의 patch를 crop하여 학습시키는 pretext task입니다. 자세한 내용은 [이전 post][post]를 참조하면 감사하겠습니다.


### Tracking Video

Video의 original object image, original과 similar object image, 완전 다른 random image를 비교하여, original와 random image의 distance보다 simiar image의 distance를 최소화 하도록 학습시키는 pretext task입니다. 아래의 그림을 보시면 조금 더 직관적으로 이해하실 수 있습니다.

![img](https://i.imgur.com/NkJRcjp.png)

### Context Encoder

image의 일부분이 누락되도 사람아 누락된 부분을 유추하는 것을 활용한 pretext task입니다. Context Encode는 Encoder와 Decoder로 이루어져있습니다. Encoder를 통해 image의 context의 feature extractor를 추출하고, Decoder를 통해 누락된 image의 context를 생성합니다. 아래의 그림을 보시면 조금 더 직관적으로 이해하실 수 있습니다.

![img](https://i.imgur.com/meRBcDz.png)

기존의 pretext task는 high-level feature보다 유사도(색상이나 질감)으로 2개의 image or patch를 학습하여, 동일한 물체일 경우 유사하다고 판단하지 않지 않았습니다. 이를 해결하기 위해 논문에서 제안하는 것은 위에서 설명드린 바와 같이 **2개의 image가 아닌 모든 patch들을 학습** 시키는 것입니다.

---

## Context Free Network & Jigsaw Puzzles

논문에서 제안하는 Pretext task에 대해서 자세히 알아보겠습니다.

![img](https://i.imgur.com/LpwAKXs.png)

위의 그림은 CFN(Context Free Network)의 mechanism과 architecture를 보여줍니다. 먼저, **mechanism** 을 아래에 순서와 함께 설명드리겠습니다.

1. image로부터 random하게 225 x 225 pixel window를 crop합니다. (자동차 image의 빨간색 점선)
1. 가로 3, 세로 3으로 grid를 나눕니다.
1. 나누어진 75 x 75 pixel cell로부터 random하게 64 x 64 pixel patch를 뽑습니다. (자동차 image의 초록색 실선, 영역별 index)
1. 9개의 patch들을 사전에 정의된 permutation set으로 부터 random하게 permutation을 뽑습니다. => **Jigsaw Puzzles**
1. permutatoin대로 나열한 patch들을 CFN으로 input시킵니다.

4번에서 **permutation** 이 의미하는 것은 patch들을 학습하는 위치를 나열하기 위해 9개를 나열하는 9!입니다. 즉, image를 3 x 3 grid로 나눌때, 영역별로 9개의 index가 설정됩니다. 이때, 9개의 patch들을 random하게 permutation함으로써 각각 영역별 index를 부여받습니다. 이 후, 부여받은 영역 index와 이에 해당하는 patch가 일치할 확률을 계산하는 것입니다.

하지만, 9개의 patch들을 나열하기엔 9! = 362,880의 많은 경우의 수가 있습니다. 이 문제를 해결하기 위해 유사한 permutation은 제거하면서 딱 1000개의 permutaion을 사전에 정의하였습니다.

mechanism에 이어서 CFN의 **architecture** 의 특징은 아래와 같습니다.

1. fc6 layer까지는 AlexNet architecture를 사용하였으며, 간단한 구조를 위해 max-pooling과 ReLU layer는 사용하지 않았습니다.
1. 각 network는 각 patch에 대한 feature extractor를 얻게 됩니다.
1. network들은 fc6 layer까지 weight를 공유합니다.
1. fc6에서 나온 output을 통합하여 fc7에 input
1. 결과적으로 random하게 뽑은 patch 배열이 해당하는 index와 일치할 확률이 output으로 나오게 됩니다.

위와 같은 architecture를 통해 기존의 AlexNet에 비해 절반 이하의 parameter를 가질 수 있었습니다. parameter의 수가 줄어들었지만, 기존의 AlexNet과 성능 차이는 미비했습니다.

---

## Result

자세한 성능의 차이의 결과는 다음과 같으며, Jigsaw Puzzles과 DFN을 Pascal VOC 데이터에 **downstream** 하였습니다.

![img](https://i.imgur.com/v3YGSAY.png)

논문에서 제안하는 pretext task와 Supervised Learning, Unsupervised Learning의 context task와 결과를 비교하습니다. 다른 pretext task에 대해 간단하게 설명하자면 다음과 같습니다.

1. Krizhevskyet al. : 기존의 AlexNet(Supervised Learning)
1. Wang and Gupta : Tracking Video(Unsupervised Learning)
1. Doersch et al. : Context Prediction(Unsupervised Learning)
1. Pathak et al. : Context Encoder(Unsupervised Learning)

기존에 존재하는 Unsupervised Learning의 pretext task보다 모든 부분에서 성능이 뛰어나다는 것을 알 수 있습니다. 하지만 Supervised Learning에 비해서는 성능이 낮지만, 거의 비슷하다고 볼 수 있습니다.

---

참고
1. [https://arxiv.org/pdf/1603.09246.pdf][참고1]
1. [https://hoya012.github.io/blog/Self-Supervised-Learning-Overview/][참고2]
1. [https://seongkyun.github.io/study/2019/11/29/unsupervised/][참고3]
1. [https://hyeonnii.tistory.com/261][참고4]
1. [https://creamnuts.github.io/paper/Jigsaw/][참고5]

---

[paper1]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf
[paper2]: https://arxiv.org/pdf/1603.09246.pdf
[post]: https://doubleby.github.io/self-supervised-learning/2020/07/02/Context-Prediction/
[참고1]: https://arxiv.org/pdf/1603.09246.pdf
[참고2]: https://hoya012.github.io/blog/Self-Supervised-Learning-Overview/
[참고3]: https://seongkyun.github.io/study/2019/11/29/unsupervised/
[참고4]: https://hyeonnii.tistory.com/261
[참고5]: https://creamnuts.github.io/paper/Jigsaw/
