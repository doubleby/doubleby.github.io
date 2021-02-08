---
layout: post
title: "Deep Learning with PyTorch (1)"
description: "Eli Stevens, Luca Antiga, Thomas Viehmann, "Deep Learning with PyTorch" - Chapter 1"
date: 2021-02-08 21:03:36 +0530
categories: Deep-Learning-with-PyTorch
mathjax : true
---
---

Deep Learning에 관심을 가지고 연구를 진행하면서, 실질적으로 code로 구현하는 것이 중요하다고 느꼈습니다. 그래서 deep learning에서 많이 활용되는 tools 중 하나인 PyTorch를 밑바닥부터 배우기 위해 "[Deep Learning with PyTorch][paper]" 를 읽어보고 시리즈처럼 포스팅을 해볼려고 합니다.

이번 포스트는 Deep Learning with PyTorch의 Chapter 1인 **Introducing deep learning and the PyTorch Library** 입니다.

---

# The deep learning revolution

지난 10년동안 machine learning은 feature engineering에 크게 의존하였습니다. 여기서 feature engineering이란, 좋은 output을 생성하기 위해 classifier에 task를 용이하게하는 input data에 대한 변환을 말합니다. 예를 들면, 손으로 쓴 숫자 image에서 0과 1을 구분하기 위해 image의 가장자리를 추정하는 filter set를 찾고 가장자리 분포가 주어지면, 올바른 숫자를 예측하도록 classifier를 훈련하는 것을 machine learning의 feature engineering이라고 볼 수 있습니다.

하지만, deep learning은 raw data에서 representations을 자동으로 찾는 것을 다룹니다. 같은 예시인 손으로 쓴 숫자 image에서 0과 1을 구분하는 예시에서, deep learning은 filter가 학습 중에 examples와 target의 labels를 반복적으로 살펴봄으로써 구체화시킵니다.

![Machine Learning과 Deep Learning의 차이](https://i.imgur.com/3lmcjrt.jpg "Machine Learning과 Deep Learning의 차이")

즉, 위의 그림과 같이 (왼쪽) machine learning은 data의 feature engnieering을 직접 정의하고 학습을 시키지만 (오른쪽) deep learning은 data의 representations을 추출하는 알고리즘이 자동적으로 제공되어 학습을 시킵니다.

---

# An overview of how PyTorch supports deep learning projects

![Basic PyTorch project](https://i.imgur.com/KJOSdKc.jpg "Basic PyTorch project")

위 그림은 PyTorch로 분석을 진행할때의 구조를 보여줍니다.

## Data source

그림의 왼쪽에 있는 data processing은 data을 가져와서 $tensors$ 로 변환합니다. 이때, process는 problem마다 다르므로 data sorcing을 직접 구현해야합니다. 이 부분은 chapter 4에서 자세히 다루겠습니다.

## Batch tensor

일반적으로 data 저장 속도가 느린 경우가 많으므로 data load를 병렬화로 진행합니다. 하지만, python에서는 이러한 작업을 batch로 쉽게 해결할 수 있습니다. 이 부분은 chapter 7에서 자세히 다루겠습니다.

## Training loop & Trained model

그림의 training loop에서는 batch tensor로 model을 평가합니다. 이때, PyTorch의 핵심적인 모듈인 $torch.nn$ 을 사용하여 neural network를 만들고 loss function과 optimizer을 사용하여 model의 outputs을 비교합니다. 여기서 Fully connected layers, convolution layers, activation functions, loss functions을 확인할 수 있습니다. 이러한 구성 요소를 사용하여 그림의 untrained model에 구축하고 초기치를 설정할 수 있습니다. 이후, task에 최적화된 parameter를 가진 trained model을 얻을 수 있습니다. 이부분은 chapter 5에서 자세히 다루겠습니다.

## Production

TorchScript와 ONNX를 통해 미리 model을 compile하는 방법을 제공하고, 표준화된 형식으로 model을 호출할 수 있습니다. 이러한 기능들은 PyTorch의 production 배포 기능을 기반으로합니다. 이부분은 chapter 15에서 자세히 다루겠습니다.

---

# Summary

+ deep learning model은 examples에서 input과 원하는 output을 연결하는 방법을 자동으로 학습합니다.
+ Pytorch를 사용하여 neural network model을 효율적으로 구축하고 훈련할 수 있습니다.
+ Pytorch는 유연성과 속도에 집중하면서 overhead를 최소화합니다. 또한, 기본적으로 작업을 죽시 실행합니다.
+ TorchScript를 사용하면 model을 사전 compile하고 python 뿐만 아니라 C++ 프로그램 및 모바일 장치에서도 모델을 호출 할 수 있습니다.
+ 2017년 초 PyTorch가 출시된 이후로 deep learning tool의 생태계는 크게 통합되었습니다.
+ Pytorch는 deep learning project를 용이하게 하기 위해 여러 utility libraries를 제공합니다.

---

참고
1. [https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf][paper]

---

[paper]: https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf
