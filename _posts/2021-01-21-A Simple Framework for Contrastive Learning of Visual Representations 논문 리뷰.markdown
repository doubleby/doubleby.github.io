---
layout: post
title: "A Simple Framework for Contrastive Learning of Visual Representations 논문 리뷰"
description: "Contrastive 방법과 3가지의 image augmentations을 사용한 SimCLR 논문 리뷰"
date: 2021-01-21 21:03:36 +0530
categories: Self-Supervised-Learning
mathjax : true
---
---

이번 포스트는 '[A Simple Framework for Contrastive Learning of Visual Representations][paper]' 논문을 리뷰하겠습니다. 이번 논문은 self-supervised learning임에도 불구하고 이전 논문에 비해 supervised learning과 성능면에서 아주 근소하게 차이밖에 나지 않아서 굉장히 의미있는 논문입니다. 또한, 방법론도 간단한 방법론을 사용하여 성능을 높여서 의미있다고 볼 수 있습니다.

---

# SimCLR

이전의 많은 방법들은 체험이나 경험에 의존하여 pretext task를 생각하였습니다. 하지만 이는 학습된 representations의 일반성을 제한할 수 있습니다. 또한, unsupervised learning의 대부분 방법들은 **generative** 와 **discriminative** 중 하나에 빠지게됩니다.

**Generative 방법들**은 input space의 fixels을 generate하거나 modeling하는 것을 학습합니다. 하지만, fixel level의 generate은 계산상 비싸고 representation learning에서 불필요할 수도 있습니다.

**Discriminative 방법들**은 supervised learning에 사용되는 것과 유사한 objective functions을 사용하여 representation을 학습합니다. Supervised learning과 유사한 objective functions을 사용하지만 unsupervised learning의 목적에 맞게, 라벨이 없는 데이터에서 pretext task를 수행하도록 network를 훈련시킵니다. 최근 discriminative 방법들은 latent space의 contrastive learning을 기반으로 합니다. 그리고 최근 방법인 contrastive learning을 기반으로 하는 discriminative 방법들이 좋은 성능을 보였습니다.

SimCLR은 visual representations의 contrastive learning을 기반으로 하는 간단한 framework이며, 이전의 방법들에 비해 높은 성능을 보였습니다. 하지만, 높은 성능에 비해 architecture가 간단하고 특별하지 않습니다. 그리고 아래와 같은 요소들을 통해 framework를 연구하였습니다.

1. 몇개의 data augmentation 작업으로 contrastive prediction task에서 효과적인 representations을 산출
2. Representation과 contrastive loss의 비선형 변형으로 representations의 퀄리티를 향상
3. Contrastive cross entropy loss를 사용한 representation을 통해 noramlization된 embeddings과 적절한 temperature 변수를 산출
4. 큰 batch size와 긴 학습 시간의 contrastive learning의 이점과 supervised learning과 비교. supervised learning과 마찬가지로 contrastive learning은 더 깊고 더 넓은 network에서 높은 성능 결과.

---

# Method

## The contrastive learning framework

최근 contrastive learning 방법에 영감을 받은 SimCLR은 latent space에서 contrastive loss를 통한 동일한 data의 서로 다른 augmented views 간의 일치를 최대화하여 representations을 학습합니다. 아래의 그림을 통해서 조금 더 자세히 알아 볼 수 있습니다.

![Framework](https://i.imgur.com/uuTL5I3.jpg "Framework")

1. 주어진 data example $x$를 무작위로 변형하는 확률적인 data augmentation 모듈은 주어진 data example과 관련된 2가지 views인 $\tilde{x_i}$와 $\tilde{x_j}$를 positive pair로 결과를 산출합니다. 산출하는 과정인 $t$~$\tau$과 ${t}'$~$\tau$은 논문에서 연속적으로 3가지의 간단한 augmentations을 적용합니다. 첫번째로, **Random Cropping** 을 수행 후 원래 크기로 다시 resize. 두번째로, **Random Color Distortions**. 세번째로, **Random Gaussian Blur**. 이 3가지의 조합이 높은 성능을 이끌어 냈습니다.

1. Augmented data examples인 $\tilde{x_i}$와 $\tilde{x_j}$의 representation을 추출하기 위해 **neural network base encoder**인 $f(\cdot)$ 을 구축하였습니다. 이때, SimCLR은 base encoder를 구축할때, 제약조건없이 다양한 network 구조를 사용할 수 있습니다. 논문에서는 다양한 network 구조 중에 $h_i = f(\tilde{x_i}) = ResNet(\tilde{x_i})$를 얻기 위해서 standard한 ResNet을 선택하였습니다. 이때, $h_i \in \mathbb{R}^d$은 average pooling layer후에 나온 output입니다.

1. Representation을 contrastive loss로 mapping하는 것으로는 작은 **neural network prejection head** $g(\cdot)$을 적용하였습니다. 그리고 $z_i = g(h_i) = W^{(2)}\sigma(W^{(1)}h_i)$를 얻기 위해 1개의 hidden layer를 가지는 MLP(Muli-Layer Perceptron)을 사용하였습니다. 이때, $\sigma$는 ReLU 비선형입니다. 이 방법을 통해 $h_i$들보다 더 좋은 contrastive loss $z_i$들을 찾을 수 있었습니다.

1. Contrastive loss function은 **contrastive prediction task**로 정의하였습니다. example인 $\tilde{x_i}$와 $\tilde{x_j}$의 positive pair를 포함한 $set$ $of$ $\tilde{x_k}$가 주어졌을때, contrastive prediction task는 $set$ $of$ $\tilde{x_k}_{k \neq i}$에서 $\tilde{x_j}$를 확인하는데 초점을 두었습니다.

1. N개의 example로 구성된 mini-batch를 무작위로 sampling하고 mini-batch에서 파생된 augmented example pair에 대한 contrastive prediction task를 정의하여 2N개의 data point를 생성하였습니다. 이때, **negative example를 sampling하지 않았습니다**. 대신에, positive pair가 주어졌을때 mini-batch내의 다른 2(N-1)개의 augmented example를 negative example로 취급하였습니다.

Framework의 mechanism에 이어서, **loss function**은 아래와 같이 정의가 됩니다.

![Loss function](https://i.imgur.com/gwl9i1R.jpg "Loss function")

이때, u와 v를 l2 normalized하는 dop product를 $sim(u, v) = u^{T}v/\left \| u \right \| \left \| v \right \|$ 라고 하고 example인 (i, j)의 positive pair의 loss function을 위와 같이 정의가 됩니다. 그리고 $\mathbb{1}$은 $k \neq i$인 경우 1로 평가되는 indicator function이며, $\tau$는 temperature parameter입니다. 학습을 수행하고 마지막 loss는 mini-batch에 있는 $(i, j)$와 $(j, i)$같이 모든 positive pairs를 거처셔 입력됩니다. 이 loss function을 논문에서는 **NT-Xent**(the normalized temperature-scaled cross entropy loss)라고 설명하였습니다.

이러한 framework를 코드로 표현한 것이 아래와 같습니다.

![Algorithm](https://i.imgur.com/JGIrxje.jpg "Algorithm")

## Training with large batch size

Framework를 간단히 유지하기 위해서 model을 Memory Bank로 학습하지 않았습니다. 대신에, batch size인 N을 256부터 8192까지 변경하였습니다. 하지만, 제일 큰 batch size인 8192 batch size는 양쪽( $\tilde{x_i}$와 $\tilde{x_j}$)의 augmentation views에서 positive pair당 16382개의 negative example를 산출하였습니다.

이처럼, 큰 batch size로 학습하는 것은 linear learning rate scaling과 standard SGD/Momentum을 사용할 때 불안정할 수도 있습니다. 이 문제를 해결하고 학습하는 과정을 안정적으로 수행하기 위해, **모든 batch size에 LARS optimizer를 사용하였습니다**.

### Global BN

Framework에 활용되는 표준적인 ResNets은 batch normalization을 사용했습니다. 또한, data 병렬 처리를 통한 분산 학습을 위해 BN의 평균과 분산은 장치마다 local로 집계되었습니다.

이때, contrastive learning에서 동일한 장치에서 positive pair가 계산될 때마다 model은 representation을 개선시키지 않고 prediction accuracy만 높이기 위해서 local information leakage를 활용할 수도 있습니다. 하지만, 이는 문제가 발생할수도 있기때문에 학습하는 동안 **모든 장치의 BN 평균 및 분산을 집계하여 문제를 해결하였습니다**.

## Evaluation protocol & default setting

+ Pretraining data
  - ImageNet ILSVRC-2012 dataset
  - CIFAR-10
+ Transfer learning
  - wide range of datasets
+ Evaluation
  - widely used linear evaluation protocol
+ Default setting
  - Data augmentation
    * random crop and resize with random flip
    * color distortions
    * Gaussian blur
  - RestNet-50 base encoder
  - 2-layer MLP projection head to project the representation to a 128-dim latent space
  - NT-Xent loss
  - LARS optimizer with learning rate of 4.8, weight decay of $10^{-6}$
  - batch size 4096 for 100 epochs
  - use linear warmup for the first 10 epochs, and decay the learning rate with the cosine decay schedule without restarts

---

# Data augmentation for contrastive representation learning

## Data augmentation defines predictive tasks

Data augmentation이 supervised와 unsupervised representation에 널리 사용되고 있지만, contrastive prediction task을 정의하는 방법은 체계적으로 잡혀있지 않습니다. 기존에 존재하는 많은 방법들은 model의 architecture를 바꿈으로써 contrastive prediction task를 정의하였습니다.

![Predictive task](https://i.imgur.com/onl0Lyv.jpg "Predictive task")

예를 들면 위 그림의 **(a)** 와 같이, network archituecture에서 receptive field를 제한하여 global-to-local view prediction을 정의하였습니다. 또 다른 예시인 그림 **(b)** 에서는 image를 분할하고 context aggregation network를 통해 인접한 view prediction을 정의하였습니다.

SimCLR은 model의 architecture를 바꾸거나 복잡한 방법이 아닌, image를 무작위로 crop하고 resize하는 방법을 사용하였습니다. 이러한 간단한 방법을 통해 위에서 언급한 2가지 방법을 포함하는 predictive task를 생성할 수 있습니다.

## Composition of data augmentation operations is crucial for learning good representations

다양한 augmentation 기법들은 연구하였고 위에서 언급한 **random crop with flip and reisze**, **color distortion**, **Gaussian blur**를 최종적으로 사용하였습니다. 사용된 augmentation의 기법들은 아래의 그림을 보시면 자세히 알 수 있습니다.

![Data augmentations](https://i.imgur.com/MsEHA3D.jpg "Data augmentations")

3가지 방법들을 사용하면 효과가 뛰어났지만, 3가지 방법들을 각각 독립적으로 하나씩만 사용하였을 때의 성능도 비교하였습니다. 이때, ImageNet을 사용하여 성능을 비교하려고 했지만 images의 size가 달랐습니다. 이 문제를 해결하기 위해, 첫번째로 무작위로 이미지를 crop, resize하였습니다. 두번째로 변환된 images을 **framework의 오직 하나 branch에만 적용하였습니다**. 세번째로 남아있는 다른 images은 $t(x_i) = x_i$처럼 그대로 두었습니다.

위와 같은 순서로 진행한 뒤, 각각의 방법을 linear evaluation으로 평가하였고 결과는 아래와 같습니다.

![Linear evaluation of data augmentations](https://i.imgur.com/HtaaTzM.jpg "Linear evaluation of data augmentations")

이 결과를 통해, 단일 방법을 사용해서는 좋은 representation을 학습할 수 없다는 것을 알 수 있었습니다. 그리고 augmentations을 구성할때, contrastive prediction task는 더 어려워졌습니다. 하지만, represnetation의 퀄리티는 드라마틱하게 향상되었습니다.

또한, 방법들 중 한쌍인 random cropping and random color distortion이 높은 성능을 보였습니다. 논문에서 추측하기로는 random cropping을 사용할때, image의 대부분 patch들이 비슷한 color 분포를 공유하는 문제때문이라고 생각했습니다. 이를 통해서 아래의 그림처럼 color histogram으로도 충분히 image가 구별되는 것을 확인할 수 있었습니다.

저자는 neural nets이 이러한 shortcut을 사용해서 predictive task를 풀지도 모른다고 생각했습니다. 그렇기 때문에 **일반화 할 수 있는 feature를 배우기 위해 color distortion과 cropping을 함께 쓰는 것이 중요했습니다**.

![Histograms of pixel intensities](https://i.imgur.com/Trrk8HV.jpg "Histograms of pixel intensities")

## Contrastive learning needs stronger data augmentation than supervised learning

Color augmentation의 중요성을 보여주기 위해, 아래의 표와 같이 color distortion의 강도를 수정하여 평가하였습니다.

![Color distortion strength](https://i.imgur.com/MSsIf4R.jpg "Color distortion strength")

결과를 보면, color distortion의 강도가 높아질수록 unsupervised model인 SimCLR은 linear evaluation이 향상시킵니다. 하지만, supervised model은 color distortion의 강도가 높아질수록 성능이 오히려 낮아졌습니다. 이 실험을 통해, **unsupervised contrastive learning은 supervised Learning보다 더욱 강한 color data augmentation으로부터 더 큰 이득을 얻을 수 있다는 것을 확인할 수 있습니다**.

---

# Architectures for encoder and head

## Unsupervised contrastive learning benefits (more) from bigger models

![Linear evaluation of models](https://i.imgur.com/aBQFfvQ.jpg "Linear evaluation of models")

위의 그림은 depth와 width가 증가함에 따라 성능도 향상되는 것을 보여줍니다. 그리고 model의 size가 증가함에 따라 SimCLR과 supervised learning의 linear classifiers evaluation의 성능 차이가 점점 작아지는 것을 확인할 수 있습니다. 즉, unsupervised learning은 supervised Learning보다 더 큰 model에서 더 많은 이득을 얻습니다.

## A nonlinear projection head improves the representation quality of the layer before it

![Linear evaluation of projection heads](https://i.imgur.com/ieXcV2f.jpg "Linear evaluation of projection heads")

이번에는 projection head인 $g(h)$를 포함하는 중요성에 대해 연구를 진행하였습니다. 위 그림은 head에 대한 3가지의 다른 architectures를 사용한 linear evaluation의 결과를 보여줍니다. head에 대한 3가지의 다른 architectures는 다음과 같습니다.

+ identity mapping (= None)
+ linear projection
+ default nonlinear projection with one additional hidden layer (and ReLU activation)

결과를 보면, nonlinear projection이 linear projection보다 3% 높았고 no projection보다 10% 높았습니다. Projection head를 사용했을때, output의 차원과 무관하게 비슷한 결과가 나왔습니다.

게다가, nonlinear projection을 사용하더라고 projection head 이전의 layer인 $h$는 layer 이후인 $z = g(h)$보다 10% 더 좋았습니다. 즉, **projection head 이전의 hidden layer가 head 이후보다 더 좋은 representation인 것을 의미합니다**.

논문에서는 nonlinear projection 이전의 representation을 사용하는 것의 중요성을 contrastive loss로부터 유발된 information loss때문이라고 추측합니다. 특히, $z = g(h)$는 data transformation에 영향을 받지 않도록 학습이 진행됩니다. 그러므로, $g$는 color나 물체의 orientation과 같은 downsteram task에 사용될 수도 있는 information를 제거할 수 있습니다. Nonlinear 변형인 $g(\cdot)$을 활용하여, 더 많은 information이 $h$에서 형성되고 유지할 수 있습니다.

이 추측을 검증하기 위해 $h$나 $g(h)$를 사용하여 pretraining 중에 적용되는 transformation을 예측하는 방법을 학습하는 실험을 수행하였습니다. 이때, $g(h) = W^{(2)}\sigma(W^{(1)}h)$으로 설정하였고 input과 output의 차원이 2048로 동일합니다. 그리고 아래의 표는 $g(h)$가 information를 잃는 반면에 $h$는 적용된 transformation에 대한 훨씬 많은 information를 포함하는 것을 확인할 수 있습니다.

![Accuracy of training additional MLP's on different representations to predict the transformation applied](https://i.imgur.com/dehHbW0.jpg  "Accuracy of training additional MLP's on different representations to predict the transformation applied")

---

# Loss functions and batch size

## Normalized cross entropy loss with adjustable temperature works better than alternatives

공통적으로 사용하는 contrastive loss functions인 logistic loss과 margin loss를 NT-Xent loss과 비교해보았습니다. 아래의 표는 loss function의 input에 대한 gradient와 objective function을 보여줍니다.

![Negative loss functions and gradients](https://i.imgur.com/VQNUrgF.png "Negative loss functions and gradients")

표의 gradient 부분을 보았을때, 아래와 같은 사항을 확인할 수 있습니다.
+ $l_2$ normalization는 temperature($\tau$)와 함께 다양한 examples에 효과적으로 가중치를 부여합니다.
+ Temperature는 model이 hard negatives에서 학습하는데 도움을 줍니다.
+ Cross entropy와 달리 다른 objective functions은 상대적인 hardness로 negatives의 값을 측정하지 않습니다.

그 결과, loss functions에 대해 semi-hard negative를 적용해야합니다. 즉, **모든 loss term에 대한 gradient를 계산하는 대신 semi-hard negative term을 사용하여 gradient를 계산할 수 있습니다** (loss margin 내에 있고 거리가 가장 가깝지만 positive인 예제보다 멀리 있는 것)

공평하게 비교하기 위해, 모든 loss functions에 대해 같은 $l_2$ normalization을 사용하였고 hyperparameters를 tune하여 아래의 표와 같이 best result를 얻었습니다. 표을 보면, 기존의 loss function에 $sh$ = semi-hard negative mining을 사용하여 도움이 되었지만 best result는 여전히 default인 NT-Xent loss보다 훨씬 안좋았습니다.

![Linear evaluation of different loss functions](https://i.imgur.com/Nm3p9qw.jpg "Linear evaluation of different loss functions")

이후, default인 NT-Xent에서 $l_2$ normalization과 temperature($\tau$)의 중요성을 테스트하였습니다. 아래의 표를 보았을때, noramlization과 적절한 temperature scaling없는 경우 상당히 낮은 성능을 보였습니다. 그리고, $l_2$ normalization 없이 contrastive task의 accuracy는 올라갔지만 representation의 결과는 linear evaluation에서 더 안좋았습니다.

![Linear evaluation with $l_2$ normalization and temperature](https://i.imgur.com/uONYyQ2.jpg "Linear evaluation with $l_2$ normalization and temperature")

## Contrastive learning benefits (more) from larger batch sizes and longer training

![Linear evaluation with different batch size and epochs](https://i.imgur.com/X9dAf5A.jpg "Linear evaluation with different batch size and epochs")

위 그림은 epoch가 다르게 model이 학습할때, batch size의 중요성을 보여줍니다. 그림을 통해, epoch가 작을때는 큰 batch size가 더 좋은 성능을 보였습니다. 하지만, epoch가 커질수록 batch가 무작위로 resampling되는 경우에 batch size별 성능의 차이가 점점 작아졌습니다.

Supervised Learning과 대조적으로, contrastive learning에서는 더 큰 batch size가 더 많은 negative example를 만들어 convergence를 촉진시킵니다. 즉, **동일한 정확도에 대해 더 작은 epoch가 필요합니다**. 그러므로, 오래 학습시키는 것 또한 많은 negative 예제를 만들어 결과를 개선시킵니다.

---

# Comparison with State-of-the-art

+ use ResNet-50 in 3 different hidden layer widths(width mulipliers of 1x, 2x, and 4x)
+ 1000 epochs

## Linear evaluation

![ImageNet accuracies of linear classifiers of methods](https://i.imgur.com/vgUS5we.jpg "ImageNet accuracies of linear classifiers of methods")

위의 표는 이전의 방법과 SimCLR의 성능을 비교한 표입니다. 결과를 통해, 특별한 architecture를 요구하는 이전 방법들과 비교했을때 standard한 network를 사용하여 더 좋은 결과를 산출하였습니다. SimCLR의 best result인 ResNet-50 (4x)가 supervised pretrained ResNet-50과 일치하는 것을 확인할 수 있습니다.

## Semi-supervised learning

Semi-supervised learning을 수행하기 위해, class가 균형있게 분포되어 있고 labeling이 된 ILSVRC-12 dataset의 1% 이나 10% sample을 사용하여 pretraining 시켰습니다. 간단하게 regularization없이 labeled된 data를 전체 base network에 fine-tune시켰습니다. 아래의 표는 최근 방법들과의 비교 결과입니다.

![ImageNet accuracy of models with few labels](https://i.imgur.com/fpmHDoX.jpg "ImageNet accuracy of models with few labels")

Supervised baseline은 augmentation을 포함한 hyper-parameters를 optimizer하였기 때문에 높은 성능을 보였습니다. SimCLR은 1%와 10%에서 높은 성능을 보였습니다.

## Transfer learning

Transfer learning을 수행하기 위해, feature extractor를 fix하고 12개의 natural image datasets를 fine-tuning함으로써 성능을 평가하였습니다. 성능을 평가할때, dataset별로 hyperparameters를 tune시키고 validation에서 best hyperparameters를 선택하였습니다. 아래의 표는 SimCLR ResNet (4x) model과의 비교 결과입니다.

![Comparison of transfer learning with 12 natural image datasets](https://i.imgur.com/K5EEp26.jpg "Comparison of transfer learning with 12 natural image datasets")

Fine-tune시켰을때, SimCLR이 5개의 datasets(Food, CIFAR10, SUN397, Cars, VOC2007)에서 supervised learning보다 높은 성능을 보였습니다. 다만, 2개의 datasets(Pets, Flowers)에서만 supervised learning이 더 높은 성능을 보였습니다. 나머지 5개 datases(CIFAR100, Birdsnap, Aircraft, DTD, Caltech-101)은 비슷한 성능을 보였습니다.

---

# Conclusion

이번 논문은 3가지의 data augmentation 방법들과 contrastive loss를 활용하여 높은 성능을 보였습니다. 그리고 이전의 방법들에 비해 복잡한 architecture가 아닌 많이 비교적 간단하게 활용되는 ResNet만을 사용하여 높은 성능을 보였습니다. 심지어, 이전 방법들에 비해 높은 성능이 아닌 동일한 architecture에서는 supervised learning과 비슷하거나 이전 방법들 보다 압도적으로 높은 성능을 보였습니다.

---

참고
1. [https://arxiv.org/pdf/2002.05709.pdf][paper]

---

[paper]: https://arxiv.org/pdf/2002.05709.pdf
