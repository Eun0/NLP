# Efficient Estimation of Word Representations in Vector Space

### 등록일 : 2013년 9월 7일

### 저자 : Google Inc


Word2Vec 논문

<br>

# Abstract


매우 큰 데이터로부터 단어의 continuous vector representation을 계산하는 두 혁신적인 모델을 제시한다  ( => CBOW와 Skip-gram )

단어 유사도(word similarity task)로 성능을 측정한다

이전의 제시된 모델에 비해 필요한 계산량이 현저히 적다 ( => 빨리 학습 가능 )

게다가 syntatic과 semantic word similarities 성능 좋다
<br>

# Introduction

예전에는 word를 atomic unit으로 씀 - 단어 사이의 유사성 알 수 없음

atomic에 3가지 장점이 있음

1. simplicity

2. robustness

3. 많은 데이터로 훈련된 간단한 모델이 적은 데이터로 훈련된 복잡한 모델보다 성능이 좋다

하지만 단점도 존재

1. 데이터 양이 제한된 경우 => 성능 not good

2. 데이터 양이 방대한 경우 => 간단한 모델 scaling up 해도 진전 x

따라서 보다 향상된 기법 필요

머신 러닝 기술들이 발전함으로써 복잡한 모델 train 가능 => 간단한 모델의 성능 능가함

<br>

## Goal of the Paper

goal : learning high-quality word vectors from huge data sets

quality 좀 더 자세히 보자면

비슷한 word는 가깝게, word가 multiple degree similarity를 가지게

다소 놀랍게도,

간단한 문법을 넘어선 word representation의 similarity를 찾았다

예를 들자면 "King"-"Man"+"Woman"을 하면 결과로 "Queen"을 내놓는다

단어들 사이의 linear regularities를 지키면서 새로운 모델을 발전시키는 방법으로 vector operations의 정확도를 최대화하고자 했다

성.공.적 이었고

어떻게 training time과 accuracy가 단어 벡터의 차원과 훈련 데이터의 양에 의존하는 지 얘기하겠다

<br>

# Model Architectures

LSA나 LDA와 같은 continuous vector를 만드는 많은 방법들이 있다.

우리는 neural network로 학습된 distributed representations of words에 집중하겠다. - LSA보다 linear regularities를 잘 보존, LDA는 계산량 너무 많음

모델 구조를 비교하기 위해 computational complexity를 정의하고,

computational complexity는 줄이면서 accuracy를 최대화해보겠다

모든 모델들의 training complexity는 다음과 같은 꼴이다

O = E x T x Q

- E : # of epochs
- T : # of words in training set
- Q : 모델 구조


모든 모델은 SGD와 backpropagation으로 학습한 것으로 가정

<br>

## Feedforward Neural Net Language Model (NNLM)

![NNLM Structure](https://github.com/Eun0/NLP/blob/master/img/nnlm.png)

(구조만 볼 것)

input,projection,hidden,output layers로 이루어져 있다

1. input : N개의 이전 단어들이 1-of-V 방식이로 인코딩된다

2. projection : NxD 차원의 shared projection matrix

3. output : softmax

NNML 구조는 projection과 hidden layer간의 계산때문에 복잡해진다

또한 hidden layer는 output에서 확률을 계산하는데 사용된다

따라서 각 training 마다 computational complexity는 다음과 같다

Q = NxD + NxDxH + HxV

- NxD : input-> projection

- NxDxH : projection -> hidden

- HxV : hidden -> output

이중에서 가장 dominating term은 __HxV__

__HxV__의 계산량을 줄이는 여러 방법 존재

1. Hierarchical version of softmax

2. Avoiding normalized models completely

따라서 1,2 방법으로 HxV는 어느정도 감소 가능

=> __NxDxH__ 가 주요 complexity


우리의 모델에서는 vocabulary는 Huffman binary tree를 이용해서 represent했고 hierarchical softmax를 사용했다

하지만 NxDxH 아직 해결 못함

Hidden layers가 없는 구조를 제시할 것이고 그렇기 때문에 sortmax의 효율성에 크게 의존한다

<br>

## Recurrent Neural Net Language Model(RNNLM)

![RNNLM Structure](https://github.com/Eun0/NLP/blob/master/img/rnnlm.png)

feedforward NNLM의 특정 한계점(ex.N을 명시)을 극복하고자 제시된 모델이 RNNLM이다

게다가 이론적으로 RNN이 더 복잡한 패턴을 효과적으로 represent

RNNLM은 input,hidden,output layers로 이루어져있다 (projection x)

이 모델의 computational complexity는 다음과 같다

Q = HxH + HxV

- HxH : input->hidden (사실 DxH인데 D가 H이다)
- HxV : hidden->output

=> HxH가 주요 complexity

<br>

# New Log-linear Models

기존의 Neural Network 기반 학습 방법에 비해 크게 달라진 것은 아니지만 계산량을 엄청나게 줄인 새로운 2가지 모델을 제시한다

앞서 본 것과 같이 주요 computational complexity는 non-linear hidden layer에서 발생했다 (따라서 없앰)

<br>

## Continuous Bag-of-Words (CBOW) 

![CBOW Structure](https://github.com/Eun0/NLP/blob/master/img/cbow.jpg)

주변 단어로 현재 단어 예측

input,projection,output layers로 이루어짐

computational complexity는 다음과 같다

Q = NxD + D x log_2(V)

<br>

## Continuous Skip-gram Model (Skip-gram)

![skip-gram Structure](https://github.com/Eun0/NLP/blob/master/img/skip-gram.jpg)

cbow와 비슷하다. 단 현재 단어로 주변 단어 예측

이 모델의 computational complexity는 다음과 같다

Q = Cx(D + D x log_2(V))

- C : maximum distance of the words


