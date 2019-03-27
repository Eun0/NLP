# Very Deep Convolutional Networks for Text Classification

### 등록일 : 2017년 1월 27일

### 저자 : Facebook AI

VDCNN 논문 번역 및 요약

# Abstract

Dominant approach for NLP task
⇒ __RNN__ and __CNN__

그러나 이런 architectures는 Computer Vision에 사용되는 NN보다 비교적 __"shallow"__

⇒ 논문에서 __deep__ 한 __VDCNN__ 제시

<br>

## VDCNN for text processing
-  character-level
- small Convolution + Pooling
- depth를 높이면서 성능 향상  (up to 29 conv layers) 
-  text processing에서 처음으로 __very deep__ CNN 쓰임

<br>

# 1. Introduction

개요
<br>

## NLP의 Goal :

**1.  Analyze** 
 ex. Sentiment analysis 

**2. Extract information**
  ex. Parsing

**3. Represent the same information differently** 
  ex. Machine translation

<br>

## The level of granularity (단위)

- characters
- subword units
- words
- sentences
- paragraphs


<br>

## CNN(Convolutional Neural Networks)

[사진]

- Feature extraction + Classification 을 한번에!
- __(convolution + (pooling)?)+__ 형태로 많이 씀 
 - layer **깊을 수록** 성능 __better__
 - Computer vision 분야에서 엄청 성공적
<br>

## Represent a sequence of words (e.g sentence)

- sequence of words의 특징
  - complicated syntactic and semantic relations
  - local and long-range dependencies

- 주로 문장을 sequence of tokens로 여기고, tokens을 RNN으로 처리

<br>

##  RNN for text processing

<img  src="http://i.imgur.com/jKodJ1u.png"  width="500"  height="400">

### 장점
- Tokens은 sequential order대로 처리됨
- internal states로 whole sequence를 ___"기억(memorize)"___
⇒ long-range dependencies 모델링



###  단점
 - __Generic__ learning machines for sequence processing
  ⇒ lacking task-specific structure
  
  - EX. 
    - 성능 : one-hidden layer < __deep problem-specific architecture__ 
     - why? search space 제한적, 효과적인 학습 가능 (with Gradient Descent) 
     
- (논문에서 제시한) 해결 방안 : Deep-CNN 
   - why CNN?
    Computer Vision에서 CNN 엄청 성공적,
    since image의 compositional structure
    ⇒ text도 비슷한 특징이 있다 ex. 문자 -> 단어,문구,문장, etc.  

<br>


# 2. Related work


## 초기의 방법

  2 단계 (feature extraction -> classification)
 
  - Typical features : bag-of-words, TF-IDF , ...  

  - More recently : projection -> combination (ex. mean) -> classification
    ⇒ token의 순서 개념이 무시돼서 잘 안됨
  
  <br>
   
##  Recursive Neural Networks (RecNN)

- parser를 이용
⇒ word embedding이 combine 될 순서를 명시함 

- top node is fed to the classifier

- RNN은 RecNN의 special case 

- ex. Input : ( ( (the)(country) )( (of)( (my)(birth) ) ) ) , [1 5]를 classifier에 넘긴다

![image](https://user-images.githubusercontent.com/33515697/53296531-16a79300-3854-11e9-95bf-25fcf243eafa.png)

 ## Histroy of CNN for NLP

- k-max pooling
- character-level
- CNN + RNN (fewer parameter?)

up to 6 conv layers

## In summary ...

- VGG나 ResNet 같은 deep CNN 사용 x
- Deeper networks were reported to not improve performance

⇒ 논문 VDCNN을 제시, 깊을 수록 성능 향상된 것을 보여줌

up to 29 conv layers

<br>

# 3. VDCNN Architecture

VGG와 ResNets을 참고

- Look-up table
- convolutional block & 3 pooling
- k-max pooling
- Fully connectec layers with ReLU
  - hidden units : 2048 
- Temporal batch normalization for regularization
  - not use drop-out  
-  Notation : 
    - __fc(Input,Output)__ : Fully connected layer
    - __3, Temp Conv , X__ : temporal convolutions with kernel size 3 and X feature maps  


![VDCNN Architecture](https://camo.githubusercontent.com/5a5c329d6edeb769887ac1604238ff3e99e298da/68747470733a2f2f66696c65732e736c61636b2e636f6d2f66696c65732d7072692f54314a3753434855372d4638544a5353454b442f5f5f5f5f5f5f5f2d312e706e673f7075625f7365637265743d37306639373230623865)



## Look-up table

- Input layer 역할
- contain the embedding of the s characters
- generates a 2D tensor of size ($f_0$,s)
  - $f_0$ : dimension of the input text  (like RGB)
  - s : input character 수, 논문에서는 s=1024


<br>

## Two Design rules
  
1. For the same output temporal resolution, the layers have the same # of feature maps
2.  When the temporal resolution is halved (pooling) , # of feature maps is doubled 
⇒ 메모리 사용량(memory footpring)을 줄여줌

## 3 Pooling operation

Design rules의 2와 관련된 부분

- The output of these convolutional blocks = 512 x $s_d$
  - $s_d=\frac{s}{2^p}$ , p : # of down-sampling operation  (여기서 3)

- convolution 결과가 input-text의 representaion ( $\because$  $s_d$ : constant )


## VGG, ResNet 기반

- 이전의 CNN for NLP
  - Shallow : up to 6 conv layers
  - Convolutions of different sizes to model short- and long-span relations

- VDCNN
  - 작고 동일한 size의 convolution을 많이 쌓았다 like VGG
  - shortcut 사용 like ResNet  

## k-max pooling

- first down-sampled to a fixed dimension
-  k개의 중요할 것 같은 features를 뽑는다 ( k=1인 경우 ⇒ max pooling )
- (512 x k) resulting features를 vector로 변환하여 FC에 input으로 넣는다
- 논문에서는 k=8 


## Convolutional Block

- [conv -> temporal BatchNorm -> ReLU] 2개가 이어진 형태 
-  모든 conv의 kernel size = 3 (like VGG)
-  padding으로 temporal resolution이 보존
- filter size 작아서 layers  깊게 쌓는 거 가능함 
![enter image description here](https://camo.githubusercontent.com/4cfc96783cd8930fad2c9dcb4fc46db1bc9b5031/68747470733a2f2f6169322d73322d7075626c69632e73332e616d617a6f6e6177732e636f6d2f666967757265732f323031362d31312d30382f383463613433303835366139323030306539306364373238343435636132323431633130646463332f332d466967757265322d312e706e67)


## Three types of down-sampling blocks

- between $K_i$ and $K_{i+1}$

- Three types :
  1.  The first convolutional layer of $K_{i+1}$ has stride 2 (ResNet-like)
  2. $K_i$ is followed by a k-max pooling layer 
     , where k is such that the resolution is halved
   3. $K_i$ is followed by max-pooling with kernel size 3 and stride 2 (VGG-like)

- 모든 방법 다 temporal resolution을 1/2배 해준다
- 마지막 conv layer의 resolution은 $s_d$


## Number of conv. layers per depth

- filter가 64,128,256,512개인 conv block 사용

- Depth가 9,17,29,49 네 가지 경우만 연구
⇒ Architecture 예시 그림은 depth가 17인 경우

- Best configurations

![image](https://user-images.githubusercontent.com/33515697/53324375-fa732700-3923-11e9-9a18-d89a3309d1a4.png)



# 4 Experimental evaluation

## 4.1 Tasks and data

- 8개의 large-scale data sets로 실험
- Task : Sentiment analysis, topic classification, news categorization
-  \# of classes : 2~14

![image](https://user-images.githubusercontent.com/33515697/53329993-c6eac980-3930-11e9-9cf8-2ac8742913f1.png)

- Thesaurus data augmentation 사용 x
- preprocessing  x  (lower-casing은 제외)
⇒ 그럼에도 불구하고 모든 data sets에서 outperform!!


## 4.2 Common model settings

- character-level
  - 66개 + special padding + space + unknown token = total 69 tokens 
    ```
    abcdefghijklmnopqrstuvwxyz0123456
    789-,;.!?:'"/|_#$%^&*~`+=<>()[]{}
    ```
  - character embedding size = 16


- Size of Input text 
  - s=1014
  - padded to s
  - larger text are truncated


- Training
  - optimizer : SGD
  - mini-batch : 128
  - initial learning rate : 0.01, momentum : 0.9  
  - temporal batch norm without dropout


## 4.3 Experimental results
![image](https://user-images.githubusercontent.com/33515697/53334876-5b5b2900-393d-11e9-9573-75126235ad33.png)

![image](https://user-images.githubusercontent.com/33515697/53334908-6b730880-393d-11e9-8c60-2d4decdb0842.png)



- 3 가지의 다른 depth와 3 가지의 다른 pooling type을 실험

  ⇒ 가장 깊은 network( 29 )와 max-pooling이 best!

- Results
  - Big data sets에서 특히 잘 됨
  - 깊을 수록 잘 됨
  -  Max-pooling이 3 가지 types 중에서 가장 잘 됨
  - SOTA CNN 보다 잘 됨
  - Short-cut이 accuracy degradation 감소
    ![image](https://user-images.githubusercontent.com/33515697/53334812-35358900-393d-11e9-8108-c71810f9dba2.png)

  - text classification with more classes sounds promising 



# 5. Conclusion

- Two design principle
  1. operate at the lowest atomic representaion of text  i.e. characters
  2. use a __deep stack__ of local operations

- Benefit of depth을 처음으로 입증

- 이 논문은 sentence classification을 집중적으로 했지만 다른 task에도 적용될 것이다  

