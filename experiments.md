# 实验记录

- [实验记录](#%E5%AE%9E%E9%AA%8C%E8%AE%B0%E5%BD%95)
  - [用于测试的pairs.txt的生成问题](#%E7%94%A8%E4%BA%8E%E6%B5%8B%E8%AF%95%E7%9A%84pairstxt%E7%9A%84%E7%94%9F%E6%88%90%E9%97%AE%E9%A2%98)
  - [数据集划分问题](#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%92%E5%88%86%E9%97%AE%E9%A2%98)
  - [实验一(fine tune with triplet loss)](#%E5%AE%9E%E9%AA%8C%E4%B8%80fine-tune-with-triplet-loss)
  - [实验二(train from scratch with triplet loss)](#%E5%AE%9E%E9%AA%8C%E4%BA%8Ctrain-from-scratch-with-triplet-loss)
  - [实验三(模型测试)](#%E5%AE%9E%E9%AA%8C%E4%B8%89%E6%A8%A1%E5%9E%8B%E6%B5%8B%E8%AF%95)
  - [实验四(train from scratch with softmax)](#%E5%AE%9E%E9%AA%8C%E5%9B%9Btrain-from-scratch-with-softmax)
  - [实验五](#%E5%AE%9E%E9%AA%8C%E4%BA%94)
  - [实验六](#%E5%AE%9E%E9%AA%8C%E5%85%AD)

## 用于测试的pairs.txt的生成问题

- 之前的做法
    1. positive pairs: 从某一类中的图片中前半部分选一张，后半部分选一张
    2. negative pairs: 选两个类，各取一张
- 现在的做法
    1. positive pairs: 从某一类中随机取两张，因为视频中大多数牛脸的出现顺序是先出现正脸，再出现侧脸，所以上面的取法会导致选出的模式可能大多数都是 **(一张正脸+一张侧脸）**
    2. negative pairs: 同上
- 目前使用的测试数据是2000个pair的图片

## 数据集划分问题

- trainset: 140 class
- testset: 40 class

## 实验一(fine tune with triplet loss)

- 使用 `pretrained model` + `triplet loss` 进行训练，得到的准确率最高是88%+， 此时的loss在**0.028**左右

## 实验二(train from scratch with triplet loss)

> 不使用fine tune，从头开始训练，看看准确率怎么样，模型是否能收敛

- 训练了**80**个epoch，准确率**88%+**，此时的loss是**0.62**左右
- 模型此时并没有完全收敛，loss还有很大的下降空间，使用作者提供的模型进行fine tune时，虽然准确率和这个模型差不多，但是那个loss只有**0.028**

## 实验三(模型测试)

> 用生成的最新的pairs.txt和马进之前训练好的模型进行测试

- Model1
  - dir: /wls/majin/models/facenet/20180405-122334
  - script: /wls/majin/develop/one_in_all/cowface/script/validate_cowface_old_model.sh
  - Accuracy: 0.87400+-0.01814
  - Validation rate: 0.29172+=0.04645
  - AUC: 0.953

- Model2
  - dir: /wls/majin/models/facenet/20180416-174504
  - script: /wls/majin/develop/one_in_all/cowface/script/validate_cowface_old_model.sh
  - Accuracy: 0.85450+-0.01005
  - Validation rate: 0.21582+-0.04406
  - AUC: 0.938

## 实验四(train from scratch with softmax)

> 使用softmax从头开始训练

## 实验五

> 这个要等到第二批数据处理好了再开始做

- 目前的数据量  
  第一批120头 + 第二批64头 = 184头

- 数据划分  
  训练集： 140头  
  测试集： 44头

- 实验范式  
  1. 将训练集划分成四部分  
      | split | class number |
      | ----- | :----------: |
      | s1    | 80           |
      | s2    | 20           |
      | s3    | 20           |
      | s4    | 20           |
  2. 首先以s1作为训练集开始训练，并在测试集上测试准确率，然后依次用s1+s2, s1+s2+s3, s1+s2+s3+s4作为训练集，依次评估模型在测试集上的准确率

- 实验目的  
  分析数据量的增加对模型效果影响多大

- 实验结果
  | train set   | accuracy |
  | :---------: | -------- |
  | s1          | -        |
  | s1+s2       | -        |
  | s1+s2+s3    | -        |
  | s1+s2+s3+s4 | -        |

## 实验六

> 使用softmax loss在训练集上进行训练，等模型收敛时，锁定模型中除了分类层的所有参数，并在模型最后添加triplet loss继续训练，只更新模型的最后一层，用这个模型**在测试集上进行训练**，并用训练好的模型在测试集上进行测试

- Motivation  
  `Deep Face Recognition`这篇论文里提到使用分类模型开始训练可以使训练更快也更容易，但是由于最终的任务是当前人脸和特征库中的人脸进行**verification**,所以 `triplet loss` 是更直接的训练方式

- 实验目的
  这种方法的好处是训练用的pairs是在测试集上选出来的，相当于是使训练好的模型在测试集合上进行了优化，当测试集中添加或者删除少量样本时，是不影响使用三元组损失学到的这个仿射变换的

- 实验结果