# 实验记录


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


## 实验一

> 使用训练好的人脸模型进行fine tune，每5个epoch保存一次模型，共100个epoch，结束后分别测试每个保存模型的准确率，判断模型大概在什么时候收敛

- 实验目的  
  确定模型微调时收敛时间

- 实验结果  
  | epoch  | accuracy |
  | ------ | -------- |
  | epoch1 | -        |


## 实验二

> 不使用fine tune，从头开始训练，看看准确率怎么样，模型是否能收敛

- 实验目的  
  确定fine tune的过程是否正确，因为现在做fine tune的时候训练结束时总是提示 `--pretrained_model` 不是可以识别的命令


## 实验三

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


## 实验四

> 使用softmax loss在训练集上进行训练，等模型收敛时，锁定模型中除了分类层的所有参数，并在模型最后添加triplet loss继续训练，只更新模型的最后一层，用这个模型**在测试集上进行训练**，并用训练好的模型在测试集上进行测试

- Motivation  
  `Deep Face Recognition`这篇论文里提到使用分类模型开始训练可以使训练更快也更容易，但是由于最终的任务是当前人脸和特征库中的人脸进行**verification**,所以 `triplet loss` 是更直接的训练方式

- 实验目的
  这种方法的好处是训练用的pairs是在测试集上选出来的，相当于是使训练好的模型在测试集合上进行了优化，当测试集中添加或者删除少量样本时，是不影响使用三元组损失学到的这个仿射变换的

- 实验结果


## 实验五

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
