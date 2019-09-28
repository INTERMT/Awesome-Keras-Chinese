目录：
- [Keras简介](#一Keras简介)
- [Keras学习手册](#二Keras学习手册)
- [Keras学习视频](#三Keras学习视频)
- [Keras代码案例](#四Keras&NLP代码案例)
    - [Keras&NLP](#四Keras&NLP代码案例)
    - [Keras&CV](#五Keras&CV代码案例)
- [Keras项目](#六Keras项目)

### 一、Keras简介

Keras是Python中以CNTK、Tensorflow或者Theano为计算后台的一个深度学习建模环境。相对于其他深度学习的框架，如Tensorflow、Theano、Caffe等，Keras在实际应用中有一些显著的优点，其中最主要的优点就是Keras已经高度模块化了，支持现有的常见模型（CNN、RNN等），更重要的是建模过程相当方便快速，使用Keras可以快速地搭建深度网络，极大的加快了开发速度。此外，Keras具有用户友好性、模块化、易扩展、与Python协作友好的特点。

### 二、Keras学习手册

接下来为大家推荐几个笔者认为不错的Keras学习手册：

1. [Keras官方手册](https://keras.io/)。非常详细的官方文档，文档中详细的介绍了从Keras每个知识点的用法，一步步带你从入门到精通。
2. [Keras中文官方手册](https://keras.io/zh/)。该中文官方手册是对对Keras英文官方手册最好的还原，适合所有阶段的Keras学习者阅读。
3. [Keras中文文档](https://keras-cn.readthedocs.io/en/latest/)。另一个非官方的Keras中文文档，笔者花了近两年的时间在维护，文档也一直在更新，包含ConvLSTM2D、SimpleRNNCellKeras、GRUCell等最新的内容，非常用心的一份Keras文档。
4. [安装Keras库进行深度学习](http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning/)。国外一篇比较火的博客，旨在演示如何安装Keras库进行深度学习。
5. [黄海广博士力荐的Keras github项目](https://github.com/erhwenkuo/deep-learning-with-keras-notebooks)。这个github的repository主要是博主在学习Keras的一些记录及练习，满满都是干货，建议大家看一下。
6. [磐创AI Keras系列教程总结](http://www.tensorflownews.com/series/keras-tutorial/)。从CNN到RNN，以入门、基础为主的讲解，适合小白学习。

### 三、Keras学习视频

1. [Waterloo大学关于Keras的课程](https://www.youtube.com/watch?v=Tp3SaRbql4k)，该视频在YouTube上有很高的播放率，课程质量非常高。
2. [CERN使用Keras进行深度学习系列教程](http://cds.cern.ch/record/2157570?ln=en)，比较详细、权威的一个Keras系列教程视频。
3. [莫烦Keras视频教程](https://www.bilibili.com/video/av16910214/)，莫烦老师的视频在B站、YouTube上都有很高的播放量，强烈推荐给大家。
4. 再为大家推荐YouTube上另一个大佬[Sentdex的Keras教学视频](https://www.youtube.com/watch?v=wQ8BIBpya2k)，还配套有相应的文本教程和笔记：https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/ 。

### 四、Keras&NLP代码案例

1. [用LSTM在IMDB影评数据集做文本分类](https://github.com/fchollet/keras/blob/master/examples/imdb_lstm.py)
2. [路透社主题分类](https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py)
3. [LSTM做文本生成](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
4. [在IMDB数据集上使用FastText](https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py)
5. [基于LSTM的BABI数据集网络](https://github.com/keras-team/keras/blob/master/examples/reuters_mlp.py)
6. [预训练词向量](https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)
7. [字符级卷积神经网络做文本分类](https://github.com/johnb30/py_crepe)
8. [LSTM预测一个人的性别](https://github.com/divamgupta/lstm-gender-predictor)

### 五、Keras&CV代码案例

1. [使用CNN进行MNIST](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py)
2. [Inception V3](https://github.com/fchollet/keras/blob/master/examples/inception_v3.py)
3. [VGG16](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3	)
4. [FractalNet](https://github.com/snf/keras-fractalnet)
5. [可视问答](https://github.com/avisingh599/visual-qa)
6. [VGG-CAM](https://github.com/tdeboissiere/VGG16CAM-keras)
7. [ResNet 50](https://github.com/keras-team/keras/pull/3266/files)
8. [对象分割](https://github.com/abbypa/NNProject_DeepMask)
9. [fcn、segnet、u-net等常用的图像分割模型](https://github.com/divamgupta/image-segmentation-keras)

### 六、Keras项目
1. [RocAlphaGo](https://github.com/Rochester-NRT/RocAlphaG)，这个项目是DeepMind 2016年《自然》杂志的一个学生主导的实施项目，使用了Python)keras实现，代码清晰性更好。
2. [BetaGo](https://github.com/maxpumperla/betago)，项目是使用keras的深度学习Go机器人。
3. [DeepJazz](https://github.com/jisungk/deepjazz)，使用Keras深度学习驱动的爵士乐生成系统；
4. [dataset-sts](https://github.com/brmson/dataset-sts)，语义文本相似度数据集集线器。
5. [NMT-Keras](https://github.com/lvapeab/nmt-keras)，利用球面进行神经机器翻译；
6. [Headline generator](https://github.com/udibr/headlines)，利用循环神经网络独立生成新闻标题的实现。


