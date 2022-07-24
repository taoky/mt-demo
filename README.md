# Machine Translation Lab

## 前言

某门 NLP 课的实验，虽然说是实验但是要求只是「实现机器翻译」这么笼统，而且给的资料里甚至包含了一份完整能运行的基于 TensorFlow 的实验代码和数据集……

这也太摆烂了，所以我想自己写一份试一试（也是基于这个原因，我准备 public 这个仓库）。我们选择的是端到端的、英文翻译到中文的任务，使用 RNN、（单个）Transformer 和 MT5 finetuning 三个模型，最后后两个都还挺好，RNN 的结果奇差（下面可以看到），可能是哪里有 bug，但是我调试了好几天都没找出来（加了 orthogonal 初始化也没用），所以我也懒得找了（

最开始分词是用的 spacy 然后扔到 torchtext.vocab 里面，但是实际训练的时候发现：

- 很慢
- vocab 太大了，显存不够用

于是最后的方案是用 ~~抱脸~~ Huggingface 的 Tokenizer，速度真的很快。

其实自己整一遍之后，虽然 `nn.Module` 是从别的地方拿过来的（见下方「参考代码」），但是至少能对 seq2seq 大致的流程有个了解。另外由于实验时间仓促，代码质量也无保证，所以仅供参考。

（虽然我对 RNN/Transformer 内部是如何工作的没有兴趣，但是看生成的句子还挺好玩的

## 参考代码

1. RNN: 
    - (MIT License) https://github.com/keon/seq2seq/blob/master/model.py
    - (MIT License) https://github.com/bentrevett/pytorch-seq2seq
    - https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
2. Transformer:
    - https://pytorch.org/tutorials/beginner/translation_transformer.html

## 参考效果

训练了 5 个 epoch。

Loss 越低越好，SacreBELU 越高越好。Loss 和 SacreBELU 保留四位小数。

|                  | RNN     | Transformer | MT5 finetune |
| ---------------- | ------- | ----------- | ------------ |
| 验证集 Loss      | 6.2052  | 3.4838      | 3.1327       |
| 验证集 SacreBELU | 11.4187 | 21.3078     | 18.0653      |
| 测试集 Loss      | 6.2017  | 3.4796      | 3.1320       |
| 测试集 SacreBELU | 11.4431 | 20.9615     | 18.0275      |

以下句子大部分来自英文维基百科（因此有可能在数据集中出现过，也有可能未出现过）。

| 英文句子                                                                                                                                                                                                                    | RNN                                                                                                                                                                                                                              | Transformer                                                                                                                                                                            | MT5 finetune                                                                                                                                                           | Google Translate                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| I have a dog.                                                                                                                                                                                                               | 狗狗狗狗。                                                                                                                                                                                                                       | 狗狗。                                                                                                                                                                                 | 有一只狗。                                                                                                                                                             | 我有一条狗。                                                                                                                                       |
| I don't have a cat.                                                                                                                                                                                                         | 没有猫猫猫。                                                                                                                                                                                                                     | 我没有猫。                                                                                                                                                                             | 我没有猫。                                                                                                                                                             | 我没有猫。                                                                                                                                         |
| Do you like me?                                                                                                                                                                                                             | 你像我我吗？                                                                                                                                                                                                                     | 吾何我？                                                                                                                                                                               | 你喜欢我吗?                                                                                                                                                            | 你喜欢我吗？                                                                                                                                       |
| I don't know how to do this.                                                                                                                                                                                                | 我不知道不知道如何做这件事。                                                                                                                                                                                                     | 我不知道如何做。                                                                                                                                                                       | 我不知道如何做这。                                                                                                                                                     | 我不知道该怎么做。                                                                                                                                 |
| Subscribe to your news feed                                                                                                                                                                                                 | 订阅新闻新闻新闻馈馈新闻新闻                                                                                                                                                                                                     | 订阅你的新闻                                                                                                                                                                           | 加入你的新闻频道                                                                                                                                                       | 订阅您的新闻提要                                                                                                                                   |
| He started using Linux 5 years ago.                                                                                                                                                                                         | 使用 5 年前使用 linux。                                                                                                                                                                                                          | 5 年前，他使用 linux。                                                                                                                                                                 | 他于 5 年前开始使用 Linux。                                                                                                                                            | 他 5 年前开始使用 Linux。                                                                                                                          |
| Natural language processing has its roots in the 1950s.                                                                                                                                                                     | 自然语言处理在 1950 年代 50 年代 50 年代。                                                                                                                                                                                       | 自然语言处理在 1950 年代具有根源。                                                                                                                                                     | 自然语言处理在 1950 年代的根源。                                                                                                                                       | 自然语言处理起源于 1950 年代。                                                                                                                     |
| Early computers were built to perform a series of single tasks, like a calculator.                                                                                                                                          | 早期的计算机计算机计算机计算机计算机，，，，计算机计算。                                                                                                                                                                         | 早期的计算机被建立起来，以执行一系列单任务，如计算器。                                                                                                                                 | 早期的计算机设计为进行一系列单一任务,如计算器。                                                                                                                        | 早期的计算机是为执行一系列单一任务而构建的，例如计算器。                                                                                           |
| In the United States, nineteen students and two teachers are killed in a mass shooting.                                                                                                                                     | 在美国，共有 19 名教师和两名教师在 2 枪击中死亡。                                                                                                                                                                                | 在美国，有 19 名学生和两名教师在大规模枪击中丧生。                                                                                                                                     | 在美国,有 9 名学生和 2 名教师在大规模枪击中被杀。                                                                                                                      | 在美国，19 名学生和 2 名教师在大规模枪击事件中丧生。                                                                                               |
| India reports its first suspected monkeypox case in a girl from Ghaziabad.                                                                                                                                                  | 印度报告首次讲述疑似疑似来自 zizizi 病例。                                                                                                                                                                                       | 印度在 ghaziabad 女孩中首次怀疑猴痘病例。                                                                                                                                              | 印度报道了第一名在格吉亚巴德的女孩的猫科案。                                                                                                                           | 印度报告了首例疑似猴痘病例，该病例来自加济阿巴德的一名女孩。                                                                                       |
| A recurrent neural network is a class of artificial neural networks where connections between nodes form a directed or undirected graph along a temporal sequence.                                                          | 神经神经网络网络（英语：神经网络网络网络网络，，，节点节点，，序列，，序列，或序列序列。                                                                                                                                         | 重复神经网络（re current neural network）是人工神经网络的一个类，节点之间连接形成一个定向或非定向图。                                                                                  | 动态神经网络(Recurrent Neural network)是一种人工神经网络,在节点之间连接的连接形成一个直接或非直接的图形。                                                              | 循环神经网络是一类人工神经网络，其中节点之间的连接沿时间序列形成有向或无向图。                                                                     |
| A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data.                                                               | 压器是一个是一个深度学习模型，，，，，，，，，，，，，数据，，，数据。                                                                                                                                                           | 跨前模型是一种深度学习模型，它采用了自我注意力机制，不同地重量的输入数据中每一个部分都具有意义。                                                                                       | 导体是一种深层次学习模型,采用自我注意机制,分辨输入数据的每一个部分的意义。                                                                                             | Transformer 是一种深度学习模型，采用 self-attention 机制，对输入数据各部分的重要性进行差分加权。                                                   |
| Google Neural Machine Translation is a neural machine translation system developed by Google and introduced in November 2016, that uses an artificial neural network to increase fluency and accuracy in Google Translate.  | google 神经机器翻译翻译是一种神经机器翻译翻译机器翻译，，2016 年 11 月，，，神经网络神经网络，，，翻译翻译，翻译翻译度和翻译度和准确度和准确度和准确度和翻译度和准确度和翻译度和准确度和翻译度和准确度和翻译度和准确度和翻译性。 | google neural machine translation（英语：google neural machine translation）是 google 开发的神经机器翻译系统，于 2016 年 11 月引入，使用人工神经网络来增加 google 翻译的流感和准确性。 | Google Neural Machine Translation(Google Neural Machine Translation)是 Google 开发的神经机器翻译系统,于 2016 年 11 月推出,使用人工神经网络来提高流畅和准确度。         | 谷歌神经机器翻译是谷歌开发并于 2016 年 11 月推出的神经机器翻译系统，它使用人工神经网络来提高谷歌翻译的流畅性和准确性。                             |
| Debian, also known as Debian GNU/Linux, is a Linux distribution composed of free and open-source software, developed by the community-supported Debian Project, which was established by Ian Murdock on August 16, 1993.    | debianbian（debianbianbianbianbian）是一个 linuxbianbianbianbianbiannulinuxnubian）是一个自由软件，由由由由由由由由由由由由由由 bian 开发的软体，由 linux 开发 bian debian，，1993 年 8 月 16 日，由 ianian murockock。          | debian，也被称为 debian gnu/linux，是 linux 发行版，由社区支持 debian 项目开发，于 1993 年 8 月 16 日由伊恩·默多克创立。                                                               | Debian(英语:Debian GNU/Linux)是 Linux 发行的 Linux 分配,由社区支持 Debian 计划(英语:Community-supported Debian Project)开发,由 Ian Murdock 于 1993 年 8 月 16 日成立。 | Debian，也称为 Debian GNU/Linux，是一个由免费和开源软件组成的 Linux 发行版，由 Ian Murdock 于 1993 年 8 月 16 日创立的社区支持的 Debian 项目开发。 |
| The anime significantly increased local tourism for the places featured, with several campgrounds reporting their number of visitors tripling.                                                                              | ，，，当地旅游旅游，，，，，，，，，，，，。                                                                                                                                                                                     | 动画大幅增加了当地旅游的潜力，一些营地的游客也报告了他们数量的三倍。                                                                                                                   | 该剧的活动影响了当地旅游业,许多营地报告了游客人数的倍增。                                                                                                              | 动画显着增加了当地旅游业的特色，几个露营地报告其游客人数增加了两倍。                                                                               |
| Along the way, they meet several other girls who are also interested in the outdoors, and begin a series of adventures on various mountains across Japan.                                                                   | 同时，他们遇到了许多其他，，，，，，，，，一系列冒险。                                                                                                                                                                           | 沿路，他们与其他几个女孩会面，他们感兴趣于门外，开始一系列冒险在日本各地。                                                                                                             | 在这次活动中,他们与其他的女孩一起参加过,并开始在日本各地的各种山脉的冒险。                                                                                             | 一路上，她们结识了其他几位对户外也很感兴趣的女孩，并开始在日本各地的山上进行一系列冒险。                                                           |
| One day, Yuko Yoshida wakes up with horns and a tail and learns that she is the descendant of a dark clan that was cursed into poverty by the opposing light clan.                                                          | 一天，，，，，，，，，尾巴，并发现尾巴，，，，。                                                                                                                                                                                 | 一天，吉田把角和尾巴醒来，得知她是一个黑暗的氏族后裔，被对立的轻氏所杀。                                                                                                               | 一天,川井一世在喉咙和尾巴上睡觉,并知道她是暗黑部落的祖先,被反对光族所杀害。                                                                                            | 有一天，吉田优子带着角和尾巴醒来，得知她是一个黑暗氏族的后裔，被对面的光明氏族诅咒而陷入贫困。                                                     |
| This was done partly by the invention of new words, but chiefly by eliminating undesirable words and by stripping such words as remained of unorthodox meanings, and so far as possible of all secondary meanings whatever. | 这部分词是由新词，，，，，，主要是消除一切词语，，，，，，，词语，，，，，，，，，，，，，，，一切意义。                                                                                                                         | 这部分是由新词发明，但主要是为了消除不受欢迎的字词，如保持正统含义，并且尽可能地尽可能地地地把所有副词都当作一种新词。                                                                 | 这部分是由新词发明的,但主要是因为消除不俗的词汇,并剥夺这些词汇仍然是非正义的,而且直到现在,所有其他的第二个词汇都可能存在。                                             | 这部分是通过发明新词来完成的，但主要是通过消除不受欢迎的词和剥离这些词的非正统含义，并尽可能去除所有次要含义。                                     |

## 安装

requirements.txt 中包含了实验整个过程安装的包，注意其中一些包后来没有用到（例如 spacy）。

以下这些包是必须安装的，建议在虚拟环境中完成：

```
torch
tokenizers
transformers
matplotlib
numpy
sacrebleu
scipy
scikit-learn
sentencepiece
protobuf
tqdm
```

## 获取数据集

实验使用的数据集需要前往 <https://statmt.org/wmt20/translation-task.html> 下载，放置于 data 目录中。每个数据集可能需要解压缩，共包含以下文件：

- news-commentary-v15.en-zh.tsv
- wikititles-v2.zh-en.tsv
- WikiMatrix.v1.en-zh.langid.tsv

后两者放于 data/wiki 目录后，执行 data 下的 wikiconvert.py 文件生成简体数据集。

## 预处理

使用以下命令生成预处理数据集和 tokenizer：

```console
python data.py --t5  # MT5 相关预处理
python data.py --s2s  # 我们实现的 seq2seq 的预处理
```

## 训练

先创建 `results-mt5`, `results-rnn`, `results-transformer` 三个目录，然后：

```console
python rnn.py --train
python transformer.py --train
python mt5.py --train
```

训练不同的模型。Batch size 参照 40G 显存配置，可能需要修改（以下同理）。

## 评估

```console
python rnn.py --model results-rnn/model.pt --eval  # --eval 是在验证集上运行，换成 --test 就是在测试集上运行，以下同理
python transformer.py --model results-transformer/model.pt --eval
python mt5.py --model results-mt5/final --eval
```

## 交互式生成

```console
python rnn.py --model results-rnn/model.pt --interactive
python transformer.py --model results-transformer/model.pt --interactive
python mt5.py --model results-mt5/final --interactive
```

## 绘制 train loss 变化曲线

参考 graph.py，注意需要在训练时重定向脚本的 stdout 和 stderr 到文件中。

## 可以改进的地方

1. WikiMatrix 没有按照数据集中的 margin score 筛掉翻译不正确的句子对；
2. 对中文用 BPE 来 tokenization 是不太合适的：完全无关的字可能也会被凑到一起；
3. 句子生成没有用 beam search；
4. 没有测试不同超参数下模型的变化，验证集其实也没用于改进参数设置。
