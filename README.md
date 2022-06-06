# Machine Translation Lab

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
