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

## 预处理

下载数据后，使用 wikiconvert.py 进行简繁转换，然后以下命令生成预处理数据集和 tokenizer：

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