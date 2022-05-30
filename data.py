from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
import datasets
import pickle
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# EN_PREFIX = "translate English to Chinese: "
bleu_metric = datasets.load_metric("bleu")
sacrebleu_metric = datasets.load_metric("sacrebleu")
t5tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

# naive seq2seq
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
# bytedecoder = ByteLevelDecoder()


def get_custom_tokenizer(lang):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=special_symbols)
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    if lang == "en":
        tokenizer.pre_tokenizer = Whitespace()
    elif lang == "zh":
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()
    else:
        raise ValueError(f"unknown lang {lang}")
    return tokenizer, trainer


def get_dataset_raw(with_prefix=True):
    # prefix is unnecessary in MT5 single task finetune
    # if with_prefix:
    #     prefix = EN_PREFIX
    # else:
    #     prefix = ""
    zh_sentences, en_sentences = [], []
    files = [
        "news-commentary-v15.en-zh.tsv",
        "wikititles-converted.tsv",
        "wikimatrix-converted.tsv",
    ]
    for fname in files:
        with open(f"data/{fname}") as f:
            for i in f:
                en, zh = i.split("\t")
                en = en.strip()
                zh = zh.strip()
                if len(en) != 0 and len(zh) != 0:
                    zh_sentences.append(zh)
                    # en_sentences.append(prefix + en)
                    en_sentences.append(en)

    zh_train, zh_test, en_train, en_test = train_test_split(
        zh_sentences,
        en_sentences,
        test_size=0.2,
        random_state=1,
        shuffle=True
        # shuffle=False
    )
    zh_train, zh_val, en_train, en_val = train_test_split(
        zh_train,
        en_train,
        test_size=0.125,
        random_state=1,
        shuffle=True
        # shuffle=False
    )
    return zh_train, zh_val, zh_test, en_train, en_val, en_test


def prepare_t5dataset():
    zh_train, zh_val, zh_test, en_train, en_val, en_test = get_dataset_raw()
    # t5tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    with t5tokenizer.as_target_tokenizer():
        zhtrain_encodings = t5tokenizer(zh_train, max_length=128, truncation=True)
        zhval_encodings = t5tokenizer(zh_val, max_length=128, truncation=True)
        zhtest_encodings = t5tokenizer(zh_test, max_length=128, truncation=True)

    entrain_encodings = t5tokenizer(en_train, max_length=128, truncation=True)
    enval_encodings = t5tokenizer(en_val, max_length=128, truncation=True)
    entest_encodings = t5tokenizer(en_test, max_length=128, truncation=True)

    with open("data/train_tokenized.pkl", "wb") as f:
        pickle.dump(
            {"zh": zhtrain_encodings, "en": entrain_encodings},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with open("data/val_tokenized.pkl", "wb") as f:
        pickle.dump(
            {"zh": zhval_encodings, "en": enval_encodings},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    with open("data/test_tokenized.pkl", "wb") as f:
        pickle.dump(
            {"zh": zhtest_encodings, "en": entest_encodings},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


# slow and generate a very large vocab not suitable for training
# def prepare_s2sdataset():
#     zh_train, zh_val, zh_test, en_train, en_val, en_test = get_dataset_raw(
#         with_prefix=False
#     )
#     import spacy
#     from spacy.lang.zh import Chinese
#     from torchtext.vocab import build_vocab_from_iterator


#     # zhtokenizer = spacy.load("zh_core_web_lg")
#     zhtokenizer = Chinese.from_config({
#         "nlp": {
#             "tokenizer": {
#                 "segmenter": "jieba"
#             }
#         }
#     })
#     entokenizer = spacy.blank("en")

#     zhtrain_encodings = [
#         [j.text.lower() for j in i]
#         for i in zhtokenizer.pipe(
#             zh_train,
#             disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"],
#             n_process=64,
#             batch_size=5000,
#         )
#     ]
#     zhval_encodings = [
#         [j.text.lower() for j in i]
#         for i in zhtokenizer.pipe(
#             zh_val,
#             disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"],
#             n_process=64,
#             batch_size=5000,
#         )
#     ]
#     zhtest_encodings = [
#         [j.text.lower() for j in i]
#         for i in zhtokenizer.pipe(
#             zh_test,
#             disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"],
#             n_process=64,
#             batch_size=5000,
#         )
#     ]

#     entrain_encodings = [
#         [j.text.lower() for j in i]
#         for i in entokenizer.pipe(
#             en_train,
#             disable=[
#                 "tok2vec",
#                 "tagger",
#                 "parser",
#                 "attribute_ruler",
#                 "lemmatizer",
#                 "ner",
#             ],
#             n_process=64,
#             batch_size=5000,
#         )
#     ]
#     enval_encodings = [
#         [j.text.lower() for j in i]
#         for i in entokenizer.pipe(
#             en_val,
#             disable=[
#                 "tok2vec",
#                 "tagger",
#                 "parser",
#                 "attribute_ruler",
#                 "lemmatizer",
#                 "ner",
#             ],
#             n_process=64,
#             batch_size=5000,
#         )
#     ]
#     entest_encodings = [
#         [j.text.lower() for j in i]
#         for i in entokenizer.pipe(
#             en_test,
#             disable=[
#                 "tok2vec",
#                 "tagger",
#                 "parser",
#                 "attribute_ruler",
#                 "lemmatizer",
#                 "ner",
#             ],
#             n_process=64,
#             batch_size=5000,
#         )
#     ]

#     # min_freq=3 to prevent embedding being too large
#     zh_vocab = build_vocab_from_iterator(
#         zhtrain_encodings, min_freq=4, specials=special_symbols, special_first=True
#     )
#     en_vocab = build_vocab_from_iterator(
#         entrain_encodings, min_freq=4, specials=special_symbols, special_first=True
#     )

#     zh_vocab.set_default_index(UNK_IDX)
#     en_vocab.set_default_index(UNK_IDX)

#     with open("data/naive_train_tokenized.pkl", "wb") as f:
#         pickle.dump(
#             {"zh": zhtrain_encodings, "en": entrain_encodings},
#             f,
#             protocol=pickle.HIGHEST_PROTOCOL,
#         )

#     with open("data/naive_val_tokenized.pkl", "wb") as f:
#         pickle.dump(
#             {"zh": zhval_encodings, "en": enval_encodings},
#             f,
#             protocol=pickle.HIGHEST_PROTOCOL,
#         )

#     with open("data/naive_test_tokenized.pkl", "wb") as f:
#         pickle.dump(
#             {"zh": zhtest_encodings, "en": entest_encodings},
#             f,
#             protocol=pickle.HIGHEST_PROTOCOL,
#         )

#     with open("data/naive_vocab.pkl", "wb") as f:
#         pickle.dump(
#             {"zh": zh_vocab, "en": en_vocab},
#             f,
#             protocol=pickle.HIGHEST_PROTOCOL,
#         )


def prepare_s2sdataset():
    zh_train, zh_val, zh_test, en_train, en_val, en_test = get_dataset_raw(
        with_prefix=False
    )
    zh_tokenizer, zh_trainer = get_custom_tokenizer("zh")
    en_tokenizer, en_trainer = get_custom_tokenizer("en")
    zh_tokenizer.train_from_iterator(zh_train, zh_trainer)
    en_tokenizer.train_from_iterator(en_train, en_trainer)

    zh_tokenizer.save("data/naive_zh_tokenizer.json")
    en_tokenizer.save("data/naive_en_tokenizer.json")

    zhtrain_encodings = [i.ids for i in zh_tokenizer.encode_batch(zh_train)]
    entrain_encodings = [i.ids for i in en_tokenizer.encode_batch(en_train)]
    zhval_encodings = [i.ids for i in zh_tokenizer.encode_batch(zh_val)]
    enval_encodings = [i.ids for i in en_tokenizer.encode_batch(en_val)]
    zhtest_encodings = [i.ids for i in zh_tokenizer.encode_batch(zh_test)]
    entest_encodings = [i.ids for i in en_tokenizer.encode_batch(en_test)]

    with open("data/naive_train_tokenized.pkl", "wb") as f:
        pickle.dump(
            {"zh": zhtrain_encodings, "en": entrain_encodings},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with open("data/naive_val_tokenized.pkl", "wb") as f:
        pickle.dump(
            {"zh": zhval_encodings, "en": enval_encodings},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with open("data/naive_test_tokenized.pkl", "wb") as f:
        pickle.dump(
            {"zh": zhtest_encodings, "en": entest_encodings},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def get_dataset_tokenized(typ, model="t5"):
    prefix = ""
    if model == "naive":
        prefix = "naive_"
    if typ == "train":
        with open(f"data/{prefix}train_tokenized.pkl", "rb") as f:
            p = pickle.load(f)
    elif typ == "eval":
        with open(f"data/{prefix}val_tokenized.pkl", "rb") as f:
            p = pickle.load(f)
    elif typ == "test":
        with open(f"data/{prefix}test_tokenized.pkl", "rb") as f:
            p = pickle.load(f)
    else:
        # if model == "naive" and typ == "vocab":
        #     with open(f"data/{prefix}vocab.pkl", "rb") as f:
        #         p = pickle.load(f)
        if model == "naive" and typ == "tokenizer":
            zh_tokenizer = Tokenizer.from_file("data/naive_zh_tokenizer.json")
            en_tokenizer = Tokenizer.from_file("data/naive_en_tokenizer.json")
            p = {"zh": zh_tokenizer, "en": en_tokenizer}
        else:
            raise ValueError("unknown type")
    return p["zh"], p["en"]


def _tokenize_decode(
    seq,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
    **kwargs,
):
    return t5tokenizer.decode(
        seq,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        **kwargs,
    )


def slowtokenizer_batch_decode(
    pool,
    seq,
):
    return list(pool.map(_tokenize_decode, seq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5", action="store_true")
    parser.add_argument("--s2s", action="store_true")
    args = parser.parse_args()
    if args.t5:
        prepare_t5dataset()
    elif args.s2s:
        prepare_s2sdataset()
