from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
import datasets
import pickle
import argparse

EN_PREFIX = "translate English to Chinese: "
bleu_metric = datasets.load_metric("bleu")
sacrebleu_metric = datasets.load_metric("sacrebleu")
t5tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")

# naive seq2seq
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]


def get_dataset_raw(with_prefix=True):
    if with_prefix:
        prefix = EN_PREFIX
    else:
        prefix = ""
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
                    en_sentences.append(prefix + en)

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


def prepare_s2sdataset():
    zh_train, zh_val, zh_test, en_train, en_val, en_test = get_dataset_raw()
    import spacy
    from torchtext.vocab import build_vocab_from_iterator

    # prepare with:
    # python -m spacy download en_core_web_lg
    # python -m spacy download zh_core_web_lg
    # spacy.prefer_gpu()
    try:
        zhtokenizer = spacy.load("zh_core_web_lg")
        # zhtokenizer_jieba = Chinese.from_config({
        #     "nlp": {
        #         "tokenizer": {
        #             "segmenter": "jieba"
        #         }
        #     }
        # })
        entokenizer = spacy.load("en_core_web_lg")
    except OSError:
        print(
            """run
        python -m spacy download en_core_web_lg
        python -m spacy download zh_core_web_lg
        first before running this script!"""
        )

    zhtrain_encodings = [
        [j.text for j in i]
        for i in zhtokenizer.pipe(
            zh_train,
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"],
            n_process=-1,
        )
    ]
    zhval_encodings = [
        [j.text for j in i]
        for i in zhtokenizer.pipe(
            zh_val,
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"],
            n_process=-1,
        )
    ]
    zhtest_encodings = [
        [j.text for j in i]
        for i in zhtokenizer.pipe(
            zh_test,
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"],
            n_process=-1,
        )
    ]

    entrain_encodings = [
        [j.text for j in i]
        for i in entokenizer.pipe(
            en_train,
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
            n_process=-1,
        )
    ]
    enval_encodings = [
        [j.text for j in i]
        for i in entokenizer.pipe(
            en_val,
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
            n_process=-1,
        )
    ]
    entest_encodings = [
        [j.text for j in i]
        for i in entokenizer.pipe(
            en_test,
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
            n_process=-1,
        )
    ]

    # min_freq=3 to prevent embedding being too large
    zh_vocab = build_vocab_from_iterator(
        zhtrain_encodings, min_freq=5, specials=special_symbols, special_first=True
    )
    en_vocab = build_vocab_from_iterator(
        entrain_encodings, min_freq=5, specials=special_symbols, special_first=True
    )

    zh_vocab.set_default_index(UNK_IDX)
    en_vocab.set_default_index(UNK_IDX)

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

    with open("data/naive_vocab.pkl", "wb") as f:
        pickle.dump(
            {"zh": zh_vocab, "en": en_vocab},
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
        if model == "naive" and typ == "vocab":
            with open(f"data/{prefix}vocab.pkl", "rb") as f:
                p = pickle.load(f)
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
