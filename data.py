from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
import datasets
import pickle
from functools import partial

EN_PREFIX = "translate English to Chinese: "
bleu_metric = datasets.load_metric("bleu")
sacrebleu_metric = datasets.load_metric("sacrebleu")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")


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
    # tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    with tokenizer.as_target_tokenizer():
        zhtrain_encodings = tokenizer(zh_train, max_length=128, truncation=True)
        zhval_encodings = tokenizer(zh_val, max_length=128, truncation=True)
        zhtest_encodings = tokenizer(zh_test, max_length=128, truncation=True)

    entrain_encodings = tokenizer(en_train, max_length=128, truncation=True)
    enval_encodings = tokenizer(en_val, max_length=128, truncation=True)
    entest_encodings = tokenizer(en_test, max_length=128, truncation=True)

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


def get_dataset_tokenized(typ):
    if typ == "train":
        with open("data/train_tokenized.pkl", "rb") as f:
            p = pickle.load(f)
    elif typ == "eval":
        with open("data/val_tokenized.pkl", "rb") as f:
            p = pickle.load(f)
    elif typ == "test":
        with open("data/test_tokenized.pkl", "rb") as f:
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
    return tokenizer.decode(
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
    prepare_t5dataset()
