import opencc

converter = opencc.OpenCC("t2s.json")

with open("wiki/wikititles-v2.zh-en.tsv") as f:
    with open("wikititles-converted.tsv", "w") as fout:
        for l in f:
            zh, en = l.split("\t")
            en = en.strip()
            zh = converter.convert(zh).strip()
            fout.write(f"{en}\t{zh}\n")

with open("wiki/WikiMatrix.v1.en-zh.langid.tsv") as f:
    with open("wikimatrix-converted.tsv", "w") as fout:
        for l in f:
            _, en, zh, from_lang, to_lang = l.split("\t")
            if to_lang != "zh-Hans":
                zh = converter.convert(zh)
            en = en.strip()
            zh = zh.strip()
            fout.write(f"{en}\t{zh}\n")
