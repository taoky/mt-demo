from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from data import (
    # EN_PREFIX,
    get_dataset_tokenized,
    # bleu_metric,
    get_dataset_tokenized,
    sacrebleu_metric,
    slowtokenizer_batch_decode,
    t5tokenizer,
)
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import argparse
from tqdm.auto import tqdm
import multiprocessing


class WMT20(torch.utils.data.Dataset):
    def __init__(self, zh_encodings, en_encodings):
        self.zh_encodings = zh_encodings
        self.en_encodings = en_encodings

    def __getitem__(self, idx):
        res = {key: torch.tensor(val[idx]) for key, val in self.en_encodings.items()}
        res["labels"] = torch.tensor(self.zh_encodings["input_ids"][idx])
        return res

    def __len__(self):
        return len(self.zh_encodings["input_ids"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mt5 finetune")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--model",
        default="google/mt5-small",
        help="Finetune from google/mt5-small or load finetuned results-mt5/final",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    # zh_train, zh_val, zh_test, en_train, en_val, en_test = get_dataset()

    # t5tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    zhtrain_encodings, entrain_encodings = get_dataset_tokenized("train")
    zhval_encodings, enval_encodings = get_dataset_tokenized("eval")

    trainset = WMT20(zhtrain_encodings, entrain_encodings)
    valset = WMT20(zhval_encodings, enval_encodings)

    data_collator = DataCollatorForSeq2Seq(tokenizer=t5tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results-mt5",
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        # per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        logging_steps=100,
        save_steps=2500,
        # eval_accumulation_steps=8,
        # fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=valset,
        tokenizer=t5tokenizer,
        data_collator=data_collator,
    )

    def eval_process(model, dataset):
        accelerator = Accelerator()
        # batch size here can be higher than per_device_eval_batch_size
        # as we don't need to cal loss here
        dataloader = DataLoader(
            dataset,
            batch_size=2048,
            collate_fn=data_collator,
        )
        model, dataloader = accelerator.prepare(model, dataloader)
        model.eval()
        # sentencepiece is too slow as it only uses one core
        # so we have to create a multiprocess pool here
        with multiprocessing.get_context("fork").Pool() as pool:
            with torch.no_grad():
                for inputs in tqdm(dataloader):
                    generated_tokens = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=128,
                    )
                    targets = inputs["labels"]
                    # Gather all predictions and targets
                    generated_tokens_gathered = accelerator.gather(
                        generated_tokens
                    ).cpu()
                    targets_gathered = accelerator.gather(targets)
                    targets_gathered = torch.where(
                        targets_gathered != -100,
                        targets_gathered,
                        t5tokenizer.pad_token_id,
                    ).cpu()

                    decoded_preds = [
                        pred.strip()
                        for pred in slowtokenizer_batch_decode(
                            pool,
                            generated_tokens_gathered,
                        )
                    ]
                    decoded_labels = [
                        [label.strip()]
                        for label in slowtokenizer_batch_decode(
                            pool,
                            targets_gathered,
                        )
                    ]
                    # panic if empty label
                    for idx, _ in enumerate(decoded_labels):
                        if len(_[0]) == 0:
                            print(decoded_labels[idx - 1], decoded_labels[idx + 1])
                            assert 0, (
                                decoded_preds[idx],
                                t5tokenizer.decode(inputs["input_ids"][idx]),
                            )
                    # print(inputs["attention_mask"])
                    # print([t5tokenizer.decode(x) for x in inputs["input_ids"]])
                    # print(decoded_preds)
                    # print(decoded_labels)
                    # input()

                    sacrebleu_metric.add_batch(
                        predictions=decoded_preds, references=decoded_labels
                    )
        results = sacrebleu_metric.compute()["score"]
        print("Sacre BLEU: ", results)

    if args.train:
        trainer.train()
        model.save_pretrained("results-mt5/final")
    if args.eval:
        # get loss
        print(trainer.evaluate())
        # get bleu
        eval_process(model, valset)
    if args.test:
        zhtest_encodings, entest_encodings = get_dataset_tokenized("test")
        testset = WMT20(zhtest_encodings, entest_encodings)
        # get loss
        print(trainer.evaluate(testset))
        # get bleu
        eval_process(model, testset)
    if args.interactive:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        while True:
            x = input("> ")
            x = "translate English to Chinese: " + x
            x = t5tokenizer(
                x, max_length=128, truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            print(x)
            result = model.generate(x, max_length=128)
            print(result)
            print(t5tokenizer.decode(result[0], skip_special_tokens=True))
