from rnn import WMT20
from torch import nn, optim
import torch
import math
from torch.nn import Transformer
from data import PAD_IDX, get_dataset_tokenized, BOS_IDX, EOS_IDX, sacrebleu_metric
import argparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src,
        trg,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_mask, src_key_padding_mask=None):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(self, tgt, memory, tgt_mask, memory_key_padding_mask=None):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def get_model(input_dim, output_dim, device):
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        input_dim,
        output_dim,
        FFN_HID_DIM,
    ).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def main(args):
    zhtrain_encodings, entrain_encodings = get_dataset_tokenized("train", model="naive")
    zhval_encodings, enval_encodings = get_dataset_tokenized("eval", model="naive")
    zhtokenizer, entokenizer = get_dataset_tokenized("tokenizer", model="naive")

    transform = lambda x: torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(x), torch.tensor([EOS_IDX])), 0
    )

    def collate_batch(batch):
        zh_list, en_list = [], []
        for (_zh, _en) in batch:
            # truncate
            _zh = _zh[:128]
            _en = _en[:128]
            zh_list.append(transform(_zh))
            en_list.append(transform(_en))

        return pad_sequence(zh_list, padding_value=PAD_IDX), pad_sequence(
            en_list, padding_value=PAD_IDX
        )

    # device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = WMT20(zhtrain_encodings, entrain_encodings)
    trainloader = DataLoader(
        trainset,
        batch_size=128,
        collate_fn=collate_batch,
        num_workers=16,
        pin_memory=True,
    )

    model = get_model(
        input_dim=entokenizer.get_vocab_size(),
        output_dim=zhtokenizer.get_vocab_size(),
        device=device,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    if args.model:
        model.load_state_dict(torch.load(args.model))

    if args.train:
        optimizer = optim.Adam(
            model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )

        model.train()

        use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        epochs = 5
        steps = 0
        for _ in range(epochs):
            for i, batch in enumerate(tqdm(trainloader)):
                src = batch[1].to(device)  # en
                trg = batch[0].to(device)  # zh
                trg_input = trg[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, trg_input, device
                )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(
                        src,
                        trg_input,
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                        src_padding_mask,
                    )
                    output = output.reshape(-1, output.shape[-1])
                    trg = trg[1:].view(-1)

                    loss = criterion(output, trg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                steps += 1
                if i % 100 == 0:
                    print(f"batch loss at {i} = {loss.item()}")
                if steps % 10000 == 0:
                    # save checkpoint
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                    }
                    torch.save(checkpoint, f"results-transformer/checkpoint-{steps}.pt")

        torch.save(model.state_dict(), "results-transformer/model.pt")

    def greedy_decode(
        model, src, src_mask, max_len, start_symbol, device, src_key_padding_mask=None
    ):
        src = src.to(device)
        src_mask = src_mask.to(device)

        # print(src.size(), src_mask.size())
        memory = model.encode(src, src_mask, src_key_padding_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len - 1):
            memory = memory.to(device)
            tgt_mask = generate_square_subsequent_mask(ys.size(0), device).type(
                torch.bool
            )
            # print(ys.size(), memory.size(), tgt_mask.size())
            out = model.decode(
                ys, memory, tgt_mask, memory_key_padding_mask=src_key_padding_mask
            )
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
            )
            if next_word == EOS_IDX:
                break
        return ys

    def eval_process(dataloader):
        model.eval()
        eval_loss = 0
        batches = 0
        with torch.no_grad():
            # loss
            for i, batch in enumerate(tqdm(dataloader)):
                src = batch[1].to(device)
                trg = batch[0].to(device)
                trg_input = trg[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, trg_input, device
                )
                output = model(
                    src,
                    trg_input,
                    src_mask,
                    tgt_mask,
                    src_padding_mask,
                    tgt_padding_mask,
                    src_padding_mask,
                )

                output_c = output.reshape(-1, output.shape[-1])
                trg_c = trg[1:].view(-1)
                loss = criterion(output_c, trg_c)

                for j in range(len(batch)):
                    s = src[:, j].unsqueeze(0).T
                    t = trg[:, j].unsqueeze(0).T
                    num_tokens = s.shape[0]
                    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
                    src_padding_mask = (s == PAD_IDX).transpose(0, 1)
                    # print(s)
                    # print(src_padding_mask)
                    sentence_ids = (
                        greedy_decode(
                            model,
                            s,
                            src_mask,
                            max_len=128,
                            start_symbol=BOS_IDX,
                            device=device,
                            src_key_padding_mask=src_padding_mask,
                        )
                        .flatten()
                        .tolist()
                    )
                    output = zhtokenizer.decode_batch([sentence_ids])
                    target = [[i] for i in zhtokenizer.decode_batch(t.T.tolist())]

                    sacrebleu_metric.add_batch(predictions=output, references=target)

                eval_loss += loss.item()
                batches += 1
            eval_loss /= batches
            print("Loss: ", eval_loss)

            results = sacrebleu_metric.compute(tokenize='zh')["score"]
            print("Sacre BLEU: ", results)

    if args.eval:
        evalset = WMT20(zhval_encodings, enval_encodings)
        evalloader = DataLoader(
            evalset,
            batch_size=512,
            collate_fn=collate_batch,
            num_workers=16,
            pin_memory=True,
        )
        eval_process(evalloader)

    if args.test:
        zhtest_encodings, entest_encodings = get_dataset_tokenized(
            "test", model="naive"
        )
        testset = WMT20(zhtest_encodings, entest_encodings)
        testloader = DataLoader(
            testset,
            batch_size=512,
            collate_fn=collate_batch,
            num_workers=16,
            pin_memory=True,
        )
        eval_process(testloader)
    if args.interactive:
        model.eval()
        with torch.no_grad():
            while True:
                x = input("> ")
                # x = [i.text for i in entokenizer(x)]
                # x = torch.tensor(en_transform(x), device=device).unsqueeze(0).T
                x = entokenizer.encode(x).ids
                x = transform(x).to(device).unsqueeze(0).T
                print(x)
                num_tokens = x.shape[0]
                src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
                sentence_ids = (
                    greedy_decode(
                        model,
                        x,
                        src_mask,
                        max_len=128,
                        start_symbol=BOS_IDX,
                        device=device,
                    )
                    .flatten()
                    .tolist()
                )
                print(zhtokenizer.decode(sentence_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("transformer seq2seq")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", help="load an existing model")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.train and args.model:
        print("Conflicted options: --train and --model")

    main(args)
