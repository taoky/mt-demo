import math
from data import EOS_IDX, get_dataset_tokenized, PAD_IDX, BOS_IDX, sacrebleu_metric

import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class WMT20(torch.utils.data.Dataset):
    def __init__(self, zh_encodings, en_encodings):
        self.zh_encodings = zh_encodings
        self.en_encodings = en_encodings
        assert len(zh_encodings) == len(en_encodings)

    def __getitem__(self, idx):
        return self.zh_encodings[idx], self.en_encodings[idx]

    def __len__(self):
        return len(self.zh_encodings)


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(
            embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True
        )

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(
            hidden_size + embed_size, hidden_size, n_layers, dropout=dropout
        )
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size, device=self.device)

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[: self.decoder.n_layers]
        output = trg.data[0, :]  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            output = trg.data[t] if is_teacher else output.data.max(1)[1]
        return outputs


def get_model(input_dim, output_dim, device):
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0
    enc = Encoder(input_dim, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(output_dim, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    return model


def main(args):
    zhtrain_encodings, entrain_encodings = get_dataset_tokenized("train", model="naive")
    zhval_encodings, enval_encodings = get_dataset_tokenized("eval", model="naive")
    zhtokenizer, entokenizer = get_dataset_tokenized("tokenizer", model="naive")

    # zh_transform = (
    #     lambda x: [zhvocab["<bos>"]]
    #     + [zhvocab[token.lower()] for token in x]
    #     + [zhvocab["<eos>"]]
    # )
    # en_transform = (
    #     lambda x: [envocab["<bos>"]]
    #     + [envocab[token.lower()] for token in x]
    #     + [envocab["<eos>"]]
    # )

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
        optimizer = optim.Adam(model.parameters())

        model.train()

        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        epochs = 3
        steps = 0
        for _ in range(epochs):
            for i, batch in enumerate(tqdm(trainloader)):
                src = batch[1].to(device)  # en
                trg = batch[0].to(device)  # zh
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(src, trg)
                    output_dim = output.shape[-1]
                    output = output[1:].view(-1, output_dim)
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
                    torch.save(checkpoint, f"results-s2s/checkpoint-{steps}.pt")

        torch.save(model.state_dict(), "results-s2s/model.pt")

    def eval_process(dataloader):
        model.eval()
        eval_loss = 0
        batches = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                src = batch[1].to(device)
                trg = batch[0].to(device)
                output = model(src, trg, 0)

                output_dim = output.shape[-1]
                output_c = output[1:].view(-1, output_dim)
                trg_c = trg[1:].view(-1)
                loss = criterion(output_c, trg_c)

                # print(output.size(), trg.size())

                _, topi = output.topk(1)
                topi = topi.squeeze()

                output = zhtokenizer.decode_batch(topi.T.tolist())
                target = [[i] for i in zhtokenizer.decode_batch(trg.T.tolist())]
                # print(output, target)

                sacrebleu_metric.add_batch(
                    predictions=output, references=target
                )

                eval_loss += loss.item()
                batches += 1
        eval_loss /= batches
        results = sacrebleu_metric.compute()["score"]
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
        zhtest_encodings, entest_encodings = get_dataset_tokenized("test", model="naive")
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
        # import spacy

        model.eval()
        with torch.no_grad():
            while True:
                x = input("> ")
                # x = [i.text for i in entokenizer(x)]
                # x = torch.tensor(en_transform(x), device=device).unsqueeze(0).T
                x = entokenizer.encode(x).ids
                x = transform(x).to(device).unsqueeze(0).T
                print(x)
                output = model(x, torch.tensor([[BOS_IDX] * 128], device=device).T, 0)
                output = output[1:]
                _, topi = output.topk(1)
                # print(topi)
                sentence_ids = []
                for i in topi:
                    if i == EOS_IDX:
                        break
                    sentence_ids.append(i)
                print(zhtokenizer.decode(sentence_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mt5 finetune")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", help="load an existing model")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.train and args.model:
        print("Conflicted options: --train and --model")

    main(args)
