from data import get_dataset_tokenized, PAD_IDX

import argparse
from torchtext.vocab import Vocab
import torch
from torch import nn, optim
import torch.functional as F
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class WMT20(torch.utils.data.Dataset):
    def __init__(self, zh_encodings, en_encodings):
        self.zh_encodings = zh_encodings
        self.en_encodings = en_encodings

    def __getitem__(self, idx):
        return self.zh_encodings[idx], self.en_encodings[idx]

    def __len__(self):
        return len(self.zh_encodings)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def get_model(input_dim, output_dim, device):
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    enc = Encoder(input_dim, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(output_dim, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    return model


def main(args):
    zhtrain_encodings, entrain_encodings = get_dataset_tokenized("train", model="naive")
    zhval_encodings, enval_encodings = get_dataset_tokenized("eval", model="naive")
    zhvocab, envocab = get_dataset_tokenized("vocab", model="naive")

    zh_transform = (
        lambda x: [zhvocab["<bos>"]] + [zhvocab[token] for token in x] + [zhvocab["<eos>"]]
    )
    en_transform = (
        lambda x: [envocab["<bos>"]] + [envocab[token] for token in x] + [envocab["<eos>"]]
    )

    def collate_batch(batch):
        zh_list, en_list = [], []
        for (_zh, _en) in batch:
            zh_list.append(torch.tensor(zh_transform(_zh)))
            en_list.append(torch.tensor(en_transform(_en)))
        return pad_sequence(zh_list, padding_value=PAD_IDX), pad_sequence(
            en_list, padding_value=PAD_IDX
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = WMT20(zhtrain_encodings, entrain_encodings)
    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_batch)

    model = get_model(input_dim=len(envocab), output_dim=len(zhvocab), device=device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(tqdm(trainloader)):
        src = batch[1].to(device)  # en
        trg = batch[0].to(device)  # zh
        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()



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
