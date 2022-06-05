import matplotlib.pyplot as plt
import json, re
import numpy as np

REGEX = re.compile("batch loss at \d+ = (\d+\.?\d+?)")

files = {
    "RNN": "output-rnn-orthogonal.log",
    "Transformer": "output-transformer.log",
    "MT5 finetune": "output-mt5.log"
}

fig, ax = plt.subplots()

def process_naive(l):
    matches = REGEX.match(l)
    if matches:
        return float(matches[1])

def process_hf(l):
    try:
        if "'loss'" in l:
            l = l.replace('\'', '"')
            j = json.loads(l)
            return float(j['loss'])
    except Exception as e:
        print(l)
        raise e


def output(filename):
    losses = []
    if "mt5" in filename:
        process = process_hf
    else:
        process = process_naive
    with open(filename) as f:
        for l in f:
            x = process(l)
            if x:
                losses.append(x)
    return losses

for legend, filename in files.items():
    losses = output(filename)
    ax.plot(np.linspace(0, 5, len(losses)), losses, label=legend)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

plt.savefig("tmp.png")
