from data import get_dataset

import argparse

parser = argparse.ArgumentParser("mt5 finetune")
parser.add_argument("--train", action='store_true')
parser.add_argument("--model", help="load an existing model")
parser.add_argument("--eval", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--interactive", action='store_true')
args = parser.parse_args()

if args.train and args.model:
    print("Conflicted options: --train and --model")

zh_train, zh_val, zh_test, en_train, en_val, en_test = get_dataset()
