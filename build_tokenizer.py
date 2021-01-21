# base tutorial : https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#training-the-tokenizer
# train a Byte-pair encoding tokenizer (BPE)
# training the tokenizer means it will learn merge rules by:
#
# Start with all the characters present in the training corpus as tokens.

#  Identify the most common pair of tokens and merge it into one token.

#  Repeat until the vocabulary (e.g., the number of tokens) has reached the size we want.

from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', "-sd", type=str, default='./save')
parser.add_argument('--data_dir', "-d", type=str, default='./data')

if __name__ == "__main__":

    args = parser.parse_args()
    # print(args)
    # sys.exit()
    # instantiate tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # splitting our inputs into words
    tokenizer.pre_tokenizer = Whitespace()

    # instantiate trainer
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=30000,
        min_frequency=1
    )

    # get files
    files = [
        f"{args.data_dir}/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]
    ]

    # train tokenizer
    tokenizer.train(files=files, trainer=trainer)

    # save tokenizer config file
    tokenizer.save(f"{args.save_dir}/tokenizer-wiki.json")

    # use tokenizer
    # load trained tokenizer
    tokenizer.from_file(f"{args.save_dir}/tokenizer-wiki.json")
    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.tokens)
    print(output.ids)
    print(output.offsets[9])
