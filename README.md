# Build tokenizer

Building a Byte-pair encoding tokenizer (BPE) for custom dataset in order to perform some NLP tasks with Hugging face Tokenizers and transformers libraries.

# Usage

Tried the pipeline on some african languages (Fongbe and Ewe)

This tutorial asume that you have all the sentences splited into different files named as \<language\>-sentences.txt.

# Instantiate and train tokenizer

### Import libraries

```python
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

```

### Create pipeline for training and model saving

Here we are training 3 tokenizers :

- one for ewe and fon (all in one)
- one for ewe specifically
- one for fon specifically

```python
args = parser.parse_args()

for f in ['ewe-fon', "ewe", "fon"]:

    # instantiate tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # splitting our inputs into words
    tokenizer.pre_tokenizer = Whitespace()

    # instantiate trainer
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=2
    )

    # get files
    files = [
        os.path.join(args.data_dir, f"{f}-sentences.txt")
    ]

    # train tokenizer
    tokenizer.train(files=files, trainer=trainer)

    # save tokenizer config file
    tokenizer.save(os.path.join(args.save_dir, f"tokenizer-{f}.json"))

```

# Use trained tokenizer

### Load trained tokenizer from the generated vocabulary (.json) file and tokenize new input sentence

```python

for f in ['ewe-fon', "ewe", "fon"]:
    print(f'Using {f} tokenizer : \n')
    try:
        tokenizer = Tokenizer.from_file(
            os.path.join(args.save_dir, f"tokenizer-{f}.json")
        )
        output = tokenizer.encode(
            "Gbadanu tɛgbɛ ɔ, Noah tuun ɖɔ e nɔ cɛ emi"
        )
        print(output.tokens)
        print(output.ids)
        print(output.offsets[9])
    except Exception as ex:
        print(ex)

    print("\n")

```

```
# outputs
Using ewe-fon tokenizer :

['Gbadanu', 'tɛgbɛ', 'ɔ', ',', 'Noah', 'tuun', 'ɖɔ', 'e', 'nɔ', 'cɛ', 'emi']
[4066, 2570, 142, 16, 5586, 1033, 274, 67, 197, 2301, 1522]
(35, 37)


Using ewe tokenizer :

['G', 'ba', 'da', 'nu', 'tɛ', 'gbɛ', 'ɔ', ',', 'Noah', 'tu', 'un', 'ɖɔ', 'e', 'nɔ', 'c', 'ɛ', 'emi']
[41, 329, 239, 165, 1867, 8619, 122, 16, 4645, 272, 3293, 462, 67, 182, 65, 125, 6886]
(22, 24)


Using fon tokenizer :

['Gbadanu', 'tɛgbɛ', 'ɔ', ',', 'Noah', 'tuun', 'ɖɔ', 'e', 'nɔ', 'cɛ', 'emi']
[1751, 1173, 112, 11, 6263, 514, 171, 57, 142, 1062, 761]
(35, 37)
```

# Run entire code

```bash
$ python build_tokenizer.py --save_dir <path to models directory> --data_dir <path to data directory>
```

# Author

- @dric2018

If you think this is useful, leave a little star.
