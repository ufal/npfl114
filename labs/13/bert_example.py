#!/usr/bin/env python3
import argparse

import numpy as np
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = transformers.TFAutoModel.from_pretrained("bert-base-multilingual-uncased")

dataset = [
    "Podmínkou koexistence jedince druhu Homo sapiens a společenství druhu Canis lupus je sjednocení akustické signální soustavy.",
    "U závodů na zpracování obilí, řízených mytologickými bytostmi je poměrně nízká produktivita práce vyvážena naprostou spolehlivostí.",
    "Vodomilní obratlovci nepatrných rozměrů nejsou ničím jiným, než vodomilnými obratlovci.",
]

print("Textual tokenization")
print([tokenizer.tokenize(sentence) for sentence in dataset])

print("Tokenization to IDs")
batch = [tokenizer.encode(sentence) for sentence in dataset]
print(batch)

max_length = max(len(sentence) for sentence in batch)
batch_ids = np.zeros([len(batch), max_length], dtype=np.int32)
batch_masks = np.zeros([len(batch), max_length], dtype=np.int32)
for i in range(len(batch)):
    batch_ids[i, :len(batch[i])] = batch[i]
    batch_masks[i, :len(batch[i])] = 1

result = model([batch_ids, batch_masks])
print([component.shape for component in result])
