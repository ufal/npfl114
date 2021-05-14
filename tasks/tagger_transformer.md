### Assignment: tagger_transformer
#### Date: Deadline: May 24, 23:59
#### Points: 3 points
#### Examples: tagger_transformer_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_transformer.py](https://github.com/ufal/npfl114/tree/master/labs/11/tagger_transformer.py)
template, implement a Transformer encoder.

#### Examples Start: tagger_transformer_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_layers=0`
```
Epoch 1/5 loss: 1.9822 - accuracy: 0.4003 - val_loss: 0.8465 - val_accuracy: 0.7235
Epoch 2/5 loss: 0.6168 - accuracy: 0.8283 - val_loss: 0.5454 - val_accuracy: 0.8280
Epoch 3/5 loss: 0.2757 - accuracy: 0.9528 - val_loss: 0.4380 - val_accuracy: 0.8416
Epoch 4/5 loss: 0.1424 - accuracy: 0.9761 - val_loss: 0.4046 - val_accuracy: 0.8468
Epoch 5/5 loss: 0.0869 - accuracy: 0.9843 - val_loss: 0.3934 - val_accuracy: 0.8480
loss: 0.4082 - accuracy: 0.8472
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=1`
```
Epoch 1/5 loss: 1.6145 - accuracy: 0.4919 - val_loss: 0.4468 - val_accuracy: 0.8265
Epoch 2/5 loss: 0.1648 - accuracy: 0.9494 - val_loss: 0.5082 - val_accuracy: 0.8356
Epoch 3/5 loss: 0.0470 - accuracy: 0.9848 - val_loss: 0.6596 - val_accuracy: 0.8202
Epoch 4/5 loss: 0.0256 - accuracy: 0.9909 - val_loss: 0.5639 - val_accuracy: 0.8291
Epoch 5/5 loss: 0.0187 - accuracy: 0.9931 - val_loss: 0.5991 - val_accuracy: 0.8387
loss: 0.6571 - accuracy: 0.8292
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4`
```
Epoch 1/5 loss: 1.6144 - accuracy: 0.4935 - val_loss: 0.4483 - val_accuracy: 0.8250
Epoch 2/5 loss: 0.1598 - accuracy: 0.9522 - val_loss: 0.5113 - val_accuracy: 0.8374
Epoch 3/5 loss: 0.0449 - accuracy: 0.9853 - val_loss: 0.7293 - val_accuracy: 0.8174
Epoch 4/5 loss: 0.0267 - accuracy: 0.9906 - val_loss: 0.7311 - val_accuracy: 0.8071
Epoch 5/5 loss: 0.0189 - accuracy: 0.9931 - val_loss: 0.6877 - val_accuracy: 0.8417
loss: 0.8193 - accuracy: 0.8206
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch 1/5 loss: 1.7227 - accuracy: 0.4576 - val_loss: 0.4702 - val_accuracy: 0.8175
Epoch 2/5 loss: 0.2176 - accuracy: 0.9332 - val_loss: 0.4847 - val_accuracy: 0.8403
Epoch 3/5 loss: 0.0621 - accuracy: 0.9813 - val_loss: 0.6176 - val_accuracy: 0.8063
Epoch 4/5 loss: 0.0385 - accuracy: 0.9869 - val_loss: 0.5598 - val_accuracy: 0.8232
Epoch 5/5 loss: 0.0312 - accuracy: 0.9893 - val_loss: 0.6466 - val_accuracy: 0.8203
loss: 0.7229 - accuracy: 0.8065
```
#### Examples End:
