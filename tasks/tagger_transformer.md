### Assignment: tagger_transformer
#### Date: Deadline: May 09, 7:59 a.m.
#### Points: 3 points
#### Examples: tagger_transformer_examples
#### Tests: tagger_transformer_tests

This assignment is a continuation of `tagger_we`. Using the
[tagger_transformer.py](https://github.com/ufal/npfl114/tree/master/labs/11/tagger_transformer.py)
template, implement a Pre-LN Transformer encoder.

#### Examples Start: tagger_transformer_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_layers=0`
```
Epoch 1/5 loss: 1.5585 - accuracy: 0.5335 - val_loss: 0.8724 - val_accuracy: 0.7128
Epoch 2/5 loss: 0.5397 - accuracy: 0.8519 - val_loss: 0.5609 - val_accuracy: 0.8247
Epoch 3/5 loss: 0.2519 - accuracy: 0.9560 - val_loss: 0.4491 - val_accuracy: 0.8407
Epoch 4/5 loss: 0.1310 - accuracy: 0.9775 - val_loss: 0.4135 - val_accuracy: 0.8476
Epoch 5/5 loss: 0.0796 - accuracy: 0.9843 - val_loss: 0.4004 - val_accuracy: 0.8478
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=1`
```
Epoch 1/5 loss: 1.0680 - accuracy: 0.6571 - val_loss: 0.6104 - val_accuracy: 0.7975
Epoch 2/5 loss: 0.2136 - accuracy: 0.9307 - val_loss: 0.5002 - val_accuracy: 0.8464
Epoch 3/5 loss: 0.0605 - accuracy: 0.9811 - val_loss: 0.7676 - val_accuracy: 0.8461
Epoch 4/5 loss: 0.0361 - accuracy: 0.9878 - val_loss: 0.9315 - val_accuracy: 0.8388
Epoch 5/5 loss: 0.0263 - accuracy: 0.9906 - val_loss: 0.9784 - val_accuracy: 0.8446
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4`
```
Epoch 1/5 loss: 1.0682 - accuracy: 0.6598 - val_loss: 0.5239 - val_accuracy: 0.8123
Epoch 2/5 loss: 0.1897 - accuracy: 0.9391 - val_loss: 0.4625 - val_accuracy: 0.8380
Epoch 3/5 loss: 0.0556 - accuracy: 0.9824 - val_loss: 0.6330 - val_accuracy: 0.8226
Epoch 4/5 loss: 0.0337 - accuracy: 0.9885 - val_loss: 0.7936 - val_accuracy: 0.8145
Epoch 5/5 loss: 0.0266 - accuracy: 0.9904 - val_loss: 0.7206 - val_accuracy: 0.8370
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch 1/5 loss: 1.1690 - accuracy: 0.6259 - val_loss: 0.5695 - val_accuracy: 0.7975
Epoch 2/5 loss: 0.2457 - accuracy: 0.9220 - val_loss: 0.4771 - val_accuracy: 0.8281
Epoch 3/5 loss: 0.0870 - accuracy: 0.9730 - val_loss: 0.6044 - val_accuracy: 0.8413
Epoch 4/5 loss: 0.0525 - accuracy: 0.9828 - val_loss: 0.7615 - val_accuracy: 0.8355
Epoch 5/5 loss: 0.0430 - accuracy: 0.9854 - val_loss: 0.7607 - val_accuracy: 0.8403
```
#### Examples End:
#### Tests Start: tagger_transformer_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_layers=0`
```
loss: 2.6393 - accuracy: 0.2170 - val_loss: 2.1583 - val_accuracy: 0.3447
```
- `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=1`
```
loss: 2.1624 - accuracy: 0.3235 - val_loss: 1.9781 - val_accuracy: 0.3119
```
- `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=4`
```
loss: 2.1716 - accuracy: 0.3277 - val_loss: 1.9632 - val_accuracy: 0.3381
```
- `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=4 --transformer_dropout=0.1`
```
loss: 2.2652 - accuracy: 0.3063 - val_loss: 1.9840 - val_accuracy: 0.3606
```
#### Tests End:
