### Assignment: tagger_transformer
#### Date: Deadline: May 09, 7:59 a.m.
#### Points: 3 points
#### Tests: tagger_transformer_tests
#### Examples: tagger_transformer_examples

This assignment is a continuation of `tagger_we`. Using the
[tagger_transformer.py](https://github.com/ufal/npfl114/tree/master/labs/11/tagger_transformer.py)
template, implement a Pre-LN Transformer encoder.

#### Tests Start: tagger_transformer_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_layers=0`
```
loss: 2.3873 - accuracy: 0.2446 - val_loss: 2.0423 - val_accuracy: 0.3588
```
2. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=1`
```
loss: 2.0967 - accuracy: 0.3463 - val_loss: 1.8760 - val_accuracy: 0.4181
```
3. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=4`
```
loss: 2.1210 - accuracy: 0.3376 - val_loss: 1.9558 - val_accuracy: 0.3937
```
4. `python3 tagger_transformer.py --epochs=1 --max_sentences=800 --transformer_heads=4 --transformer_dropout=0.1`
```
loss: 2.2215 - accuracy: 0.3050 - val_loss: 2.0125 - val_accuracy: 0.3264
```
#### Tests End:
#### Examples Start: tagger_transformer_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_layers=0`
```
Epoch 1/5 loss: 1.5015 - accuracy: 0.5407 - val_loss: 0.8692 - val_accuracy: 0.7163
Epoch 2/5 loss: 0.5383 - accuracy: 0.8481 - val_loss: 0.5703 - val_accuracy: 0.8240
Epoch 3/5 loss: 0.2542 - accuracy: 0.9589 - val_loss: 0.4611 - val_accuracy: 0.8308
Epoch 4/5 loss: 0.1315 - accuracy: 0.9795 - val_loss: 0.4289 - val_accuracy: 0.8330
Epoch 5/5 loss: 0.0791 - accuracy: 0.9852 - val_loss: 0.4171 - val_accuracy: 0.8348
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=1`
```
Epoch 1/5 loss: 1.0808 - accuracy: 0.6464 - val_loss: 0.5847 - val_accuracy: 0.7878
Epoch 2/5 loss: 0.2452 - accuracy: 0.9226 - val_loss: 0.4712 - val_accuracy: 0.8389
Epoch 3/5 loss: 0.0752 - accuracy: 0.9779 - val_loss: 0.7052 - val_accuracy: 0.8136
Epoch 4/5 loss: 0.0432 - accuracy: 0.9860 - val_loss: 0.6045 - val_accuracy: 0.8314
Epoch 5/5 loss: 0.0324 - accuracy: 0.9888 - val_loss: 0.6385 - val_accuracy: 0.8323
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4`
```
Epoch 1/5 loss: 1.0461 - accuracy: 0.6636 - val_loss: 0.5026 - val_accuracy: 0.8155
Epoch 2/5 loss: 0.1966 - accuracy: 0.9391 - val_loss: 0.4557 - val_accuracy: 0.8386
Epoch 3/5 loss: 0.0712 - accuracy: 0.9777 - val_loss: 0.5322 - val_accuracy: 0.8262
Epoch 4/5 loss: 0.0424 - accuracy: 0.9858 - val_loss: 0.5099 - val_accuracy: 0.8474
Epoch 5/5 loss: 0.0309 - accuracy: 0.9891 - val_loss: 0.6569 - val_accuracy: 0.8404
```
- `python3 tagger_transformer.py --max_sentences=5000 --transformer_heads=4 --transformer_dropout=0.1`
```
Epoch 1/5 loss: 1.1482 - accuracy: 0.6274 - val_loss: 0.5542 - val_accuracy: 0.7950
Epoch 2/5 loss: 0.2579 - accuracy: 0.9176 - val_loss: 0.4971 - val_accuracy: 0.8091
Epoch 3/5 loss: 0.0944 - accuracy: 0.9727 - val_loss: 0.5654 - val_accuracy: 0.8082
Epoch 4/5 loss: 0.0573 - accuracy: 0.9820 - val_loss: 0.5170 - val_accuracy: 0.8340
Epoch 5/5 loss: 0.0448 - accuracy: 0.9850 - val_loss: 0.5199 - val_accuracy: 0.8465
```
#### Examples End:
