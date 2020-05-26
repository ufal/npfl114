### Assignment: sentiment_analysis
#### Date: Deadline: Jun 7, 23:59
#### Points: 5 points

In this assignment you should try finetuning the mBERT model to perform
sentiment analysis. We will use Czech dataset of Facebook comments,
which can be loaded by the
[text_classification_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/13/text_classification_dataset.py)
module.

Use the BERT implementation from the
[🤗 Transformers library](https://github.com/huggingface/transformers), which
you can install by `pip3 install [--user] transformers`. Start by looking at the
[bert_example.py](https://github.com/ufal/npfl114/tree/master/labs/13/bert_example.py)
example demonstrating loading, tokenizing and calling a BERT model, and you can
also read [the documentation](https://huggingface.co/transformers/), specifically
for the [tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html)
and for [TFBertModel.call](https://huggingface.co/transformers/model_doc/bert.html#transformers.TFBertModel.call).

The assignment is an _open-data task_, where you submit only the test set annotations
together with the training script (which will not be executed, it will be
only used to understand the approach you took, and to indicate teams).
Explicitly, submit **exactly one .txt file** and **at least one .py file**.
You pass if your test set accuracy is at least 75%.

You can start with the
[sentiment_analysis.py](https://github.com/ufal/npfl114/tree/master/labs/13/sentiment_analysis.py)
template, which among others generates test set annotations in the required format.
