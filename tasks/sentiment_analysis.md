### Assignment: sentiment_analysis
#### Date: Deadline: May 31, 23:59
#### Points: 3 points

Perform sentiment analysis on Czech Facebook data using provided pre-trained
Czech Electra small. The dataset consists of pairs of _(document, label)_
and can be (down)loaded using the
[text_classification_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/11/text_classification_dataset.py)
module. When loading the dataset, a `tokenizer` might be provided, and if it is,
the _document_ is also passed through the tokenizer and the resulting tokens are
added to the dataset.

Even though this assignment is not a competition, your goal is to submit test
set annotations with at least 77% accuracy. As usual, you can evaluate your
predictions using the [text_classification_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/11/text_classification_dataset.py)
module, either by running with `--evaluate=path` arguments, or using its
`evaluate_file` method.

Note that contrary to working with EfficientNet, you **need** to **finetune**
the Electra model in order to achieve the required accuracy.

You can start with the
[sentiment_analysis.py](https://github.com/ufal/npfl114/tree/master/labs/11/sentiment_analysis.py)
template, which among others loads the Electra Czech model and generates test
set annotations in the required format. Note that [bert_example.py](https://github.com/ufal/npfl114/tree/master/labs/11/bert_example.py)
module illustrate the usage of both the Electra tokenizer and the Electra model.
