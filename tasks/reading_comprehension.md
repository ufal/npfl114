### Assignment: reading_comprehension
#### Date: Deadline: May 31, 23:59; non-competition part extended to Jun 30
#### Points: 4 points+5 bonus

Implement the best possible model for reading comprehension task using
a translated version of the SQuAD 2.0 dataset, utilizing the provided
pre-trained Czech Electra small.

The dataset can be loaded using the
[reading_comprehension_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/11/reading_comprehension_dataset.py)
module. The loaded dataset is the direct reprentation of the data and not yet
ready to be directly trained on. Each of the `train`, `dev` and `test` datasets
are composed of a list of paragraphs, each consisting of:
- `context`: text with the information;
- `qas`: list of questions and answers, where each item consists of:
  - `question`: text of the question;
  - `answers`: a list of answers, each answer is composed of:
    - `text`: string of the text, exactly as appearing in the context;
    - `start`: character offset of the answer text in the context.

Note that a question might not be answerable given the context, in which case
the list of answers is empty. In the `train` and `dev` sets, each question has
at most one answer, while in the `test` set there might be several answers.
We evaluate the reading comprehension task using _accuracy_, where an answer is
considered correct if its text is exactly equal to some correct answer.
You can evaluate your predictions as usual with the
[reading_comprehension_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/11/reading_comprehension_dataset.py)
module, either by running with `--evaluate=path` arguments, or using its
`evaluate_file` method.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl114/2021-summer#competitions). Everyone who submits a solution
a solution with at least **TO BE FILLED LATER**% answer accuracy gets 4 points; the rest 5 points
will be distributed depending on relative ordering of your solutions.

You can start with the
[reading_comprehension.py](https://github.com/ufal/npfl114/tree/master/labs/11/reading_comprehension.py)
template, which among others (down)loads the data and Czech Electra small model, and describes
the format of the required test set annotations.
