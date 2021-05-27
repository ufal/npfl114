import os
import sys
import urllib.request
import zipfile

# Loads a reading comprehension data.
#
# The data consists of three datasets:
# - train
# - dev
# - test
# and each of the datasets is composed of list of paragraphs.
# Each paragraph consists of
# - context: text
# - qas: list of questions and answers, where each qa consists of
#   - question: text of the question
#   - answers: a list of answers, each answer is composed of
#     - text: string of the text, exactly as appearing in the context
#     - start: character offset of the answer text in the context
class ReadingComprehensionDataset:
    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/datasets/"

    class Dataset:
        def __init__(self, data_file):
            # Load the data
            self._paragraphs = []
            in_paragraph = False
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                if line:
                    if not in_paragraph:
                        self._paragraphs.append({"context": line, "qas": []})
                        in_paragraph = True
                    else:
                        question, *qas = line.split("\t")
                        assert len(qas) % 2 == 0

                        self._paragraphs[-1]["qas"].append({
                            "question": question,
                            "answers": [
                                {"text": qas[i], "start": int(qas[i + 1])} for i in range(0, len(qas), 2)]})
                else:
                    in_paragraph = False

        @property
        def paragraphs(self):
            return self._paragraphs

    def __init__(self, name="reading_comprehension"):
        path = "{}.zip".format(name)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file))

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset, predictions, skip_empty=True):
        gold = [qa["answers"] for paragraph in gold_dataset.paragraphs for qa in paragraph["qas"]]
        if len(predictions) != len(gold):
            raise RuntimeError("The predictions contain different number of answers than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct, total = 0, 0
        for prediction, gold_answers in zip(predictions, gold):
            if len(gold_answers):
                correct += any(prediction == gold_answer["text"] for gold_answer in gold_answers)
                total += 1
            elif not skip_empty:
                correct += not prediction
                total += 1

        return 100 * correct / total

    @staticmethod
    def evaluate_file(gold_dataset, predictions_file):
        predictions = [answer.strip() for answer in predictions_file]
        return ReadingComprehensionDataset.evaluate(gold_dataset, predictions)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--corpus", default="reading_comprehension", type=str, help="The corpus to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="The dataset to evaluate (dev/test)")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = ReadingComprehensionDataset.evaluate_file(
                getattr(ReadingComprehensionDataset(args.corpus), args.dataset), predictions_file)
        print("Reading comprehension accuracy: {:.2f}%".format(accuracy))
