#!/usr/bin/env python3
import os
import sys
import urllib.request
import zipfile

class ElectraCzechSmallLc:
    _URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/models/"

    def __init__(self, model="electra_czech_small_lc"):
        try:
            import transformers
        except:
            raise RuntimeError("The `transformers` package was not found, please install it")

        if not os.path.exists(model):
            model_zip = "{}.zip".format(model)

            print("Downloading file {}...".format(model_zip), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, model_zip), filename=model_zip)

            print("Extracting file {}...".format(model_zip), file=sys.stderr)
            with zipfile.ZipFile(model_zip, "r") as model_file:
                model_file.extractall()
            os.remove(model_zip)

        self._model, self._transformers = model, transformers

    def create_tokenizer(self, *args, **kwargs):
        return self._transformers.AutoTokenizer.from_pretrained(self._model, *args, **kwargs)

    def create_model(self, *args, **kwargs):
        return self._transformers.TFAutoModel.from_pretrained(self._model, *args, **kwargs)
