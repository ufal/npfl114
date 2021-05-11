#!/usr/bin/env python3
import os
import sys
import urllib.request
import zipfile

URL = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/models/"
MODEL = "electra_czech_small_lc"

if not os.path.exists(MODEL):
    print("Downloading file {}...".format(MODEL), file=sys.stderr)
    urllib.request.urlretrieve("{}/{}.zip".format(URL, MODEL), filename="{}.zip".format(MODEL))

    print("Extracting file {}...".format(MODEL), file=sys.stderr)
    with zipfile.ZipFile("{}.zip".format(MODEL), "r") as model_file:
        model_file.extractall()
    os.remove("{}.zip".format(MODEL))
