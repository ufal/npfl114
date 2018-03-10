#!/usr/bin/env python3
import base64
import io
import zipfile

with open("gym_cartpole_recodex_submission.py", "w") as submission:
    print("import gym_cartpole_recodex_evaluation", file=submission)

    with io.BytesIO() as zip_data:
        with zipfile.ZipFile(zip_data, mode="w", compression=zipfile.ZIP_LZMA) as zip:
            zip.write("gym_cartpole/checkpoint")
            zip.write("gym_cartpole/model.data-00000-of-00001")
            zip.write("gym_cartpole/model.index")
            zip.write("gym_cartpole/model.meta")

        print("model = ", base64.b85encode(zip_data.getbuffer()), file=submission)

    print("gym_cartpole_recodex_evaluation.evaluate(model)", file=submission)
