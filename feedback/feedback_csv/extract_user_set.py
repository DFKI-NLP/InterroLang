import os

import pandas as pd

path = os.getcwd()
files = os.listdir(path)

user_set_path = "../../experiments/parsing_interrolang_dev/user_set_interrolang.txt"

is_exist = False

for file in files:
    if file.endswith(".csv"):
        df = pd.read_csv(file)
        user_text = list(df["User text"])
        golden_label = list(df["Golden label"])

        assert len(user_text) == len(golden_label)

        if os.path.exists(user_set_path):
            is_exist = True

        with open(user_set_path, "a+") as f:
            if is_exist:
                f.write("\n")

            for i in range(len(user_text)):
                f.write(user_text[i] + '\n')

                if not golden_label[i].endswith("[e]"):
                    f.write(golden_label[i] + "[e] " + '\n')
                else:
                    f.write(golden_label[i] + '\n')

                if i != len(user_text) - 1:
                    f.write("\n")
        f.close()
