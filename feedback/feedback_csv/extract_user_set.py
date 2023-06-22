import os

import pandas as pd

path = os.getcwd()
files = os.listdir(path)

boolq_user_set_path = "../../experiments/parsing_interrolang_dev/user_set_interrolang_boolq.txt"
olid_user_set_path = "../../experiments/parsing_interrolang_dev/user_set_interrolang_olid.txt"
da_user_set_path = "../../experiments/parsing_interrolang_dev/user_set_interrolang_daily_dialog.txt"

name2path = {"boolq": boolq_user_set_path, "olid": olid_user_set_path, "daily_dialog": da_user_set_path}

for file in files:
    is_exist = False

    if file.endswith(".csv") and (file.startswith("boolq") or file.startswith("olid") or file.startswith("daily")):
        name = file[:file.find("_")]

        if name == "daily":
            name += "_dialog"

        df = pd.read_csv(file, on_bad_lines='skip')
        user_text = list(df["User text"])
        golden_label = list(df["Golden label"])

        assert len(user_text) == len(golden_label)
        if os.path.exists(name2path[name]):
            is_exist = True

        with open(name2path[name], "a+") as f:
            if is_exist:
                f.write("\n")

            for i in range(len(user_text)):
                f.write(user_text[i] + '\n')

                if not golden_label[i].endswith("[e]"):
                    f.write(golden_label[i] + " [e]" + '\n')
                else:
                    f.write(golden_label[i] + '\n')

                if i != len(user_text) - 1:
                    f.write("\n")
        f.close()
