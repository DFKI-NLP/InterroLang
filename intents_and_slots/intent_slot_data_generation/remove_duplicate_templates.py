import os

dirname = "unfiltered_templates"
for fname in os.listdir(dirname):
    if ".txt" in fname:
        with open(dirname+"/"+fname) as f:
            orig_lines = f.readlines()
            lines = list(set(orig_lines))
            print(fname, len(orig_lines), len(lines))
            with open("templates/"+fname, "w") as f2:
                for line in lines:
                    f2.write(line.lower())

