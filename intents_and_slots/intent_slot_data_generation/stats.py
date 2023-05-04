import os

dirname = "templates"
op2samples = dict()
for fname in os.listdir(dirname):
    if ".txt" in fname:
        with open(dirname+"/"+fname) as f:
            lines = f.readlines()
            op2samples[fname.replace(".txt","")] = len(lines)
for op, samples in op2samples.items():
    print(op, samples)
