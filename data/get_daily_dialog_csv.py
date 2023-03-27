import csv
import torch
from datasets import load_dataset

test_data = load_dataset('daily_dialog', split='test')
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)

with open("./daily_dialog_test.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["dialog", "act"])

    for idx_batch, b in enumerate(test_loader):
        for i in range(len(b["act"])):
            dialog = b["dialog"][i][0]
            act = b["act"][i].item()
            writer.writerow([dialog, act])
