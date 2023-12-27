import os
import datasets
from datasets import load_dataset

# Load the daily dialogue dataset.
dataset = load_dataset("daily_dialog")

# Select the first 5000 dialogues from the dataset.
dialogues = dataset["train"]["dialog"][:5000]

# # Create the file if it does not exist.
# os.makedirs("dialogues.txt", exist_ok=True)

# Write the dialogues to the text file.
with open("dialogues.txt", "w") as f:
    for dialogue in [str(dialogue) + "\n" for dialogue in dialogues]:
        f.write(dialogue)
        print("writing ....")