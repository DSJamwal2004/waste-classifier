import os

base = "data/raw"

print("Exists:", os.path.exists(base))

for folder in os.listdir(base):
    print(folder, "->", len(os.listdir(os.path.join(base, folder))))