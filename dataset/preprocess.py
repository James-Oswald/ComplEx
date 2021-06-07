

import csv
import pickle

files = ["test.txt","train.txt","valid.txt"]

entities = {}
relations = {}
csvData = {}

for filename in files:
    with open(filename, "r") as file:
        print(f"Building Dict With {filename}")
        csvData[filename] = list(csv.reader(file, delimiter = '\t'))
        for i, row in enumerate(csvData[filename]):
            s, r, o = row
            if s not in entities:
                entities[s] = len(entities)
            if r not in relations:
                relations[r] = len(relations)
            if o not in entities:
                entities[o] = len(entities)
            if i % 1000:
                print(f"Finished {i}")
        file.close()
for filename in files:
    with open(f"./{filename}.nums.txt", "w") as file:
        print(f"Building Enumerated With {filename}")
        for row in csvData[filename]:
            s, r, o = row
            file.write(f"{entities[s]},{relations[r]},{entities[o]}\n")
        file.close()
with open("entityDict.pkl", "wb") as file:
    pickle.dump(entities, file, pickle.HIGHEST_PROTOCOL)
with open("relationDict.pkl", "wb") as file:
    pickle.dump(relations, file, pickle.HIGHEST_PROTOCOL)