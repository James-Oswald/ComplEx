

import csv
import pickle

files = ["test.txt","train.txt","valid.txt"]

entities = {}
relations = {}
csvData = {}

#We go through all data and build the entity name to enumeration dictionary
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

#We go through all the datasets and convert names to enumerations
for filename in files:
    with open(f"./{filename}.nums.txt", "w") as file:
        print(f"Building Enumerated With {filename}")
        for row in csvData[filename]:
            s, r, o = row
            file.write(f"{entities[s]},{relations[r]},{entities[o]}\n")
        file.close()

#We save our entity -> num dicts for later use
with open("entityDict.pkl", "wb") as file:
    pickle.dump(entities, file, pickle.HIGHEST_PROTOCOL)
with open("relationDict.pkl", "wb") as file:
    pickle.dump(relations, file, pickle.HIGHEST_PROTOCOL)