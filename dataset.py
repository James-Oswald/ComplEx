
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
CsvDataset = tf.data.experimental.CsvDataset

class FreeBase15kDataset():
    def __init__(self) -> None:
        self.train:CsvDataset = self._loadDataset("./dataset/train.txt.nums.txt", "Train")
        self.valid:CsvDataset = self._loadDataset("./dataset/valid.txt.nums.txt", "Valid")
        self.test:CsvDataset = self._loadDataset("./dataset/test.txt.nums.txt", "Test")
        print(f"Loading Dataset Dicts")
        self.entityDict = pickle.load(open("./dataset/entityDict.pkl","rb"))
        self.relationDict = pickle.load(open("./dataset/relationDict.pkl","rb"))
        self.numEntities = len(self.entityDict)
        self.numRelations = len(self.relationDict)

    #Load a FreeBase15k Datafile as a Tensorflow dataset
    def _loadDataset(self, datasetPath:str, name:str)->CsvDataset:
        print(f"Loading {name} Datset")
        return CsvDataset(
            filenames=datasetPath, 
            record_defaults=[tf.int32, tf.int32, tf.int32], 
            field_delim=','
        )

#Testing Dataset Loading
if __name__ == "__main__":
    #tf.config.run_functions_eagerly(True)
    dataset = FreeBase15kDataset().train
    dataset.batch(32)
    i = 0
    for element in dataset.as_numpy_iterator():   #print the fist 10 dataset elements as a test
        print(element)
        i = i + 1
        if i > 10:
            break