
import os
from typing import Tuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#from tensorflow.python.data.experimental import CsvDataset
CsvDataset = tf.data.experimental.CsvDataset

class FreeBase15kDataset():
    @tf.function
    def __init__(self) -> None:
        self.trainStrings:CsvDataset = self._loadDataset("./dataset/train.txt", "Train")
        self.validStrings:CsvDataset = self._loadDataset("./dataset/valid.txt", "Valid")
        self.testStrings:CsvDataset = self._loadDataset("./dataset/test.txt", "Test")
        tf.print("Combining Datasets")
        combinedDataset = self.trainStrings.concatenate(self.validStrings).concatenate(self.testStrings)
        #self.entityTokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", lower=False)
        #self.relationTokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", lower=False)
        tf.print("Creating Entity&Relation Dictionary")
        entityKeyTensor, entityValueTensor = tf.zeros((0,1), dtype=tf.string), tf.zeros((0,1), dtype=tf.int32)
        relationKeyTensor, relationValueTensor = tf.zeros((0,1), dtype=tf.string), tf.zeros((0,1), dtype=tf.int32)
        for (i, (s, r, o)) in combinedDataset.enumerate():
            if tf.math.count_nonzero(tf.math.equal(s, entityKeyTensor)) == 0:
                newIndex = tf.shape(entityValueTensor)[0]
                entityKeyTensor = tf.stack([entityKeyTensor, s])
                entityValueTensor = tf.stack([entityValueTensor, newIndex])
            if tf.math.count_nonzero(tf.math.equal(r, relationKeyTensor)) == 0:
                newIndex = tf.shape(entityValueTensor)[0]
                relationKeyTensor = tf.stack([relationKeyTensor, r])
                relationValueTensor = tf.stack([relationValueTensor, newIndex])
            if tf.math.count_nonzero(tf.math.equal(o, entityKeyTensor)) == 0:
                newIndex = tf.shape(entityValueTensor)[0]
                entityKeyTensor = tf.stack([entityKeyTensor, o])
                entityValueTensor = tf.stack([entityValueTensor, newIndex])
            if i % 1000 == 0:
                tf.print(f"Wrote Data {i}")
        tf.print("Creating Entity&Relation Hashtables")
        entityTable = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(entityKeyTensor, entityValueTensor),
            default_value=-1)
        relattionTable = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(relationKeyTensor, relationValueTensor),
            default_value=-1)
        tf.print("Mapping Dataset Strings to Indicies")
        self.train = self.trainStrings.map(self._mapToIndicies, tf.data.AUTOTUNE)
        self.valid = self.trainStrings.map(self._mapToIndicies, tf.data.AUTOTUNE)
        self.test = self.trainStrings.map(self._mapToIndicies, tf.data.AUTOTUNE)
        tf.print("Configuring dataset info")
        #for entity in entities:


    #Load a FreeBase15k Datafile as a Tensorflow dataset
    @tf.function
    def _loadDataset(self, datasetPath:str, name:str)->CsvDataset:
        print(f"Loading {name} Datset")
        return CsvDataset(
            filenames=datasetPath, 
            record_defaults=[tf.string, tf.string, tf.string], 
            field_delim='\t'
        )

    @tf.function
    def _mapToIndicies(self, s, r, o):
        subjectIndex, objectIndex = self.entityTokenizer([s, o])
        relationIndex = self.relationTokenizer([r])
        return subjectIndex, relationIndex, objectIndex

    @tf.function
    def _fitOnTexts(self, tripplet):
        (s,r,o) = tripplet
        

#Testing Dataset Loading
if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    dataset = FreeBase15kDataset()
    i = 0
    for element in dataset.train.as_numpy_iterator():   #print the fist 10 dataset elements as a test
        print(element)
        i = i + 1
        if i > 10:
            break