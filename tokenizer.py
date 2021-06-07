
#This file existis to spite the janky ass tf.keras.preprocessing.text.Tokenizer which for some god unknown reason
#wasn't developed to run in graph mode. Idiots. https://github.com/tensorflow/tensorflow/issues/46907

import tensorflow as tf

class WordTokenizer():
    def __init__(self):
        tf.lookup.experimental.MutableHashTable 
