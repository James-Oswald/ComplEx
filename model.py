
import tensorflow as tf

class ComplEx(tf.keras.Model):
    def __init__(self,
        #Model Hyperparamters
        k = 50,    #K, imposed low rank on embedings
        l = 0.001, #\lambda, L^2 regularization hyperparamater
        a = 0.5,  #initial learning rate
        n = 10,   #negitive samples per positive sample
    ):
        super(ComplEx, self).__init__()
        self.k = 
