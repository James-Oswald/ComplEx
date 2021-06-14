
import tensorflow as tf
from dataset import FreeBase15kDataset
from tensorflow.python.keras.backend import l2_normalize

class ComplEx(tf.keras.Model):
    def __init__(self,
        #Model Hyperparamters
        dataset:FreeBase15kDataset,
        k:int = 50,    #K, imposed low rank on embedings
        l:float = 0.001, #\lambda, L^2 regularization hyperparamater
    ):
        super(ComplEx, self).__init__()
        self.dataset = dataset
        self.k = k
        self.l = l

        #K by n 
        self.rE = tf.Variable(tf.experimental.numpy.random.randn(dataset.numEntities, k))
        self.iE = tf.Variable(tf.experimental.numpy.random.randn(dataset.numEntities, k))
        self.rR = tf.Variable(tf.experimental.numpy.random.randn(dataset.numRelations, k))
        self.iR = tf.Variable(tf.experimental.numpy.random.randn(dataset.numRelations, k))
        #self.trainable =[self.rE, self.iE, self.rR, self.iR]

    #See Equations 9-11
    @tf.function
    def score(self, tripplets):
        ss, rs, os = tripplets[:,0], tripplets[:,1], tripplets[:,2]
        #s1 = tf.math.multiply(self.rR[rs], tf.math.multiply(self.rE[ss], self.rE[os]))
        #s2 = tf.math.multiply(self.rR[rs], tf.math.multiply(self.iE[ss], self.iE[os]))
        #s3 = tf.math.multiply(self.iR[rs], tf.math.multiply(self.rE[ss], self.iE[os]))
        #s4 = tf.math.multiply(self.iR[rs], tf.math.multiply(self.iE[ss], self.rE[os]))
        s1 = tf.math.reduce_sum(tf.gather(self.rR, rs) * tf.gather(self.rE, ss) * tf.gather(self.rE, os), 1)
        s2 = tf.math.reduce_sum(tf.gather(self.rR, rs) * tf.gather(self.iE, ss) * tf.gather(self.iE, os), 1)
        s3 = tf.math.reduce_sum(tf.gather(self.iR, rs) * tf.gather(self.rE, ss) * tf.gather(self.iE, os), 1)
        s4 = tf.math.reduce_sum(tf.gather(self.iR, rs) * tf.gather(self.iE, ss) * tf.gather(self.rE, os), 1)
        return s1 + s2 + s3 - s4

    #See Appendix A
    @tf.function
    def loss(self, 
        tripplets,      #The batch of tripplets
        scores         #The scoring function outputs
    ):
        ss, rs, os, ys = tripplets[:,0], tripplets[:,1], tripplets[:,2], tripplets[:,3]
        ys = tf.cast(ys, tf.float64)
        t1 = tf.reduce_sum(tf.keras.activations.softplus(-1*ys*scores))
        t2 = tf.nn.l2_loss(tf.gather(self.rE, ss)) + tf.nn.l2_loss(tf.gather(self.iE, ss)) + \
            tf.nn.l2_loss(tf.gather(self.rR, rs)) + tf.nn.l2_loss(tf.gather(self.iR, rs)) + \
            tf.nn.l2_loss(tf.gather(self.rE, os)) + tf.nn.l2_loss(tf.gather(self.iE, os))
        return t1 + self.l * t2

    @tf.function
    def call(self, inputs):
        return self.score(inputs)
    

#if __name__ == "__main__":
#    a = tf.constant([1,2,3])
#    b = tf.constant([1,2,3])
#    c = tf.constant([1,2,3])
#    print(a * b * c)