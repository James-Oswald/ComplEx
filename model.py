
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


        # numEntities by K
        self.rE = tf.Variable(tf.experimental.numpy.random.randn(dataset.numEntities, k))
        # numEntities by K
        self.iE = tf.Variable(tf.experimental.numpy.random.randn(dataset.numEntities, k))
        # numRelations by K
        self.rR = tf.Variable(tf.experimental.numpy.random.randn(dataset.numRelations, k))
        # numRelations by K
        self.iR = tf.Variable(tf.experimental.numpy.random.randn(dataset.numRelations, k))

    #See Equations 9-11
    @tf.function
    def score(self, tripplets):
        ss, rs, os = tripplets[:,0], tripplets[:,1], tripplets[:,2]
        s1 = tf.math.reduce_sum(tf.gather(self.rR, rs) * tf.gather(self.rE, ss) * tf.gather(self.rE, os), 1)
        s2 = tf.math.reduce_sum(tf.gather(self.rR, rs) * tf.gather(self.iE, ss) * tf.gather(self.iE, os), 1)
        s3 = tf.math.reduce_sum(tf.gather(self.iR, rs) * tf.gather(self.rE, ss) * tf.gather(self.iE, os), 1)
        s4 = tf.math.reduce_sum(tf.gather(self.iR, rs) * tf.gather(self.iE, ss) * tf.gather(self.rE, os), 1)
        return s1 + s2 + s3 - s4

    #See Appendix A
    #Returns sum(log(1 + exp(-Y_{sro}*Phi(s,r,o,Theta)))) + L2(Theta)^2
    @tf.function
    def loss(self, tripplets:tf.Tensor, scores:tf.Tensor)->tf.float64:
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