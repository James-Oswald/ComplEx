

import tensorflow as tf
from tensorflow.python.ops.numpy_ops.np_math_ops import negative
from dataset import FreeBase15kDataset
from model import ComplEx

class Trainer():
    def __init__(self,
        model:ComplEx, 
        maxEpochs=100,
        earlyStopping=10,
        batchSize=32,
        a = 0.5,  #initial learning rate
        n = 10,   #negitive samples per positive sample
    ): 
        self.model = model
        self.maxEpochs = maxEpochs
        self.earlyStopping = earlyStopping
        self.batchSize = batchSize
        self.a = a
        self.n = n
        
        self.dataset = self.model.dataset
        self.dataset.train = self.dataset.train.batch(self.batchSize)
        self.optimizer = tf.keras.optimizers.Adam(a)

    @tf.function
    def _rangeWithout(end, without):
        r = tf.range(0, end, dtype=tf.int32)
        return tf.concat([tf.slice(r, 0, without), tf.slice(r, without, -1)], 0)

    @tf.function
    def genNegitiveSamples(self, batch):
        #batch * 
        trueTripplets = tf.stack([batch[0],batch[1],batch[2], tf.ones([32],dtype=tf.int32)], axis=1)
        allSamples = tf.ones([self.batchSize, 4, self.n],dtype=tf.int32) * tf.expand_dims(trueTripplets, axis=2)
        allSamples[:,3,1:] = 0
        for i in tf.range(batch.size[0]): 
            allSamples[i,2,1:] = tf.random.shuffle(self._rangeWithout(self.dataset.numEntities, allSamples[i,0,0]))[:self.n-1]
        tf.random.shuffle(tf.transpose(allSamples, [2,0,1]))
        return samples, observations

    def train(self):
        for epoch in range(self.maxEpochs):
            for batch in self.dataset.train:
                with tf.GradientTape() as tape:
                    #generate n negitive samples
                    self.genNegitiveSamples(batch)
                    score = self.model.score(batch)
                    prob = tf.sigmoid(score)            #computed Y
                    loss = self.model.loss(score)       
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable))
            print(f"Epoch {epoch}: Loss {loss}")

if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    dataset = FreeBase15kDataset()
    model = ComplEx(dataset)           
    trainer = Trainer(model)
    trainer.train()