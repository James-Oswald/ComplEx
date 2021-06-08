

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
    def genNegitiveSamples(tripplets):
        #batch * 
        tf.concat([*tripplets])
        return samples, observations

    def train(self):
        for epoch in range(self.maxEpochs):
            for batch in self.dataset.train:
                with tf.GradientTape() as tape:
                    #generate n negitive samples
                    


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