

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

    #Generate a range without some given set of values inside of it
    @tf.function
    def _rangeWithout(self, end, without):
        without = tf.sort(without)
        without = tf.concat([without, [-1]], 0)
        r = tf.range(0, end, dtype=tf.int32)
        rv = tf.slice(r, [0], [without[0]])
        for i in range(without.shape[0] - 2):
            rv = tf.concat([rv, tf.slice(r, [without[i]], [without[i+1] - without[i]])], 0)
        rv = tf.concat([rv, tf.slice(r, [without[-2]], [-1])], 0)
        return rv

    # 
    #
    #
    #
    #@tf.function
    def _genNegitiveSamples(self, tripplet, numNegSamples):
        """Generates numNegSamples negitive samples for a given tripplet using the local closed world assumption.
        Params:
        -tripplet: A True KG tripplet Tensor containing (Subject, Relation, Object, True) 
        -numNegSamples: The number of negitive samples to generate for the tripplet 
        Returns:
        -A Tensor ( shape (numNegSamples+1)x4 ) of corrupted tripplets, including the original tripplet as its final element
            [s, r, o, 1] => [[#, r, o, 0], [s, r, #, 0], ..., [s, r, o, 1]]
        """
        s, _, o, _ = tripplet
        #The list of corrupted head/tails, local closed world assumption. See Trouillon et al. 2016 section 4.2 paragraph 2
        #generate a range excluding head and tail (these are entity IDs exculding the true entities)
        #ISSUE with this is you cant ever generate the inverse as a negitive sample, this really needs to be revisted
        # _rangeWithout(5, [2]) => [0,1,3,4]
        corruptedEntities = self._rangeWithout(self.dataset.numEntities, [s,o])
        # randomize the order of the list
        # [0,1,3,4] =shuffle=> [4,1,0,3] (or any other permutaion)
        corruptedEntities = tf.random.shuffle(corruptedEntities)[:numNegSamples]
        #tiled corrupt entities, just make this the same length as our data,
        #we elementwise multiply this with coruptPos later on. 
        #[4,1, ...] => [[4,4,4,4], [1,1,1,1], ...]
        tiledCorEnt = tf.transpose(tf.tile([corruptedEntities], [4, 1]))

        #random vector of 0s and 1s specifying if we're corrupting the head or the tail of the tripplet
        # [0, 1, ...]
        headOrTail = tf.random.uniform([numNegSamples], dtype=tf.int32, maxval=2)
        #since tails are in index 2 of the tripplet, we multiply all 1s by 2 
        # [0, 1, ...] => [0, 2, ...]
        headOrTail = 2 * headOrTail
        #We one hot encode these for a matrix with the exact postions we're corrupting in the neg sample matrix
        # [0, 2, ...] => [[1, 0, 0, 0], 
        #                 [0, 0, 1, 0], ...]
        corPos = tf.one_hot(headOrTail, 4, dtype=tf.int32)
        # [[1, 0, 0, 0],   => [[0, 1, 1, 1],
        #  [0, 0, 1, 0], ...]  [1, 1, 0, 1], ...]
        invCorPos = tf.cast(tf.math.logical_not(tf.cast(corPos, tf.bool)), tf.int32)

        #[s,r,o,1] => [s,r,o,0]
        negTripplet = tf.concat([tripplet[:-1],[0]],axis=0)
        #corrupted Samples
        #[s,r,o,0] => [[s,r,o,0],
        #              [s,r,o,0], ...] (repeated numNegSamples times)
        tiledSamples = tf.tile([negTripplet],[numNegSamples,1])
        
        #The corrupted data in the right location
        #  [1, 0, 0, 0] * [4, 4, 4, 4] = [4, 0, 0, 0]
        #  [0, 0, 1, 0]   [1, 1, 1, 1]   [0, 0, 1, 0]
        corEntMat = corPos * tiledCorEnt
        #Samples with corrupted entries set to 0
        #  [0, 1, 1, 1] * [s, r, o, 0] = [0, r, o, 0]
        #  [1, 1, 0, 1]   [s, r, o, 0]   [s, r, 0, 0]
        cor0Samples = invCorPos * tiledSamples
        #True corrupted Samples
        #  [4, 0, 0, 0] * [0, r, o, 0] = [4, r, o, 0]
        #  [0, 0, 1, 0]   [s, r, 0, 0]   [s, r, 1, 0]
        corSamples = corEntMat + cor0Samples
        #combine the corrupt samples with true samples
        samples = tf.concat([corSamples, [tripplet]],axis=0)
        return samples

    #@tf.function
    def _genNegSamplesForBatch(self, batch):
        batchSamples = tf.zeros([0, 4], dtype=tf.int32)
        for tripplet in batch:
            trippletSamples = self._genNegitiveSamples(tripplet, self.n)
            batchSamples = tf.concat([batchSamples, trippletSamples], axis=0)
        batchSamples = tf.random.shuffle(batchSamples)
        return batchSamples 

    def train(self):
        for epoch in range(self.maxEpochs):
            for batchNum, batch in enumerate(self.dataset.train):
                with tf.GradientTape() as tape:
                    #generate n negitive samples
                    batch = self._genNegSamplesForBatch(batch)
                    score = self.model.score(batch)
                    prob = tf.sigmoid(score)            #computed Y
                    loss = self.model.loss(batch, score)       
                grads = tape.gradient(loss, self.model.trainable)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable))
                print(f"Batch {batchNum}/{272155//32}: Loss {loss}")
            print(f"Epoch {epoch}: Loss {loss}")

if __name__ == "__main__":
    #tf.config.run_functions_eagerly(True)
    dataset = FreeBase15kDataset()
    model = ComplEx(dataset)           
    trainer = Trainer(model)
    trainer.train()