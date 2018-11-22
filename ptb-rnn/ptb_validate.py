import tensorflow as tf
import forward_propagation
import numpy as np
import os
from tensorflow.models.tutorials.rnn.ptb import reader


DATA_PATH = "ptb_dataset"
MODEL_SAVE_PATH = "model"
MODEL_NAME = "model.ckpt"                 
HIDDEN_LAYER_NODE = 200
LEARNING_RATE = 1.0
BATCH_SIZE = 20
NUM_STEPS = 35
NUM_LAYERS = 2
MAX_GRAD_NORM = 5

def train(batch_size, num_steps):
    #step 1------define placeholder of input data
    input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name = "input_data")
    output_targets = tf.placeholder(tf.int32, [batch_size, num_steps], name = "output_targets")
    
    #step 2------define network structure
    gru_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_LAYER_NODE)
    cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * NUM_LAYERS)
    initial_state = cell.zero_state(batch_size, tf.float32)
 
    #step 3------calculate forward propagation
    logits, final_state = forward_propagation.forward_propagation(cell, input_data, batch_size, num_steps, initial_state)
    
    #step 4------define loss
    total_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits], 
        [tf.reshape(output_targets, [-1])], 
        [tf.ones([batch_size * num_steps], dtype = tf.float32)])
    loss = tf.reduce_sum(total_loss) / batch_size  
    
    #step 6------define a object to load model
    saver = tf.train.Saver()

    #step 7------generate a queue of input data and targets batchs
    _1, _2, test_data, _3 = reader.ptb_raw_data(DATA_PATH)
    batch = reader.ptb_producer(test_data, batch_size, num_steps)
    
    #step 8------execution
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))        
        state = sess.run(initial_state)
        total_loss = 0.0
        steps = 0
        epoch_num = len(test_data) // (batch_size * num_steps) 
        #creat threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        for step in range(epoch_num):
            (x, y) = sess.run(batch)
            loss_value, state = sess.run([loss, final_state], feed_dict = {input_data: x, output_targets: y, initial_state: state})
            total_loss += loss_value
            steps += num_steps
        perlexity = np.exp(total_loss / steps)
        print("perplexity on test is %g" %(perlexity))
        #stop threads
        coord.request_stop()
        coord.join(threads) 

def main(argv = None):
    train(BATCH_SIZE, NUM_STEPS)

if __name__ == "__main__":
    tf.app.run()
    
