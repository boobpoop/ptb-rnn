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

    #step 5------select a optimizer and train
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    trainable_variables = tf.trainable_variables()
    #step 5.1------compute grandients
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_variables), MAX_GRAD_NORM)
    #step 5.2------update grandients
    train_step = optimizer.apply_gradients(zip(grads, trainable_variables))
    
    #step 6------define a object to save model
    saver = tf.train.Saver()

    #step 7------generate a queue of input data and targets batchs
    train_data, _1, _2, _3 = reader.ptb_raw_data(DATA_PATH)
    batch = reader.ptb_producer(train_data, batch_size, num_steps)
    
    #step 8------execution
    with tf.Session() as sess:
        total_loss = 0.0
        steps = 0
        epoch_num = len(train_data) // (batch_size * num_steps) 
        sess.run(tf.global_variables_initializer())
        state = sess.run(initial_state)
        #creat threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        for step in range(epoch_num):
            (x, y) = sess.run(batch)
            loss_value, state, _ = sess.run([loss, final_state, train_step], feed_dict = {input_data: x, output_targets: y, initial_state: state})
            total_loss += loss_value
            steps += num_steps
            perlexity = np.exp(total_loss / steps)
            if step % 100 == 0:
                print("After %d steps, loss on training batch is %g. perplexity is %g" %(step, loss_value, perlexity))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        #stop threads
        coord.request_stop()
        coord.join(threads) 

def main(argv = None):
    train(BATCH_SIZE, NUM_STEPS)

if __name__ == "__main__":
    tf.app.run()
    
