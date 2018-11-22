import tensorflow as tf
import numpy as np

HIDDEN_SIZE = 200
VOCAB_SIZE = 10000


def forward_propagation(cell, input_data, batch_size, num_steps, initial_state):

    embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE], initializer = tf.random_uniform_initializer(-0.05, 0.05))
    inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs = []
    state = initial_state
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0: 
                tf.get_variable_scope().reuse_variables()
            cell_output, state = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)
   
    output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
    weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE], initializer = tf.random_uniform_initializer(-0.05, 0.05))
    bias = tf.get_variable("bias", [VOCAB_SIZE], initializer = tf.random_uniform_initializer(-0.05, 0.05))
    logits = tf.matmul(output, weight) + bias
    return (logits, state)
             
    
