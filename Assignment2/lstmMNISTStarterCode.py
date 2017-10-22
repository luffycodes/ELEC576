import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200
test_step = 1000

nInput = 28  # we want the input to take the 28 pixels
nSteps = 28  # every 28
nHidden = 128  # number of neurons for the RNN
nClasses = 10  # this is MNIST so you know
result_dir = './results_RNN/'

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
    'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
    'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(x, nSteps, 0)
    # x = unfoldRNN(x)
    lstm_cell = rnn_cell.BasicLSTMCell(nHidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def unfoldRNN(x):
    return tf.unstack(x, nSteps, 1)


pred = RNN(x, weights, biases)
prediction = tf.nn.softmax(pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model (with test logits, for dropout to be disabled)
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss_summary = tf.summary.scalar("loss", cost)
accuracy_summary = tf.summary.scalar("accuracy", accuracy)
test_accuracy_summary = tf.summary.scalar("test_accuracy", accuracy)

# Initialize the variables (i.e. assign their default value)
init = tf.initialize_all_variables()

# Start training

with tf.Session() as sess:
    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
    sess.run(init)

    for step in range(1, training_steps + 1):
        batchX, batchY = mnist.train.next_batch(batch_size)
        batchX = batchX.reshape((batch_size, nSteps, nInput))
        sess.run(optimizer, feed_dict={x: batchX, y: batchY})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc, loss_summ, accuracy_summ = sess.run([cost, accuracy, loss_summary, accuracy_summary],
                                                           feed_dict={x: batchX,
                                                                      y: batchY})
            summary_writer.add_summary(loss_summ, step)
            summary_writer.add_summary(accuracy_summ, step)
            summary_writer.flush()

            print("Training Step " + str(step) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))

        if step % test_step == 0 or step == 1:
            loss, acc, loss_summ, test_accuracy_summ = sess.run([cost, accuracy, loss_summary, test_accuracy_summary],
                                                                feed_dict={
                                                                    x: mnist.test.images.reshape(len(mnist.test.images),
                                                                                                 nSteps, nInput),
                                                                    y: mnist.test.labels})
            summary_writer.add_summary(test_accuracy_summ, step)
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step)
            summary_writer.flush()

            print("Test Step " + str(step) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Testing Accuracy= " +
                  "{:.5f}".format(acc))

    print("Optimization Finished!")

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
