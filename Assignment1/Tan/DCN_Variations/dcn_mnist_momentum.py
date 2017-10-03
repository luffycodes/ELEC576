__author__ = 'tan_nguyen'

import os
import time

# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import Tensorflow and start a session
import tensorflow as tf

sess = tf.InteractiveSession()


def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    # Specify training parameters
    # result_dir = './results/'  - directory where the results from the training are saved
    max_step = 5500  # the maximum iterations. After max_step iterations, the training will stop no matter what

    start_time = time.time()  # start timing

    # FILL IN THE CODE BELOW TO BUILD YOUR NETWORK

    # placeholders for input data and input labels
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # reshape the input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    stats("W_conv1", W_conv1)
    stats("b_conv1", b_conv1)
    stats("h_conv1", h_conv1)
    stats("h_pool1", h_pool1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    stats("W_conv2", W_conv2)
    stats("b_conv2", b_conv2)
    stats("h_conv2", h_conv2)
    stats("h_pool2", h_pool2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    stats("W_fc1", W_fc1)
    stats("b_fc1", b_fc1)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    stats("h_fc1", h_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    stats("W_fc2", W_fc2)
    stats("b_fc2", b_fc2)

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

    # setup training
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.MomentumOptimizer(1e-4, 0.8).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(cross_entropy.op.name, cross_entropy)

    training_summary = tf.summary.scalar("training_accuracy", accuracy)
    validation_summary = tf.summary.scalar("validation_accuracy", accuracy)
    test_summary = tf.summary.scalar("test_accuracy", accuracy)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # run the training
    for i in range(max_step):
        batch = mnist.train.next_batch(50)  # make the data batch, which is used in the training iteration.

        # validation & test accuracies every 1100 iterations
        if i % 1100 == 0:
            validation_accuracy, validation_summ = sess.run([accuracy, validation_summary],
                                                            feed_dict={x: mnist.validation.images,
                                                                       y_: mnist.validation.labels,
                                                                       keep_prob: 1.0})
            summary_writer.add_summary(validation_summ, i)
            print("validation: step %d, accuracy %g" % (i, validation_accuracy))

            test_accuracy, test_summ = sess.run([accuracy, test_summary],
                                                feed_dict={x: mnist.test.images,
                                                           y_: mnist.test.labels,
                                                           keep_prob: 1.0})
            summary_writer.add_summary(test_summ, i)
            print("test: step %d, accuracy %g" % (i, test_accuracy))

        # the batch size is 50
        if i % 100 == 0:
            # output the training accuracy every 100 iterations
            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            train_accuracy, summary_str = sess.run([accuracy, summary_op],
                                                   feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            print("training: step %d, accuracy %g" % (i, train_accuracy))
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        # save the checkpoints every 1100 iterations
        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # run one train_step

    # print test error
    accuracy_eval = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print("test accuracy final %g" % accuracy_eval)

    stop_time = time.time()
    print('The training takes %f second to finish' % (stop_time - start_time))


def stats(prefix, metric):
    tf.summary.scalar(prefix + "_min", tf.reduce_min(metric))
    tf.summary.scalar(prefix + "_max", tf.reduce_max(metric))
    W_conv1_mean = tf.reduce_mean(metric)
    tf.summary.scalar(prefix + "_mean", W_conv1_mean)
    tf.summary.scalar(prefix + "_std_dev", tf.sqrt(tf.reduce_mean(tf.square(metric - W_conv1_mean))))
    tf.summary.histogram(prefix + "_histogram", metric)


result_dir = './results_momentum/'  # directory where the results from the training are saved
if __name__ == "__main__":
    main()
