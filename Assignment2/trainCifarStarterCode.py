from scipy import misc
import numpy as np
import tensorflow as tf
import os


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


result_dir = './results/'
ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 100

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot label
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path)  # 28 by 28
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot label

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, imsize, imsize,
                                   nchannels])  # tf variable for the data, remember shape is [None, width, height, numberOfChannels]
y_ = tf.placeholder("float", shape=[None, nclass])  # tf variable for labels

# model

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, nclass])
b_fc2 = bias_variable([nclass])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Credits: https://github.com/tensorflow/tensorflow/issues/908 - Martin-Gorner
W1_a = W_conv1
W1pad = tf.zeros([5, 5, 1, 1])
W1_b = tf.concat([W1_a, W1pad, W1pad, W1pad, W1pad], 3)
W1_c = tf.split(W1_b, 36, 3)
W1_row0 = tf.concat(W1_c[0:6], 0)
W1_row1 = tf.concat(W1_c[6:12], 0)
W1_row2 = tf.concat(W1_c[12:18], 0)
W1_row3 = tf.concat(W1_c[18:24], 0)
W1_row4 = tf.concat(W1_c[24:30], 0)
W1_row5 = tf.concat(W1_c[30:36], 0)
W1_d = tf.concat([W1_row0, W1_row1, W1_row2, W1_row3, W1_row4, W1_row5], 1)
W1_e = tf.reshape(W1_d, [1, 30, 30, 1])
Wtag = tf.placeholder(tf.string, None)
image_summary_t = tf.summary.image("Visualize_kernels", W1_e)

# loss
# loss, optimization, evaluation, and accuracy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add a scalar summary for the snapshot loss.
tf.summary.scalar(cross_entropy.op.name, cross_entropy)

training_summary = tf.summary.scalar("training_accuracy", accuracy)
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

sess.run(tf.initialize_all_variables())
batch_xs = np.zeros(
    [batchsize, imsize, imsize, nchannels])  # setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize, nclass])  # setup as [batchsize, the how many classes]

nsamples = 10000
max_step = 5500
for i in range(5600):  # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :]

    if i % 1100 == 0:
        test_accuracy, test_summ = sess.run([accuracy, test_summary],
                                            feed_dict={x: Test,
                                                       y_: LTest,
                                                       keep_prob: 1.0})
        summary_writer.add_summary(test_summ, i)
        print("test: step %d, accuracy %g" % (i, test_accuracy))

    if i % 100 == 0:
        # output the training accuracy every 100 iterations
        # Update the events file which is used to monitor the training (in this case,
        # only the training loss is monitored)
        train_accuracy, summary_str = sess.run([accuracy, summary_op],
                                               feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        print("training: step %d, accuracy %g" % (i, train_accuracy))
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

    # save the checkpoints every 1100 iterations
    if i % 1100 == 0 or i == max_step:
        checkpoint_file = os.path.join(result_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=i)

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})  # run one train_step

# --------------------------------------------------
# test
print("test accuracy %g" % accuracy.eval(feed_dict={x: Test,
                                                    y_: LTest, keep_prob: 1.0}))

sess.close()
