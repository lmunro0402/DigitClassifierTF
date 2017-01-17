# Handwritten digit classifier
#
# @author Luke Munro

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Set parameters
learning_rate = 0.01
num_iterations = 30
batch_size = 100
display_step = 2

# TF graph input
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

# MODEL

# Initialize weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
	# linear model
	model = tf.nn.softmax(tf.matmul(x, W) + b)

# Summary operations for visualization 
w_h = tf.histogram_summary("weights", W)
b_h = tf.histogram_summary("biases", b)


with tf.name_scope("cost_function") as scope:
	# Cross entropy loss functino
	cost = -tf.reduce_sum(y*tf.log(model))
	# Summary to monitor cost cost function
	tf.scalar_summary("cost_function", cost)

with tf.name_scope("train") as scope:
	# Gradient descent
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize variables
init = tf.initialize_all_variables()

# Merge summaries
merged_summary_op = tf.merge_all_summaries()

#Launch grapp
with tf.Session() as sess:
	sess.run(init)

	# Log writer to ~/Clones/DigitClassifier/tf_logs
	summary_writer = tf.train.SummaryWriter('tf_logs', graph_def=sess.graph_def)

	# Training
	for iteration in range(num_iterations):
		avg_cost = 0.
		total_batches = int(mnist.train.num_examples/batch_size)
		for i in range(total_batches):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			# Avg loss
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batches
			# Write logs
			summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
			summary_writer.add_summary(summary_str, iteration*total_batches+i)

		if not iteration%display_step:
			print "Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost)

	print "Training complete!"

	# Testing
	predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
	print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
