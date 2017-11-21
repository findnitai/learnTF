import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 1000
n_nodes_hl3 =1500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 2
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)


# import tensorflow as tf
# import time as mytime

# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# n_nodes_hl1 = 250
# n_nodes_hl2 = 80
# n_nodes_hl3 = 30

# n_classes = 10
# batch_size = 100

# #height X width
# x = tf.placeholder('float',[None, 784])
# y = tf.placeholder('float')

# def neural_network_model(data):
# 	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
# 					  'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}
# 	# tensor flow variable
# 	# input data is mul by weights then summed, bias adds to that
# 	# (input data * weights) + biases (model for each layer)
# 	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
# 					  'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}
# 	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
# 					  'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}
# 	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
# 					  'biases':tf.Variable(tf.random_normal([n_classes]))}

# 	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
# 	l1 = tf.nn.relu(l1)
# 	#rectified linear - activation/threshold fucntion

# 	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases']) 
# 	l2 = tf.nn.relu(l2)

# 	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases']) 
# 	l3 = tf.nn.relu(l3)

# 	output = tf.matmul(l3, output_layer['weights']) , output_layer['biases']

# 	return output

# 	#model is coded, not we need to tell tensor flow what to do
# 	#how we want to run data through that model in the session
# 	# output shape will be of traning and testing set labes
# 	#  learning rate for adamoptimizer can be modified

# def train_neural_network(x):
# 	prediction = neural_network_model(x)
# 	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels= y))
# 	optimizer = tf.train.AdamOptimizer().minimize(cost)
# 	start = mytime.time()
# 	n_epochs = 2
# 	#cycles feed forward and back propagation
# 	with tf.Session() as sess:
# 		sess.run(tf.initialize_all_variables())

# 		for epoch in n_epochs:
# 			epoch_loss = 0
# 			for _ in range(int(mnist.train.num_examples/batch_size)):
# 				x, y = mnist.train.next_batch(batch_size)
# 				_,c = sess.run([optimizer, cost], feed_dict = {x: x, y: y})
# 				epoch_loss += c
# 			print('Epoch', epoch, 'completed out of ', n_epochs, 'loss:', epoch_loss)
# 			print(mytime.time() - start)
# 		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
# 		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
# 		print ('accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labes}))


# train_neural_network(x)



















# #60k traning samples of digits written by hand
# #10k testing samples
# #amnest 28*28 images 
# #each feature is a 0/1 - is it part of the number or not. NN will model
# #this relationship

# '''
# input > weight > HL1 (activation fxn) > weights > HL2 (activation fxn)
# > weights > output layer

# compare output to intened output > cost fxn (cross entropy, how
# close or not close we are to the inteded target)

# optimization fucntion (optimizer) > minimize cost (AdamOptimizer, SGD, AdaGrad)

# does backpropagation 

# feed forward + backprop = is one epoch (10-20 times epoch is done)
# each time lowering the cost fucntion, cost high to low then saturates (may be)

# '''
# '''
# one_hot = one componenet is on rest are off
# 0 = [1,0,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0,0]
# 2 = and so on
# '''

