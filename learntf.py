#test
import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.mul(x1,x2)
print(result)

with tf.Session() as sess:
	output = sess.run(result)
	print(output)
#computation graph - model everything - no of nodes, layers, starting
#values etc 
#session is run via optimiser 
#cost funtion (our) build inside the computation graph
#we do not have to code the logic that will modify the weights 
#we just tell tensorflow to minimize the cost fxn defined by us.
	

# sess = tf.Session()
# print(sess.run(result))
# sess.close