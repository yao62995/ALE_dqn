
import tensorflow as tf

#################################
def device_for_node(n):
  if n.type == "MatMul":
    return "/gpu:0"
  else:
    return "/cpu:0"

graph = tf.Graph()
with graph.as_default():
  with graph.device(device_for_node):
    pass


#################################
# Creates a graph.
with tf.device('/gpu:3'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print sess.run(c)






