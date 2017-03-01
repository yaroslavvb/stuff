import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops
import portpicker

port = portpicker.pick_unused_port()
host = "127.0.0.1"
job_name = "worker"
cluster = {job_name: [host+":"+str(port)]}
cluster_spec = tf.train.ClusterSpec(cluster).as_cluster_def()

server = tf.train.Server(cluster_spec, job_name=job_name)
sess = tf.Session(server.target)

x = tf.get_variable("x", shape=[], dtype=tf.float32,
                    initializer=tf.constant_initializer(2), use_resource=True)
sess.run(tf.global_variables_initializer())
print(sess.run(x))
