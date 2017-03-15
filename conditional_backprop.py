# Example of conditionally enabling backprop based on a variable.
# variable "switches" determines which entries of "y" will be backpropagated
# through.
#
# IE, switches.assign([1,0]) enables backprop through first value but not
# second.
#
# Running it you should see following on stdout:
# Value 2.0, gradient 2.0
# Value 2.0, gradient 0.0
# Value 2.0, gradient 1.0

import tensorflow as tf

def conditional_backprop(do_backprop, tensor):
    do_backprop = tf.Print(do_backprop, [do_backprop], "switch query")
    t = tf.cond(tf.cast(do_backprop, tf.bool),
                lambda: tf.Print(tensor, [0],
                                 "backprop enabled for "+tensor.op.name),
                lambda: tf.zeros_like(tensor))
    y = t + tf.stop_gradient(tensor - t)
    return y

x = tf.ones((), name="x")
y0 = tf.add(x, 0, name="y0")
y1 = tf.add(x, 0, name="y1")

switches = tf.Variable(tf.ones((2)))
doit = tf.constant(True)
yy0 = conditional_backprop(switches[0], y0)
yy1 = conditional_backprop(switches[1], y1)
y = tf.stack([yy0, yy1], name="y")

z = tf.reduce_sum(y)

grad = tf.gradients(z, [x])[0]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Value %.1f, gradient %.1f"%tuple(sess.run([z, grad])))

sess.run(switches.assign([0,0]))
print("Value %.1f, gradient %.1f"%tuple(sess.run([z, grad])))

sess.run(switches.assign([1,0]))
print("Value %.1f, gradient %.1f"%tuple(sess.run([z, grad])))
