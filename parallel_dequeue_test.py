# http://stackoverflow.com/questions/41830206/how-to-share-a-queue-containing-variable-length-sequences-batches-between-multip
#
# 0.12.1
# v0.12.0-10-g4d924e7-dirty
# [array([1, 2, 3, 4, 0], dtype=int32), True]
# [array([8, 6, 7, 9, 5], dtype=int32), True]
# [array([11, 12, 13, 14, 10], dtype=int32), True]
# [array([16, 17, 18, 19, 15], dtype=int32), True]
#
# In HEAD (from Jan 17)
# 0.12.head
# 0.12.1-1878-g76d5960-dirty
# [array([0, 0, 0, 0, 0], dtype=int32), False]
# [array([1, 1, 1, 1, 1], dtype=int32), False]
# [array([2, 2, 2, 2, 2], dtype=int32), False]
# [array([3, 3, 3, 3, 3], dtype=int32), False]

import os, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf

def create_session():
    #    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    config = tf.ConfigProto()
    sess = tf.InteractiveSession("", config=config)
    return sess

import time
import threading
import os
os.environ['PYTHONUNBUFFERED'] = 'True'

n = 100
num_parallel = 5
dtype = tf.int32
queue = tf.FIFOQueue(capacity=n, dtypes=[dtype], shapes=[()])
enqueue_op = queue.enqueue_many(tf.range(n))
size_op = queue.size()

dequeue_ops = []
for i in range(num_parallel):
    dequeue_ops.append(queue.dequeue())

if hasattr(tf, "stack"):
    batch = tf.stack(dequeue_ops)
else:
    batch = tf.pack(dequeue_ops)
all_unique = tf.equal(tf.size(tf.unique(batch)[0]), num_parallel)
sess = create_session()
sess.run(enqueue_op)
print(tf.__version__)
print(tf.__git_version__)
for i in range(n//num_parallel):
    print(sess.run([batch, all_unique, size_op]))
print(tf.get_default_graph().as_graph_def())

# node {
#   name: "fifo_queue"
#   op: "FIFOQueueV2"
#   attr {
#     key: "capacity"
#     value {
#       i: 100
#     }
#   }
#   attr {
#     key: "component_types"
#     value {
#       list {
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     key: "container"
#     value {
#       s: ""
#     }
#   }
#   attr {
#     key: "shapes"
#     value {
#       list {
#         shape {
#         }
#       }
#     }
#   }
#   attr {
#     key: "shared_name"
#     value {
#       s: ""
#     }
#   }
# }
# node {
#   name: "range/start"
#   op: "Const"
#   attr {
#     key: "dtype"
#     value {
#       type: DT_INT32
#     }
#   }
#   attr {
#     key: "value"
#     value {
#       tensor {
#         dtype: DT_INT32
#         tensor_shape {
#         }
#         int_val: 0
#       }
#     }
#   }
# }
# node {
#   name: "range/limit"
#   op: "Const"
#   attr {
#     key: "dtype"
#     value {
#       type: DT_INT32
#     }
#   }
#   attr {
#     key: "value"
#     value {
#       tensor {
#         dtype: DT_INT32
#         tensor_shape {
#         }
#         int_val: 100
#       }
#     }
#   }
# }
# node {
#   name: "range/delta"
#   op: "Const"
#   attr {
#     key: "dtype"
#     value {
#       type: DT_INT32
#     }
#   }
#   attr {
#     key: "value"
#     value {
#       tensor {
#         dtype: DT_INT32
#         tensor_shape {
#         }
#         int_val: 1
#       }
#     }
#   }
# }
# node {
#   name: "range"
#   op: "Range"
#   input: "range/start"
#   input: "range/limit"
#   input: "range/delta"
#   attr {
#     key: "Tidx"
#     value {
#       type: DT_INT32
#     }
#   }
# }
# node {
#   name: "fifo_queue_EnqueueMany"
#   op: "QueueEnqueueManyV2"
#   input: "fifo_queue"
#   input: "range"
#   attr {
#     key: "Tcomponents"
#     value {
#       list {
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     key: "timeout_ms"
#     value {
#       i: -1
#     }
#   }
# }
# node {
#   name: "fifo_queue_Size"
#   op: "QueueSizeV2"
#   input: "fifo_queue"
# }
# node {
#   name: "fifo_queue_Dequeue"
#   op: "QueueDequeueV2"
#   input: "fifo_queue"
#   attr {
#     key: "component_types"
#     value {
#       list {
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     key: "timeout_ms"
#     value {
#       i: -1
#     }
#   }
# }
# node {
#   name: "fifo_queue_Dequeue_1"
#   op: "QueueDequeueV2"
#   input: "fifo_queue"
#   attr {
#     key: "component_types"
#     value {
#       list {
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     key: "timeout_ms"
#     value {
#       i: -1
#     }
#   }
# }
# node {
#   name: "fifo_queue_Dequeue_2"
#   op: "QueueDequeueV2"
#   input: "fifo_queue"
#   attr {
#     key: "component_types"
#     value {
#       list {
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     key: "timeout_ms"
#     value {
#       i: -1
#     }
#   }
# }
# node {
#   name: "fifo_queue_Dequeue_3"
#   op: "QueueDequeueV2"
#   input: "fifo_queue"
#   attr {
#     key: "component_types"
#     value {
#       list {
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     key: "timeout_ms"
#     value {
#       i: -1
#     }
#   }
# }
# node {
#   name: "fifo_queue_Dequeue_4"
#   op: "QueueDequeueV2"
#   input: "fifo_queue"
#   attr {
#     key: "component_types"
#     value {
#       list {
#         type: DT_INT32
#       }
#     }
#   }
#   attr {
#     key: "timeout_ms"
#     value {
#       i: -1
#     }
#   }
# }
# node {
#   name: "stack"
#   op: "Pack"
#   input: "fifo_queue_Dequeue"
#   input: "fifo_queue_Dequeue_1"
#   input: "fifo_queue_Dequeue_2"
#   input: "fifo_queue_Dequeue_3"
#   input: "fifo_queue_Dequeue_4"
#   attr {
#     key: "N"
#     value {
#       i: 5
#     }
