import tensorflow as tf

def dot_product(tensor1, tensor2, axis=1):
    """ Dot product of two input tensors along axis """
    dot_product = tf.reduce_sum(tensor1 * tensor2, axis=axis)
    return dot_product


