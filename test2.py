import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])

b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
c = tf.matmul(a, b)


print(c)
