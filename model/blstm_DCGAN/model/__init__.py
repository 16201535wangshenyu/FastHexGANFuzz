# import tensorflow as tf
#
# if __name__ == "__main__":
#     a1 = tf.random_normal([3, 3], mean=0, stddev=2)
#     a = tf.nn.softmax(a1)
#     b1 = tf.convert_to_tensor([1, 2, 3])
#     b = tf.one_hot(b1, 3)
#     cross_entropy = b * tf.log(a)
#     cross_entropy_sum = -tf.reduce_sum(cross_entropy)
#     y = tf.ones([260,1])
#
#     y3 = tf.pad(y, [[0, 5],[0,0]], "CONSTANT")
#
#     with tf.Session() as session:
#         a1_, a_, b1_, b_, cross_entropy_, cross_entropy_sum_, y3_ = session.run(
#             [a1, a, b1, b, cross_entropy, cross_entropy_sum, y3])
#
#         print(a1_)
#         print("\n")
#         print(a_)
#         print("\n")
#         print(b1_)
#         print("\n")
#         print(b_)
#         print("\n")
#         print(cross_entropy_)
#         print("\n")
#         print(cross_entropy_sum_)
#         print("\n")
#         print(y3_)
