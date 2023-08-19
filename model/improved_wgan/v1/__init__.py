# import tensorflow as tf
# from model.v1.ops import *
#
# #
# # input = tf.convert_to_tensor([
# #     [2, 1, 3,0],
# #     [1, 1, 2,0]
# # ],dtype=tf.int32)
# #
# Q = tf.convert_to_tensor(
#     [
#          [[0.4266, 1.2146, 0.2948, 0.7982],
#          [3.3630, 6.0530, 2.8162, 4.5184]]
#     ]
# )
# Q = tf.transpose(Q,[0,2,1])
# Q = tf.concat(tf.split(Q, 2, axis=2),axis=0)
#
# K = tf.convert_to_tensor(
#     [[[-3.1740, -5.1786, -3.0295, -4.3418],
#     [-1.1436,  0.9968, -3.3553, -2.2970]]]
#                          )
# K = tf.transpose(K,[0,2,1])
# K = tf.concat(tf.split(K, 2, axis=2),axis=0)
#
# V = tf.convert_to_tensor(
#     [
#         [[-1.3323, -0.9149, -2.6045, -2.5382],
#         [0.5526, 3.0459, -0.9211, 0.4955]]
#      ]
# )
# V = tf.transpose(V,[0,2,1])
# V = tf.concat(tf.split(V, 2, axis=2),axis=0)
#
# attention = linear_scaled_dot_product_attention(Q, K, V,
#                                                 causality=False, dropout_rate=0.,
#                                                 training=True,
#                                                 key_masks=None,
#                                                 query_masks=None,
#                                                 num_heads=2,
#                                                 scope="scaled_dot_product_attention")
# # attention ,output = multihead_attention(queries=input,
# #                              keys=input,
# #                              values=input,
# #                              num_heads=1,
# #                              dropout_rate=0,
# #                              key_masks=key_mask,
# #                              query_masks=query_mask,
# #                              training=True,
# #                              causality=False
# #                              )
# # print(output)
# if __name__ == "__main__":
#     with tf.Session() as sess:
#         # attention = sess.run(tf.global_variables_initializer())
#         attention = sess.run([attention, ])
#         print(attention)
# #         print(input)
# #         print(tf.shape(input))
# #         print(query_mask)
# #
# #
# #         print(output)
# #         print(tf.shape(output))
# #
# #         print(attention)
# #
# # """
# #
#
# #
# #
# #
# # """
