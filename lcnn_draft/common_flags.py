import tensorflow as tf

tf.app.flags.DEFINE_string('model', 'lcnn_resnet_18',
                            """Model type.""")
tf.app.flags.DEFINE_integer('resnet_block_type', 0,
                            """.""")
tf.app.flags.DEFINE_boolean('first_conv_3x3', False,
                            """.""")

tf.app.flags.DEFINE_float('c', 0.001,
                           'Const from paper https://arxiv.org/pdf/1611.06473v1.pdf')
tf.app.flags.DEFINE_integer('channels_reduction_ratio', 4,
                           'How much to reduce number of channels in lcnn residual layer')

tf.app.flags.DEFINE_float('alpha', 0.001,
                           'Const from paper https://arxiv.org/pdf/1611.06473v1.pdf')
tf.app.flags.DEFINE_integer('lcnn_alpha_start_iter', 0,
                           'Iteration number when we should start to increase lcnn alpha coefficient')
tf.app.flags.DEFINE_float('lcnn_alpha_decay', 1,
                           'By which amount should we increase alpha every lcnn_alpha_step iterations')
tf.app.flags.DEFINE_integer('lcnn_alpha_decay_step', 1000,
                           'Number of step to apply lcnn alpha schedule')

tf.app.flags.DEFINE_float('weight_decay', 0.0001,
                           'Global weight decay')
tf.app.flags.DEFINE_boolean('batch_norm_after_conv_1x1', False,
                           'Should we put batch norm layer after conv1x1 in lcnn convolution decomposition')
tf.app.flags.DEFINE_string('augmentation_type', 'no',
                           'Type of data augmentation during training')
tf.app.flags.DEFINE_boolean('glorot_init', False,
                           'Use glorot initialization')
