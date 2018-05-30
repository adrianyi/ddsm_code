import tensorflow as tf

def add_layer(input_tensor, variable_scope,
              n_filters=16, kernel_size=(3, 3), strides=2,
              dropout=0.2):
    with tf.variable_scope(variable_scope):
        conv1 = tf.layers.conv2d(
            input_tensor,
            filters = n_filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = 'valid',
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            bias_initializer = tf.zeros_initializer(),
            kernel_regularizer = None,
            name = 'conv1',
            activation = tf.nn.relu
        )
        conv2 = tf.layers.conv2d(
            conv1,
            filters = n_filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = 'same',
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
    		bias_initializer = tf.zeros_initializer(),
    		kernel_regularizer = None,
    		name = 'conv2',
    		activation = tf.nn.relu
        )
        drop = tf.layers.dropout(conv2, dropout, name='dropout')
    return drop

def get_model(input_tensor, FLAGS, n_layers=4, dropout=0.2):
    n_filters = FLAGS.initial_filters
    ksize = FLAGS.initial_kernel_size
    conv_output = add_layer(input_tensor, 'layer_1', n_filters, kernel_size=(ksize, ksize), dropout=dropout)
    for i in range(2, n_layers+1):
        n_filters *= 2
        conv_output = add_layer(conv_output, 'layer_{:d}'.format(i), n_filters=n_filters, dropout=dropout)
    with tf.variable_scope('final_layer'):
        pool = tf.layers.max_pooling2d(conv_output, conv_output.shape[1:3], strides=1, name='max_pool')
        flat = tf.squeeze(pool, axis=[1,2], name='flat')
        dense = tf.layers.dense(flat, 512,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        		bias_initializer=tf.zeros_initializer(),
                                name='dense1')
        pred = tf.layers.dense(dense, 1,
                               # activation=tf.nn.sigmoid,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        	   bias_initializer=tf.zeros_initializer(),
                               name='dense2')
    return tf.squeeze(pred, name='prob')
