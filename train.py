import os

import tensorflow as tf
from clusterone import get_data_path, get_logs_path

from model import get_model

PATH_TO_LOCAL_LOGS = os.path.abspath(os.path.expanduser('./logs'))
ROOT_PATH_TO_LOCAL_DATA = os.path.abspath(os.path.expanduser('./data'))

try:
    job_name = os.environ['JOB_NAME']
    task_index = os.environ['TASK_INDEX']
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except:
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None

flags = tf.app.flags

# Flags for configuring the distributed task
flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the chief worker task the performs the variable "
                     "initialization")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")

# Training related flags
flags.DEFINE_string("data_dir",
                    get_data_path(
                        dataset_name = "sjay87/", #all mounted repo
                        local_root = ROOT_PATH_TO_LOCAL_DATA,
                        local_repo = "",
                        path = ""
                    ),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to ClusterOne without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")
flags.DEFINE_string("log_dir",
                    get_logs_path(root = PATH_TO_LOCAL_LOGS),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")

n_files = len([f for f in os.listdir(flags.FLAGS.data_dir) if f[-9:]=='tfrecords'])
flags.DEFINE_integer("n_training_samples", 11178*(n_files-1),
                     "Number of training samples")
flags.DEFINE_integer("n_test_samples", 11178,
                     "Number of test samples")

flags.DEFINE_integer("logging_period", 1,
                     "Number of iterations between each logging")

# You can change below numbers.
flags.DEFINE_float("threshold", 0.5,
                   "Probability threshold for predictions")
flags.DEFINE_float("positive_weight", 4.0,
                   "Ratio between loss for positive labels:negative labels")

flags.DEFINE_integer("initial_filters", 64,
                     "Number of conv filters in the 1st hidden layer")
flags.DEFINE_integer("initial_kernel_size", 5,
                     "Size of conv kernel in the 1st hidden layer")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("validation_size", 2048, "Number of samples to use for validation")
flags.DEFINE_integer("epochs", 1000, "Number of epochs")
flags.DEFINE_float("dropout", 0.3, "Dropout probability")

FLAGS = flags.FLAGS

print(FLAGS.data_dir)
print(FLAGS.log_dir)

def device_and_target():
  # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
  # Don't set a device.
  if FLAGS.job_name is None:
    print("Running single-machine training")
    return (None, "")

  # Otherwise we're running distributed TensorFlow.
  print("%s.%d  -- Running distributed training"%(FLAGS.job_name, FLAGS.task_index))
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")
  if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
    raise ValueError("Must specify an explicit `ps_hosts`")
  if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
    raise ValueError("Must specify an explicit `worker_hosts`")

  cluster_spec = tf.train.ClusterSpec({
      "ps": FLAGS.ps_hosts.split(","),
      "worker": FLAGS.worker_hosts.split(","),
  })
  server = tf.train.Server(
      cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()

  worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  return (tf.train.replica_device_setter(
              worker_device=worker_device,
              cluster=cluster_spec),
          server.target)

def read_and_decode(filename_queue, train):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'label_normal': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })

    label = features['label_normal']
    image = tf.reshape(tf.decode_raw(features['image'], tf.uint8), [299, 299, 1])

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if train:
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=FLAGS.batch_size,
                                                capacity=2000,
                                                num_threads=2,
                                                min_after_dequeue=1000)
    else:
        images, labels = tf.train.batch([image, label],
                                        batch_size=FLAGS.batch_size,
                                        capacity=2000,
                                        num_threads=2)
    return images, labels

def main():
    device, target = device_and_target()

    FILENAMES = [os.path.abspath(os.path.join(FLAGS.data_dir, f))
                 for f in sorted(os.listdir(FLAGS.data_dir)) if f[:5]=='train' and f[-9:]=='tfrecords']
    train_filenames = FILENAMES[:4]
    test_filenames = FILENAMES[4:5]

    with tf.device(device):
        train_filename_queue = tf.train.string_input_producer(train_filenames, num_epochs=FLAGS.epochs)
        test_filename_queue = tf.train.string_input_producer(test_filenames, num_epochs=FLAGS.epochs)
        train_images, train_labels = read_and_decode(train_filename_queue, train=True)
        test_images, test_labels = read_and_decode(test_filename_queue, train=False)

        X = tf.placeholder(tf.float32, (None, 299, 299, 1), name='image_input')
        y = tf.placeholder(tf.float32, (None,), name='label_input')
        dropout = tf.placeholder_with_default(1.0, shape=())
        logits = get_model(X, FLAGS, n_layers=4, dropout=dropout)
        preds = tf.cast(tf.greater(logits, 0), tf.int64)

        log_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(preds, tf.cast(y, tf.int64))))
        positive_rate = tf.reduce_mean(tf.to_float(preds))

        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(log_loss, global_step=global_step)

        tf.summary.scalar("loss", log_loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("positive_rate", positive_rate)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'), graph=tf.get_default_graph())
        valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'valid'), graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 1
        for _ in range(FLAGS.n_training_samples*FLAGS.epochs//FLAGS.batch_size):
            # Training
            train_X, train_y = sess.run([train_images, train_labels])
            i, _ = sess.run([global_step, train_op],
                            feed_dict={X: train_X, y: train_y, dropout: FLAGS.dropout})
            if i == 1 or i%FLAGS.logging_period == 0:
                feed_dict = {X: train_X, y: train_y, dropout: FLAGS.dropout}
                train_summary = sess.run(merged, feed_dict=feed_dict)
                train_writer.add_summary(train_summary, i)
                print(i,'\n','-'*20)
                print(train_summary)
                # Validation
                test_X, test_y = sess.run([test_images, test_labels])
                valid_summary = sess.run(merged, feed_dict={X: test_X, y: test_y})
                valid_writer.add_summary(valid_summary, i)
                print(valid_summary)
                step += 1

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
