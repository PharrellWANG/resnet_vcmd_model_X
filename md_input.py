"""md dataset input module.
"""

import tensorflow as tf

data_path_x = '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32.csv'


def build_input(data_path=data_path_x, batch_size=100, mode='train', num_classes=37):
    """Build md image and labels.
  
    Args:
      data_path: Filename for data.
      batch_size: Input batch size.
      mode: Either 'train' or 'eval'.
      num_classes: number of final classes
      
    Returns:
      image: Batches of image. [batch_size, image_size, image_size, 1]
      labels: Batches of labels. [batch_size, num_classes]
      
    Raises:
      ValueError: when the specified dataset is not supported.
    """
    image_size = 16
    num_classes = num_classes

    depth = 1

    filename_queue = tf.train.string_input_producer([data_path])

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.

    record_defaults = [[1] for _ in range(257)]

    list_of_256plus1_columns = tf.decode_csv(value, record_defaults=record_defaults)

    image = tf.stack(list_of_256plus1_columns[0:len(list_of_256plus1_columns) - 1])
    label = tf.stack(list_of_256plus1_columns[len(list_of_256plus1_columns) - 1])

    depth_major = tf.reshape(image,
                             [depth, image_size, image_size])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    label = tf.cast(label, tf.int32)
    label = tf.reshape(label, (1,))
    # print('===================')
    # print(image.shape)
    # print(label.shape)
    # image = np.multiply(image, 1.0 / 255.0)

    if mode == 'train':
        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 16
    else:
        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + image from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 1
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # with tf.Session() as sess:
    #     # Start populating the filename queue.
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     for i in range(1):
    #         # Retrieve a single instance:
    #         image, label = sess.run([image, label])
    #         print(image)
    #         print(image.shape)
    #         print(label)
    #
    #     coord.request_stop()
    #     coord.join(threads)

    # Display the training image in the visualizer.
    tf.summary.image('images', images, 10)
    return images, labels

# build_input()
