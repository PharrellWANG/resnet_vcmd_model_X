"""ResNet Eval module.
"""
import time
import six

import md_input
import numpy as np
import md_resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'eval', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path',
                           '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/train_data_32x32/training_32x32_equal.csv',
                           'File pattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path',
                           '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32.csv',
                           'File pattern for eval data')
tf.app.flags.DEFINE_string('train_dir',
                           '/Users/Pharrell_WANG/PycharmProjects/resnet_vcmd_model_X/32x32_wrn_model/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '/Users/Pharrell_WANG/PycharmProjects/resnet_vcmd_model_X/32x32_wrn_model/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 10,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '/Users/Pharrell_WANG/PycharmProjects/resnet_vcmd_model_X/32x32_wrn_model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')


def evaluate(hps):
    """Eval loop."""
    with tf.device('/cpu:0'):
        images, labels = md_input.build_input(FLAGS.eval_data_path, hps.batch_size, FLAGS.mode, hps.num_classes)
        model = md_resnet_model.ResNet(hps, images, labels, FLAGS.mode)
        model.build_graph()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)

        # sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        tf.train.start_queue_runners(sess)

        best_precision = 0.0
        while True:
            try:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
                continue
            if not (ckpt_state and ckpt_state.model_checkpoint_path):
                tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
                continue
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)

            total_prediction, correct_prediction = 0, 0
            for _ in six.moves.range(FLAGS.eval_batch_count):
                (summaries, loss, predictions, truth, train_step) = sess.run(
                    [model.summaries, model.cost, model.predictions,
                     model.labels, model.global_step])

                truth = np.argmax(truth, axis=1)
                predictions = np.argmax(predictions, axis=1)
                correct_prediction += np.sum(truth == predictions)
                total_prediction += predictions.shape[0]

            precision = 1.0 * correct_prediction / total_prediction
            best_precision = max(precision, best_precision)

            precision_summ = tf.Summary()
            precision_summ.value.add(
                tag='Precision', simple_value=precision)
            summary_writer.add_summary(precision_summ, train_step)
            best_precision_summ = tf.Summary()
            best_precision_summ.value.add(
                tag='Best Precision', simple_value=best_precision)
            summary_writer.add_summary(best_precision_summ, train_step)
            summary_writer.add_summary(summaries, train_step)
            tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                            (loss, precision, best_precision))
            summary_writer.flush()

            if FLAGS.eval_once:
                break

            time.sleep(600)


def main(_):
    hps = md_resnet_model.HParams(batch_size=100,
                                  num_classes=37,
                                  lrn_rate=0.3,
                                  # num_residual_units=5,
                                  num_residual_units=4,
                                  use_bottleneck=False,
                                  weight_decay_rate=0.0002,
                                  relu_leakiness=0.1,
                                  optimizer='mom')
    evaluate(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
