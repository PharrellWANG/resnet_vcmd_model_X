# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import md_input
import numpy as np
import md_resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path',
                           '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/train_data_32x32/training_32x32_equal.csv',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path',
                           '/Users/Pharrell_WANG/PycharmProjects/vcmd_data_prepare/test_data_32x32/testing_32x32.csv',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_string('train_dir',
                           '/Users/Pharrell_WANG/PycharmProjects/resnet_vcmd_model_X/32x32_wrn_model/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '/Users/Pharrell_WANG/PycharmProjects/resnet_vcmd_model_X/32x32_wrn_model/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '/Users/Pharrell_WANG/PycharmProjects/resnet_vcmd_model_X/32x32_wrn_model',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')


def train(hps):
    """Training loop."""
    with tf.device('/cpu:0'):
        images, labels = md_input.build_input(
            FLAGS.train_data_path, hps.batch_size, FLAGS.mode, hps.num_classes)
    with tf.device('/gpu:0'):
        model = md_resnet_model.ResNet(hps, images, labels, FLAGS.mode)
        model.build_graph()

        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

        truth = tf.argmax(model.labels, axis=1)
        predictions = tf.argmax(model.predictions, axis=1)
        precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

        summary_hook = tf.train.SummarySaverHook(
            save_steps=100,
            output_dir=FLAGS.train_dir,
            summary_op=tf.summary.merge([model.summaries,
                                         tf.summary.scalar('Precision', precision)]))

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': model.global_step,
                     'loss': model.cost,
                     'precision': precision},
            every_n_iter=100)

        class _LearningRateSetterHook(tf.train.SessionRunHook):
            """Sets learning_rate based on global step."""

            def begin(self):
                self._lrn_rate = 0.1

            def before_run(self, run_context):
                return tf.train.SessionRunArgs(
                    model.global_step,  # Asks for global step value.
                    feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

            def after_run(self, run_context, run_values):
                train_step = run_values.results
                if train_step < 5000:
                    self._lrn_rate = 0.1
                elif train_step < 15000:
                    self._lrn_rate = 0.07
                elif train_step < 20000:
                    self._lrn_rate = 0.05
                elif train_step < 30000:
                    self._lrn_rate = 0.01
                elif train_step < 45000:
                    self._lrn_rate = 0.008
                elif train_step < 50000:
                    self._lrn_rate = 0.006
                elif train_step < 60000:
                    self._lrn_rate = 0.005
                elif train_step < 70000:
                    self._lrn_rate = 0.004
                elif train_step < 80000:
                    self._lrn_rate = 0.003
                elif train_step < 90000:
                    self._lrn_rate = 0.002
                elif train_step < 100000:
                    self._lrn_rate = 0.001
                elif train_step < 110000:
                    self._lrn_rate = 0.0009
                elif train_step < 115000:
                    self._lrn_rate = 0.0008
                elif train_step < 120000:
                    self._lrn_rate = 0.0007
                elif train_step < 125000:
                    self._lrn_rate = 0.0006
                elif train_step < 130000:
                    self._lrn_rate = 0.0005
                elif train_step < 135000:
                    self._lrn_rate = 0.0004
                elif train_step < 140000:
                    self._lrn_rate = 0.0003
                elif train_step < 145000:
                    self._lrn_rate = 0.0002
                elif train_step < 150000:
                    self._lrn_rate = 0.0001
                elif train_step < 155000:
                    self._lrn_rate = 0.00009
                elif train_step < 160000:
                    self._lrn_rate = 0.00008
                elif train_step < 180000:
                    self._lrn_rate = 0.00005
                else:
                    self._lrn_rate = 0.00001

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.log_root,
                hooks=[logging_hook, _LearningRateSetterHook()],
                chief_only_hooks=[summary_hook],
                # Since we provide a SummarySaverHook, we need to disable default
                # SummarySaverHook. To do that we set save_summaries_steps to 0.
                save_summaries_steps=0,
                config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True, )) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(model.train_op)


def main(_):
    hps = md_resnet_model.HParams(batch_size=100,
                                  num_classes=37,
                                  min_lrn_rate=0.00001,
                                  lrn_rate=0.1,
                                  # num_residual_units=5,
                                  num_residual_units=4,
                                  use_bottleneck=False,
                                  weight_decay_rate=0.0002,
                                  relu_leakiness=0.1,
                                  optimizer='mom')
    train(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
