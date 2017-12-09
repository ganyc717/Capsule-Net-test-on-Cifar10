import tensorflow as tf
import CapsuleNet
import Cifar10
import config
import datetime
import os
import numpy as np

class Solver:
    def __init__(self, net,data):
        self.net = net
        self.data = data
        self.weights_file = config.WEIGHTS_FILE
        self.max_iter = config.TRAIN_ITER_LOOP
        self.learning_rate = config.LEARNING_RATE
        self.save_iter = config.SAVE_ITER
        self.summary_iter = config.SUMMARY_ITER

        self.saver = tf.train.Saver()
        self.weights_file = config.WEIGHTS_FILE
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(config.SUMMARY_OUTPUT_DIR, flush_secs=60)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.net.total_loss)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allocator_type='BFC', allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

        try:
            print('Try restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)
        except:
            print("failed to restore weight, random initilize parameters")
        self.writer.add_graph(self.sess.graph)

    def train(self):
        for step in range(1, self.max_iter + 1):
            images, labels = self.data.fetch_batch()
            images = images / 255.0
            feed_dict = {self.net.input_img: images, self.net.input_label: labels}
            if step % self.summary_iter == 0:
                _, accuracy,summary_str = self.sess.run(
                        [self.train_op, self.net.accuracy,self.summary_op],feed_dict=feed_dict)

                print(datetime.datetime.now().strftime('%m/%d %H:%M:%S ')+"step: %d" % step)
                print("accuracy is %f" % accuracy)
                self.writer.add_summary(summary_str, step)

            else:
                self.sess.run([self.train_op], feed_dict=feed_dict)


            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                self.weights_file))
                self.saver.save(self.sess, self.weights_file)
            tf.reset_default_graph()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    capsuleNet = CapsuleNet.CapsuleNet()
    cifar = Cifar10.Cifar()

    para_data_path = config.WEIGHTS_FILE

    solver = Solver(capsuleNet, cifar)

    print('Start training ...')
    solver.train()
    print('Done training.')

if __name__ == '__main__':
    main()