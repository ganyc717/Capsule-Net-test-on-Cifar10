import tensorflow as tf
import CapsuleNet
import Cifar10
import config
import datetime
import os
import numpy as np

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    capsuleNet = CapsuleNet.CapsuleNet(is_training=False)
    cifar = Cifar10.Cifar()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allocator_type='BFC', allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    saver = tf.train.Saver()
    try:
        print('Try restoring weights from: ' + config.WEIGHTS_FILE)
        saver.restore(sess, config.WEIGHTS_FILE)
    except:
        print("Failed to initilize weight, return")
        return None

    correct = 0
    for step in range(10000//config.BATCH_SIZE):
        images, labels = cifar.fetch_batch(dataset="test")
        images = images / 255.0
        feed_dict = {capsuleNet.input_img: images}
        output = sess.run(capsuleNet.output_len, feed_dict=feed_dict)
        pred_label = np.argmax(output,axis=1)
        result = np.sum((pred_label == labels).astype(np.int32))
        correct += result
    print("correct rate on test set is ", correct/10000)

if __name__ == '__main__':
    main()