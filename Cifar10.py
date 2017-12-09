import requests
import os
import config as cfg
import urllib
import sys
import tarfile
import pickle
import numpy as np

LABEL_SIZE = 1
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_CLASSES = 10

TRAIN_NUM = 10000
TRAIN_NUMS = 50000


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Cifar:
    def __init__(self):
        cifarFile = cfg.CIFAR_DATADIR + cfg.CIFAR_FILENAME
        if not os.path.exists(cifarFile):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading cifar10 %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(cfg.CIFAR_DATA_DOWNLOAD_URL, cifarFile,
                                                         reporthook=_progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', cfg.CIFAR_FILENAME, statinfo.st_size, 'bytes.')

        tarfile.open(cifarFile, 'r:gz').extractall(cfg.CIFAR_DATADIR)
        testfile = "test_batch"
        trainfiles = ["data_batch_" + str(i) for i in range(1,6)]
        self.train_images = []
        self.train_labels = []

        for file in trainfiles:
            data,label = self.extract_file(cfg.CIFAR_RAW_FILE_DIR + file)
            self.train_images.append(data)
            self.train_labels.append(label)
        self.train_images = np.reshape(self.train_images,[50000,32,32,3])
        self.train_labels = np.reshape(self.train_labels,[-1])

        data, label = self.extract_file(cfg.CIFAR_RAW_FILE_DIR + testfile)
        self.test_images = np.reshape(data,[10000,32,32,3])
        self.test_labels = np.reshape(label, [-1])

    def fetch_batch(self,batch = None,dataset = "train"):
        if batch is None:
            batch = cfg.BATCH_SIZE
        if dataset == "train":
            sample = np.random.choice(50000, batch)
            return self.train_images[sample],self.train_labels[sample]
        elif dataset == "test":
            if not hasattr(self,"testset_count"):
                self.testset_count = 0
            if self.testset_count + batch >= 10000:
                sample = range(self.testset_count,10000)
                self.testset_count = (self.testset_count + batch) % 10000
                sample = list(sample) + list(range(self.testset_count))
            else:
                sample = list(range(self.testset_count,self.testset_count + batch))
                self.testset_count += batch

            return self.test_images[sample],self.test_labels[sample]
        else:
            return None


    def extract_file(self,file_path):
        dict = unpickle(file_path)
        data = np.array(dict[b'data'])
        label = np.array(dict[b'labels'])
        data = np.reshape(data,[10000,3,1024])
        data = np.transpose(data,[0,2,1])
        return data, label