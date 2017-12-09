CIFAR_FILENAME = "cifar-10-python.tar.gz"
CIFAR_DATADIR = "data/"
CIFAR_RAW_FILE_DIR = "data/cifar-10-batches-py/"
CIFAR_DATA_DOWNLOAD_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
IMAGE_SIZE = 32
CATEGORY_NUM = 10

BATCH_SIZE = 50


CAPSULE1_VEC_NUM = 16 # The real capsule1 vector num should be CAPSULE1_VEC_NUM * featuremap_size * featuremap_size
CAPSULE1_VEC_LEN = 8

CAPSULE2_VEC_NUM = 64
CAPSULE2_VEC_LEN = 16

CAPSULE3_VEC_NUM = CATEGORY_NUM
CAPSULE3_VEC_LEN = 16

ROUTE_ITER_COUNT = 3

MAX_POSITIVE_THRES = 0.9
MIN_NEGATIVE_THRES = 0.1
NEGATIVE_LOSS_WEIGHT = 0.5
REGULARIZATION_WEIGHT = 0.005


TRAIN_ITER_LOOP = 3000000
SAVE_ITER = 3000
SUMMARY_ITER = 100
LEARNING_RATE = 1e-5
WEIGHTS_FILE = '.\\save\\model.ckpt'
SUMMARY_OUTPUT_DIR = "./"
