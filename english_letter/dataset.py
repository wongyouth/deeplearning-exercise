import struct as st
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

np.set_printoptions(linewidth=300)

# Use dataset EMNIST Letters from https://www.nist.gov/itl/iad/image-group/emnist-dataset
# Download link: http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip

DATA_DIR = 'data/'
f_x_train = 'emnist-letters-train-images-idx3-ubyte'
f_y_train = 'emnist-letters-train-labels-idx1-ubyte'
f_x_test = 'emnist-letters-test-images-idx3-ubyte'
f_y_test = 'emnist-letters-test-labels-idx1-ubyte'

def load_data():
    x_train = load_feature(open(DATA_DIR + f_x_train, 'rb'))
    y_train = load_label(open(DATA_DIR + f_y_train, 'rb'))
    x_test  = load_feature(open(DATA_DIR + f_x_test, 'rb'))
    y_test  = load_label(open(DATA_DIR + f_y_test, 'rb'))

    return (x_train, y_train, x_test, y_test)

def load_feature(file):
    return idx2array(file, 'idx3')

def load_label(file):
    return idx2array(file, 'idx1')

# idx convert to array
# idx format http://yann.lecun.com/exdb/mnist/
# https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
#
# the original data should be rotate 90° clockwise and flip 180°
def idx2array(file, ftype):
    file.seek(0)
    magic = st.unpack('>4B', file.read(4))
    num = st.unpack('>I', file.read(4))[0]

    if ftype == 'idx3':
        nR = st.unpack('>I', file.read(4))[0]
        nC = st.unpack('>I', file.read(4))[0]
        total = num * nR * nC * 1
        array = np.array(st.unpack('>' + 'B' * total, file.read(total)))
        array = array.reshape((num, nR, nC)).astype('uint8')
        # flip
        array = np.flip(array, 1)
        # rotate 90 deg clockwise
        array = np.rot90(array, -1, (1, 2))
        array = 255 - array
    else:
        total = num
        array = np.array(st.unpack('>' + 'B' * total, file.read(total)))
        array = array.reshape((num,)).astype('uint8')
        array = array - 1

    return array


if __name__ == '__main__':
    x = load_feature(open(DATA_DIR + f_x_test, 'rb'))
    y = load_label(open(DATA_DIR + f_y_test, 'rb'))

    print('max', np.max(y))

    img = Image.fromarray(x[0], 'L')
    img = img.resize((280, 280))
    img.show()

    for n in range(1):
        print(x[n])
        print(y[n])
