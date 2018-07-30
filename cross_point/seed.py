import numpy as np
np.set_printoptions(linewidth=300)

np.random.seed(42)

IMG_HEIGHT = 28
IMG_WIDTH  = 28
RADIUS = 5

def create_data(n):
    X = []
    Y = []
    for i in range(n):
        x = np.random.randint(0, IMG_WIDTH)
        y = np.random.randint(0, IMG_HEIGHT)
        X.append(create_x(x, y, RADIUS))
        Y.append(create_y(x, y))
        if i % 1000 == 0:
            print('create %d data' % i)

    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)

def create_x(x, y, radius):
    data = np.ones((IMG_HEIGHT, IMG_WIDTH), 'uint8') * 255
    # 1. horizontal line
    start = max(x - radius, 0)
    end = min(x + radius + 1, IMG_WIDTH)

    for v in range(start, end):
        data[y, v] = np.random.randint(0, 50)

    # 2. vertical line
    start = max(y - radius, 0)
    end = min(y + radius + 1, IMG_HEIGHT)

    for v in range(start, end):
        data[v, x] = np.random.randint(0, 50)

    # print(data)
    return data

# y is index from the very first point
def create_y(x, y):
    return y * IMG_HEIGHT + x

def load_data(path):
    data = np.load(path)
    x_train, y_train, x_test, y_test = [data[name] for name in ['x_train', 'y_train', 'x_test', 'y_test']]
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    x_train, y_train = create_data(50000)
    x_test, y_test = create_data(10000)
    np.savez('data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
