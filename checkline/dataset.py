import numpy as np
from skimage.draw import line_aa
import random

np.set_printoptions(linewidth=300)
random.seed(42)

##
# box data with line
#
# @param positive {int} postive line or negative line
# @param w {int} width
# @param h {int} height
# @param strokewidth {int} strokewidth
# @param delta {int} the max distance of the next point to previous point,
#                    if DELTA value is larger than STROKEWIDTH the line may become discontinuous
def lineBox(positive = True, w = 28, h = 28, strokewidth = 3, delta = 1):
    if random.random() > 0.5:
        data = createBox1(w, h)
    else:
        data = createBox2(w, h, strokewidth, delta)

    if positive:
        return data
    else:
        if random.random() > 0.5:
            data = negative1(w, h, data)
        else:
            data = negative2(w, h, data)

        return data

# the length of line is less than 50%
def negative1(w, h, data):
    data2 = np.zeros((h, w), dtype=np.uint8)

    # get segmental line as negative line
    offset = random.randint(0, h // 2)
    length = random.randint(1, h // 2)
    zone = range(offset, offset + length)

    data2[zone, :] = 1
    data = data2 * data

    return data

# cross line case
def negative2(w, h, data):
    y_start = random.randint(0, h-1)
    y_end = random.randint(0, h-1)

    index = np.nonzero(data[y_start])[0][0]
    x_start = random.randint(0, max(0, index-2))

    index = np.nonzero(data[y_end])[0][-1]
    x_end = random.randint(min(w-1, index + 2), w-1)

    rr, cc, val = line_aa(y_start, x_start, y_end,  x_end)
    data[rr, cc] = val * 255

    return data

# anti-alias line
def createBox1(w = 28, h = 28):
    data = np.zeros((h, w), dtype=np.uint8)

    start = random.randint(0, w-1)
    end = random.randint(0, w-1)

    rr, cc, val = line_aa(0, start, h - 1, end)
    data[rr, cc] = val * 255

    return data

def createBox2(w = 28, h = 28, strokewidth = 3, delta = 1):
    data = np.zeros((h, w), dtype=np.uint8)

    for (x,y) in line(w, delta, h-1):
        width = random.randint(1, strokewidth)
        start = max(x - width // 2, 0)
        data[y][start:start+width] = 255

    return data

# line data
def line(w, delta, y):
    if y == 0:
        x = random.randrange(w)
        return [(x, y)]
    else:
        prev = line(w, delta, y - 1)

        # change location every 3 points
        if y % 3 == 0:
            x = prev[-1][0] + random.randint(-delta, delta)
        else:
            x = prev[-1][0]

        x = max(x, 0)
        x = min(x, w-1)
        prev.append((x,y))
        return prev

def load_data(size):
    positive = size // 10 * 6
    negative = size - positive
    
    X = []
    Y = []

    for _ in range(positive):
        X.append(lineBox(positive=True))
        Y.append(1)

    for _ in range(negative):
        X.append(lineBox(positive=False))
        Y.append(0)
    
    X = np.array(X)
    Y = np.array(Y)

    return (X, Y)

if __name__ == '__main__':
    # canvas padding size
    PADDING = 4
    # box size
    W = H = 28

    # canvas size
    BOX_W = W + 2 * PADDING
    BOX_H = H + 2 * PADDING

    img = np.zeros((BOX_H * 10, BOX_W * 10)).astype('uint8')
    # print(img.dtype)

    for row in range(10):
        img[row * BOX_H, :] = 64
        img[:, row * BOX_H] = 64

        for col in range(10):
            positive = True if row < 5 else False
            box = lineBox(w = W, h = H, positive = positive)
            y = row * BOX_H + PADDING
            x = col * BOX_W + PADDING
            img[y:y + H, x:x + W] = box

    import cv2 as cv
    cv.imshow('img', img)
    cv.waitKey()

    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
