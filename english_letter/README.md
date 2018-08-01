# Recognize English character letters

TO recoginize handwritten 26 English character letters

## Dataset

Use dataset EMNIST Letters from https://www.nist.gov/itl/iad/image-group/emnist-dataset
And the download link is http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip

After unzip, the files located as

<pre>
data
├── emnist-letters-mapping.txt
├── emnist-letters-test-images-idx3-ubyte
├── emnist-letters-test-labels-idx1-ubyte
├── emnist-letters-train-images-idx3-ubyte
├── emnist-letters-train-labels-idx1-ubyte
</pre>


## Data preprocessing

The original image is like:

![orginal letter](img.png)

So it need to be flipped and rotate 90° clockwise, turns to:

![rotate 90](img2.png)


## Network

Network:

<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 64)        18496
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 64)        0
_________________________________________________________________
flatten (Flatten)            (None, 12544)             0
_________________________________________________________________
dense (Dense)                (None, 128)               1605760
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 26)                3354
=================================================================
Total params: 1,627,930
Trainable params: 1,627,930
Non-trainable params: 0
_________________________________________________________________

</pre>

## Data Sample

<pre>
[[255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255]
 [255 255 255 255 252 251 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255]
 [255 255 255 235 146 141 210 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255]
 [255 255 255 141  10   2  41 250 255 255 255 255 255 235 155 222 254 255 255 255 255 255 255 255 255 255 255 255]
 [255 255 255 130   6   1  22 233 255 255 255 255 245 132  17  92 222 255 255 255 255 253 251 251 255 255 255 255]
 [255 255 255 173  22   1   4 173 253 255 255 252 176  35   1  11 129 255 255 255 252 178 130 142 247 255 255 255]
 [255 255 255 218  38   1   1 127 250 255 255 221  51   1   0   1  39 250 255 233 101   8   1   2 144 252 255 255]
 [255 255 255 218  38   1   1  95 239 255 252 171  22   1   0   1  22 233 245 160  23   1   1   9 178 253 255 255]
 [255 255 255 216  38   1   1  71 230 255 221  80   4   1   0   1   5 173 127  34   2   0   3  78 247 255 255 255]
 [255 255 255 140  10   1   1  38 218 246  80   4   1   0   0   0   1   4   1   1   1   1  23 146 255 255 255 255]
 [255 255 254  98   4   1   0  38 215 178   9   1   0   0   0   0   0   0   0   0   1  13 124 233 255 255 255 255]
 [255 255 255 161  21   1   0  37 205 115   1   1   0   0   0   0   0   0   0   1   3 124 223 255 255 255 255 255]
 [255 255 255 234  83   3   0   6  28   8   1   2  16   7   1   0   0   0   1   2  48 246 255 255 255 255 255 255]
 [255 255 255 245 113   4   0   1   1   1   1  21 139  90   2   1   0   0   1  21 160 255 255 255 255 255 255 255]
 [255 255 255 223  52   1   0   0   0   1   2  53 220 218  20   1   0   0   1  38 216 255 255 255 255 255 255 255]
 [255 255 255 218  38   1   0   0   0   1  34 164 248 234  21   1   0   0   1  40 218 255 255 255 255 255 255 255]
 [255 255 255 218  38   1   0   0   1   1 140 251 255 223  10   1   0   1   9 128 247 255 255 255 255 255 255 255]
 [255 255 255 218  38   1   0   0   1  11 209 255 255 246  33   1   0   1  47 209 255 255 255 255 255 255 255 255]
 [255 255 255 218  38   1   0   0   3  78 248 255 255 251  38   1   0   5 115 246 255 255 255 255 255 255 255 255]
 [255 255 255 218  38   1   0   1  10 141 255 255 255 251  38   1   0  22 173 253 255 255 255 255 255 255 255 255]
 [255 255 255 218  38   1   0   1  35 205 255 255 255 251  38   1   1  38 216 255 255 255 255 255 255 255 255 255]
 [255 255 255 218  38   1   0   8  96 235 255 255 255 251  52   1   1  52 223 255 255 255 255 255 255 255 255 255]
 [255 255 255 173  22   1   1  39 217 255 255 255 255 255 140  15  22 145 251 255 255 255 255 255 255 255 255 255]
 [255 255 255 130   6   1   1  85 234 255 255 255 255 255 223 144 173 237 255 255 255 255 255 255 255 255 255 255]
 [255 255 255 141  10   1   4 172 252 255 255 255 255 255 255 252 253 255 255 255 255 255 255 255 255 255 255 255]
 [255 255 255 235 146 128 142 248 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255]
 [255 255 255 255 252 251 251 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255]
 [255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255 255]]
</pre>

 ## Train result

Train on 124800 samples, validate on 20800 samples

<pre>
Epoch 1/15
124800/124800 [==============================] - 274s 2ms/step - loss: 0.9946 - acc: 0.6965 - val_loss: 0.3594 - val_acc: 0.8898
Epoch 2/15
124800/124800 [==============================] - 273s 2ms/step - loss: 0.5672 - acc: 0.8198 - val_loss: 0.3003 - val_acc: 0.9054
Epoch 3/15
124800/124800 [==============================] - 905s 7ms/step - loss: 0.4885 - acc: 0.8437 - val_loss: 0.2696 - val_acc: 0.9150
Epoch 4/15
124800/124800 [==============================] - 273s 2ms/step - loss: 0.4415 - acc: 0.8588 - val_loss: 0.2530 - val_acc: 0.9171
Epoch 5/15
124800/124800 [==============================] - 3047s 24ms/step - loss: 0.4058 - acc: 0.8683 - val_loss: 0.2402 - val_acc: 0.9231
Epoch 6/15
124800/124800 [==============================] - 275s 2ms/step - loss: 0.3717 - acc: 0.8795 - val_loss: 0.2322 - val_acc: 0.9256
Epoch 7/15
124800/124800 [==============================] - 273s 2ms/step - loss: 0.3425 - acc: 0.8883 - val_loss: 0.2284 - val_acc: 0.9263
Epoch 8/15
124800/124800 [==============================] - 272s 2ms/step - loss: 0.3228 - acc: 0.8938 - val_loss: 0.2207 - val_acc: 0.9291
Epoch 9/15
124800/124800 [==============================] - 281s 2ms/step - loss: 0.3080 - acc: 0.8983 - val_loss: 0.2166 - val_acc: 0.9311
Epoch 10/15
124800/124800 [==============================] - 279s 2ms/step - loss: 0.2948 - acc: 0.9009 - val_loss: 0.2117 - val_acc: 0.9325
Epoch 11/15
124800/124800 [==============================] - 280s 2ms/step - loss: 0.2781 - acc: 0.9065 - val_loss: 0.2077 - val_acc: 0.9326
Epoch 12/15
124800/124800 [==============================] - 282s 2ms/step - loss: 0.2671 - acc: 0.9091 - val_loss: 0.2017 - val_acc: 0.9353
Epoch 13/15
124800/124800 [==============================] - 308s 2ms/step - loss: 0.2564 - acc: 0.9116 - val_loss: 0.2186 - val_acc: 0.9337
Epoch 14/15
124800/124800 [==============================] - 288s 2ms/step - loss: 0.2470 - acc: 0.9155 - val_loss: 0.2085 - val_acc: 0.9350
Epoch 15/15
124800/124800 [==============================] - 293s 2ms/step - loss: 0.2358 - acc: 0.9185 - val_loss: 0.2066 - val_acc: 0.9362
</pre>

Test loss: 0.20661989430769329
Test accuracy: 0.9361538461538461


## Concolusion

93% Accuracy may not high enough for production. There is also space for inprovment.

- Add more network
- Tuning adam parameters

But I am pleasure of it as ian exercise.