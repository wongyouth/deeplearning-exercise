# Find cross point position by Tensorflow

This is an example of deep learning to find the cross point from cross lines.

## Network

<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        832
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
_________________________________________________________________
gaussian_noise (GaussianNois (None, 14, 14, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0
_________________________________________________________________
dense (Dense)                (None, 784)               2459408
=================================================================
Total params: 2,478,736
Trainable params: 2,478,736
Non-trainable params: 0
_________________________________________________________________
</pre>

## Prequiments

You need to install some packages:

- tensorflow
- numpy

## Run

    python main.py

## Result

Result from 50000 train samples and 5 epoch

<pre>
Train on 50000 samples, validate on 10000 samples
Epoch 1/5
50000/50000 [==============================] - 57s 1ms/step - loss: 6.3021 - acc: 0.0165 - val_loss: 3.3563 - val_acc: 0.1796
Epoch 2/5
50000/50000 [==============================] - 59s 1ms/step - loss: 1.3473 - acc: 0.6318 - val_loss: 0.1209 - val_acc: 0.9942
Epoch 3/5
50000/50000 [==============================] - 60s 1ms/step - loss: 0.0315 - acc: 0.9990 - val_loss: 0.0047 - val_acc: 1.0000
Epoch 4/5
50000/50000 [==============================] - 64s 1ms/step - loss: 0.0036 - acc: 1.0000 - val_loss: 0.0016 - val_acc: 1.0000
Epoch 5/5
50000/50000 [==============================] - 60s 1ms/step - loss: 0.0014 - acc: 1.0000 - val_loss: 7.3396e-04 - val_acc: 1.0000
</pre>