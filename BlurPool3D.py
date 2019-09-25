from typing import Tuple, Union

import keras
from keras import backend as K
from keras.layers import Layer

import numpy as np


class BlurPool3D(Layer):
    """
        https://arxiv.org/abs/1904.11486

        Keras implementation of BlurPool3D layer
         for "channels_last" image data format

        Original 1D and 2D PyTorch implementation can be found at
        https://github.com/adobe/antialiased-cnns
    """

    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int, int]],
        kernel_size: Union[int, Tuple[int, int, int]],
        **kwargs,
    ):
        if isinstance(pool_size, int):
            self.pool_size = (pool_size,) * 3
        else:
            self.pool_size = pool_size

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,) * 3
        else:
            self.kernel_size = kernel_size

        self.blur_kernel = None

        self.padding = tuple(
            (int(1.0 * (size - 1) / 2), int(np.ceil(1.0 * (size - 1) / 2)))
            for size in self.kernel_size
        )

        super().__init__(**kwargs)

    def build(self, input_shape):

        kernel_to_array = {
            1: np.array([1.0]),
            2: np.array([1.0, 1.0]),
            3: np.array([1.0, 2.0, 1.0]),
            4: np.array([1.0, 3.0, 3.0, 1.0]),
            5: np.array([1.0, 4.0, 6.0, 4.0, 1.0]),
            6: np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0]),
            7: np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0]),
        }

        a = kernel_to_array[self.kernel_size[0]]
        b = kernel_to_array[self.kernel_size[1]]
        c = kernel_to_array[self.kernel_size[2]]

        bk = a[:, None, None] * b[None, :, None] * c[None, None, :]
        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[4])

        new_shape = (*self.kernel_size, input_shape[4], 1)
        bk = np.reshape(bk, new_shape)
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(
            name='blur_kernel',
            shape=new_shape,
            initializer=blur_init,
            trainable=False,
        )

        super().build(input_shape)

    def call(self, x, **kwargs):
        x = K.spatial_3d_padding(x, padding=self.padding)

        # we imitate depthwise_conv3d actually
        channels = x.shape[-1]
        x = K.concatenate(
            [
                K.conv3d(
                    x=x[:, :, :, :, i : i + 1],
                    kernel=self.blur_kernel[..., i : i + 1, :],
                    strides=self.pool_size,
                    padding='valid',
                )
                for i in range(0, channels)
            ],
            axis=-1,
        )

        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            int(np.ceil(input_shape[1] / self.pool_size[0])),
            int(np.ceil(input_shape[2] / self.pool_size[1])),
            int(np.ceil(input_shape[3] / self.pool_size[2])),
            input_shape[4],
        )

    def get_config(self):
        base_config = super().get_config()
        base_config['pool_size'] = self.pool_size
        base_config['kernel_size'] = self.kernel_size

        return base_config

