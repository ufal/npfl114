# Modified by Milan Straka <straka@ufal.mff.cuni.cz> for the NPFL114 course
# from https://github.com/qubvel/efficientnet.

# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy, Björn Barz. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

# Code of this model implementation is mostly written by
# Björn Barz ([@Callidior](https://github.com/Callidior))

import collections
import math
import os
import string
import sys
import urllib.request

import tensorflow as tf

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))


def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='', ):
    """Mobile Inverted Residual Bottleneck."""

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = tf.keras.layers.Conv2D(filters, 1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=CONV_KERNEL_INITIALIZER,
                                   name=prefix + 'expand_conv')(inputs)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'expand_bn')(x)
        x = tf.keras.layers.Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = tf.keras.layers.DepthwiseConv2D(block_args.kernel_size,
                                        strides=block_args.strides,
                                        padding='same',
                                        use_bias=False,
                                        depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                        name=prefix + 'dwconv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'bn')(x)
    x = tf.keras.layers.Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = tf.keras.layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if tf.keras.backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = tf.keras.layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = tf.keras.layers.Conv2D(num_reduced_filters, 1,
                                           activation=activation,
                                           padding='same',
                                           use_bias=True,
                                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                                           name=prefix + 'se_reduce')(se_tensor)
        se_tensor = tf.keras.layers.Conv2D(filters, 1,
                                           activation='sigmoid',
                                           padding='same',
                                           use_bias=True,
                                           kernel_initializer=CONV_KERNEL_INITIALIZER,
                                           name=prefix + 'se_expand')(se_tensor)
        x = tf.keras.layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = tf.keras.layers.Conv2D(block_args.output_filters, 1,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=CONV_KERNEL_INITIALIZER,
                               name=prefix + 'project_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(
            s == 1 for s in block_args.strides
    ) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = tf.keras.layers.Dropout(drop_rate,
                                        noise_shape=(None, 1, 1, 1),
                                        name=prefix + 'drop')(x)
        x = tf.keras.layers.add([x, inputs], name=prefix + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 blocks_args=DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=None,
                 classes=1000,
                 **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_resolution: int, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: int.
        blocks_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` contains path to nonexisting file.')

    if weights is not None and include_top and classes != 1000:
        raise ValueError('If using `weights` with `include_top` as true, `classes` should be 1000')

    # Determine proper input shape
    if input_shape is None:
        if tf.keras.backend.image_data_format() == "channels_last":
            input_shape = [default_resolution, default_resolution, 3]
        else:
            input_shape = [3, default_resolution, default_resolution]

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    activation = tf.nn.swish

    # Build stem
    x = img_input
    x = tf.keras.layers.Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,
                               strides=(2, 2),
                               padding='same',
                               use_bias=False,
                               kernel_initializer=CONV_KERNEL_INITIALIZER,
                               name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = tf.keras.layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    outputs = []
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        if block_args.strides != [1, 1]:
            outputs.append(x)
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in range(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(
                    idx + 1,
                    string.ascii_lowercase[bidx + 1]
                )
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  prefix=block_prefix)
                block_num += 1

    # Build top
    x = tf.keras.layers.Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=CONV_KERNEL_INITIALIZER,
                               name='top_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = tf.keras.layers.Activation(activation, name='top_activation')(x)
    outputs.append(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate and dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = tf.keras.layers.Dense(classes,
                                  activation='softmax',
                                  kernel_initializer=DENSE_KERNEL_INITIALIZER,
                                  name='probs')(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs.append(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    outputs.reverse()
    model = tf.keras.Model(inputs, outputs, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model


def EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.0, 224, 0.2,
        model_name='efficientnet-b0',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetB0, '__doc__', EfficientNet.__doc__)


def EfficientNetB1(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.0, 1.1, 240, 0.2,
        model_name='efficientnet-b1',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetB1, '__doc__', EfficientNet.__doc__)


def EfficientNetB2(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(
        1.1, 1.2, 260, 0.3,
        model_name='efficientnet-b2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetB2, '__doc__', EfficientNet.__doc__)


def EfficientNetB3(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(
        1.2, 1.4, 300, 0.3,
        model_name='efficientnet-b3',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetB3, '__doc__', EfficientNet.__doc__)


def EfficientNetB4(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.4, 1.8, 380, 0.4,
        model_name='efficientnet-b4',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetB4, '__doc__', EfficientNet.__doc__)


def EfficientNetB5(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.6, 2.2, 456, 0.4,
        model_name='efficientnet-b5',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetB5, '__doc__', EfficientNet.__doc__)


def EfficientNetB6(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        1.8, 2.6, 528, 0.5,
        model_name='efficientnet-b6',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetB6, '__doc__', EfficientNet.__doc__)


def EfficientNetB7(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        2.0, 3.1, 600, 0.5,
        model_name='efficientnet-b7',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetB7, '__doc__', EfficientNet.__doc__)


def EfficientNetL2(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        classes=1000,
        **kwargs
):
    return EfficientNet(
        4.3, 5.3, 800, 0.5,
        model_name='efficientnet-l2',
        include_top=include_top, weights=weights,
        input_tensor=input_tensor, input_shape=input_shape,
        classes=classes,
        **kwargs
    )
setattr(EfficientNetL2, '__doc__', EfficientNet.__doc__)


def pretrained_efficientnet_b0(include_top, dynamic_shape=False):
    url = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/models/"
    path = "efficientnet-b0_noisy-student.h5"

    if not os.path.exists(path):
        print("Downloading file {}...".format(path), file=sys.stderr)
        urllib.request.urlretrieve("{}/{}".format(url, path), filename=path)

    return EfficientNetB0(include_top, weights="efficientnet-b0_noisy-student.h5", input_shape=[None, None, 3] if dynamic_shape else None)
