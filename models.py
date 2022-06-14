import numpy as np
import tensorflow as tf
from stn import spatial_transformer_network as stn_transformer
# from tensorflow.python.keras.models import *
# from tensorflow.python.keras.layers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import util
import lpips.lpips_tf as lpips_tf
from moire import *
rnd_moire = generator()

class Uplusplus(Layer):
    def __init__(self, height, width, base_num: int = 32):
        super(Uplusplus, self).__init__()
        self.hidden = Conv2D(base_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.down00 = Conv2D(base_num * 2, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down10 = Conv2D(base_num * 4, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down20 = Conv2D(base_num * 8, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down30 = Conv2D(base_num * 16, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half

        # up
        self.up10 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up20 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up11 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up30 = Conv2D(base_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up21 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up12 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up40 = Conv2D(base_num * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up31 = Conv2D(base_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up22 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up13 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')

    def call(self, image):
        # v begin
        x00 = self.hidden(image)
        # print("x00:", x00.shape)
        # batch x 512 x 512 x 32

        x10 = self.down00(x00)
        # print("x10:", x10.shape)
        # batch x 256 x 256 x 64

        up01 = UpSampling2D(size=(2, 2))(x10)
        merge_00_01 = concatenate([x00, up01], axis=-1)
        x01 = self.up10(merge_00_01)
        # print("x01", x01.shape)
        # batch x 512 x 512 x 32
        # v end

        # v begin
        x20 = self.down10(x10)
        # print("x20:", x20.shape)
        # batch x 128 x 128 x 128
        up11 = UpSampling2D(size=(2, 2))(x20)
        merge_10_11 = concatenate([x10, up11], axis=-1)
        x11 = self.up20(merge_10_11)
        # print("x11:", x11.shape)
        # batch x 256 x 256 x 64

        up02 = UpSampling2D(size=(2, 2))(x11)
        merge_00_01_02 = concatenate([x00, x01, up02], axis=-1)
        x02 = self.up11(merge_00_01_02)
        # print("x02:", x02.shape)
        # batch x 512 x 512 x 32

        # v end
        # v begin
        x30 = self.down20(x20)
        # print("x30:", x30.shape)
        # batch x 64 x 64 x 256
        up21 = UpSampling2D(size=(2, 2))(x30)
        merge_20_21 = concatenate([x20, up21], axis=-1)
        x21 = self.up30(merge_20_21)
        # print("x21:", x21.shape)
        # batch x 128 x 128 x 128

        up12 = UpSampling2D(size=(2, 2))(x21)
        merge_10_11_12 = concatenate([x10, x11, up12], axis=-1)
        x12 = self.up21(merge_10_11_12)
        # print("x12:", x12.shape)
        # batch x 256 x 256 x 64

        up03 = UpSampling2D(size=(2, 2))(x12)
        merge_00_01_02_03 = concatenate([x00, x01, x02, up03], axis=-1)
        x03 = self.up12(merge_00_01_02_03)
        # print("x03:", x03.shape)
        # batch x 512 x 512 x 32

        # v end
        # v begin
        x40 = self.down30(x30)
        # print("x40:", x40.shape)
        # batch x 32 x 32 x 512

        up31 = UpSampling2D(size=(2, 2))(x40)
        merge_30_31 = concatenate([x30, up31], axis=-1)
        x31 = self.up40(merge_30_31)
        # print("x31:", x31.shape)
        # batch x 64 x 64 x 256

        up22 = UpSampling2D(size=(2, 2))(x31)
        merge_20_21_22 = concatenate([x20, x21, up22], axis=-1)
        x22 = self.up31(merge_20_21_22)
        # print("x22:", x22.shape)
        # batch x 128 x 128 x 128

        up13 = UpSampling2D(size=(2, 2))(x22)
        merge_10_11_12_13 = concatenate([x10, x11, x12, up13], axis=-1)
        x13 = self.up22(merge_10_11_12_13)
        # print("x13:", x13.shape)
        # batch x 256 x 256 x 64

        up04 = UpSampling2D(size=(2, 2))(x13)
        merge_00_01_02_03_04 = concatenate([x00, x01, x02, x03, up04], axis=-1)
        x04 = self.up13(merge_00_01_02_03_04)
        # print("x04:", x04.shape)
        # batch x 512 x 512 x 32
        # v end
        return x04


class WatermarkEncoder2(Layer):
    def __init__(self, height, width, base_num: int = 32):
        super(WatermarkEncoder2, self).__init__()
        self.fc = Dense(64 * 64 * 3, activation='relu', kernel_initializer='he_normal')
        self.base_filter_num = base_num
        self.Encoder = Uplusplus(height=height, width=width, base_num=base_num)
        self.RGB_recover = Conv2D(3, 1, activation='tanh', padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5
        secret = self.fc(secret)
        secret = Reshape((64, 64, 3))(secret)
        # secret = batch x 50 x 50 x 3
        secret_up = UpSampling2D(size=(8, 8))(secret)
        inputs = concatenate([secret_up, image], axis=-1)
        x04 = self.Encoder(inputs)
        output = self.RGB_recover(x04)
        return output


class WatermarkEncoder(Layer):
    def __init__(self, height, width, base_num: int = 32):
        super(WatermarkEncoder, self).__init__()
        self.multiple = 1
        if height % 8 == 0:
            self.multiple = 8
        elif height % 4 == 0:
            self.multiple = 4
        elif height % 2 == 0:
            self.multiple = 2
        self.pre_h = height // self.multiple
        self.pre_w = width // self.multiple
        self.fc = Dense(self.pre_h * self.pre_w * 3, activation='relu', kernel_initializer='he_normal')

        self.base_filter_num = base_num
        self.hidden = Conv2D(base_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.down00 = Conv2D(base_num * 2, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down10 = Conv2D(base_num * 4, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down20 = Conv2D(base_num * 8, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down30 = Conv2D(base_num * 16, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half

        # up
        self.up10 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up20 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up11 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up30 = Conv2D(base_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up21 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up12 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up40 = Conv2D(base_num * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up31 = Conv2D(base_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up22 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up13 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')

        self.RGB_recover = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.fc(secret)
        secret = Reshape((self.pre_h, self.pre_w, 3))(secret)
        # secret = batch x 50 x 50 x 3
        secret_up = UpSampling2D(size=(self.multiple, self.multiple))(secret)
        inputs = concatenate([secret_up, image], axis=-1)
        # print("input:", inputs.shape)
        # batch x 512 x 512 x 6

        # v begin
        x00 = self.hidden(inputs)
        # print("x00:", x00.shape)
        # batch x 512 x 512 x 32

        x10 = self.down00(x00)
        # print("x10:", x10.shape)
        # batch x 256 x 256 x 64

        up01 = UpSampling2D(size=(2, 2))(x10)
        merge_00_01 = concatenate([x00, up01], axis=-1)
        x01 = self.up10(merge_00_01)
        # print("x01", x01.shape)
        # batch x 512 x 512 x 32
        # v end

        # v begin
        x20 = self.down10(x10)
        # print("x20:", x20.shape)
        # batch x 128 x 128 x 128
        up11 = UpSampling2D(size=(2, 2))(x20)
        merge_10_11 = concatenate([x10, up11], axis=-1)
        x11 = self.up20(merge_10_11)
        # print("x11:", x11.shape)
        # batch x 256 x 256 x 64

        up02 = UpSampling2D(size=(2, 2))(x11)
        merge_00_01_02 = concatenate([x00, x01, up02], axis=-1)
        x02 = self.up11(merge_00_01_02)
        # print("x02:", x02.shape)
        # batch x 512 x 512 x 32

        # v end
        # v begin
        x30 = self.down20(x20)
        # print("x30:", x30.shape)
        # batch x 64 x 64 x 256
        up21 = UpSampling2D(size=(2, 2))(x30)
        merge_20_21 = concatenate([x20, up21], axis=-1)
        x21 = self.up30(merge_20_21)
        # print("x21:", x21.shape)
        # batch x 128 x 128 x 128

        up12 = UpSampling2D(size=(2, 2))(x21)
        merge_10_11_12 = concatenate([x10, x11, up12], axis=-1)
        x12 = self.up21(merge_10_11_12)
        # print("x12:", x12.shape)
        # batch x 256 x 256 x 64

        up03 = UpSampling2D(size=(2, 2))(x12)
        merge_00_01_02_03 = concatenate([x00, x01, x02, up03], axis=-1)
        x03 = self.up12(merge_00_01_02_03)
        # print("x03:", x03.shape)
        # batch x 512 x 512 x 32

        # v end
        # v begin
        x40 = self.down30(x30)
        # print("x40:", x40.shape)
        # batch x 32 x 32 x 512

        up31 = UpSampling2D(size=(2, 2))(x40)
        merge_30_31 = concatenate([x30, up31], axis=-1)
        x31 = self.up40(merge_30_31)
        # print("x31:", x31.shape)
        # batch x 64 x 64 x 256

        up22 = UpSampling2D(size=(2, 2))(x31)
        merge_20_21_22 = concatenate([x20, x21, up22], axis=-1)
        x22 = self.up31(merge_20_21_22)
        # print("x22:", x22.shape)
        # batch x 128 x 128 x 128

        up13 = UpSampling2D(size=(2, 2))(x22)
        merge_10_11_12_13 = concatenate([x10, x11, x12, up13], axis=-1)
        x13 = self.up22(merge_10_11_12_13)
        # print("x13:", x13.shape)
        # batch x 256 x 256 x 64

        up04 = UpSampling2D(size=(2, 2))(x13)
        merge_00_01_02_03_04 = concatenate([x00, x01, x02, x03, up04], axis=-1)
        x04 = self.up13(merge_00_01_02_03_04)
        # print("x04:", x04.shape)
        # batch x 512 x 512 x 32
        # v end
        output = self.RGB_recover(x04)
        return output


class WatermarkDecoder(Layer):
    def __init__(self, secret_size, height, width, base_num: int = 32):
        super(WatermarkDecoder, self).__init__()
        self.height = height
        self.width = width
        self.stn_params = Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu')
        ])
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32').flatten()

        self.W_fc1 = tf.Variable(tf.zeros([128, 6]), name='W_fc1')
        self.b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        # self.decoder = Uplusplus(width=width, height=height, base_num=base_num)
        self.de_secret = Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
            Flatten(),
            Dense(512, activation='relu', kernel_initializer='he_normal'),
            Dense(secret_size)
        ])

    def call(self, image):
        image = image - .5
        stn_params = self.stn_params(image)
        x = tf.matmul(stn_params, self.W_fc1) + self.b_fc1
        transformed_image = stn_transformer(image, x, [self.height, self.width, 3])
        # de_image = self.decoder(transformed_image)
        # pre_secret = self.de_secret(de_image)
        pre_secret = self.de_secret(transformed_image)
        return pre_secret


def image_to_summary(image, name, family='Visual_Result'):
    image = tf.clip_by_value(image, 0, 1)
    image = tf.cast(image * 255, dtype=tf.uint8)
    summary = tf.summary.image(name, image, max_outputs=1, family=family)
    return summary


def calculate_acc(secret_pre, secret_true):
    secret_pre_r = tf.round(secret_pre)
    secret_pre_r = tf.cast(secret_pre_r, dtype=tf.int8)
    secret_label = tf.cast(secret_true, dtype=tf.int8)
    return tf.contrib.metrics.accuracy(secret_pre_r, secret_label)


def get_secret_acc(secret_true, secret_pred):
    with tf.variable_scope("acc"):
        secret_pred = tf.round(tf.sigmoid(secret_pred))
        correct_pred = tf.count_nonzero(secret_pred - secret_true, axis=1)
        bit_acc = 1 - tf.reduce_sum(correct_pred) / tf.size(secret_pred, out_type=tf.int64)
        return bit_acc


def noise_attack(watermarked_image, TFM, args, global_step):
    # use_second = tf.constant(value=args.use_second, dtype=tf.bool)
    linear_fn = lambda linear_total_step: tf.minimum(tf.to_float(global_step) / linear_total_step, 1.)

    # random blur
    # k = util.random_blur_kernel(N_blur=7)
    # attacked_image = tf.nn.conv2d(watermarked_image, k, [1, 1, 1, 1], padding='SAME')
    attacked_image = util.random_blur(watermarked_image, kenel_size=7)

    # gaussian nosie
    # gaussian_noise = tf.random_normal(shape=tf.shape(attacked_image), mean=0.0, stddev=args.gauss_stddev, dtype=tf.float32)
    # attacked_image = attacked_image + gaussian_noise
    # attacked_image = tf.clip_by_value(attacked_image, 0, 1)

    rnd_stddev = tf.random.uniform([]) * linear_fn(args.gaussian_step) * args.gauss_stddev
    attacked_image = util.rnd_gaussain_noise(attacked_image, gauss_stddev=rnd_stddev)

    # contrast & brightness shift
    # contrast_params = [.5, 1.5]
    # rnd_bri = .3
    # rnd_hue = .1
    #
    # contrast_scale = tf.random_uniform(shape=[tf.shape(attacked_image)[0]], minval=contrast_params[0],
    #                                    maxval=contrast_params[1])
    # contrast_scale = tf.reshape(contrast_scale, shape=[tf.shape(attacked_image)[0], 1, 1, 1])
    # rnd_brightness = util.get_rnd_brightness_tf(rnd_bri, rnd_hue, args.batch_size)
    #
    # attacked_image = attacked_image * contrast_scale
    # attacked_image = attacked_image + rnd_brightness
    # attacked_image = tf.clip_by_value(attacked_image, 0, 1)
    cts_low = 1. - (1. - args.cts_low) * linear_fn(args.cts_step)
    cts_high = 1. + (args.cts_high - 1.) * linear_fn(args.cts_step)
    rnd_bri = linear_fn(args.bri_step) * args.max_bri
    rnd_hue = linear_fn(args.hue_step) * args.max_hue

    if not args.use_second:
        attacked_image = util.rnd_bri_cts(attacked_image, args.batch_size, contrast_low=cts_low, contrast_high=cts_high,
                                          rnd_bri=rnd_bri, rnd_hue=rnd_hue)
    else:
        attacked_image = tf.image.random_contrast(attacked_image, lower=cts_low, upper=cts_high)
        attacked_image = tf.image.random_hue(attacked_image, max_delta=rnd_hue)
        attacked_image = tf.image.random_brightness(attacked_image, max_delta=rnd_bri)

    # random rat

    rnd_sat = tf.random.uniform([]) * linear_fn(args.sat_step) * args.rnd_sat
    if not args.use_second:
        # attacked_image_lum = tf.expand_dims(tf.reduce_sum(attacked_image * tf.constant([.3, .6, .1]), axis=3), 3)
        # attacked_image = (1 - rnd_sat) * attacked_image + rnd_sat * attacked_image_lum
        attacked_image = util.random_saturation(attacked_image, rnd_sat=rnd_sat)
    else:
        # my
        attacked_image = tf.image.adjust_saturation(attacked_image, saturation_factor=rnd_sat)
        # attacked_image = tf.image.random_saturation(attacked_image, lower=.0, upper=1.0)

    # jpeg compression
    if not args.use_second:
        rnd_factor = tf.random.uniform([]) + 0.1
        attacked_image = util.jpeg_compress_decompress(attacked_image, factor=rnd_factor)
        # attacked_image = util.batch_jepg_attack(attacked_image, low=50, high=100)
    else:
        # attacked_image = util.batch_jepg_attack(attacked_image, low=50, high=100)
        rnd_factor = tf.random.uniform([]) + 0.1
        attacked_image = util.jpeg_compress_decompress(attacked_image, factor=rnd_factor)

    # moire
    # moire_noise = rnd_moire(args.cover_h, args.cover_w) / 255
    # moire_noise = tf.convert_to_tensor(moire_noise, dtype= tf.float32)
    # moire_noise = tf.expand_dims(moire_noise, axis=0)
    # alpha = 0.3
    # attacked_image = (1-alpha) * attacked_image + alpha * moire_noise


    # warp
    if args.is_in_warp:
        attacked_image = util.random_warp(attacked_image, TFM, H=args.cover_h, max_factor=args.max_warp)
    noise_config = [tf.summary.scalar('nosie_config/rnd_bri', rnd_bri, family='noise_config'),
                    tf.summary.scalar('nosie_config/rnd_sat', rnd_sat, family='noise_config'),
                    tf.summary.scalar('nosie_config/rnd_hue', rnd_hue, family='noise_config'),
                    tf.summary.scalar('nosie_config/rnd_noise', rnd_stddev, family='noise_config'),
                    tf.summary.scalar('nosie_config/contrast_low', cts_low, family='noise_config'),
                    tf.summary.scalar('nosie_config/contrast_high', cts_high, family='noise_config'),
                    tf.summary.scalar('nosie_config/jpeg_attack_strength', rnd_factor, family='noise_config')]
    return attacked_image, noise_config


def make_graph3(Encoder, Decoder, cover_batch, secret_batch, loss_ratio_pl, args, TFM, global_step):
    M_map = Encoder([secret_batch, cover_batch])

    T_map = util.caclu_complex(cover_batch) * 255
    T_map = tf.abs(T_map)
    T_map = tf.clip_by_value(T_map, clip_value_min=0.01, clip_value_max=255)

    epsilon = 0.8
    N_map = tf.tanh(M_map)
    M_map_visual = N_map
    N_map = epsilon * N_map * T_map

    N_map_n = util.MaxMinNormalization(N_map) - 0.5
    damping = tf.cond(tf.greater_equal(global_step, args.restart),
                      lambda: (1 - args.damping_end) * tf.minimum(tf.to_float(global_step - args.restart) / 10000, 1.0),
                      lambda: .0)
    damping = 1 - damping
    N_map_n = N_map_n * damping

    last_map = tf.cond(tf.greater_equal(global_step, args.fast_step),
                       lambda: N_map_n,
                       lambda: N_map)

    N_map = tf.cond(tf.greater_equal(global_step, args.attention_start),
                    lambda: last_map,
                    lambda: M_map)

    watermark_image = cover_batch + N_map
    attack_stego, noise_config = noise_attack(watermark_image, TFM=TFM, args=args, global_step=global_step)
    pre_secret = Decoder(attack_stego)

    # loss
    # bit_acc = calculate_acc(pre_secret, secret_batch)
    bit_acc = get_secret_acc(secret_batch, pre_secret)
    psnr_cover = tf.reduce_mean(tf.image.psnr(cover_batch, watermark_image, max_val=1.0))

    loss_secret = tf.losses.sigmoid_cross_entropy(secret_batch, pre_secret)
    loss_lpips = tf.reduce_mean(lpips_tf.lpips(watermark_image, cover_batch))
    loss_mse = tf.losses.mean_squared_error(cover_batch, watermark_image)

    loss_total = loss_secret * loss_ratio_pl[0] + loss_lpips * loss_ratio_pl[1] + loss_mse * loss_ratio_pl[2]
    config_op = tf.summary.merge([
                                     tf.summary.scalar('bit_acc', bit_acc, family='loss_config'),
                                     tf.summary.scalar('psnr_cover', psnr_cover, family='loss_config'),
                                     tf.summary.scalar('total_loss', loss_total, family='loss_config'),
                                     tf.summary.scalar('MSE_loss', loss_mse, family='loss_config'),
                                     tf.summary.scalar('Lpip_loss', loss_lpips, family='loss_config'),
                                     tf.summary.scalar('secret_loss', loss_secret, family='loss_config'),
                                     tf.summary.scalar('MSE_loss_ratio', loss_ratio_pl[2], family='loss_config'),
                                     tf.summary.scalar('Lpip_loss_ratio', loss_ratio_pl[1], family='loss_config'),
                                     tf.summary.scalar('secret_loss_ratio', loss_ratio_pl[0], family='loss_config'),
                                 ] + noise_config)

    image_summary_op = tf.summary.merge([
        image_to_summary(cover_batch, 'cover_image', family='Visual_Result'),
        image_to_summary(M_map_visual, 'M_map', family='Visual_Result'),
        image_to_summary(T_map / 255.0, 'T_map', family='Visual_Result'),
        image_to_summary(N_map, 'N_map', family='Visual_Result'),
        image_to_summary(watermark_image, 'watermarked_image', family='Visual_Result'),
        image_to_summary(attack_stego, 'attacked_image', family='Visual_Result'),
    ])
    return loss_total, loss_secret, config_op, image_summary_op, bit_acc


def make_encode_graph(Encoder, cover_batch, secret_batch, damping_end):
    M_map = Encoder([secret_batch, cover_batch])

    T_map = util.caclu_complex(cover_batch) * 255
    T_map = tf.abs(T_map)
    T_map = tf.clip_by_value(T_map, clip_value_min=0.01, clip_value_max=255)

    epsilon = 0.8
    N_map = tf.tanh(M_map)
    M_map_visual = N_map
    N_map = epsilon * N_map * T_map
    N_map_n = (util.MaxMinNormalization(N_map) - 0.5) * damping_end

    watermark_image = cover_batch + N_map_n
    watered_image = tf.clip_by_value(watermark_image, 0, 1)
    return watered_image, M_map_visual, T_map, N_map_n


def make_decode_graph(Decoder: WatermarkDecoder, watered_image):
    predict_secret = Decoder(watered_image)
    return tf.round(tf.sigmoid(predict_secret))


def mymodel_test():
    myEncoder = WatermarkEncoder(width=512, height=512)

    secret = np.random.binomial(1, .5, 100)
    secret = tf.cast(secret, dtype=tf.float32)
    secret = tf.reshape(secret, [1, 100])
    img = tf.random_normal([1, 512, 512, 3])
    res = myEncoder([secret, img])

    myDecoder = WatermarkDecoder(100, 512, 512)
    de_secret = myDecoder(res)
    x = tf.random_normal([1, 400, 400, 3])
    y = tf.pad(x, [[0, 0], [40, 40], [40, 40], [0, 0]])
    y_shape = y.shape
    z = y[:, 40:-40, 40:-40, :]
    z_shape = z.shape
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(de_secret))
