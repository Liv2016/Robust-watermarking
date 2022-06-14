import itertools
import random
import tensorflow as tf
import numpy as np
import cv2

def build_dataset(file_names, batch_size, epoch, H, W, secret_size):
    file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
    file_queue = tf.train.string_input_producer(file_names, num_epochs=epoch, shuffle=True, capacity=16)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(file_queue)
    cover = tf.image.decode_jpeg(img_bytes)
    cover = tf.image.resize_images(cover, [H, W]) / 255
    secret = tf.keras.backend.random_binomial(shape=[secret_size], p=.5, dtype=tf.float32)
    cover_batch = tf.train.batch([cover], batch_size=batch_size, shapes=[H, W, 3])
    secret_batch = tf.train.batch([secret], batch_size=batch_size, shapes=[secret_size])
    return cover_batch, secret_batch

def normalize_RGB(img): # 10.18 add
    # img H W C
    img_max = tf.reduce_max(img, axis=[0, 1])
    img_min = tf.reduce_min(img, axis=[0, 1])
    return (img - img_min) / ((img_max - img_min) + 1e-9)


def MaxMinNormalization(batch): # 10.18 change
    return tf.map_fn(fn=normalize_RGB, elems=batch)

def cast_uint8(img):
    min_, max_ = tf.reduce_min(img), tf.reduce_max(img)
    img = (img - min_) / (max_ - min_)
    img = tf.cast(img, dtype=tf.float32)
    return img


def gausskern(kernlen=5, nsig=0.8):
    # Returns a 2D Gaussian kernel array.
    import scipy.stats as st
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)
    return kernel


def gassian_fiter(img_B):
    # B H W C
    shape = tf.shape(img_B)
    img_t = tf.transpose(img_B, [0, 3, 1, 2])
    img_r = tf.reshape(img_t, [shape[0] * shape[3], shape[1], shape[2]])
    img_g = tf.map_fn(fn=fg, elems=img_r)
    img_g = tf.reshape(img_g, [shape[0], shape[3], shape[1], shape[2]])
    img_g = tf.transpose(img_g, [0, 2, 3, 1])
    return img_g


def fg(img):
    img = tf.expand_dims(img, axis=-1)
    img = tf.expand_dims(img, axis=0)
    knernel = gausskern(5, 0.8)
    k = tf.reshape(knernel, [5, 5, 1, 1])
    img_gauss = tf.nn.conv2d(img, filter=k, strides=[1, 1, 1, 1], padding="SAME")
    img_gauss = tf.squeeze(img_gauss, axis=0)
    img_gauss = tf.squeeze(img_gauss, axis=-1)
    return img_gauss


def caclu_complex(img_B):
    # k1 = tf.convert_to_tensor(np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]]), dtype= tf.float32)
    # k2 = tf.convert_to_tensor(np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]]), dtype= tf.float32)
    # k3 = tf.convert_to_tensor(np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]]), dtype= tf.float32)
    # k4 = tf.convert_to_tensor(np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]]), dtype= tf.float32)
    # k1 = tf.reshape(k1, [3,3,1,1])
    # k2 = tf.reshape(k2, [3,3,1,1])
    # k3 = tf.reshape(k3, [3,3,1,1])
    # k4 = tf.reshape(k4, [3,3,1,1])
    img_pre = gassian_fiter(img_B)
    img_sobel = tf.image.sobel_edges(img_pre)
    img_sobel = tf.abs(img_sobel)
    img_sobel = tf.reduce_sum(img_sobel, axis=-1)
    img_sobel = tf.map_fn(fn=cast_uint8, elems=img_sobel)
    return img_sobel

def random_warp(attacked_image, M, H:int=512,max_factor:float = 0.05):
    axis_shift = np.floor(H * max_factor)
    pad_H = axis_shift.astype('int32')
    pad_W = axis_shift.astype('int32')
    stego_pad = tf.pad(attacked_image, [[0, 0], [pad_H, pad_H], [pad_W, pad_W], [0, 0]])
    warp_stego_pad = tf.contrib.image.transform(stego_pad, M[:, 1, :], interpolation='BILINEAR')
    pad_recover = tf.contrib.image.transform(warp_stego_pad, M[:, 0, :], interpolation='BILINEAR')
    result = pad_recover[:,pad_H:-pad_H, pad_W:-pad_W,:]
    return result

def batch_jepg_attack(batch_img, low=50, high=100):
    return tf.map_fn(lambda img: tf.image.random_jpeg_quality(img, low, high), elems=batch_img)

def random_blur_kernel(N_blur, probs=[.25, .25], sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.], wmin_line=3):
    N = N_blur
    coords = tf.to_float(tf.stack(tf.meshgrid(tf.range(N_blur), tf.range(N_blur), indexing='ij'), -1)) - (.5 * (N - 1))
    # coords = tf.to_float(coords)
    manhat = tf.reduce_sum(tf.abs(coords), -1)
    # nothing, default
    vals_nothing = tf.to_float(manhat < .5)
    # gauss
    sig_gauss = tf.random.uniform([], sigrange_gauss[0], sigrange_gauss[1])
    vals_gauss = tf.exp(-tf.reduce_sum(coords ** 2, -1) / 2. / sig_gauss ** 2)
    # line
    theta = tf.random_uniform([], 0, 2. * np.pi)
    v = tf.convert_to_tensor([tf.cos(theta), tf.sin(theta)])
    dists = tf.reduce_sum(coords * v, -1)
    sig_line = tf.random.uniform([], sigrange_line[0], sigrange_line[1])
    w_line = tf.random.uniform([], wmin_line, .5 * (N - 1) + .1)
    vals_line = tf.exp(-dists ** 2 / 2. / sig_line ** 2) * tf.to_float(manhat < w_line)
    t = tf.random_uniform([])
    vals = vals_nothing
    vals = tf.cond(t < probs[0] + probs[1], lambda: vals_line, lambda: vals)
    vals = tf.cond(t < probs[0], lambda: vals_gauss, lambda: vals)
    v = vals / tf.reduce_sum(vals)
    z = tf.zeros_like(v)
    f = tf.reshape(tf.stack([v, z, z, z, v, z, z, z, v], -1), [N, N, 3, 3])
    return f


def random_blur(watermarked_image, kenel_size: int = 7):
    # random blur
    k = random_blur_kernel(N_blur=kenel_size)
    attacked_image = tf.nn.conv2d(watermarked_image, k, [1, 1, 1, 1], padding='SAME')
    return attacked_image


def rnd_gaussain_noise(attacked_image, gauss_stddev):
    gaussian_noise = tf.random_normal(shape=tf.shape(attacked_image), mean=0.0, stddev=gauss_stddev, dtype=tf.float32)
    attacked_image = attacked_image + gaussian_noise
    # attacked_image = tf.clip_by_value(attacked_image, 0, 1)
    return attacked_image


def get_rnd_brightness_tf(rnd_bri, rnd_hue, batch_size):
    rnd_hue = tf.random.uniform((batch_size, 1, 1, 3), -rnd_hue, rnd_hue)
    rnd_brightness = tf.random.uniform((batch_size, 1, 1, 1), -rnd_bri, rnd_bri)
    return rnd_hue + rnd_brightness


def rnd_bri_cts(attacked_image, batch_size, contrast_low: float = .5, contrast_high: float = 1.5, rnd_bri: float = .3,
                rnd_hue: float = .1):
    contrast_scale = tf.random_uniform(shape=[tf.shape(attacked_image)[0]], minval=contrast_low, maxval=contrast_high)
    contrast_scale = tf.reshape(contrast_scale, shape=[tf.shape(attacked_image)[0], 1, 1, 1])
    rnd_brightness = get_rnd_brightness_tf(rnd_bri, rnd_hue, batch_size)

    attacked_image = attacked_image * contrast_scale
    attacked_image = attacked_image + rnd_brightness
    # attacked_image = tf.clip_by_value(attacked_image, 0, 1)
    return attacked_image


def random_saturation(attacked_image, rnd_sat):
    attacked_image_lum = tf.expand_dims(tf.reduce_sum(attacked_image * tf.constant([.3, .6, .1]), axis=3), 3)
    attacked_image = (1 - rnd_sat) * attacked_image + rnd_sat * attacked_image_lum
    return attacked_image


# jpeg compression start
def diff_round(x):
    return tf.round(x) + (x - tf.round(x)) ** 3


def round_only_at_0(x):
    cond = tf.cast(tf.abs(x) < 0.5, tf.float32)
    return cond * (x ** 3) + (1 - cond) * x


def rgb_to_ycbcr(image):
    matrix = np.array(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=np.float32).T
    shift = [0., 128., 128.]

    result = tf.tensordot(image, matrix, axes=1) + shift
    result.set_shape(image.shape.as_list())
    return result


def downsampling(image):
    # input: batch x height x width x 3
    # output: tuple of length 3
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    y, cb, cr = tf.split(image, 3, axis=3)
    cb = tf.nn.avg_pool(cb, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    cr = tf.nn.avg_pool(cr, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.squeeze(y, axis=-1), tf.squeeze(cb, axis=-1), tf.squeeze(cr, axis=-1)


def image_to_patches(image):
    # input: batch x h x w
    # output: batch x h*w/64 x h' x w'
    k = 8
    height, width = image.shape.as_list()[1:3]
    batch_size = tf.shape(image)[0]
    image_reshaped = tf.reshape(image, [batch_size, height // k, k, -1, k])
    image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
    return tf.reshape(image_transposed, [batch_size, -1, k, k])


def dct_8x8(image):
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    result = scale * tf.tensordot(image, tensor, axes=2)
    result.set_shape(image.shape.as_list())
    return result


def c_quantize(image, rounding, factor=1):
    c_table = np.empty((8, 8), dtype=np.float32)
    c_table.fill(99)
    c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                [24, 26, 56, 99], [47, 66, 99, 99]]).T
    image = image / (c_table * factor)
    image = rounding(image)
    return image


def c_dequantize(image, factor=1):
    c_table = np.empty((8, 8), dtype=np.float32)
    c_table.fill(99)
    c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                [24, 26, 56, 99], [47, 66, 99, 99]]).T
    return image * (c_table * factor)


def y_quantize(image, rounding, factor=1):
    y_table = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]],
        dtype=np.float32).T
    image = image / (y_table * factor)
    image = rounding(image)
    return image


def y_dequantize(image, factor=1):
    y_table = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]],
        dtype=np.float32).T
    return image * (y_table * factor)


def idct_8x8(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * alpha

    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)
    result = 0.25 * tf.tensordot(image, tensor, axes=2) + 128
    result.set_shape(image.shape.as_list())
    return result


def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    k = 8
    batch_size = tf.shape(patches)[0]
    image_reshaped = tf.reshape(patches, [batch_size, height // k, width // k, k, k])
    image_transposed = tf.transpose(image_reshaped, [0, 1, 3, 2, 4])
    return tf.reshape(image_transposed, [batch_size, height, width])


def upsampling_420(y, cb, cr):
    # input:
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    # output:
    #   image: batch x height x width x 3
    def repeat(x, k=2):
        height, width = x.shape.as_list()[1:3]
        x = tf.expand_dims(x, -1)
        x = tf.tile(x, [1, 1, k, k])
        x = tf.reshape(x, [-1, height * k, width * k])
        return x

    cb = repeat(cb)
    cr = repeat(cr)
    return tf.stack((y, cb, cr), axis=-1)


def ycbcr_to_rgb(image):
    matrix = np.array(
        [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
        dtype=np.float32).T
    shift = [0, -128, -128]

    result = tf.tensordot(image + shift, matrix, axes=1)
    result.set_shape(image.shape.as_list())
    return result


def jpeg_compress_decompress(image, downsample_c=True, rounding=round_only_at_0, factor=1):
    image *= 255
    height, width = image.shape.as_list()[1:3]
    orig_height, orig_width = height, width
    if height % 16 != 0 or width % 16 != 0:
        # Round up to next multiple of 16
        height = ((height - 1) // 16 + 1) * 16
        width = ((width - 1) // 16 + 1) * 16

        vpad = height - orig_height
        wpad = width - orig_width
        top = vpad // 2
        bottom = vpad - top
        left = wpad // 2
        right = wpad - left

        # image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
        image = tf.pad(image, [[0, 0], [0, vpad], [0, wpad], [0, 0]], 'SYMMETRIC')

    # "Compression"
    image = rgb_to_ycbcr(image)
    if downsample_c:
        y, cb, cr = downsampling(image)
    else:
        y, cb, cr = tf.split(image, 3, axis=3)
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = components[k]
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)
        comp = c_quantize(comp, rounding, factor) if k in ('cb', 'cr') else y_quantize(
            comp, rounding, factor)
        components[k] = comp

    # "Decompression"
    for k in components.keys():
        comp = components[k]
        comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(
            comp, factor)
        comp = idct_8x8(comp)
        if k in ('cb', 'cr'):
            if downsample_c:
                comp = patches_to_image(comp, int(height / 2), int(width / 2))
            else:
                comp = patches_to_image(comp, height, width)
        else:
            comp = patches_to_image(comp, height, width)
        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']
    if downsample_c:
        image = upsampling_420(y, cb, cr)
    else:
        image = tf.stack((y, cb, cr), axis=-1)
    image = ycbcr_to_rgb(image)

    # Crop to original size
    if orig_height != height or orig_width != width:
        # image = image[:, top:-bottom, left:-right]
        image = image[:, :-vpad, :-wpad]

    # Hack: RGB -> YUV -> RGB sometimes results in incorrect values
    #    min_value = tf.minimum(tf.reduce_min(image), 0.)
    #    max_value = tf.maximum(tf.reduce_max(image), 255.)
    #    value_range = max_value - min_value
    #    image = 255 * (image - min_value) / value_range
    image = tf.minimum(255., tf.maximum(0., image))
    image /= 255
    return image


def get_transform_matrix(image_size, d, batch_size):
    Ms = np.zeros((batch_size, 2, 8))
    for i in range(batch_size):
        tl_x = random.uniform(-d, d)  # Top left corner, top
        tl_y = random.uniform(-d, d)  # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)  # Bot left corner, left
        tr_x = random.uniform(-d, d)  # Top right corner, top
        tr_y = random.uniform(-d, d)  # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)  # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size]], dtype="float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i, 0, :] = M_inv.flatten()[:8]
        Ms[i, 1, :] = M.flatten()[:8]
    return Ms
