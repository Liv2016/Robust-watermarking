import numpy as np
import cv2
import os
from math import ceil


class moire_noise_Options:
    """
    Configuration options for the moire_noise
    """

    def __init__(self,
                 datapath: str = "../data/sample_images",
                 file: str = "lena512color.png",
                 save_path: str = '../data/output',
                 save_format: str = "jpg",
                 canvas_dim: int = 512,
                 empty: bool = True,
                 gamma: int = 1,
                 type: str = "sine",
                 recapture_verbose: bool = False,
                 psnr: bool = False,
                 show_mask: bool = True,
                 H: int = 400,
                 W: int = 400,
                 save_name=None,
                 seed=None,
                 ):
        self.datapath = datapath
        self.file = file
        self.save_path = save_path
        self.save_name = save_name
        self.save_format = save_format
        self.canvas_dim = canvas_dim
        self.empty = empty
        self.gamma = gamma
        self.type = type
        self.recapture_verbose = recapture_verbose
        self.psnr = psnr
        self.seed = seed
        self.show_mask = show_mask
        self.H = H
        self.W = W


def generator():
    op = moire_noise_Options()
    if op.empty:
        original = np.zeros([op.canvas_dim, op.canvas_dim, 3])
        canvas_dim = op.canvas_dim if type(op.canvas_dim) == list else [op.canvas_dim, ] * 2
        canvas = np.ones(canvas_dim + [3, ], np.uint8) * 255  # blank white image
    else:
        canvas = cv2.imread(os.path.join(op.datapath, op.file), cv2.IMREAD_COLOR)
        original = canvas.copy()
    H, W, _ = canvas.shape

    dst_H, dst_W, _ = original.shape

    src_pt = np.zeros((4, 2), dtype="float32")
    src_pt[0] = [W // 4, H // 4]
    src_pt[1] = [W // 4 * 3, H // 4]
    src_pt[2] = [W // 4 * 3, H // 4 * 3]
    src_pt[3] = [W // 4, H // 4 * 3]

    recap_module = RecaptureModule(nl_moire=True, nl_dir='b', nl_type='sine', nl_skew=0,
                                   nl_cont=10, nl_dev=1, nl_tb=0.20, nl_lr=0.20,seed=op.seed)
    """
                                       v_moire=0, v_type='sg', v_skew=[20, 80], v_cont=10, v_dev=3,
                                   h_moire=0, h_type='f', h_skew=[20, 80], h_cont=10, h_dev=3,
                                   nl_moire=True, nl_dir='b', nl_type='sine', nl_skew=0,
                                   nl_cont=10, nl_dev=3, nl_tb=0.15, nl_lr=0.15,
                                   gamma=op.gamma, margins=None, seed=op.seed)
    """
    return recap_module


class RecaptureModule(object):
    def __init__(self,nl_moire=False, nl_dir=None, nl_type=None, nl_skew=None, nl_cont=None, nl_dev=None,
                 nl_tb=None, nl_lr=None,seed=None):
        self._seed = seed
        # ================================== Non-linear Moire pattern ========================================
        self._nl_moire  = nl_moire
        self._nl_dir    = nl_dir
        self._nl_type   = nl_type
        self._nl_skew	= nl_skew
        self._nl_cont	= nl_cont
        self._nl_dev	= nl_dev
        self._nl_tb	= nl_tb
        self._nl_lr	= nl_lr

    def __call__(self, H, W, verbose=False):
        '''
        :param image: the input image to transform (HWC format)
        :type image: np.array
        :param verbose: whether to print out the processing log or not
        :type verbose: bool

        :return: the image transformed as specified (HWC format)
        :rtype: np.array
        '''
        # Non-linear moire pattern insertion
        if self._nl_moire:
            nl_mask = nonlinear_wave([H, W, 3], directions=self._nl_dir, pattern=self._nl_type,
                                     skew=self._nl_skew, contrast=self._nl_cont, dev=self._nl_dev,
                                     tb_margins=self._nl_tb, lr_margins=self._nl_lr, seed=self._seed)
            if verbose:
                print('(Non-linear moire call) direction: {}, type: {}, skew: {}, contrast: {}, dev: {}, \
                        margins: {}, {}' \
                      .format(self._nl_dir, self._nl_type, self._nl_skew, self._nl_cont, self._nl_dev,
                              self._nl_tb, self._nl_lr))
        else:
            nl_mask = np.zeros([H, W, 3])
        return nl_mask


def nonlinear_wave(out_shape, gap=4, skew=0, thick=1, directions='b',
                   pattern='fixed', contrast=7, color=None, dev=3,
                   tb_margins=0, lr_margins=0, seed=None):
    # Initialize the shape of the mask
    mask_shape = out_shape

    # Leave additional space for warping margins
    assert tb_margins >= 0 and tb_margins < 0.5, "Please provide a valid 'tb_margins' value in [0,0.5)."
    assert lr_margins >= 0 and lr_margins < 0.5, "Please provide a valid 'lr_margins' value in [0,0.5)."
    tb_extra = ceil((mask_shape[0] / (1 - 2 * tb_margins) - mask_shape[0]) / 2)
    lr_extra = ceil((mask_shape[1] / (1 - 2 * lr_margins) - mask_shape[1]) / 2)
    mask_shape[0] += 2 * tb_extra
    mask_shape[1] += 2 * lr_extra

    # Check which directions to draw lines in
    rowwise = colwise = False
    if directions == 'b':
        rowwise = True
        colwise = True
    elif directions == 'h':
        rowwise = True
    elif directions == 'v':
        colwise = True
    else:
        raise ValueError("Please provide a valid argument for 'directions' parameter, among {'b','h','v'}.")

    # Leave additional space for full skewing
    if rowwise:
        mask_shape[0] += 2 * np.abs(skew)
    if colwise:
        mask_shape[1] += 2 * np.abs(skew)
    mask = np.zeros(mask_shape)
    H, W, _ = mask_shape

    # Set color
    contrast = 64
    if color is None:
        color = (contrast,) * 3

    # Draw lines onto mask
    if rowwise:
        ### Generate color map
        num_lines = len(list(range(0, H, gap)))
        if pattern == 'gaussian':
            if seed:
                np.random.seed(seed)
            color_map = [[int(round(np.clip(np.random.randn() * dev + mean, \
                                            max(mean - 2 * dev, 0), min(mean + 2 * dev, 255)))) \
                          for mean in color]
                         for _ in range(num_lines)]
        elif pattern == 'sine':
            color_map = [[int(round(np.clip(np.sin(line_num) * dev + mean, 0, 255))) \
                          for mean in color] \
                         for line_num in range(num_lines)]

        elif pattern == 'fixed':
            color_map = [color, ] * num_lines
        else:
            raise NotImplementedError('Please choose a valid pattern type.')

        ### Draw lines as specified
        for row, color in zip(range(0, H, gap), color_map):
            if skew <= 0:
                cv2.line(mask, (0, row), (W, row + skew), color, thickness=thick)
            else:
                cv2.line(mask, (0, row - skew), (W, row), color, thickness=thick)
    if colwise:
        ### Generate color map
        num_lines = len(list(range(0, W, gap)))
        if pattern == 'gaussian':
            if seed:
                np.random.seed(seed)
            color_map = [[int(round(np.clip(np.random.randn() * dev + mean, \
                                            max(mean - 2 * dev, 0), min(mean + 2 * dev, 255)))) \
                          for mean in color]
                         for _ in range(num_lines)]
        elif pattern == 'sine':
            color_map = [[int(round(np.clip(np.sin(line_num) * dev + mean, 0, 255))) \
                          for mean in color] \
                         for line_num in range(num_lines)]

        elif pattern == 'fixed':
            color_map = [color, ] * num_lines
        else:
            raise NotImplementedError('Please choose a valid pattern type.')

        ### Draw lines as specified
        for col, color in zip(range(0, W, gap), color_map):
            if skew <= 0:
                cv2.line(mask, (col, 0), (col + skew, H), color, thickness=thick)
            else:
                cv2.line(mask, (col - skew, 0), (col, H), color, thickness=thick)

    # Add noise
    if skew:
        if rowwise:
            mask = mask[np.abs(skew):-np.abs(skew)]
        if colwise:
            mask = mask[:, np.abs(skew):-np.abs(skew)]

    # Distort mask
    ### Set the corners of the image as source points
    H, W, _ = mask.shape
    src_points = np.zeros((4, 2), dtype="float32")
    src_points[0] = [0, 0]  # top-left (w,h)
    src_points[1] = [W - 1, 0]  # top-right (w,h)
    src_points[2] = [W - 1, H - 1]  # bottom-right (w,h)
    src_points[3] = [0, H - 1]  # bottom-left (w,h)

    ### Randomly generate dest points within the given margins
    t_margin = [0, tb_margins]
    b_margin = [1 - tb_margins, 1]
    l_margin = [0, lr_margins]
    r_margin = [1 - lr_margins, 1]
    if seed:
        np.random.seed(seed)
    tl_h, tr_h = np.random.randint(*[H * val for val in t_margin], size=2)
    bl_h, br_h = np.random.randint(*[H * val for val in b_margin], size=2)
    tl_w, bl_w = np.random.randint(*[W * val for val in l_margin], size=2)
    tr_w, br_w = np.random.randint(*[W * val for val in r_margin], size=2)
    dst_points = np.zeros((4, 2), dtype="float32")
    dst_points[0] = [tl_w, tl_h]  # top-left
    dst_points[1] = [tr_w, tr_h]  # top-right
    dst_points[2] = [br_w, br_h]  # bottom-right
    dst_points[3] = [bl_w, bl_h]  # bottom-left

    # Compute warp matrix and warp the mask
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_mask = cv2.warpPerspective(mask, M, (W, H))

    # Remove (potential) black regions by removing the margins
    warped_mask = warped_mask[tb_extra:-tb_extra, lr_extra:-lr_extra]
    # out = (canvas + warped_mask).clip(0, 255)
    return np.uint8(warped_mask)

def gamma_correction(img, gamma=2.2):
    invgamma = 1.0/gamma
    table = np.array([(val / 255.0)**invgamma * 255 \
            for val in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)
