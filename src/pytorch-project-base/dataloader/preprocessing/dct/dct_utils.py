import numpy as np
from scipy.fft import dctn
import cv2

from img_utils.utils import pad_to_block_size, blocking, unblock, cut_ac


def dct_deal(data, block_size, alpha, mode='scipy'):
    height, width = data.shape[:2]
    padded_data = pad_to_block_size(data, block_size=block_size)
    blocked_array = blocking(padded_data, block_size=block_size)
    if alpha > 1.0 or alpha < 0:
        raise RuntimeError('alpha must be plus value.')
    cutted_block_size = int(block_size*alpha)
    dct_data = np.zeros((cutted_block_size, cutted_block_size, blocked_array.shape[2]))
    
    for i in range(blocked_array.shape[2]):
        if mode == 'scipy':
            dct_data[:,:,i] = cut_ac(dctn(blocked_array[:,:,i], type=2, norm='ortho'), alpha=alpha)
        elif mode == 'cv2':
            dct_data[:,:,i] = cut_ac(cv2.dct(blocked_array[:,:,i].astype(np.float32)), alpha=alpha)
        else:
            raise RuntimeError('mode must be scipy or cv2.')
    unblocked_data = unblock(dct_data, int(block_size*alpha), int(height*alpha), int(width*alpha))
    return unblocked_data


def dct_transform(data, block_size, alpha, mode='scipy'):
    """DCT変換を行う

    Parameters
    ----------
    data : ndarray
        画像データ
    block_size : int
        ブロックサイズ
    alpha : float
        画像のサイズ(height,width)を何倍にするか. デフォルトだと0.5で画像サイズは1/4になる.
    """
    ch1 = dct_deal(data[:,:,0], block_size, alpha, mode)
    ch2 = dct_deal(data[:,:,1], block_size, alpha, mode)
    ch3 = dct_deal(data[:,:,2], block_size, alpha, mode)
    return np.dstack([ch1, ch2, ch3])