import numpy as np


def normalization_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """画像の正規化を行う

    Parameters
    ----------
    image : ndarray
        画像データ

    Returns
    -------
    ndarray
        正規化された画像データ
    """
    normalized_image = np.zeros_like(image).astype(np.float32)
    image = image.astype(np.float32)
    image = image / 255.0
    normalized_image[:,:,0] = (image[:,:,0] - mean[0]) / (std[0])
    normalized_image[:,:,1] = (image[:,:,1] - mean[1]) / (std[1])
    normalized_image[:,:,2] = (image[:,:,2] - mean[2]) / (std[2])
    return normalized_image


def scale_image(image, orig_range=(0, 255), new_range=(-128, 127), mode='int8'):
    """画像のスケーリング処理"""
    orig_min, orig_max = orig_range
    new_min, new_max = new_range
    
    # スケーリング処理
    scaled_image = ((image - orig_min) / (orig_max - orig_min)) * (new_max - new_min) + new_min

    # データ型変換
    if mode == 'int8':
        scaled_image = scaled_image.astype(np.int8)
    elif mode == 'int16':
        scaled_image = scaled_image.astype(np.int16)
    
    return scaled_image


def pad_to_block_size(data, block_size=8):
    """画像を指定したブロックサイズにパディングする"""
    height = data.shape[0]
    width = data.shape[1]
    new_height = (height + block_size - 1) // block_size * block_size
    new_width = (width + block_size - 1) // block_size * block_size
    padded_data = np.zeros((new_height, new_width))
    padded_data[:height, :width] = data
    return padded_data


def blocking(data, block_size):
    """2次元画像をブロック化して3次元配列にする"""
    height, width = data.shape
    blocked_data = np.zeros((block_size, block_size, int(height/block_size)*int(width/block_size)))
    count = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            blocked_data[:,:,count] = data[i:i+block_size, j:j+block_size]
            count += 1
    return blocked_data


def unblock(blocked_data, block_size, height, width):
    """3次元配列のブロック化されたデータを元の2次元画像に戻す"""
    data = np.zeros((height, width))
    # print(f'data: {data.shape}')
    count = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            data[i:i+block_size, j:j+block_size] = blocked_data[:, :, count]
            count += 1
    return data


def cut_ac(blocked_data, alpha=0.5):
    """AC成分を削る

    Parameters
    ----------
    blocked_data : ndarray
        ブロック化されたデータ(２次元配列)
    alpha : float, optional
        画像のサイズ(height,width)を何倍にするか. デフォルトだと0.5で画像サイズは1/4になる., by default 0.5

    Returns
    -------
    ndarray
        AC成分がカットされた配列
    """
    block_size = blocked_data.shape[0]
    cutted_block_size = int(block_size*alpha)
    cutted_block = blocked_data[:cutted_block_size, :cutted_block_size]
    return cutted_block