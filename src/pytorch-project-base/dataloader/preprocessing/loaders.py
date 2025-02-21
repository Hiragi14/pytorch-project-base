import cv2
from .dct.dct_utils import dct_transform
from .img_utils.utils import scale_image, normalization_image


def return_loader(block_size, alpha, mode='scipy', size=224):
    """
    Returns a custom loader that converts RGB images to YUV using OpenCV and applies DCT.
    
    Args:
        size (int): Size to resize the image to.
        block_size (int): Size of the block for DCT.
        alpha (float): Compression factor for DCT.
        mode (str): Mode to use for DCT (scipy or cv2).
    """
    def loader(path):
        """
        Custom loader to convert RGB images to YUV using OpenCV.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: YUV image as a PyTorch Tensor.
        """
        # RGB画像を読み込み (BGR形式で読み込まれる)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {path}")

        # BGR -> YUV 変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # リサイズ
        image = cv2.resize(image, (size, size))
        
        # スケーリング
        image = scale_image(image)

        # DCT変換
        dct_image = dct_transform(image, block_size=block_size, alpha=alpha, mode=mode)
        return dct_image
    return loader


def return_loader_normalize(block_size, alpha, mode='scipy', size=224):
    """
    Returns a custom loader that converts RGB images to YUV using OpenCV and applies DCT.

    Args:
        size (int): Size to resize the image to.
        block_size (int): Size of the block for DCT.
        alpha (float): Compression factor for DCT.
        mode (str): Mode to use for DCT (scipy or cv2).
    """
    def loader(path):
        """
        Custom loader to convert RGB images to YUV using OpenCV.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: YUV image as a PyTorch Tensor.
        """
        # RGB画像を読み込み (BGR形式で読み込まれる)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {path}")

        # BGR -> YUV 変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # リサイズ
        image = cv2.resize(image, (size, size))

        # スケーリング
        image = normalization_image(image)
        # DCT変換
        dct_image = dct_transform(image, block_size=block_size, alpha=alpha, mode=mode)
        return dct_image
    return loader