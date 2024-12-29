import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_low_pass_filter(image, cutoff):
    """
    生成高斯低通滤波器
    :param image: 输入图像（灰度图像）
    :param cutoff: 截止频率
    :return: 高斯低通滤波器（与输入图像尺寸相同的二维数组）
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(0, cols, cols)
    y = np.linspace(0, rows, rows)
    xv, yv = np.meshgrid(x, y)
    distance = np.sqrt((xv - ccol) ** 2 + (yv - crow) ** 2)
    mask = np.exp(-(distance ** 2) / (2 * cutoff ** 2))
    return mask


def gaussian_high_pass_filter(image, cutoff):
    """
    生成高斯高通滤波器
    :param image: 输入图像（灰度图像）
    :param cutoff: 截止频率
    :return: 高斯高通滤波器（与输入图像尺寸相同的二维数组）
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(0, cols, cols)
    y = np.linspace(0, rows, rows)
    xv, yv = np.meshgrid(x, y)
    distance = np.sqrt((xv - ccol) ** 2 + (yv - crow) ** 2)
    mask = 1 - np.exp(-(distance ** 2) / (2 * cutoff ** 2))
    return mask


def frequency_domain_filtering(image, filter_mask):
    """
    频域滤波函数
    :param image: 输入图像（灰度图像）
    :param filter_mask: 滤波器掩码
    :return: 滤波后的图像（灰度图像）
    """
    # 进行傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 应用滤波器
    fshift_filtered = fshift * filter_mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    filtered_image = np.fft.ifft2(f_ishift)
    filtered_image = np.abs(filtered_image)
    filtered_image = np.uint8(cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX))
    return filtered_image


def main():
    # 读取图像
    image = cv2.imread('./lv1.png')
    if image is None:
        raise ValueError("Image not found")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 显示原始图像
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 高斯低通滤波
    cutoff_low = 70
    glpf_mask = gaussian_low_pass_filter(gray_image, cutoff_low)
    glpf_image = frequency_domain_filtering(gray_image, glpf_mask)
    plt.subplot(232)
    plt.imshow(glpf_image, cmap='gray')
    plt.title(f'Gaussian Low Pass Filter (Cutoff={cutoff_low})')
    plt.axis('off')

    # 高斯高通滤波
    cutoff_high = 8
    ghpf_mask = gaussian_high_pass_filter(gray_image, cutoff_high)
    ghpf_image = frequency_domain_filtering(gray_image, ghpf_mask)
    plt.subplot(233)
    plt.imshow(ghpf_image, cmap='gray')
    plt.title(f'Gaussian High Pass Filter (Cutoff={cutoff_high})')
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()