import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny_edge_detection(image_path, low_threshold=100, high_threshold=200):
    """
    对输入图像进行 Canny 边缘检测
    :param image_path: 图像文件的路径
    :param low_threshold: Canny 边缘检测的低阈值
    :param high_threshold: Canny 边缘检测的高阈值
    :return: 原始图像、灰度图像、Canny 边缘检测结果
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊平滑图像（可选）
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 应用 Canny 边缘检测
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    return image, gray_image, edges


def display_images(original_image, gray_image, edges):
    """
    显示原始图像、灰度图像和 Canny 边缘检测结果
    :param original_image: 原始图像
    :param gray_image: 灰度图像
    :param edges: Canny 边缘检测结果
    """
    plt.figure(figsize=(15, 5))

    # 显示原始图像
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 显示灰度图像
    plt.subplot(132)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # 显示 Canny 边缘检测结果
    plt.subplot(133)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.show()


def main():
    image_path = './car.png'
    original_image, gray_image, edges = canny_edge_detection(image_path, low_threshold=100, high_threshold=200)
    display_images(original_image, gray_image, edges)


if __name__ == "__main__":
    main()