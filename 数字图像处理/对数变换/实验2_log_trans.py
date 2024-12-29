'''
【功能】拓展实验：利用对数变换，压缩傅里叶频谱值的动态范围，使其能够被人眼观察到更多细节
【B站/YouTube】轩辕十四很nice
【开源协议】The MIT License (MIT)
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d


def set_chinese():  # 中文显示工具函数
    import matplotlib
    print("[INFO] matplotlib版本为：%s" % matplotlib.__version__)
    matplotlib.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文标签 ‘SimHei’，'FangSong'
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def image_log(input):
	return np.log(1 + input)

if __name__ == '__main__':
    set_chinese()

    gray_img = np.asarray(Image.open('./squared_paper.jpg').convert('L'))
    print("[INFO] 原图尺寸为：", gray_img.shape)

    fft_arr = np.fft.fftshift(np.abs(np.fft.fft2(gray_img)))  # 对原图做傅立叶变换,得到一个频谱矩阵
    print("[INFO] 频谱矩阵最大值为：%d " % int(np.max(fft_arr)))


    fig = plt.figure() #创建第一个显示窗口

    ax1 = fig.add_subplot(221)
    ax1.set_title('原图')
    ax1.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('频谱幅值', fontsize=6)
    ax2.set_ylabel('出现次数', fontsize=6)
    ax2.grid(True, linestyle=':', linewidth=1)
    ax2.set_title('频谱幅值分布直方图', fontsize=12)
    ax2.hist(fft_arr.flatten(), density=False, color='r', edgecolor='k', log=True)

    ax3 = fig.add_subplot(223)
    ax3.set_title('频谱矩阵')  # 直接显示频谱值，不利于人眼观察
    ax3.imshow(fft_arr, cmap='gray')

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title('频谱矩阵3D图', fontsize=12)
    Y = np.arange(0, fft_arr.shape[0], 1)
    X = np.arange(0, fft_arr.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    ax4.plot_surface(X, Y, fft_arr, cmap='coolwarm')



    fig2 = plt.figure() #创建第二个显示窗口

    ax5 = fig2.add_subplot(121)
    ax5.set_title('经对数变换的频谱矩阵')
    log_fft_arr = image_log(fft_arr)  # 对频谱矩阵进行对数变换
    ax5.imshow(log_fft_arr, cmap='gray')


    ax6 = fig2.add_subplot(122, projection='3d')
    ax6.set_title('经对数变换的频谱矩阵3D图', fontsize=12)
    Y = np.arange(0, fft_arr.shape[0], 1)
    X = np.arange(0, fft_arr.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    ax6.set_xlabel('宽', fontsize=12)
    ax6.set_ylabel('高', fontsize=12)
    ax6.set_zlabel('幅值', fontsize=12)
    ax6.plot_surface(X, Y, log_fft_arr, rstride=1, cstride=1, cmap='coolwarm',antialiased=True)
    ax6.set_zlim(-np.max(log_fft_arr), np.max(log_fft_arr))  # 拉开坐标轴范围显示投影
    ax6.contour(X, Y, log_fft_arr, zdir='z', offset=-np.max(log_fft_arr), cmap="coolwarm")  # 生成z方向投影，投到x-y平面

    plt.show()