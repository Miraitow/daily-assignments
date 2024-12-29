from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def set_chinese():   # 中文显示工具函数
	import matplotlib
	print("[INFO] matplotlib版本为：%s" % matplotlib.__version__)
	matplotlib.rcParams['font.sans-serif'] = ['FangSong']
	matplotlib.rcParams['axes.unicode_minus'] = False

def gamma_trans(input, gamma=2, eps=0 ):
    return 255. * (((input + eps)/255.) ** gamma)


def update_gamma(val):

    # 获取滑块数值，作为γ
    gamma_ = slider1.val

    # 对原图执行γ变换
    output = gamma_trans(gray_img, gamma_, 0.2)

    # 显示γ变换结果图像
    ax3.clear()
    ax3.set_title("伽马变换结果，gamma = " + str(round(gamma_,2)))
    ax3.imshow(output, cmap='gray',vmin=0,vmax=255)
    # 显示γ变换结果图像的灰度分布直方图
    ax4.clear()
    ax4.set_xlim(0, 255)  # 设置x轴分布范围
    ax4.set_ylim(0, 0.15)  # 设置y轴分布范围
    ax4.grid(True, linestyle=':', linewidth=1)
    ax4.set_title('伽马变换后，灰度分布直方图', fontsize=12)
    ax4.hist(output.flatten(),bins=50,density=True,color='r',edgecolor='k')


if __name__ == '__main__':

    set_chinese()

    # 读入原图
    gray_img = np.asarray(Image.open('./washed_out.tif').convert('L'))

    # 创建一个显示主体，并分成四个显示区域
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # 显示原图
    ax1.set_title("原始图片")
    ax1.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

    # 显示原图的灰度分布直方图
    ax2.grid(True, linestyle=':', linewidth=1)
    ax2.set_title('原图灰度分布直方图', fontsize=12)
    ax2.set_xlim(0, 255)  # 设置x轴分布范围
    ax2.set_ylim(0, 0.15)  # 设置y轴分布范围
    ax2.hist(gray_img.flatten(), bins=50,density=True,color='r',edgecolor='k')

    # 在显示主体下方创建滑动条，用于交互控制γ值
    plt.subplots_adjust(bottom=0.3)
    s1 = plt.axes([0.25, 0.1, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    slider1 = Slider(s1, '参数gamma', 0.0, 2.0,
                      valfmt='%.f', valinit=1.0, valstep=0.1)
    slider1.on_changed(update_gamma)
    slider1.reset()
    slider1.set_val(1.0)



    plt.show()



