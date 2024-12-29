'''
【功能】利用γ变换提前矫正图像，从而实现图像在CRT显示器中正常显示
【B站/YouTube】轩辕十四很nice
【开源协议】The MIT License (MIT)
'''


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

def crt_distortion(input, gamma=2):  # 模拟CRT失真（做了归一化）
    return 255. * ((input/255.) ** gamma)


def update_gamma(val):

    gamma = slider1.val  # 获取"失真γ"


    # 图像送入CRT前，先做伽马变换预处理
    gamma_ = 1 / gamma   #  "矫正γ" 为 "失真γ" 的倒数
    correct_img = gamma_trans(gray_img, gamma_, 0)  #
    ax1.set_title("伽马矫正，矫正γ = 1/" + str(round(gamma,2)))
    ax1.imshow(correct_img, cmap='gray',vmin=0,vmax=255)

    # 简易模拟CRT输出
    output = crt_distortion(correct_img, gamma)
    print(output)
    ax2.set_title("模拟CRT输出，失真γ = " + str(round(gamma,2)))
    ax2.imshow(output, cmap='gray',vmin=0,vmax=255)


if __name__ == '__main__':
    set_chinese()

    gray_img = np.asarray(Image.open('./intensity_ramp.tif').convert('L'))

    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)


    ax0.set_title("原始图片")
    ax0.imshow(gray_img, cmap='gray',vmin=0,vmax=255)


    plt.subplots_adjust(bottom=0.3)
    s1 = plt.axes([0.25, 0.1, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    slider1 = Slider(s1, 'CRT失真γ： ', 0.0, 4.0,
                      valfmt='%.f', valinit=1.0, valstep=0.1)
    slider1.on_changed(update_gamma)
    slider1.reset()
    slider1.set_val(2)

    plt.show()



