'''
【功能】验证γ变换公式中，补偿系数对输出图像的影响
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


def update_gamma(val):
    gamma = slider1.val
    output = gamma_trans(input_arr, gamma=gamma, eps=0.5)
    print("--------\n", output)
    ax1.set_title("伽马变换后，gamma = " + str(gamma))
    ax1.imshow(output, cmap='gray', vmin=0, vmax=255)


if __name__ == '__main__':

    set_chinese()

    input_arr = np.array( [ [0,  50,  100, 150],
                            [0,  50,  100, 150],
                            [0,  50,  100, 150],
                            [0,  50,  100, 150]] )

    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    ax0.set_title("输入矩阵")
    ax0.imshow(input_arr, cmap='gray',vmin=0,vmax=255)

    ax1 = fig.add_subplot(122)

    plt.subplots_adjust(bottom=0.3)
    s1 = plt.axes([0.25, 0.1, 0.55, 0.03], facecolor='lightgoldenrodyellow')
    slider1 = Slider(s1, '参数gamma', 0.0, 2.0,
                     valfmt='%.f', valinit=1.0, valstep=0.1)
    slider1.on_changed(update_gamma)
    slider1.reset()
    slider1.set_val(1)

    plt.show()







