'''
【功能】利用三段线性变换，大幅拉伸图像对比度，实现图像增强目的
【B站/YouTube】轩辕十四很nice
【开源协议】The MIT License (MIT)
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def set_chinese():  # 中文显示工具函数
    import matplotlib
    print("[INFO] matplotlib版本为：%s" % matplotlib.__version__)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False


# 三段对比度拉伸变换，其中x1,y1和x2,y2为分段点
def three_linear_trans(x, x1, y1, x2, y2):
    # 1. 检查参数，避免分母为0
    if x1 == x2 or x2 == 255:
        print("[INFO] x1=%d,x2=%d ->调用此函数必须满足:x1≠x2且x2≠255" % (x1, x2))
        return None

    # 2. 执行分段线性变换
    # 【快速算法】
    m1 = (x < x1)
    m2 = (x1 <= x) & (x <= x2)  # 注意这边要用“&”，且注意运算顺序
    m3 = (x > x2)

    # 处理可能的零除问题
    if x1 == 0:
        x1 = 1e-10  # 给 x1 一个极小的非零值，避免除以零
    if x2 == 255:
        x2 = 254.9  # 给 x2 一个小于 255 的值，避免除以零

    out = (y1 / x1 * x) * m1 \
          + ((y2 - y1) / (x2 - x1) * (x - x1) + y1) * m2 \
          + ((255 - y2) / (255 - x2) * (x - x2) + y2) * m3

    # 3. 获取分段线性函数的点集，用于绘制函数图像
    x_point = np.arange(0, 256, 1)
    cond2 = [(True if (i >= x1 and i <= x2) else False) for i in x_point]
    y_point = (y1 / x1 * x_point) * (x_point < x1) \
              + ((y2 - y1) / (x2 - x1) * (x_point - x1) + y1) * cond2 \
              + ((255 - y2) / (255 - x2) * (x_point - x2) + y2) * (x_point > x2)

    return out, x_point, y_point


def update_trans(val):
    # 读入 4 个滑动条的值
    x1, y1 = slider_x1.val, slider_y1.val
    x2, y2 = slider_x2.val, slider_y2.val

    # 下面这段代码用于确保x2>x1,y2>y1，从而保证函数单调
    if x1 >= x2:
        x1 = x2 - 1
    if y1 >= y2:
        y1 = y2 - 1

    # 执行分段线性变换
    out, x_point, y_point = three_linear_trans(gray_img,
                                       x1, y1,
                                       x2, y2)

    # 显示变换结果图像
    ax2.clear()
    ax2.set_title("分段线性变换结果", fontsize=8)
    ax2.imshow(out, cmap='gray', vmin=0, vmax=255)

    # 绘制函数图像
    ax3.clear()
    ax3.annotate('( %d,%d )' % (x1, y1), xy=(x1, y1), xytext=(x1 - 15, y1 + 15))
    ax3.annotate('( %d,%d )' % (x2, y2), xy=(x2, y2), xytext=(x2 + 15, y2 - 15))
    ax3.set_title("分段线性函数图像", fontsize=8)
    ax3.grid(True, linestyle=':', linewidth=1)
    ax3.plot([x1, x2], [y1, y2], 'ro')
    ax3.plot(x_point, y_point, 'g')

    # 显示变换结果的灰度分布直方图
    ax5.clear()
    ax5.grid(True, linestyle=':', linewidth=1)
    ax5.set_title('变换结果灰度分布直方图', fontsize=8)
    ax5.set_xlim(0, 255)  # 设置x轴分布范围
    ax5.set_ylim(0, 0.15)  # 设置y轴分布范围
    ax5.hist(out.flatten(), bins=50, density=True, color='r', edgecolor='k')


if __name__ == '__main__':
    set_chinese()

    # 以灰度方式读入图像
    gray_img = np.asarray(Image.open('./washed_out_pollen_image.tif').convert('L'))
    print("[INFO] 原图尺寸为：", gray_img.shape)

    # 创建一个显示主体，并分成五个显示区域
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)

    # 显示原图及其灰度分布直方图
    ax1.set_title("原始输入图片", fontsize=8)
    ax1.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

    ax4.grid(True, linestyle=':', linewidth=1)
    ax4.set_title('原图灰度分布直方图', fontsize=8)
    ax4.set_xlim(0, 255)  # 设置x轴分布范围
    ax4.set_ylim(0, 0.15)  # 设置y轴分布范围
    ax4.hist(gray_img.flatten(), bins=50, density=True, color='r', edgecolor='k')

    # 创建四个滑动条，用于调节x1，y1，x2，y2四个值
    plt.subplots_adjust(bottom=0.3)
    x1 = plt.axes([0.25, 0.2, 0.45, 0.03],
                facecolor='lightgoldenrodyellow')
    slider_x1 = Slider(x1, '参数x1', 0.0, 255.,
                    valfmt='%.f', valinit=91, valstep=1)
    slider_x1.on_changed(update_trans)

    y1 = plt.axes([0.25, 0.16, 0.45, 0.03],
                facecolor='lightgoldenrodyellow')
    slider_y1 = Slider(y1, '参数y1', 0.0, 255.,
                    valfmt='%.f', valinit=0, valstep=1)
    slider_y1.on_changed(update_trans)

    x2 = plt.axes([0.25, 0.06, 0.45, 0.03],
                facecolor='white')
    slider_x2 = Slider(x2, '参数x2', 0.0, 254.,
                    valfmt='%.f', valinit=138, valstep=1)
    slider_x2.on_changed(update_trans)

    y2 = plt.axes([0.25, 0.02, 0.45, 0.03],
                facecolor='white')
    slider_y2 = Slider(y2, '参数y2', 0.0, 255.,
                    valfmt='%.f', valinit=255, valstep=1)
    slider_y2.on_changed(update_trans)

    slider_x1.set_val(91)
    slider_y1.set_val(91)
    slider_x2.set_val(138)
    slider_y2.set_val(138)

    plt.show()