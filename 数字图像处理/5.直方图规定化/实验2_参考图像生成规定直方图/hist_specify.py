'''
【功能】通过参考无雾图像的规定直方图，实现类似去雾滤镜的效果。
【B站/YouTube】十四阿哥很nice
【开源协议】The MIT License (MIT)
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 中文支持函数
def set_chinese():
    import matplotlib
    print("[INFO] matplotlib版本为：%s" % matplotlib.__version__)
    matplotlib.rcParams['font.sans-serif'] = ['FangSong']
    matplotlib.rcParams['axes.unicode_minus'] = False

# 获取图像的概率密度
def get_pdf(in_img):
    total = in_img.shape[0] * in_img.shape[1] #计算图片总像素数
    return [ np.sum(in_img == i)/total for i in range(256) ] #求概率密度

# 构造目标图像的映射表（核心代码）
def gen_target_table(Pv):
    table = []
    SUMq = 0.
    for i in range(256):
        SUMq = SUMq + Pv[i]
        table.append(round(SUMq*255, 0))   #四舍五入
    return table


# 直方图均衡化（核心代码）
def hist_equal(in_img):

    # 1.求原始图像的概率密度
    Pr = get_pdf(in_img)

    # 2.构造输出图像（初始化成输入）
    out_img = np.copy(in_img)

    # 3.执行“直方图均衡化”（执行累积分布函数变换）
    y_points = [] # 存储点集，用于画图
    SUMk = 0  # 累加值存储变量
    for i in range(256):
        SUMk = SUMk + Pr[i]
        out_img[(in_img == i)] = SUMk*255 #灰度值逆归一化
        y_points.append(SUMk*255) #构造绘制函数图像的点集（非核心代码，可忽略）

    return out_img.astype("int32"), y_points


# 直方图规定化（核心代码）
def hist_specify(in_img=None,ref_img=None):

    # 1.拿到目标图像规定的概率密度,并构造映射表
    Pv = get_pdf(ref_img) #从参考图像中拿到规定直方图
    table = gen_target_table(Pv)

    # 2.对原始图像做直方图均衡
    ori_eq_img, T_points = hist_equal(in_img)

    # 3.构造输出图像（初始化成输入）
    out_img = np.copy(ori_eq_img)

    # 4.执行“直方图规定化”（利用映射表，实现反映射）
    temp = 0
    for val in range(256):
        if val in ori_eq_img:
            if val in table:
                temp = len(table)-table[::-1].index(val)-1 #拿到指定值最后出现的索引
                #temp = table.index(val)  #index拿到指定值首次出现的索引
            out_img[(ori_eq_img == val)] = temp # 找不到映射关系时，取前一个映射值

    return out_img, T_points, table, Pv




if __name__ == '__main__':

    set_chinese()

    # 读入原图 和 参考图像
    gray_img = np.asarray(Image.open('./fog.png').convert('L'))
    ref_img = np.asarray(Image.open('./ref_de_fog.png').convert('L'))

    # 对原图执行“直方图规定化”
    out_img, T_pts, G_pts, spec_hist = hist_specify(gray_img,ref_img)
	
    # 创建1个显示主体，并分成6个显示区域
    fig = plt.figure()
    ax1, ax2, ax3 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)
    ax4, ax5, ax6 = fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)

    # 窗口显示：原图，原图灰度分布，结果图像，结果图像灰度分布
    ax1.set_title('原图', fontsize=8)
    ax1.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

    ax2.grid(True, linestyle=':', linewidth=1)
    ax2.set_title('原图分布', fontsize=8)
    ax2.set_xlim(0, 255)  # 设置x轴分布范围
    ax2.set_ylim(0, 0.5)  # 设置y轴分布范围
    ax2.hist(gray_img.flatten(),bins=255,density=True,color='black',edgecolor='black')

    ax4.set_title('直方图规定化结果', fontsize=8)
    ax4.imshow(out_img, cmap='gray', vmin=0, vmax=255)

    ax5.grid(True, linestyle=':', linewidth=1)
    ax5.set_title('结果分布', fontsize=8)
    ax5.set_xlim(0, 255)  # 设置x轴分布范围
    ax5.set_ylim(0, 0.5)  # 设置y轴分布范围
    ax5.hist(out_img.flatten(),bins=255,density=True,color='black',edgecolor='black')



    # 窗口显示：原图，原图灰度分布，结果图像，结果图像灰度分布
    ax3.set_title('参考图像', fontsize=16)
    ax3.imshow(ref_img, cmap='gray', vmin=0, vmax=255)

    # 窗口显示：绘制规定概率密度函数
    ax6.set_title("参考图像PDF（规定直方图）", fontsize=14)
    ax6.grid(True, linestyle=':', linewidth=1)
    ax6.plot(np.arange(0, 256, 1), spec_hist, color='r',lw=1)

    plt.show()










