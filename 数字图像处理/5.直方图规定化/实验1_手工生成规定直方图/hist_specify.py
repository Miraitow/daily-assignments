'''
【功能】利用直方图规定化，保持原图色调的同时，增强图像暗部细节
【B站/YouTube】十四阿哥很nice
【开源协议】The MIT License (MIT)
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 中文设置函数
def set_chinese():
    import matplotlib
    print("[INFO] matplotlib版本为：%s" % matplotlib.__version__)
    matplotlib.rcParams['font.sans-serif'] = ['FangSong']
    matplotlib.rcParams['axes.unicode_minus'] = False

# 获取图像的概率密度
def get_pdf(in_img):
    total = in_img.shape[0] * in_img.shape[1] #计算图片总像素数
    return [ np.sum(in_img == i)/total for i in range(256) ] #求概率密度

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

# 生成均匀分布
def gen_eq_pdf():
    return [ 0.0039 for i in range(256) ] #均匀概率密度 1/256=0.0039

# 生成多峰正态分布
# means 各峰均值
# stds  各峰标准差
# ampl  各正态分布幅值（和为1）
# bias  概率密度函数总体抬升量（若没有抬升量，容易造成图像过暗）
def gen_mul_norm_pdf(x, means, stds, ampl, bias):
    pdf = np.zeros([256,], dtype=float)
    for i in range(len(means)):
        pdf_temp = ampl[i]*np.exp(-((x-means[i])**2)/(2*stds[i]**2))\
                   /(stds[i] * np.sqrt(2 * np.pi))
        pdf = pdf + pdf_temp
    pdf = pdf + bias #总体抬升概率密度，避免出现零值
    pdf2 = pdf/np.sum(pdf) #由于做了抬升，所以要重新归一化
    return pdf2.tolist()


# 构造目标图像的映射表（核心代码）
def gen_target_table(Pv):
    table = []
    SUMq = 0.
    for i in range(256):
        SUMq = SUMq + Pv[i]
        table.append(round(SUMq*255, 0))   #四舍五入
    return table

# 直方图规定化（核心代码）
def hist_specify(in_img=None):

    # 1.拿到目标图像规定概率密度,并构造映射表
    Pv = gen_mul_norm_pdf(np.arange(0,256,1),[38,191],[13,13],[0.93,0.07],0.002)
    table = gen_target_table(Pv)
    print(table)

    # 2.对原始图像做直方图均衡
    ori_eq_img, T_points = hist_equal(in_img)

    # 3.构造输出图像（初始化成输入）
    out_img = np.copy(ori_eq_img)

    # 4.执行“直方图规定化”（根据映射表，做逆映射 B'->B）
    map_val = 0  # 逆映射值初始化为0
    for v in range(256):
        if v in ori_eq_img: # 存在于B'
            if v in table:  # 存在于映射表
                map_val = len(table)-table[::-1].index(v)-1 #拿到指定值最后出现的索引
            out_img[(ori_eq_img == v)] = map_val # 找不到映射关系时，取前一个映射值

    return out_img, T_points, table, Pv, Pv2




if __name__ == '__main__':

    set_chinese()

    # 读入原图
    gray_img = np.asarray(Image.open('./mars_moon_phobos.tif').convert('L'))

    # 对原图执行“直方图规定化”
    out_img, T_pts, G_pts, spec_hist = hist_specify(gray_img)

    # 创建1个显示主体，并分成6个显示区域
    fig = plt.figure()
    ax1, ax2, ax3 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)
    ax4, ax5, ax6 = fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)

    # 窗口显示：原图，原图灰度分布，规定PDF，结果图像，结果图像灰度分布，对应变换函数
    ax1.set_title('原图', fontsize=8)
    ax1.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

    ax2.grid(True, linestyle=':', linewidth=1)
    ax2.set_title('原图分布', fontsize=8)
    ax2.set_xlim(0, 255)  # 设置x轴分布范围
    ax2.set_ylim(0, 1)  # 设置y轴分布范围
    ax2.hist(gray_img.flatten(),bins=255,density=True,color='black',edgecolor='black')

    ax4.set_title('直方图规定化结果', fontsize=8)
    ax4.imshow(out_img, cmap='gray', vmin=0, vmax=255)

    ax5.grid(True, linestyle=':', linewidth=1)
    ax5.set_title('结果分布', fontsize=8)
    ax5.set_xlim(0, 255)  # 设置x轴分布范围
    ax5.set_ylim(0, 1)  # 设置y轴分布范围
    ax5.hist(out_img.flatten(),bins=255,density=True,color='black',edgecolor='black')

    # 窗口显示：绘制规定概率密度函数
    ax3.set_title("规定PDF", fontsize=12)
    ax3.grid(True, linestyle=':', linewidth=1)
    ax3.plot(np.arange(0, 256, 1), spec_hist, color='r',lw=1)

    # 窗口显示：绘制对应的灰度变换函数
    ax6.set_title("灰度变换：橙色T(k)，绿色G(q)", fontsize=10)
    ax6.grid(True, linestyle=':', linewidth=1)
    ax6.plot(np.arange(0, 256, 1), T_pts, '-',color='orange',lw=2)
    ax6.plot(np.arange(0, 256, 1), G_pts, 'k--',color='green', lw=2)

    plt.show()












