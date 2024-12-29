'''
【功能】利用对数变换，压缩矩阵值域的动态范围，使其能够被人眼观察到更多细节
【B站/YouTube】轩辕十四很nice
【开源协议】The MIT License (MIT)
'''

import numpy as np
import matplotlib.pyplot as plt

def set_chinese():   # 中文显示工具函数
	import matplotlib
	matplotlib.rcParams['font.sans-serif'] = ['SimHei']
	matplotlib.rcParams['axes.unicode_minus'] = False

def image_log(input):
	return np.log(1 + input)  # 由于numpy广播机制，数字1会自动变成和input相同尺寸的全为1的矩阵

if __name__ == '__main__':

    set_chinese() # 调用此函数，窗口就可以正确显示中文

    # input1 = np.array([[-9, -8],
                      # [ 1,  2]])
					  
	
    input2 = np.array( [[10,   150  ],
                       [250,  25500]] )

    output2 = image_log(input2)
    print(output2)



    fig = plt.figure()
	
ax1 = fig.add_subplot(121)
ax1.set_title('对数变换前', fontsize=12)
ax1.imshow(input2, cmap='gray', vmin=0, vmax=25500)

ax2 = fig.add_subplot(122)
ax2.set_title('对数变换后', fontsize=12)
ax2.imshow(output2, cmap='gray')

plt.show()


