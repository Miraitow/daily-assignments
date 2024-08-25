import numpy as np
import copy
import matplotlib.pyplot as plt
import random

#准备好距离矩阵
city_num = 5
city_dist_mat = np.zeros([city_num, city_num])
city_dist_mat[0][1] = city_dist_mat[1][0] = 11
city_dist_mat[0][2] = city_dist_mat[2][0] = 22
city_dist_mat[0][3] = city_dist_mat[3][0] = 33
city_dist_mat[0][4] = city_dist_mat[4][0] = 44
city_dist_mat[1][2] = city_dist_mat[2][1] = 55
city_dist_mat[1][3] = city_dist_mat[3][1] = 66
city_dist_mat[1][4] = city_dist_mat[4][1] = 77
city_dist_mat[2][3] = city_dist_mat[3][2] = 88
city_dist_mat[2][4] = city_dist_mat[4][2] = 99
city_dist_mat[3][4] = city_dist_mat[4][3] = 100


# 定义全局变量，用于记录个体数量和距离列表
num_person_idx = 0  # 个体索引
num_person = 0      # 个体数量
dis_list = []       # 距离列表

class Individual:
    def __init__(self, genes=None):
        global num_person
        global dis_list
        global num_person_idx
        num_person_idx += 1
        # 每训练20次增加一个个体数量
        if num_person_idx % 20 == 0:
            num_person += 1
        
        self.genes = genes
        
        # 初始化基因
        if self.genes == None:
            genes = [0] * 5  # 城市数量
            temp = [0] * 4
            temp = [i for i in range(1, city_num)]  # 城市编号
            random.shuffle(temp)  # 打乱城市顺序
            genes[1:] = temp
            genes[0] = 0
            self.genes = genes
            self.fitness = self.evaluate_fitness()  # 计算适应度
        else:
            self.fitness = float(self.evaluate_fitness())
    
    # 计算个体的适应度
    def evaluate_fitness(self):
        dis = 0
        # 计算路径长度
        for i in range(city_num - 1):
            dis += city_dist_mat[self.genes[i]][self.genes[i+1]]
            if i == city_num - 2:
                dis += city_dist_mat[self.genes[i + 1]][0]  # 回到起点
        # 每20个个体记录一次距离
        if num_person_idx % 20 == 0:
            dis_list.append(dis)
        # 适应度为距离的倒数
        return 1 / dis

# 复制列表函数
def copy_list(old):
    new = []
    for element in old:
        new.append(element)
    return new

# 按适应度排序函数
def sort_win_num(group):
    for i in range(len(group)):
        for j in range(len(group) - i - 1):
            if group[j].fitness < group[j+1].fitness:
                temp = group[j]
                group[j] = group[j+1]
                group[j+1] = temp
    return group
#定义Ga类 
#3~5，交叉、变异、更新种群，全部在Ga类中实现
class Ga:
    #input_为城市间的距离矩阵
    def __init__(self, input_):
        #声明全局变量
        global city_dist_mat
        city_dist_mat = input_
        #当代的最佳个体
        self.best = Individual(None)
        #种群
        self.individual_list = []
        #每一代的最佳个体
        self.result_list = []
        #每一代个体对应的最佳适应度
        self.fitness_list = []
   
    
    # 交叉操作，采用交叉变异方法
    def cross(self):
        new_gen = []  # 用于存放新一代的个体
        num_cross = 3  # 交叉时选择的城市数量

        # 对每两个相邻的个体进行交叉
        for i in range(0, len(self.individual_list) - 1, 2):
            parent_gen1 = copy_list(self.individual_list[i].genes)  # 父代1的基因
            parent_gen2 = copy_list(self.individual_list[i+1].genes)  # 父代2的基因
            index1_1 = 0
            index1_2 = 0
            index2_1 = 0
            index2_2 = 0
            index_list = [0] * 3  # 存放交叉的起始索引的列表

            # 随机选择交叉的起始索引
            for i in range(city_num - 3):  # 这里应该是range(city_num - num_cross)，即0，1
                index_list[i] = i + 1
            index1_1 = random.choice(index_list)
            index1_2 = index1_1 + 2
            index2_1 = random.choice(index_list)
            index2_2 = index2_1 + 2

            # 获取交叉段的城市列表
            choice_list1 = parent_gen1[index1_1:index1_2 + 1]
            choice_list2 = parent_gen2[index2_1:index2_2 + 1]

            # 初始化两个子代的基因列表
            son_gen1 = [0] * city_num
            son_gen2 = [0] * city_num

            # 将交叉段复制到子代中
            son_gen1[index1_1: index1_2 + 1] = choice_list1
            son_gen2[index2_1: index2_2 + 1] = choice_list2

            temp1 = choice_list1  # 临时保存交叉段的城市列表
            temp2 = choice_list2

            # 处理未被交叉的城市
            if index1_1 == 0:
                pass
            else:
                for i in range(index1_1):
                    for j in range(city_num):
                        if parent_gen2[j] not in choice_list1:
                            son_gen1[i] = parent_gen2[j]
                            choice_list1.append(parent_gen2[j])
                            break

            choice_list1 = temp1

            if index1_2 == city_num - 1:
                pass
            else:
                for i in range(index1_2 + 1, city_num):
                    for j in range(city_num):
                        if parent_gen2[j] not in choice_list1:
                            son_gen1[i] = parent_gen2[j]
                            choice_list1.append(parent_gen2[j])
                            break

            # 处理子代2中未被交叉的城市，同理
            if index2_1 == 0:
                pass
            else:
                for i in range(index2_1):
                    for j in range(city_num):
                        if parent_gen1[j] not in choice_list2:
                            son_gen2[i] = parent_gen1[j]
                            choice_list2.append(parent_gen1[j])
                            break

            choice_list2 = temp2

            if index2_2 == city_num - 1:
                pass
            else:
                for i in range(index2_2 + 1, city_num):
                    for j in range(city_num):
                        if parent_gen1[j] not in choice_list2:
                            son_gen2[i] = parent_gen1[j]
                            choice_list2.append(parent_gen1[j])
                            break

            # 将新生成的子代加入新一代的列表中
            new_gen.append(Individual(son_gen1))
            new_gen.append(Individual(son_gen2))

        return new_gen
    #变异
    def mutate(self, new_gen):
        mutate_p = 0.02  # 变异概率，待调参数
        index_list = [0] * (city_num - 1)  # 初始化索引列表
        index_1 = 1  # 变异位置索引1
        index_2 = 1  # 变异位置索引2
        
        # 初始化索引列表，表示基因中除了起点之外的位置
        for i in range(city_num - 1):
            index_list[i] = i + 1
        
        # 对于新生成的个体群中的每个个体
        for individual in new_gen:
            # 根据变异概率决定是否进行变异
            if random.random() < mutate_p:
                # 随机选择两个不同的位置进行交换
                index_1 = random.choice(index_list)
                index_2 = random.choice(index_list)
                while index_1 == index_2:  # 确保选择的两个位置不相同
                    index_2 = random.choice(index_list)
                
                # 交换选定位置上的基因值
                temp = individual.genes[index_1]
                individual.genes[index_1] = individual.genes[index_2]
                individual.genes[index_2] = temp
        
        # 变异结束，将新生成的个体与老一代进行合并
        self.individual_list += new_gen
    #选择
    def select(self):
        #在此选用轮盘赌算法
        #考虑到5的阶乘是120，所以可供选择的个体基数应该适当大一些，
        #在此每次从种群中选择6个，进行轮盘赌，初始化60个个体，同时适当调高变异的概率
        select_num = 6
        select_list = []
        for i in range(select_num):
            
            gambler = random.choice(self.individual_list)
            gambler = Individual(gambler.genes)
            select_list.append(gambler)
        #求出这些fitness之和
        sum = 0
        for i in range(select_num):
            sum += select_list[i].fitness
        sum_m = [0]*select_num
        #实现概率累加
        for i in range(select_num):
            for j in range(i+1):
                sum_m[i] += select_list[j].fitness
            sum_m[i] /= sum
        new_select_list = []
        p_num = 0#随机数
        for i in range(select_num):
            p_num = random.uniform(0,1)
            if p_num>0 and p_num < sum_m[0]:
                new_select_list.append(select_list[0])
            elif p_num>= sum_m[0] and p_num < sum_m[1]:
                new_select_list.append(select_list[1])
            elif p_num >= sum_m[1] and p_num < sum_m[2]:
                new_select_list.append(select_list[2])
            elif p_num >= sum_m[2] and p_num < sum_m[3]:
                new_select_list.append(select_list[3])
            elif p_num >= sum_m[3] and p_num < sum_m[4]:
                new_select_list.append(select_list[4])
            elif p_num >= sum_m[4] and p_num < sum_m[5]:
                new_select_list.append(select_list[5])
            else:
                pass
        #将新生成的一代替代父代种群
        self.individual_list = new_select_list
    #更新种群
    def next_gen(self):
        #交叉
        new_gene = self.cross()
        #变异
        self.mutate(new_gene)
        #选择
        self.select()
        #获得这一代的最佳个体
        for individual in self.individual_list:
            if individual.fitness > self.best.fitness:
                self.best = individual
                
                
    def train(self):
        #随机出初代种群#
        individual_num = 60
        self.individual_list = [Individual() for _ in range(individual_num)]
        #迭代
        gen_num = 100
        for i in range(gen_num):
            #从当代种群中交叉、变异、选择出适应度最佳的个体，获得子代产生新的种群
            self.next_gen()
            #连接首位
            result = copy.deepcopy(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        print(self.result_list[-1])
        print('距离总和是：', 1/self.fitness_list[-1])
        
        
    def draw(self):
        # 创建x轴坐标，即迭代次数
        x_list = [i for i in range(num_person)]
        # 获取每一代最佳路径的长度列表作为y轴坐标
        y_list = dis_list
        # 设置图表大小
        plt.rcParams['figure.figsize'] = (60, 45)
        # 绘制折线图
        plt.plot(x_list, y_list, color='g')
        # 设置x轴标签
        plt.xlabel('Cycles', size=50)
        # 设置y轴标签
        plt.ylabel('Route', size=50)
        # 设置x轴刻度
        x = np.arange(0, 80, 5)
        plt.xticks(x)
        y = np.arange(0, 1000, 20)
        plt.yticks(y)
        # 设置图表标题
        plt.title('Trends in distance changes', size=50)
        # 设置刻度标签大小
        plt.tick_params(labelsize=30)
        plt.show()

route = Ga(city_dist_mat)  # 初始化遗传算法对象
route.train()  # 训练遗传算法获取最优路径
route.draw()  # 绘制路径长度变化趋势图
