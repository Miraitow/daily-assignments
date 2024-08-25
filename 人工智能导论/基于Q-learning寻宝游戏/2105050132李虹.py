import pandas as pd
import numpy as np
import time
import tkinter as tk

UNIT = 40  # 像素
ENV_H = 8  # 格子高度
ENV_W = 8  # 格子宽度


class env(tk.Tk, object):
    def __init__(self):
        super(env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  #上下左右四个动作
        self.n_actions = len(self.action_space)
        self.title('寻宝游戏')
        self.geometry('{0}x{1}'.format(ENV_H * UNIT, ENV_H * UNIT))  #设置窗口的大小
        self._build_ENV()

    def _build_ENV(self):
        # 创建游戏画布
        self.canvas = tk.Canvas(self, bg='white',
                                height=ENV_H * UNIT,
                                width=ENV_W * UNIT)

        # 绘制网格
        for c in range(0, ENV_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, ENV_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, ENV_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, ENV_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # 创建陷阱
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # 创建陷阱
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # 创建陷阱
        hell3_center = origin + np.array([UNIT * 2, UNIT * 6])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')

        # 创建陷阱
        hell4_center = origin + np.array([UNIT * 6, UNIT * 2])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')

        # 创建陷阱
        hell5_center = origin + np.array([UNIT * 4, UNIT * 4])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')

        # 创建陷阱
        hell6_center = origin + np.array([UNIT * 4, UNIT * 1])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')

        # 创建陷阱
        hell7_center = origin + np.array([UNIT * 1, UNIT * 3])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')

        # 创建陷阱
        hell8_center = origin + np.array([UNIT * 2, UNIT * 4])
        self.hell8 = self.canvas.create_rectangle(
            hell8_center[0] - 15, hell8_center[1] - 15,
            hell8_center[0] + 15, hell8_center[1] + 15,
            fill='black')

        # 创建陷阱
        hell9_center = origin + np.array([UNIT * 3, UNIT * 2])
        self.hell9 = self.canvas.create_rectangle(
            hell9_center[0] - 15, hell9_center[1] - 15,
            hell9_center[0] + 15, hell9_center[1] + 15,
            fill='black')

        # 创建宝藏
        oval_center = origin + UNIT * 3
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # 创建寻宝人
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 显示画布
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # 返回观察结果
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect) #获取当前寻宝人位置
        base_action = np.array([0, 0])
        if action == 0:  # 上
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 下
            if s[1] < (ENV_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 右
            if s[0] < (ENV_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # 左
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # 移动寻宝人

        s_ = self.canvas.coords(self.rect)  # 获取新位置

        # 奖励函数
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4), self.canvas.coords(self.hell5), self.canvas.coords(self.hell6),
                    self.canvas.coords(self.hell7),
                    self.canvas.coords(self.hell8), self.canvas.coords(self.hell9)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update() #强制画布重绘


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) #存储每个状态-动作对的 Q 值的数据帧

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            #根据Q表选择最大的Q值对应的动作
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            #随机选择一个动作
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            #结合当前奖励和未来预期的奖励
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


    def check_state_exist(self, state): #检查状态 state 是否存在于 Q 表中
        if state not in self.q_table.index:
            self.q_table = self.q_table._append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


def update():
    for episode in range(150):
        observation = env.reset()
        print(episode)
        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                break
    print('game over')
    env.destroy()


if __name__ == '__main__':
    env = env()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
