import random
import pygame
from pygame.locals import *
import sys
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

SCREENWIDTH = 822
SCREENHEIGHT = 199
FPS = 30  # 每秒帧数，降低以减小压力

# Q-learning parameters
alpha = 0.5  # 学习率
gamma = 0.95  # 折扣因子
epsilon = 0.5  # 探索-利用折中

# Initialize Q-table
y_ranges = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # 根据游戏中玛丽可能的y坐标范围,更细化，优化状态空间的离散化
jump_states = ['ground', 'rising', 'falling']  # 跳跃状态（例如：上升、下降、地面）
# 更精细的障碍物距离分段（近距离细分，远距离合并）
obstacle_x_ranges = [0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500]

number_of_states = len(y_ranges) * len(jump_states) * len(obstacle_x_ranges) * len(y_ranges)
actions = ['left', 'right','jump', 'no_action']
number_of_actions = len(actions)

Q = np.zeros((number_of_states, number_of_actions))

def plot_q_values(episode_numbers, all_scores, all_epsilons,q_values_history):

    # 绘制图表
    plt.figure(figsize=(12, 8))

    # 平均奖励
    plt.subplot(2, 2, 1)
    plt.plot(episode_numbers, all_scores, label='Average Reward', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode')
    plt.legend()
    plt.grid(True)
    # 显示最终平均奖励
    final_reward = all_scores[-1]
    plt.text(episode_numbers[-1], final_reward, f'Final: {final_reward:.2f}',
             fontsize=10, color='black', ha='right', va='bottom')

    # 平均 Q 值
    plt.subplot(2, 2, 2)
    plt.plot(episode_numbers, q_values_history, label='Average Q-value', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average Q-value')
    plt.title('Average Q-value per Episode')
    plt.legend()
    plt.grid(True)
    # 显示最终平均 Q 值
    final_q_value = q_values_history[-1]
    plt.text(episode_numbers[-1], final_q_value, f'Final: {final_q_value:.2f}',
             fontsize=10, color='black', ha='right', va='bottom')

    # 探索率 (Epsilon)
    plt.subplot(2, 2, 3)
    plt.plot(episode_numbers, all_epsilons, label='Epsilon', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate (Epsilon)')
    plt.legend()
    plt.grid(True)
    # 显示最终探索率
    final_epsilon = all_epsilons[-1]
    plt.text(episode_numbers[-1], final_epsilon, f'Final: {final_epsilon:.3f}',
             fontsize=10, color='black', ha='right', va='bottom')

    # 成功次数（假设成功次数为分数大于 0 的次数）
    success_counts = [sum(1 for score in all_scores[:i + 1] if score > 0) for i in range(len(all_scores))]
    plt.subplot(2, 2, 4)
    plt.plot(episode_numbers, success_counts, label='Success Count', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Success Count')
    plt.title('Success Count per Episode')
    plt.legend()
    plt.grid(True)
    # 显示最终成功次数
    final_success_count = success_counts[-1]
    plt.text(episode_numbers[-1], final_success_count, f'Final: {final_success_count}',
             fontsize=10, color='black', ha='right', va='bottom')
    plt.tight_layout()

    plt.savefig("result_pic/q_learning_results.png")  # 将图像保存到 result_pic 文件夹
    plt.show()

def mainGame():
    global epsilon

    # 初始化字体
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 24)  # 使用系统字体，大小为 24

    episode_numbers = []
    all_scores = []
    all_epsilons = []
    q_values_history = []

    score = 0
    over = False
    global SCREEN, FPSCLOCK, Q  # 将 Q 表设为全局变量
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption("RUN! MATCHSTICK MAN")

    addObstackeTimer = 0
    obstacles = []  # 障碍物列表

    # 初始化游戏对象
    muscic_button = Music_Button()
    btu_img = muscic_button.open_img
    muscic_button.bg_music.play(-1)
    bg1 = MyMap(0, 0)
    bg2 = MyMap(800, 0)
    marie = Marie()

    # Q-learning 相关参数
    total_episodes = 1000  # 训练的总回合数
    current_episode = 0
    epsilon_decay = 0.995  # 探索率衰减系数

    # 终点标志
    finish_line = SCREENWIDTH - 50  # 终点位于屏幕最右侧
    finish_reached = False  # 是否到达终点

    def game_over():
        bump_audio = pygame.mixer.Sound("audio/bump.wav")
        bump_audio.play()

        screen_w = pygame.display.Info().current_w
        screen_h = pygame.display.Info().current_h
        over_img = pygame.image.load("image/gameover.png").convert_alpha()
        SCREEN.blit(over_img, ((screen_w - over_img.get_width()) / 2, (screen_h - over_img.get_height()) / 2))

    def game_win():
        win_audio = pygame.mixer.Sound("audio/win.wav")
        win_audio.play()

        screen_w = pygame.display.Info().current_w
        screen_h = pygame.display.Info().current_h
        win_img = pygame.image.load("image/win.png").convert_alpha()
        SCREEN.blit(win_img, ((screen_w - win_img.get_width()) / 2, (screen_h - win_img.get_height()) / 2))

    step = 0
    rewards = []

    while current_episode < total_episodes:
        current_episode += 1
        over = False
        finish_reached = False  # 重置终点标志
        score = 0
        marie = Marie()  # 重置玛丽
        obstacles = []  # 清空障碍物
        state = None  # 初始状态

        # 单个回合的训练循环
        while not over:
            step += 1
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONUP:
                    if muscic_button.is_select():
                        if muscic_button.is_open:
                            btu_img = muscic_button.close_img
                            muscic_button.is_open = False
                            muscic_button.bg_music.stop()
                        else:
                            btu_img = muscic_button.open_img
                            muscic_button.is_open = True
                            muscic_button.bg_music.play(-1)

            if not over and not finish_reached:
                # 1. 获取当前状态
                nearest_obstacle = None
                if obstacles:
                    # 只考虑右侧的障碍物（x坐标大于玛丽的位置）
                    valid_obstacles = [obj for obj in obstacles if obj.rect.x > marie.rect.x]
                    if valid_obstacles:
                        nearest_obstacle = min(valid_obstacles, key=lambda obj: obj.rect.x - marie.rect.x)

                if nearest_obstacle:
                    # 获取当前状态索引
                    current_state = marie.get_state(nearest_obstacle)

                    # 2. 选择动作 (epsilon-greedy)
                    if random.uniform(0, 1) < epsilon:
                        action = random.choice(actions)  # 随机探索
                    else:
                        action_idx = np.argmax(Q[current_state, :])
                        action = actions[action_idx]

                    # 3. 执行动作（替代原来的键盘输入）
                    marie.move(action)

                # --- 游戏逻辑更新 ---
                # 更新背景和玛丽位置
                if not finish_reached:  # 如果未到达终点，继续滚动背景
                    bg1.map_rolling()
                    bg2.map_rolling()

                #marie.move()

                # 生成新障碍物（保持原有逻辑）
                addObstackeTimer += 10
                if addObstackeTimer >= 100:
                    if random.randint(0, 50) > 15:
                        obstacles.append(Obstacle())
                    addObstackeTimer = 0

                # 更新障碍物位置并检测碰撞
                reward = 0  # 默认奖励
                for obstacle in obstacles:
                    obstacle.obstacle_move()

                    # 碰撞检测
                    if pygame.sprite.collide_rect(marie, obstacle):
                        over = True
                        reward = -50  # 碰撞惩罚
                        game_over()
                    elif (obstacle.rect.x + obstacle.rect.width) < marie.rect.x:
                        score += obstacle.getSocre()
                        reward = +10  # 成功躲避奖励
                    else:
                        # 根据障碍物的垂直距离给予奖励
                        vertical_distance = abs(obstacle.rect.y - marie.rect.y)
                        if vertical_distance < 20:  # 如果垂直距离较小，给予额外奖励
                            reward = +2
                        elif marie.jumpState and vertical_distance > 50:  # 如果角色在不必要时跳跃，给予惩罚
                            reward = -0.5
                        else:
                            reward = +0.1  # 默认奖励
                    # 将奖励累加到分数
                    score += reward

                # 4. 获取新状态并更新 Q 表
                if nearest_obstacle and not over:
                    new_state = marie.get_state(nearest_obstacle)
                    action_idx = actions.index(action) if action else 0
                    # Q-learning 更新公式
                    Q[current_state][action_idx] += alpha * (
                          reward + gamma * np.max(Q[new_state]) - Q[current_state][action_idx]
                    )
                # 5. 检测是否到达终点
                if marie.rect.x >= finish_line:
                    finish_reached = True
                    #reward = +100  # 到达终点的奖励
                    game_win()


                # 渲染画面
                SCREEN.fill((0, 0, 0))
                bg1.map_update()
                bg2.map_update()
                marie.draw_marie()
                for obstacle in obstacles:
                    obstacle.draw_obstacle()
                SCREEN.blit(btu_img, (20, 20))

                # 渲染并显示分数
                score_text = font.render(f"Score: {score}", True, (100, 100, 100))  # 白色文本
                SCREEN.blit(score_text, (SCREENWIDTH - 150, 20))  # 将分数显示在屏幕右上角

                pygame.display.update()
                FPSCLOCK.tick(FPS)

            # 回合结束处理
            if over:
                epsilon *= epsilon_decay  # 衰减探索率
                print(f"Episode: {current_episode}, Score: {score}, Epsilon: {epsilon:.3f}")
                episode_numbers.append(current_episode)
                all_scores.append(score)
                all_epsilons.append(epsilon)
                q_values_history.append(np.mean(Q))
                break

    # 可视化Q值收敛情况
    plot_q_values(episode_numbers, all_scores, all_epsilons, q_values_history)

    pygame.quit()


# 玛丽类
class Marie:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.jumpState = False
        self.jumpHeight = 130
        self.lowest_y = 140
        self.jumpValue = 0

        self.marieIndex = 0
        self.marieIndexGen = cycle([0, 1, 2])

        self.adventure_img = (
            pygame.image.load("image/adventure1.png").convert_alpha(),
            pygame.image.load("image/adventure2.png").convert_alpha(),
            pygame.image.load("image/adventure3.png").convert_alpha(),
        )

        self.jump_audio = pygame.mixer.Sound("audio/jump.wav")
        self.rect.size = self.adventure_img[0].get_size()

        self.x = 50
        self.y = self.lowest_y
        self.rect.topleft = (self.x, self.y)

        # 左右移动速度
        self.move_speed = 5

    # 跳方法
    def jump(self):
        self.jumpState = True

    # 玛丽移动
    def move(self, action):
        # 执行动作
        if action == "left" and self.rect.x > 0:  # 防止角色移出屏幕左侧
            self.rect.x -= self.move_speed
        elif action == "right" and self.rect.x < SCREENWIDTH - self.rect.width:  # 防止角色移出屏幕右侧
            self.rect.x += self.move_speed
        elif action == "jump" and self.rect.y >= self.lowest_y:
            self.jump_audio.play()
            self.jump()
        if self.jumpState:
            if self.rect.y >= self.lowest_y:
                self.jumpValue = -15

            if self.rect.y <= self.lowest_y - self.jumpHeight:
                self.jumpValue = 15  # 增大绝对值，使角色下降更快

            self.rect.y += self.jumpValue

            if self.rect.y >= self.lowest_y:
                self.jumpState = False

    # 绘制玛丽
    def draw_marie(self):
        marieIndex = next(self.marieIndexGen)
        SCREEN.blit(self.adventure_img[marieIndex], (self.x, self.rect.y))


    def can_jump(self):
        return self.rect.y >= self.lowest_y and not self.jumpState

    def get_jump_state(self):
        if not self.jumpState:
            return 0  # ground
        return 1 if self.jumpValue < 0 else 2  # rising/falling

    def get_state(self, obstacle):
        # 优化后的状态计算
        y_state = np.digitize(self.rect.y, y_ranges)
        jump_state = self.get_jump_state()
        # 障碍物的水平距离
        obs_x_state = np.digitize(obstacle.rect.x - self.rect.x, obstacle_x_ranges)
        # 障碍物的垂直距离
        obs_y_state = np.digitize(obstacle.rect.y - self.rect.y, y_ranges)
        # 障碍物类型（0: missile, 1: pipe）
        obs_type = 0 if obstacle.image == obstacle.missile else 1

        state_index = (y_state * len(jump_states) * len(obstacle_x_ranges) * len(y_ranges) * 2 +
                       jump_state * len(obstacle_x_ranges) * len(y_ranges) * 2 +
                       obs_x_state * len(y_ranges) * 2 +
                       obs_y_state +
                       obs_type)
        state_index = state_index % number_of_states  # 确保索引在 Q 表范围内


        return state_index

    def get_jump_state(self):
        if not self.jumpState:
            return 0  # ground
        return 1 if self.jumpValue < 0 else 2  # rising/falling

# 绘制地图
class MyMap:
    def __init__(self, x, y):
        self.bg = pygame.image.load("image/bg.png").convert_alpha()
        self.x = x
        self.y = y

    def map_rolling(self):
        if self.x < -790:
            self.x = 800
        else:
            self.x -= 5

    def map_update(self):
        SCREEN.blit(self.bg, (self.x, self.y))


# 障碍物类
class Obstacle:
    score = 1
    move = 5
    obstacle_y = 150

    def __init__(self):
        self.move = 10 if random.random() < 0.3 else 7  # 导弹/管道不同速度
        self.obstacle_y = 100 if self.move == 10 else 150

        self.rect = pygame.Rect(0, 0, 0, 0)
        self.missile = pygame.image.load("image/missile.png").convert_alpha()
        self.pipe = pygame.image.load("image/pipe.png").convert_alpha()
        self.numbers = (
            pygame.image.load("image/0.png").convert_alpha(),
            pygame.image.load("image/1.png").convert_alpha(),
            pygame.image.load("image/2.png").convert_alpha(),
            pygame.image.load("image/3.png").convert_alpha(),
            pygame.image.load("image/4.png").convert_alpha(),
            pygame.image.load("image/5.png").convert_alpha(),
            pygame.image.load("image/6.png").convert_alpha(),
            pygame.image.load("image/7.png").convert_alpha(),
            pygame.image.load("image/8.png").convert_alpha(),
            pygame.image.load("image/9.png").convert_alpha(),
        )

        self.score_audio = pygame.mixer.Sound("audio/score.wav")

        r = random.randint(0, 1)

        if r == 0:
            self.image = self.missile
            self.move = 15  # 导弹速度较快
            self.obstacle_y = 100   # 导弹出现在天空
        else:
            self.image = self.pipe
            self.move = 7  # 管道速度较慢
            self.obstacle_y = 150  # 管道固定在地面

        self.rect.size = self.image.get_size()
        self.width, self.height = self.rect.size
        self.x = 800
        self.y = self.obstacle_y
        self.rect.center = (self.x, self.y)

    def obstacle_move(self):
        self.rect.x -= self.move

    def draw_obstacle(self):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))

    def getSocre(self):
        self.score
        tmp = self.score
        if tmp == 1:
            self.score_audio.play()
        self.score = 0
        return tmp

    # 显示分数
    def showScore(self, score):
        self.scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0

        for digit in self.scoreDigits:
            totalWidth += self.numbers[digit].get_width()

        Xoffset = (SCREENWIDTH - (totalWidth + 30))

        for digit in self.scoreDigits:
            SCREEN.blit(self.numbers[digit], (Xoffset, SCREENHEIGHT * 0.1))
            Xoffset += self.numbers[digit].get_width()


# 背景音乐按钮
class Music_Button:
    is_open = True

    def __init__(self):
        self.open_img = pygame.image.load("image/btn_open.png").convert_alpha()
        self.close_img = pygame.image.load("image/btn_close.png").convert_alpha()
        self.bg_music = pygame.mixer.Sound("audio/bg_music.wav")

    def is_select(self):
        point_x, point_y = pygame.mouse.get_pos()
        w, h = self.open_img.get_size()
        in_x = point_x > 20 and point_x < 20 + w
        in_y = point_y > 20 and point_y < 20 + h

        return in_x and in_y

if __name__ == "__main__":
    mainGame()