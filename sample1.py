﻿import numpy as np
import gym
from gym import spaces

from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv


class GoLeftEnv(gym.Env):
  """
  Gymのインターフェースに従うカスタム環境
  エージェントが常に左に行くことを学ぶ環境
  """
  # ColabのためGUIを実装できない
  metadata = {'render.modes': ['console']}

  # 定数を定義
  LEFT = 0
  RIGHT = 1

  def __init__(self, grid_size=10):
    super(GoLeftEnv, self).__init__()

    # 1Dグリッドのサイズ
    self.grid_size = grid_size

    # グリッドの右側でエージェントを初期化
    self.agent_pos = grid_size - 1

    # 行動空間と状態空間を定義
    # gym.spacesオブジェクトでなければならない
    # 離散行動を使用する場合の例には、左と右の2つがある
    n_actions = 2
    self.action_space = spaces.Discrete(n_actions)

    # 状態はエージェントの座標になる
    # Discrete空間とBox空間の両方で表現できる
    self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                       shape=(1,), dtype=np.float32)

  def reset(self):
    """
    【重要】観測はnumpy配列でなければならない
    :return: (np.array)
    """
    # グリッドの右側でエージェントを初期化
    self.agent_pos = self.grid_size - 1

    # float32に変換してより一般的なものにします（連続行動を使用する場合）
    return np.array(self.agent_pos).astype(np.float32)

  def step(self, action):
    if action == self.LEFT:
      self.agent_pos -= 1
    elif action == self.RIGHT:
      self.agent_pos += 1
    else:
      raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    # グリッドの境界を表現
    self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

    # グリッドの左側にいるか
    done = self.agent_pos == 0

    # ゴールを除くすべての場所で0の報酬
    reward = 1 if self.agent_pos == 0 else 0

    # 必要に応じて情報を渡すことができるが、現在は未使用
    info = {}

    return np.array(self.agent_pos).astype(np.float32), reward, done, info

  def render(self, mode='console', close=False):
    if mode != 'console':
      raise NotImplementedError()

    # エージェントは「x」、残りは「.」として表現
    print("." * self.agent_pos, end="")
    print("x", end="")
    print("." * (self.grid_size - self.agent_pos))
    
def main1():
  env = GoLeftEnv(grid_size=10)

  obs = env.reset()
  env.render()

  print(env.observation_space)
  print(env.action_space)
  print(env.action_space.sample())

  GO_LEFT = 0

  # ハードコードされた最高のエージェント：常に左に行く
  n_steps = 20
  for step in range(n_steps):
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(GO_LEFT)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render()
    if done:
      print("Goal reached!", "reward=", reward)
      break

def main():
  # 環境の生成
  env = GoLeftEnv(grid_size=10)

  # 環境のラップ
  env = Monitor(env, filename=None, allow_early_resets=True)
  env = DummyVecEnv([lambda: env])

  # エージェントの訓練
  model = ACKTR('MlpPolicy', env, verbose=1).learn(5000)

  # 訓練済みエージェントのテスト
  obs = env.reset()
  n_steps = 20
  for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render(mode='console')
    if done:
      # VecEnvは、エピソード完了に遭遇すると自動的にリセットされることに注意
      print("Goal reached!", "reward=", reward)
      break

if __name__ == '__main__':
    main()