import gym
import keyboard
import time
env = gym.make('SpaceInvadersDeterministic-v4')
print('env.action_space')
print(env.action_space)
for i_episode in range(20):
    observation = env.reset()
    t = 0
    while True:
        env.render()
        time.sleep(0.05)
        if keyboard.is_pressed('j') and keyboard.is_pressed('k'):
            action = 5 # mover esquerda disparando
        elif keyboard.is_pressed('l') and keyboard.is_pressed('k'):
            action = 4 # mover dereita disparando
        elif keyboard.is_pressed('j'):
            action = 3 # mover esquerda
        elif keyboard.is_pressed('l'):
            action = 2 # mover dereita
        elif keyboard.is_pressed('k'):
            action = 1 # dispara
        else:
            action = 0 # No-Op
        observation, reward, done, info = env.step(action)
        remaining_lives = info['ale.lives']
        print(remaining_lives)
        t += 1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
