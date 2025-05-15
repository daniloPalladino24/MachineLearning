import pygame
import numpy as np
import time
import random
import pickle
import matplotlib.pyplot as plt
from cat_mouse_cheese_env import CatMouseCheeseEnv
from graphics import init_graphics, draw_grid, render_entities, load_images

def train_q_learning(env, episodes=1000000, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
    q_table = np.zeros((env.grid_size, env.grid_size, env.grid_size, env.grid_size, env.grid_size, env.grid_size, 4))
    rewards_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state[0], state[1], state[2], state[3], state[4], state[5]])

            next_state, reward, done, _, _ = env.step(action)

            q_table[state[0], state[1], state[2], state[3], state[4], state[5], action] = (1 - alpha) * q_table[state[0], state[1], state[2], state[3], state[4], state[5], action] + \
                alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], next_state[5]]))

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    return q_table, rewards_per_episode

def plot_learning_curve(rewards_per_episode, filename="learning_curve.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode, label='Total Reward per Episode')
    
    # Calcolare la media mobile
    window_size = 100
    moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(rewards_per_episode)), moving_avg, color='red', label='Media')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def save_q_table(q_table, filename="q_table.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(q_table, file)

def load_q_table(filename="q_table.pkl"):
    with open(filename, "rb") as file:
        return pickle.load(file)

def test_q_learning(env, q_table, episodes=10, delay=0.5):
    pygame.init()
    screen, clock = init_graphics(env.grid_size)
    mouse_img, cat_img, cheese_img = load_images()

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"\n🎬 Episodio {i+1}/{episodes} 🎬\n")
        time.sleep(1)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            action = np.argmax(q_table[state[0], state[1], state[2], state[3], state[4], state[5]])
            state, _, done, _, _ = env.step(action)

            draw_grid(screen, env.grid_size)
            render_entities(screen, env.mouse_pos, env.cat_pos, env.cheese_pos, env.walls, env.grid_size, mouse_img, cat_img, cheese_img)
            pygame.display.flip()
            clock.tick(30)
            time.sleep(delay)

        if state[0:2].tolist() == env.cheese_pos:
            print("🎉 Il topo ha raggiunto il formaggio! 🧀")
        else:
            print("💀 Il topo è stato preso dal gatto! 😿")
        time.sleep(1)

    pygame.quit()

def calculate_accuracy(env, q_table, test_episodes=10000):
    successes = 0

    for _ in range(test_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(q_table[state[0], state[1], state[2], state[3], state[4], state[5]])
            state, _, done, _, _ = env.step(action)

        if state[0:2].tolist() == env.cheese_pos:
            successes += 1
        elif state[0:2].tolist() == env.cat_pos:
            pass

    accuracy = (successes / test_episodes) * 100
    print(f"✅ Accuratezza: {accuracy:.2f}% su {test_episodes} episodi di test")
    return accuracy

def main():
    env = CatMouseCheeseEnv(grid_size=5)
    
    # Addestramento e salvataggio della Q-table
    #q_table, rewards_per_episode = train_q_learning(env)
    #save_q_table(q_table, "q_table.pkl")
    #plot_learning_curve(rewards_per_episode, filename="learning_curve.png")
    
    # Caricamento della Q-table e test
    q_table = load_q_table("q_table1000000.pkl")
    test_q_learning(env, q_table, episodes=10, delay=0.5)
    calculate_accuracy(env, q_table, test_episodes=10000)

if __name__ == "__main__":
    main()