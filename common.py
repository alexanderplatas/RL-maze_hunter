import cv2


def evaluate_policy(model, env, n_eval_episodes: int = 100):

    total_steps = 0
    total_reward = 0

    # Poner render a True para ver la evaluación
    env.unwrapped.set_render(True)

    for n in range(n_eval_episodes):

        episode_steps = 0
        episode_reward = 0
        done = False

        obs, info = env.reset()

        while not done:
            action, _states = model.predict(obs)
            if episode_steps == 0 and action == 0: continue
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

        print(f"Epidose {n+1} -->  Reward: {episode_reward} - Steps: {episode_steps}")
        total_reward += episode_reward
        total_steps += episode_steps

    print(f"\n  - Average reward per episode: {total_reward / n_eval_episodes}")
    print(f"  - Average steps per episode: {total_steps / n_eval_episodes}")

    # Volver a poner el render a False
    env.unwrapped.set_render(False)

    cv2.destroyAllWindows()
