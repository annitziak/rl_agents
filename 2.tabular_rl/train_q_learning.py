import gymnasium as gym
from tqdm import tqdm

from rl2025.constants import EX2_QL_CONSTANTS as CONSTANTS
from rl2025.exercise2.agents import QLearningAgent
from rl2025.exercise2.utils import evaluate

#also you can try this imports if you want to test with Run class for generalizability
#import numpy as np
#import os


#from rl2025.util.result_processing import Run
#from rl2025.util.result_processing import rank_runs, get_best_saved_run, get_mean_rank_runs, get_std_rank_runs, save_ranked_results_to_file


CONFIG = {
    "eval_freq": 1000, # keep this unchanged
    "alpha": 0.05,
    "epsilon": 0.9,
    "gamma": 0.99,  # for question 2.1 change this to 0.8
}
CONFIG.update(CONSTANTS)


def q_learning_eval(
        env,
        config,
        q_table,
        render=False):
    """
    Evaluate configuration of Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    if render:
        eval_env = gym.make(CONFIG["env"], render_mode="human")
    else:
        eval_env = env
    return evaluate(eval_env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"])


def train(env, config):
    """
    Train and evaluate Q-Learning on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        total reward over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table
    """
    agent = QLearningAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in tqdm(range(1, config["total_eps"]+1)):
        obs, _ = env.reset()
        episodic_return = 0
        t = 0

        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)
            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            agent.learn(obs, act, reward, n_obs, done)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = q_learning_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


if __name__ == "__main__":
    #print(f"Training Q-Learning agent with gamma={CONFIG['gamma']})")
    env = gym.make(CONFIG["env"]) 
    total_reward, _, _, q_table = train(env, CONFIG)
    env.close() #added to close the environment

    #if you want to try out multiple runs you can also run this code
    # this has shown consistently >0.6 mean return for all seeds so the agent is learning
    #SEEDS = np.random.randint(0, 10000, 10)
    # save these seeds 
    #np.save("exercise2/data/q_learning_seeds.npy", SEEDS)
    #runs_list = []

    #with open("exercise2/data/q_learning_results.txt", "w") as result_file:
    #    for seed in SEEDS:
    #        message = f"Training Q-Learning agent with seed={seed}"
    #        print(message)
    #        result_file.write(message + "\n")

    #        config = CONFIG.copy()
    #        config["save_filename"] = f"qlearning_seed_{seed}"
    #        config["seed"] = seed

    #        run = Run(config)
    #        run.run_name = f"q_learning_{seed}"

    #        env = gym.make(CONFIG["env"])
    #        total_reward, eval_means, eval_stds, q_table = train(env, config)

    #        run.update(eval_means, eval_stds)
    #        runs_list.append(run)
    #        env.close()

    #    ranked_runs = rank_runs(runs_list)
    #    message = "Ranked runs by final return mean:"
    #    print(message)
    #    result_file.write(message + "\n")

    #    for i, r in enumerate(ranked_runs):
    #        message = f"{i + 1}. {r.run_name}: Mean Final Return = {r.final_return_mean:.2f}"
    #        print(message)
    #        result_file.write(message + "\n")

    #    best_run, best_weights_file = get_best_saved_run(runs_list)
    #    message = f"Best run: {best_run.run_name}, Best saved model: {best_weights_file}"
    #    print(message)
    #    result_file.write(message + "\n")

    #    mean_rank = get_mean_rank_runs(runs_list)
    #    message = f"Mean rank of all {len(SEEDS)} runs: {mean_rank:.2f}"
    #    print(message)
    #    result_file.write(message + "\n")

    #    std_rank = get_std_rank_runs(runs_list)
    #    message = f"Standard deviation of all {len(SEEDS)} runs: {std_rank:.2f}"
    #    print(message)
    #    result_file.write(message + "\n")

    #print("Done!")
