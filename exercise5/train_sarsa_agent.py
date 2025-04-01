import gymnasium as gym
import numpy as np
from tqdm import tqdm

from rl2025.exercise5.sarsa import SARSAAgent
from rl2025.constants import EX2_QL_CONSTANTS as CONSTANTS
from rl2025.exercise2.utils import evaluate

from rl2025.util.result_processing import Run, rank_runs, get_best_saved_run, get_mean_rank_runs, get_std_rank_runs

CONFIG = {
    "eval_freq": 1000,
    "alpha": 0.05,
    "epsilon": 0.9,
    "gamma": 0.99,
}
CONFIG.update(CONSTANTS)


def sarsa_eval(env, config, q_table, render=False):
    eval_agent = SARSAAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        alpha=config["alpha"],
        epsilon=0.0,
    )
    eval_agent.q_table = q_table
    eval_env = gym.make(CONFIG["env"], render_mode="human") if render else env

    return evaluate(
        eval_env, eval_agent, config["eval_eps_max_steps"], config["eval_episodes"]
    )


def train_sarsa(env, config, file_handle):
    agent = SARSAAgent(
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

    for eps_num in tqdm(range(1, config["total_eps"] + 1), file=file_handle):
        obs, _ = env.reset()
        episodic_return = 0
        t = 0

        agent.schedule_hyperparameters(step_counter, max_steps)
        act = agent.act(obs)

        while t < config["eps_max_steps"]:
            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            n_act = agent.act(n_obs) if not done else None
            agent.learn(obs, act, reward, n_obs, n_act, done)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs, act = n_obs, n_act

        total_reward += episodic_return

        if eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = sarsa_eval(env, config, agent.q_table)
            eval_message = f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}"
            tqdm.write(eval_message, file=file_handle)
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


if __name__ == "__main__":

    # Load the seeds from the file : they should be ranom but the same as the ones used in the Q-learning agent

    with open("exercise5/data/q_learning_results.txt", "r") as seeds_file:
        SEEDS = list(map(int, seeds_file.readline().split("Using seeds [")[1].split("]")[0].split(",")))

    runs_list = []

    with open("exercise5/data/sarsa_results_linear_decay.txt", "w") as result_file:  #change this to save in other file 
        print(f"Using seeds {SEEDS}")
        result_file.write(f"Using seeds {SEEDS}\n")

        for seed in SEEDS:
            message = f"Training SARSA agent with seed={seed}"
            print(message)
            result_file.write(message + "\n")

            config = CONFIG.copy()
            config["save_filename"] = f"sarsa_seed_{seed}"
            config["seed"] = seed

            run = Run(config)
            run.run_name = f"sarsa_{seed}"

            env = gym.make(CONFIG["env"], is_slippery=True) 
            total_reward, eval_means, eval_stds, q_table = train_sarsa(env, config, result_file)

            run.update(eval_means, eval_stds)
            runs_list.append(run)
            env.close()

        ranked_runs = rank_runs(runs_list)
        summary_messages = ["Ranked runs by final return mean:"]
        for i, r in enumerate(ranked_runs):
            summary_messages.append(f"{i + 1}. {r.run_name}: Mean Final Return = {r.final_return_mean:.2f}")

        best_run, best_weights_file = get_best_saved_run(runs_list)
        summary_messages.append(f"Best run: {best_run.run_name}, Best saved model: {best_weights_file}")

        mean_rank = get_mean_rank_runs(runs_list)
        summary_messages.append(f"Mean rank of all {len(SEEDS)} runs: {mean_rank:.2f}")

        std_rank = get_std_rank_runs(runs_list)
        summary_messages.append(f"Standard deviation of all {len(SEEDS)} runs: {std_rank:.2f}")

        for msg in summary_messages:
            print(msg)
            result_file.write(msg + "\n")

    print("Done!")