import gymnasium as gym

from rl2025.constants import EX2_MC_CONSTANTS as CONSTANTS
from rl2025.exercise2.agents import MonteCarloAgent
from rl2025.exercise2.utils import evaluate
from tqdm import tqdm

CONFIG = {
    "eval_freq": 5000, # keep this unchanged
    "epsilon": 0.9,
    "gamma": 0.99,  # for question 2.2 change this to 0.8
}
CONFIG.update(CONSTANTS)



def monte_carlo_eval(
        env,
        config,
        q_table,
        render=False):
    """
    Evaluate configuration of MC on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_table (Dict[(Obs, Act), float]): Q-table mapping observation-action to Q-values
    :param render (bool): flag whether evaluation runs should be rendered
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agent = MonteCarloAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=CONFIG["gamma"],
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
    Train and evaluate MC on given environment with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :return (float, List[float], List[float], Dict[(Obs, Act), float]):
        returns over all episodes, list of means and standard deviations of evaluation
        returns, final Q-table, final state-action counts
    """
    agent = MonteCarloAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )

    step_counter = 0
    max_steps = config["total_eps"] * config["eps_max_steps"]

    total_reward = 0
    evaluation_return_means = []
    evaluation_negative_returns = []

    for eps_num in tqdm(range(1, config["total_eps"] + 1)):
        obs, _ = env.reset()

        t = 0
        episodic_return = 0

        obs_list, act_list, rew_list = [], [], []
        while t < config["eps_max_steps"]:
            agent.schedule_hyperparameters(step_counter, max_steps)
            act = agent.act(obs)

            n_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            obs_list.append(obs)
            rew_list.append(reward)
            act_list.append(act)

            t += 1
            step_counter += 1
            episodic_return += reward

            if done:
                break

            obs = n_obs

        agent.learn(obs_list, act_list, rew_list)
        total_reward += episodic_return

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, negative_returns = monte_carlo_eval(env, config, agent.q_table)
            tqdm.write(f"EVALUATION: EP {eps_num} - MEAN RETURN {mean_return}")
            evaluation_return_means.append(mean_return)
            evaluation_negative_returns.append(negative_returns)

    return total_reward, evaluation_return_means, evaluation_negative_returns, agent.q_table


if __name__ == "__main__":
    print(f"Training MC with gamma={CONFIG['gamma']})")
    env = gym.make(CONFIG["env"]) 
    total_reward, _, _, q_table = train(env, CONFIG)

    #if you want to try out multiple runs you can also run this code
    # this has shown consistently >0.6 mean return for all seeds so the agent is learning
    """SEEDS = np.random.randint(0, 10000, 5)
    runs_list = []
    for seed in SEEDS:
        config = CONFIG.copy()
        config["save_filename"] = f"mc_seed_{seed}" 
        config["seed"] = seed #make sure to put seed at correct place
        run = Run(config)
        run.run_name = f"mc_{seed}"
        env = gym.make(CONFIG["env"]) #change this flag when needed is_slippery
        total_reward, eval_means, eval_stds, q_table = train(env, config)

        run.update(eval_means, eval_stds)
        runs_list.append(run)
        env.close()

    # Rank the runs by performance
    ranked_runs = rank_runs(runs_list)
    print("Ranked runs by final return mean:")
    for i, r in enumerate(ranked_runs):
        print(f"{i + 1}. {r.run_name}: Mean Final Return = {r.final_return_mean:.2f}")

    # Get the best run
    best_run, best_weights_file = get_best_saved_run(runs_list)
    print(f"Best run: {best_run.run_name}, Best saved model: {best_weights_file}")

    mean_rank = get_mean_rank_runs(runs_list)
    print(f"Mean rank of all {len(SEEDS)} runs: {mean_rank:.2f}")

    std_rank = get_std_rank_runs(runs_list)
    print(f"Standard deviation of all {len(SEEDS)} runs: {std_rank:.2f}")

    # Save the results to a file
    save_ranked_results_to_file(runs_list, "exercise2/data/mc_results.txt")

    print("Done!") """

