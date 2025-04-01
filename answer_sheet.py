
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the Every-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and Every-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) Every-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / Every-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """

    answer = 'Monte Carlo (MC) is more affected by gamma as it computes returns as sums of exponentially discounted rewards over ENTIRE episodes (R₁ + γR₂ + γ²R₃ + …), making it highly sensitive to gamma. Small reductions greatly diminish distant rewards, especially in longer episodes with delayed rewards.  Conversely, Q-learning bootstraps and updates Q-values INCREMENTALLY via the Bellman equation  (Q(s,a) = R + γ maxₐQ(snew,anew)), using immediate rewards and a one-step future estimate. This gradual adjustment makes it less sensitive to gamma changes. Larger gamma values increase the weight of distant rewards. These differences significantly impact convergence speed and policy quality, especially in environments with longterm dependencies.'
    return answer

def question2_5() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) on the differences between the non-slippery and the slippery varian of the problem.
    by [Q-learning / Every-Visit Monte Carlo].
    return: answer (str): your answer as a string (100 words max)
    """
    
    answer = """The slippery variant adds stochasticity, making transitions unpredictable and learning harder, 
    as actions dont lead to the intended state (1/3 chance of slipping). Q-learning adapts faster due to 
    incremental updates, reaching stability quicker. In contrast, MC suffers from higher variance and slower
    convergence, resulting in lower average rewards. Both methods perform worse than in the deterministic case. 
    Lowering gamma reduces returns further, as short-term rewards dominate. In the non-slippery setting, 
    deterministic transitions simplify learning: Q-learning quickly achieves optimal returns, while MC converges 
    but more slowly. Overall, Q-learning is more robust to stochastic environments and outperforms MC."""
                
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In the DiscreteRL algorithm, which learning rate achieves the highest mean returns at the end of training?
    a) 2e-2
    b) 2e-3
    c) 2e-4
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a", "b" or "c" 
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.99
    b) 0.75
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c" 
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.5
    c) 1e-5
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c" 
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0? 
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e" #it will be epsilon start
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.95?
    a) 0.95
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "e"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e" 
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = """
    A strategy based on exploration fraction ensures a controlled transition from exploration to exploitation,
    independent of timesteps, making it predictable and consistent across environments. It is less sensitive 
    to parameter choices and works well with varying episode lengths, and cases where premature exploitation could
    hinder learning, as it provides a more predictable exploration schedule, easier generalization. In contrast, exponential decay uses a
    fixed rate, which may cause epsilon to decay too quickly or slowly depending on timesteps, requiring 
    fine-tuning per environment and is too dependant on training duration.
    """

    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = """ 
    In DQN, the loss does not steadily decrease like in supervised learning because the target Q-values 
    change dynamically throughout training, whereas in supervised learning, targets remain fixed. In DQN, 
    the target network is updated periodically, causing abrupt shifts in value estimation and fluctuations(increases) in the loss. 
    Additionally, DQN has correlated, non-stationary data due to evolving state-action transitions as the policy updates, 
    unlike the independent, stationary data assumed in supervised learning. Smaller fluctuations in loss also occur non-periodically as the agent 
    shifts to exploration which can temporarily decrease performance (not necessarily optimal action). 
    These factors result in periodic loss spikes rather than a smooth decline, making DQN training 
    more unstable than typical supervised learning.
    """
    # TYPE YOUR ANSWER HERE (150 words max)
    return answer

def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = """
    Spikes in DQN training occur due to periodic target network updates (every 2000 timesteps), which cause abrupt 
    shifts in Q-value estimates. Since the target Q-values suddenly change, the difference between predicted and target 
    values generally increases, leading to higher loss. The frequency of the spikes (i.e the target network updates) is a parameter
    with a trade-off between stability and learning speed. Too frequent updates can lead to instability, while too infrequent
    updates can slow down learning. Larger discrepancies between Q-values and target values lead to more significant increases
    in loss, causing higher loss spikes in the training process.
    """  # TYPE YOUR ANSWER HERE (100 words max)

    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = """
    Please for all this refer to exercise5 directory where i have the pdf file of my answer and findings as well as
    all the code and data used to generate it. The main code is in the train_sarsa_agent.py and sarsa.py where you can find
    the code for SARSA and also the alternative linear decay strategy I used. The directory data also saves the output to ensure
    that you can check the results. They are plotted for your convenience at the pdf along with the insights I found.
    """  # TYPE YOUR ANSWER HERE (200 words max)
    return answer
