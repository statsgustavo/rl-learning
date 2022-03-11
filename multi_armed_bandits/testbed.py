import os

import fire
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def _bandit(action_value):
    return np.random.normal(
        loc=action_value,
        scale=1,
    )


def _add_white_noise(actions, sigma=0.01):
    return actions + np.random.normal(0, sigma, size=actions.size)


def _select_optimal_action_and_break_ties_randomly(actions_values, actions_rewards):
    optimal_action_idx = np.argmax(actions_rewards)

    if optimal_action_idx.size > 1:
        optimal_action_idx = np.random.choice(optimal_action_idx, size=1)

    return optimal_action_idx, actions_rewards[optimal_action_idx]


def _explore(explore_probability):
    p = np.random.uniform(low=0, high=1, size=1)
    return True if p < explore_probability else False


def _choose_one_action(actions_values, explore_probability):
    actions_rewards = _bandit(actions_values)

    (
        optimal_action_id,
        optimal_action_reward,
    ) = _select_optimal_action_and_break_ties_randomly(actions_values, actions_rewards)

    if _explore(explore_probability):
        # Exploration
        chosen_action_id = np.random.choice(np.arange(actions_values.size))
        chosen_action_reward = actions_rewards[chosen_action_id]
        is_greedy = 0
    else:
        # Exploitation
        chosen_action_id, chosen_action_reward = (
            optimal_action_id,
            optimal_action_reward,
        )

        is_greedy = 1

    return chosen_action_id, chosen_action_reward, is_greedy


def incremental_average(current_estimate, current_num_steps, reward):
    return current_estimate + (1 / current_num_steps) * (reward - current_estimate)


def _log_experiment(average_reward, step_number, num_optimal_actions):
    log_msg = "[Experiment run #{0}] Avg. reward: {1}\t|\t% of optimal actions: {2}\n"
    pct_optimal_actions = num_optimal_actions / step_number

    log_msg = log_msg.format(step_number, average_reward, pct_optimal_actions)

    print(log_msg, end="\r", flush=True)


def plot_evolution(steps, average_reward, proportion_optimal, epsilon, num_arms):
    figure, axes = plt.subplots(2, 1, figsize=(20, 10))
    axes[0].plot(steps, average_reward)
    axes[1].hlines(1 - epsilon, steps.min(), steps.max(), ls="--", color="black")
    axes[1].plot(steps, proportion_optimal)

    _ = [ax.set_xlabel("Step") for ax in axes]
    axes[0].set_ylabel("Average reward")
    axes[1].set_ylabel("% Optimal action")
    axes[0].set_title(f"{num_arms} armed badit - $\epsilon$ = {epsilon}")
    figure.savefig(f"./outputs/{num_arms}-armed-badit-epsilon-{epsilon}.png")


def plot_reward_distribution(actions, rewards, epsilon, num_arms):
    figure, axis = plt.subplots(1, 1, figsize=(20, 10))
    axis = sns.violinplot(x=actions, y=rewards, ax=axis, order=np.arange(num_arms))
    axis.set_xlabel("Action")
    axis.set_ylabel("Reward distribution")
    axis.set_title(
        f"Reward distribution\n{num_arms} armed badit - $\epsilon$ = {epsilon}"
    )
    figure.savefig(
        f"./outputs/{num_arms}-armed-badit-epsilon-{epsilon}-reward-distribution.png"
    )


def run_experiment(
    num_actions: int, epsilon: float, max_runs: int = 10000, keep_track: bool = False
):
    """
    Run multi-armed bandits experiment with non-stationary problem.

    :param num_actions: Number of arms bandit has, i.e. number of possible actions to
    be taken.
    :param epsilon: Probability of exploration, i.e. <epsilon>% of the times, the
    algorithm will try different random actions in order to achieve reward improvement.
    :param keep_track: If True, keeps track of all experiment repetiotions and
    generates plots for the average reward and percentage of optimal actions taken over
    time. If False logs data to screen.
    :param max_runs: If keep_track is set to True, experiments will run <max_run> times
    and then will stop.

    :returns steps, reward, optimal_actions: If keep track is True, returns an array of
    all results up to <max_runs>. Otherwise, returns last run's estimates.
    """

    actions_values = np.random.normal(loc=0, scale=1, size=num_actions)
    average_reward = 0
    num_steps_taken = 0
    num_optimal_actions = 0

    if keep_track:
        history_actions = [None]
        history_reward = [average_reward]
        history_steps = [num_steps_taken]
        history_optimal_actions = [num_optimal_actions]

    while True:
        try:
            action_id, action_reward, is_greedy = _choose_one_action(
                actions_values, epsilon
            )

            num_steps_taken += 1

            average_reward = incremental_average(
                average_reward, num_steps_taken, action_reward
            )

            num_optimal_actions += is_greedy

            if not keep_track:
                _log_experiment(average_reward, num_steps_taken, num_optimal_actions)

        except KeyboardInterrupt:
            break
        finally:
            actions_values = _add_white_noise(actions_values)
            if keep_track:
                history_actions.append(action_id)
                history_steps.append(num_steps_taken)
                history_reward.append(average_reward)
                history_optimal_actions.append(num_optimal_actions / num_steps_taken)

                if num_steps_taken == max_runs:
                    break

    if keep_track:
        history_actions = np.array(history_actions)
        history_steps = np.array(history_steps)
        history_reward = np.array(history_reward)
        history_optimal_actions = np.array(history_optimal_actions)

        plot_evolution(
            history_steps, history_reward, history_optimal_actions, epsilon, num_actions
        )
        plot_reward_distribution(history_actions, history_reward, epsilon, num_actions)
        return history_steps, history_reward, history_optimal_actions
    else:
        return num_steps_taken, average_reward, num_optimal_actions


if __name__ == "__main__":
    os.system("clear")
    fire.Fire(run_experiment)
