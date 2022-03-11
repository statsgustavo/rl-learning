from collections import abc

import hypothesis as hp
import numpy as np
import pytest
from hypothesis import strategies as st
from multi_armed_bandits import testbed


@hp.given(st.integers(1, 10))
def test_action_reward_shape(n):
    action_values = np.random.normal(loc=0, scale=1, size=n)
    reward = testbed._bandit(action_values)
    assert isinstance(reward, np.ndarray) and (reward.size == n)


@hp.given(st.integers(1, 10))
def test_add_white_noise(n):
    action_values = np.random.normal(loc=0, scale=1, size=n)
    assert testbed._add_white_noise(action_values, 0.01).size == n


def test_optimal_action_selection():
    actions_values = np.array([1, 1, 2, 3, 5, 8, 11, 13])
    actions_rewards = np.array([0.1, 0.1, 0.2, 10.3, 3.5, 0.8, 1.1, 1.3])
    (
        optimal_action,
        optimal_reward,
    ) = testbed._select_optimal_action_and_break_ties_randomly(
        actions_values, actions_rewards
    )

    assert optimal_action == 3
    assert optimal_reward == 10.3


def test_optimal_action_selection_tie_removal():
    actions_values = np.array([1, 1, 2, 3, 5, 8, 11, 13])
    actions_rewards = np.array([0.1, 0.1, 0.2, 10.3, 3.5, 0.8, 1.1, 10.3])
    (
        optimal_action,
        optimal_reward,
    ) = testbed._select_optimal_action_and_break_ties_randomly(
        actions_values, actions_rewards
    )

    assert not isinstance(optimal_action, abc.Iterable)
    assert not isinstance(optimal_reward, abc.Iterable)


@hp.given(n=st.integers(1, 10), epsilon=st.floats(0.01, 0.05))
def test_choose_one_action(n, epsilon):
    actions_values = np.random.normal(loc=0, scale=1, size=n)
    proportion_of_greedy_actions = 0
    num_steps = 1000
    for i in range(num_steps):
        (
            chosen_action_value,
            chosen_action_reward,
            is_greedy,
        ) = testbed._choose_one_action(actions_values, epsilon)
        proportion_of_greedy_actions += int(is_greedy)

    assert not isinstance(chosen_action_value, abc.Iterable)
    assert not isinstance(chosen_action_reward, abc.Iterable)
    assert (proportion_of_greedy_actions / num_steps) == pytest.approx(
        1 - epsilon, rel=1e-01
    )
