"""Exercise the installed native extension in a fresh Python process."""

from engram import Runtime


def main() -> None:
    runtime = Runtime(input_dims=2, num_actions=2, seed=7)
    initial_state = runtime.learning_state_hash

    for _ in range(20):
        runtime.step([0.9, 0.1])
    runtime.reward(1.0)
    runtime.step([0.1, 0.9])
    assert runtime.total_reward == 1.0
    assert runtime.learning_state_hash != initial_state

    runtime.reward(2.0)
    runtime.end_episode()
    assert runtime.total_reward == 3.0

    learned_state = runtime.learning_state_hash
    evaluation = runtime.evaluation_copy()
    evaluation_state = evaluation.learning_state_hash
    assert not evaluation.learning_enabled
    for _ in range(10):
        evaluation.step([0.8, 0.2])
    evaluation.reward(-4.0)
    evaluation.end_episode()
    assert evaluation.learning_state_hash == evaluation_state
    assert runtime.learning_state_hash == learned_state

    runtime.reset()
    assert runtime.learning_state_hash == initial_state
    assert runtime.total_reward == 0.0
    assert runtime.tick_count == 0


if __name__ == "__main__":
    main()
