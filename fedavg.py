import torch


def fedavg_round(global_model, client_models, sample_counts):
    if not client_models or len(client_models) != len(sample_counts):
        raise ValueError("Provide one sample count for each client model")
    if any(count <= 0 for count in sample_counts):
        raise ValueError("Sample counts must be positive")

    total = sum(sample_counts)
    weights = [count / total for count in sample_counts]
    states = [model.state_dict() for model in client_models]
    averaged = {}

    for key, value in states[0].items():
        if torch.is_floating_point(value):
            averaged[key] = sum(
                state[key] * weight for state, weight in zip(states, weights)
            )
        else:
            averaged[key] = value

    global_model.load_state_dict(averaged)
    for model in client_models:
        model.load_state_dict(averaged)
    return global_model, client_models
