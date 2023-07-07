import numpy as np 

        
def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep 
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(final_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep 
    return schedule


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 

        if name.endswith('.bias') or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

