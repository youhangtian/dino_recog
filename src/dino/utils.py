import numpy as np 
import cv2 


def make_square(img_bgr):
    width, height = img_bgr.shape[1], img_bgr.shape[0]
    if width > height:
        padding_size= width - height 
        padding = (padding_size // 2, padding_size // 2, 0, 0)
    else:
        padding_size = height - width 
        padding = (0, 0, padding_size // 2, padding_size // 2)

    padding_img = cv2.copyMakeBorder(img_bgr, *padding, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padding_img 


def rotate_img(img_bgr, rotate_degree):
    width, height = img_bgr.shape[1], img_bgr.shape[0]
    center = (width / 2, height / 2)
    degree = np.random.rand() * rotate_degree * 2 - rotate_degree 
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    rotated_img = cv2.warpAffine(img_bgr, M, (width, height), borderValue=(114, 114, 114))
    return rotated_img
        
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

