import torch
import torch.nn as nn
import torch.nn.functional as F


def prediction_coherence_loss(p_clean, p_deg):
    return torch.mean((p_clean - p_deg) ** 2)


def ftsc_loss(s_clean, s_deg):
    cos = (s_clean * s_deg).sum(dim=-1)
    return torch.mean(1.0 - cos)


def type_gating_weights(types, type_alpha, p_clean):
    alpha_map = type_alpha
    default = 1.0
    weights = []

    for type_name in types:
        weights.append(alpha_map.get(type_name, default))

    alpha = torch.tensor(weights, device=p_clean.device, dtype=p_clean.dtype)
    confidence = (p_clean.detach() - 0.5).abs() * 2.0
    return alpha * confidence


def ft_rcl_loss(
    logits_clean,
    logits_deg,
    labels,
    s_clean,
    s_deg,
    types,
    severity_eta,
    type_alpha,
    beta=0.5,
    gamma_pc=1.0,
    gamma_ftsc=0.5,
    criterion=None,
):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    p_clean = F.softmax(logits_clean, dim=1)[:, 1]
    p_deg = F.softmax(logits_deg, dim=1)[:, 1]

    loss_cls_clean = criterion(logits_clean, labels)
    loss_cls_deg = criterion(logits_deg, labels)

    l_pc = prediction_coherence_loss(p_clean, p_deg)
    l_ftsc = ftsc_loss(s_clean, s_deg)

    w_rob = type_gating_weights(types, type_alpha, p_clean) * severity_eta
    w_rob = w_rob.mean()

    robust = gamma_pc * l_pc + gamma_ftsc * l_ftsc
    total = loss_cls_clean + beta * loss_cls_deg + w_rob * robust

    parts = {
        "loss_cls_clean": loss_cls_clean.detach(),
        "loss_cls_deg": loss_cls_deg.detach(),
        "l_pc": l_pc.detach(),
        "l_ftsc": l_ftsc.detach(),
        "w_rob": w_rob.detach(),
    }

    return total, parts
