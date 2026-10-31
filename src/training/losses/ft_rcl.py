import torch
import torch.nn as nn
import torch.nn.functional as F

_JS_EPS = 1e-6


def _as_distribution(p, eps=_JS_EPS):
    """Two-class probabilities, clamped and renormalized, in FP32."""
    p = p.float()
    if p.ndim == 1:
        p = p.clamp(eps, 1.0 - eps)
        p = torch.stack((1.0 - p, p), dim=-1)
    else:
        p = p.clamp(min=eps)
        p = p / p.sum(dim=-1, keepdim=True)
    return p


def jensen_shannon_divergence(p, q, eps=_JS_EPS):
    """Mean Jensen-Shannon divergence between two-class distributions, in FP32.

    Probabilities are clamped away from zero before the logarithm. The per-example
    value is clamped at zero so a tiny negative from rounding cannot enter the loss.
    """
    p = _as_distribution(p, eps)
    q = _as_distribution(q, eps)
    m = (0.5 * (p + q)).clamp(min=eps)
    m = m / m.sum(dim=-1, keepdim=True)
    kl_pm = torch.sum(p * (p.log() - m.log()), dim=-1)
    kl_qm = torch.sum(q * (q.log() - m.log()), dim=-1)
    return (0.5 * (kl_pm + kl_qm)).clamp(min=0.0).mean()


def prediction_coherence_loss(p_clean, p_deg, pc_loss="mse"):
    """Coherence between the clean and degraded predictions.

    ``mse`` is the mean squared difference of P(fake). ``js`` is the
    Jensen-Shannon divergence of the two-class distributions.
    """
    if pc_loss == "mse":
        p_clean = p_clean.float()
        p_deg = p_deg.float()
        if p_clean.ndim == 2:
            p_clean = p_clean[:, 1]
            p_deg = p_deg[:, 1]
        return torch.mean((p_clean - p_deg) ** 2)

    if pc_loss == "js":
        return jensen_shannon_divergence(p_clean, p_deg)

    raise ValueError(f"Unknown pc_loss: {pc_loss}")


def ftsc_loss(s_clean, s_deg):
    cos = (s_clean * s_deg).sum(dim=-1)
    return torch.mean(1.0 - cos)


def type_gating_weights(types, type_alpha, p_clean, gate="full", uniform_type_weights=False):
    """Per-example type and confidence factors. Severity is applied by the caller.

    ``full`` uses both factors. ``no_confidence`` keeps type weights only.
    ``no_type`` keeps confidence only. ``uniform_type_weights`` sets every type
    weight to 1.
    """
    if gate not in ("full", "no_confidence", "no_type"):
        raise ValueError(f"type_gating_weights does not apply to gate {gate!r}")

    if uniform_type_weights or gate == "no_type":
        alpha = torch.ones(len(types), device=p_clean.device, dtype=torch.float32)
    else:
        weights = [float(type_alpha.get(type_name, 1.0)) for type_name in types]
        alpha = torch.tensor(weights, device=p_clean.device, dtype=torch.float32)

    if gate == "no_confidence":
        confidence = torch.ones_like(alpha)
    else:
        confidence = (p_clean.detach().float() - 0.5).abs() * 2.0

    return alpha * confidence


def robustness_weight(
    types,
    type_alpha,
    p_clean,
    severity_eta,
    gate="full",
    uniform_type_weights=False,
):
    """Scalar weight on the robustness terms.

    ``full``: mean(type weight × clean-view confidence × severity).
    ``no_confidence``: mean(type weight × severity).
    ``no_type``: mean(confidence × severity).
    ``none``: severity only.
    """
    if gate == "none":
        return p_clean.new_tensor(float(severity_eta))

    if gate not in ("full", "no_confidence", "no_type"):
        raise ValueError(f"Unknown gate mode: {gate}")

    per_example = type_gating_weights(
        types,
        type_alpha,
        p_clean,
        gate=gate,
        uniform_type_weights=uniform_type_weights,
    )
    return (per_example * float(severity_eta)).mean()


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
    gate="full",
    pc_loss="mse",
    uniform_type_weights=False,
):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Loss terms are always computed in FP32, even if the model ran under autocast.
    logits_clean = logits_clean.float()
    logits_deg = logits_deg.float()
    s_clean = s_clean.float()
    s_deg = s_deg.float()

    p_clean_full = F.softmax(logits_clean, dim=1)
    p_deg_full = F.softmax(logits_deg, dim=1)
    p_clean = p_clean_full[:, 1]
    p_deg = p_deg_full[:, 1]

    loss_cls_clean = criterion(logits_clean, labels)
    loss_cls_deg = criterion(logits_deg, labels)

    l_pc = prediction_coherence_loss(p_clean_full, p_deg_full, pc_loss=pc_loss)
    l_ftsc = ftsc_loss(s_clean, s_deg)
    w_rob = robustness_weight(
        types,
        type_alpha,
        p_clean,
        severity_eta,
        gate=gate,
        uniform_type_weights=uniform_type_weights,
    )

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
