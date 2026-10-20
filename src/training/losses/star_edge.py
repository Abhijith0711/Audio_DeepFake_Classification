import torch
import torch.nn.functional as F


def stc_loss(signatures, probs):
    if len(signatures) < 2:
        zero = signatures[0].new_tensor(0.0)
        return zero

    sig_loss = 0.0
    prob_loss = 0.0

    for t in range(len(signatures) - 1):
        cos = (signatures[t] * signatures[t + 1]).sum(dim=-1)
        sig_loss = sig_loss + torch.mean(1.0 - cos)
        prob_loss = prob_loss + torch.mean((probs[t] - probs[t + 1]) ** 2)

    return sig_loss + prob_loss


def abr_loss(p_clean, p_deg, labels, margin=0.05):
    real_mask = labels == 0

    if real_mask.sum() == 0:
        return p_clean.new_tensor(0.0)

    p_c = p_clean[real_mask]
    p_d = p_deg[real_mask]
    return torch.mean(F.relu(p_d - p_c + margin))


def channel_aux_loss(channel_logits, channel_targets, criterion):
    if channel_logits is None or channel_targets is None:
        return channel_logits.new_tensor(0.0) if channel_logits is not None else torch.tensor(0.0)

    return criterion(channel_logits, channel_targets)
