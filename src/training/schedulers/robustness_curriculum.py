def max_severity_for_epoch(epoch, epoch_milestones, severity_milestones):
    max_sev = severity_milestones[0]

    for boundary, sev in zip(epoch_milestones, severity_milestones):
        if epoch >= boundary:
            max_sev = sev

    return max_sev


def severity_eta(severity, max_severity):
    if max_severity <= 0:
        return 0.0

    return float(min(1.0, max(0.0, severity / max_severity)))


def early_stop_floor(epoch_milestones):
    """Epoch after the hardest curriculum severity has started.

    With milestones [1, 6, 13] and severities [1, 3, 4], severity 4 starts at
    epoch 13, so early stopping is first allowed at epoch 14.
    """
    if not epoch_milestones:
        return 1
    return int(epoch_milestones[-1]) + 1


def should_early_stop(epoch, epochs_without_improvement, patience, min_epoch):
    """Patience can end a run only once `min_epoch` has been reached.

    For M3, `min_epoch` is the epoch after severity 4 starts, so the checkpoint
    has trained for one epoch on the full curriculum before stopping is allowed.
    """
    return int(epoch) >= int(min_epoch) and int(epochs_without_improvement) >= int(patience)
