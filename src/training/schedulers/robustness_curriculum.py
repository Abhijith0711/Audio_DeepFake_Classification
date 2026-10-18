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
