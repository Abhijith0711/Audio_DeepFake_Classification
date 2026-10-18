import torch


def project_artifact(z, u, detach_u=False):
    if detach_u:
        u = u.detach()

    u_norm = torch.nn.functional.normalize(u, p=2, dim=-1)
    coeff = (z * u_norm).sum(dim=-1, keepdim=True)
    return z - coeff * u_norm
