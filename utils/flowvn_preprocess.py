import torch


def finalize_flowvn_deferred_adjoint(kdata_unnorm, coil_sens, mask, fe_ifft=False):
    """Finalize deferred FlowVN test preprocessing on the active torch device."""
    if kdata_unnorm.ndim != 7:
        raise RuntimeError(f"Expected kdata shape (B,Nv,Nc,Nt,FE,PE,SPE), got {tuple(kdata_unnorm.shape)}")
    if coil_sens.ndim != 5:
        raise RuntimeError(f"Expected coil_sens shape (B,Nc,FE,PE,SPE), got {tuple(coil_sens.shape)}")
    if mask.ndim != 7:
        raise RuntimeError(f"Expected mask shape (B,1,1,Nt,1,PE,SPE), got {tuple(mask.shape)}")

    if fe_ifft:
        kdata_unnorm = torch.fft.ifftshift(kdata_unnorm, dim=-3)
        kdata_unnorm = torch.fft.ifft(kdata_unnorm, dim=-3, norm="ortho")
        kdata_unnorm = torch.fft.fftshift(kdata_unnorm, dim=-3)

    masked = kdata_unnorm * mask
    finv = torch.fft.ifftshift(masked, dim=(-2, -1))
    finv = torch.fft.ifft2(finv, dim=(-2, -1), norm="ortho")
    finv = torch.fft.fftshift(finv, dim=(-2, -1))

    sens = coil_sens.unsqueeze(1).unsqueeze(3)
    imdata = torch.sum(finv * torch.conj(sens), dim=2)

    batch = int(kdata_unnorm.shape[0])
    denom = torch.linalg.vector_norm((kdata_unnorm != 0).to(torch.float32).reshape(batch, -1), dim=1)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    norm = torch.linalg.vector_norm(kdata_unnorm.reshape(batch, -1), dim=1) / denom

    imdata = imdata / norm.view(batch, 1, 1, 1, 1, 1)
    kdata = kdata_unnorm / norm.view(batch, 1, 1, 1, 1, 1, 1)
    return imdata, kdata, norm
