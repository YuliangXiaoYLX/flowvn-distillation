"""
Adopted from
https://github.com/rixez/pytorch_mri_variationalnetwork
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from utils.misc_utils import *

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _triton_available():
    return triton is not None and tl is not None


if _triton_available():

    @triton.jit
    def _adaptive_interp_forward_kernel(
        x_ptr,
        yk_ptr,
        maxx_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        n_flt: tl.constexpr,
        n_knots: tl.constexpr,
        output_residual: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        maxx = tl.load(maxx_ptr)
        minx = -maxx
        width = (maxx - minx) / (n_knots - 1)

        x_scaled = (x - minx) / width
        x_scaled = tl.minimum(tl.maximum(x_scaled, 0.00001), n_knots - 1.00001)
        idx_f_f = tl.floor(x_scaled)
        frac = x_scaled - idx_f_f
        idx_f = idx_f_f.to(tl.int64)
        idx_f = tl.minimum(tl.maximum(idx_f, 0), n_knots - 2)
        idx_c = idx_f + 1
        col = offsets % n_flt

        y_f = tl.load(yk_ptr + idx_f * n_flt + col, mask=mask, other=0.0)
        y_c = tl.load(yk_ptr + idx_c * n_flt + col, mask=mask, other=0.0)
        y = y_f * (1.0 - frac) + y_c * frac
        if output_residual:
            y += x
        tl.store(out_ptr + offsets, y, mask=mask)


    @triton.jit
    def _adaptive_interp_backward_kernel(
        grad_out_ptr,
        x_ptr,
        yk_ptr,
        maxx_ptr,
        grad_x_ptr,
        grad_yk_ptr,
        n_elements: tl.constexpr,
        n_flt: tl.constexpr,
        n_knots: tl.constexpr,
        output_residual: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        maxx = tl.load(maxx_ptr)
        minx = -maxx
        width = (maxx - minx) / (n_knots - 1)

        x_unclamped = (x - minx) / width
        inside = (x_unclamped >= 0.00001) & (x_unclamped <= n_knots - 1.00001)
        x_scaled = tl.minimum(tl.maximum(x_unclamped, 0.00001), n_knots - 1.00001)
        idx_f_f = tl.floor(x_scaled)
        frac = x_scaled - idx_f_f
        idx_f = idx_f_f.to(tl.int64)
        idx_f = tl.minimum(tl.maximum(idx_f, 0), n_knots - 2)
        idx_c = idx_f + 1
        col = offsets % n_flt

        y_f = tl.load(yk_ptr + idx_f * n_flt + col, mask=mask, other=0.0)
        y_c = tl.load(yk_ptr + idx_c * n_flt + col, mask=mask, other=0.0)
        grad_x = ((y_c - y_f) / width) * grad_out
        grad_x = tl.where(inside, grad_x, 0.0)
        if output_residual:
            grad_x += grad_out
        tl.store(grad_x_ptr + offsets, grad_x, mask=mask)

        tl.atomic_add(
            grad_yk_ptr + idx_f * n_flt + col,
            grad_out * (1.0 - frac),
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(
            grad_yk_ptr + idx_c * n_flt + col,
            grad_out * frac,
            sem="relaxed",
            mask=mask,
        )

else:
    _adaptive_interp_forward_kernel = None
    _adaptive_interp_backward_kernel = None


class AdaptiveInterpolatorTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xin, yk, flt_response_var, n_flt, n_interp_knots, output_residual):
        if not _triton_available():
            raise RuntimeError("flowvn_activation_backend='triton' requires Triton")
        if not xin.is_cuda:
            raise RuntimeError("flowvn_activation_backend='triton' requires CUDA tensors")
        if xin.dtype != torch.float32 or yk.dtype != torch.float32:
            raise RuntimeError("flowvn_activation_backend='triton' currently supports float32 only")

        x = xin.contiguous()
        yk_contig = yk.contiguous()
        out = torch.empty_like(x)
        n_elements = x.numel()
        block = 256
        grid = (triton.cdiv(n_elements, block),)
        _adaptive_interp_forward_kernel[grid](
            x,
            yk_contig,
            flt_response_var,
            out,
            n_elements,
            int(n_flt),
            int(n_interp_knots),
            bool(output_residual),
            BLOCK=block,
        )
        ctx.save_for_backward(x, yk_contig, flt_response_var)
        ctx.n_flt = int(n_flt)
        ctx.n_interp_knots = int(n_interp_knots)
        ctx.output_residual = bool(output_residual)
        ctx.input_shape = tuple(xin.shape)
        return out.reshape_as(xin)

    @staticmethod
    def backward(ctx, grad_output):
        x, yk, flt_response_var = ctx.saved_tensors
        grad_out = grad_output.contiguous()
        grad_x = torch.empty_like(x)
        grad_yk = torch.zeros_like(yk)
        n_elements = x.numel()
        block = 256
        grid = (triton.cdiv(n_elements, block),)
        _adaptive_interp_backward_kernel[grid](
            grad_out,
            x,
            yk,
            flt_response_var,
            grad_x,
            grad_yk,
            n_elements,
            int(ctx.n_flt),
            int(ctx.n_interp_knots),
            bool(ctx.output_residual),
            BLOCK=block,
        )
        return grad_x.reshape(ctx.input_shape), grad_yk, None, None, None, None


def zero_mean_norm_ball(
    x, zero_mean=True, normalize=True, norm_bound=1.0, mask=None, axis=(0, ...)
):
    """https://github.com/VLOGroup/mri-variationalnetwork/blob/master/vn/proxmaps.py"""
    if mask is None:
        shape = []
        for i in range(len(x.shape)):
            if i in axis:
                shape.append(x.shape[i])
            else:
                shape.append(1)
        mask = torch.ones(shape, dtype=torch.float32, device=x.device)

    x_masked = x * mask

    if zero_mean:
        x_mean = torch.mean(x_masked, dim=axis, keepdim=True)
        x_zm = x_masked - x_mean
    else:
        x_zm = x_masked

    if normalize:
        magnitude = torch.sqrt(torch.sum(torch.square(x_zm), dim=axis, keepdim=True))
        x_proj = x_zm / magnitude * norm_bound
        return x_proj

    return x_zm


class RBFActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, mu, sigma):
        """
        input: input tensor ((N*D)xCxTxHxW)
        w: weight of the RBF kernels (1 x C x 1 x 1 x 1 x # of RBF kernels)
        mu: center of the RBF (# of RBF kernels)
        sigma: std of the RBF (1)
        """
        output = input.new_zeros(input.shape)
        rbf_grad_input = input.new_zeros(input.shape)
        for i in range(w.shape[-1]):
            tmp = w[:, :, :, :, :, i] * torch.exp(
                -torch.square(input - mu[i]) / (2 * sigma**2)
            )
            output += tmp
            rbf_grad_input += tmp * (-(input - mu[i])) / (sigma**2)
        del tmp

        ctx.save_for_backward(input, w, mu, sigma, rbf_grad_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w, mu, sigma, rbf_grad_input = ctx.saved_tensors

        grad_input = grad_output * rbf_grad_input

        grad_w = w.new_zeros(w.shape)
        for i in range(w.shape[-1]):
            tmp = (
                grad_output * torch.exp(-torch.square(input - mu[i]) / (2 * sigma**2))
            ).sum((0, 2, 3, 4))
            grad_w[:, :, :, :, :, i] = tmp.view(w.shape[0:-1])

        return grad_input, grad_w, None, None


class RBFActivation(nn.Module):
    def __init__(self, feat, **kwargs):
        super().__init__()
        self.options = kwargs

        x_0 = np.linspace(
            kwargs["vmin"], kwargs["vmax"], kwargs["num_act_weights"], dtype=np.float32
        )
        mu = np.linspace(
            kwargs["vmin"], kwargs["vmax"], kwargs["num_act_weights"], dtype=np.float32
        )

        sigma = 2 * kwargs["vmax"] / (kwargs["num_act_weights"] - 1)

        w_0 = kwargs["weight"] * x_0
        w_0 = np.reshape(w_0, (1, 1, 1, 1, 1, kwargs["num_act_weights"]))
        w_0 = np.repeat(w_0, feat, 1)

        self.w = torch.nn.Parameter(torch.from_numpy(w_0))

        self.register_buffer("mu", torch.from_numpy(mu))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

        self.rbf_act = RBFActivationFunction.apply

    def forward(self, x):
        return self.rbf_act(x, self.w, self.mu, self.sigma)


class LinearActivationFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, zero_knot_indexes, size):
        x_clamped = x.clamp(
            min=-(grid.item() * (size // 2)), max=(grid.item() * (size // 2 - 1))
        )

        floored_x = torch.floor(x_clamped / grid)
        fracs = x_clamped / grid - floored_x

        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()
        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)

        out = coefficients_vect[indexes + 1] * fracs + coefficients_vect[indexes] * (
            1 - fracs
        )
        out[x < -(grid.item() * (size // 2))] = x[x < -(grid.item() * (size // 2))]
        out[x > (grid.item() * (size // 2 - 1))] = x[
            x > (grid.item() * (size // 2 - 1))
        ]
        return out

    @staticmethod
    def backward(ctx, grad_out):
        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = (
            (coefficients_vect[indexes + 1] - coefficients_vect[indexes])
            / grid
            * grad_out
        )

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1) + 1, (fracs * grad_out).view(-1)
        )
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1), ((1 - fracs) * grad_out).view(-1)
        )
        return grad_x, grad_coefficients_vect, None, None, None


class LinearActivation(torch.nn.Module):
    def __init__(self, num_activations, **kwargs):
        super().__init__()
        self.options = kwargs

        self.register_buffer(
            "grid", torch.tensor([self.options["grid"]], dtype=torch.float32)
        )

        grid_tensor = (
            torch.arange(
                -(self.options["num_act_weights"] // 2),
                (self.options["num_act_weights"] // 2) + 1,
                dtype=torch.float32,
            )
            .mul(self.grid)
            .expand((num_activations, self.options["num_act_weights"]))
        )
        grid_tensor = grid_tensor * 0.01
        self.coefficients_vect = torch.nn.Parameter(grid_tensor.contiguous().view(-1))

        self.register_buffer(
            "zero_knot_indexes",
            (
                torch.arange(0, num_activations, dtype=torch.long)
                * self.options["num_act_weights"]
                + (self.options["num_act_weights"] // 2)
            ),
        )

    def forward(self, input):
        return LinearActivationFunc.apply(
            input,
            self.coefficients_vect,
            self.grid,
            self.zero_knot_indexes,
            self.options["num_act_weights"],
        )


class AdaptiveInterpolatorTorchTF(nn.Module):
    def __init__(
        self,
        n_flt: int,
        maxx: float = 3.0,
        n_interp_knots: int = 91,
        stddev_init: float = 0.05,
        init_yk=None,
        lowmem: bool = True,
        output_residual: bool = False,
        backend: str = "torch",
    ):
        super().__init__()
        self.n_flt = int(n_flt)
        self.n_interp_knots = int(n_interp_knots)
        self.lowmem = bool(lowmem)
        self.output_residual = bool(output_residual)
        self.backend = str(backend)
        if self.backend not in ("torch", "triton"):
            raise ValueError("flowvn_activation_backend must be 'torch' or 'triton'")

        self.register_buffer(
            "flt_response_var", torch.tensor(float(maxx), dtype=torch.float32)
        )
        self.register_buffer(
            "flt_response_max", torch.tensor(float(maxx), dtype=torch.float32)
        )
        self.register_buffer(
            "init_maxx", torch.tensor(float(maxx), dtype=torch.float32)
        )

        if init_yk is None:
            yk_init = torch.empty(self.n_interp_knots, self.n_flt, dtype=torch.float32)
            nn.init.trunc_normal_(
                yk_init,
                mean=0.0,
                std=float(stddev_init),
                a=-2.0 * float(stddev_init),
                b=2.0 * float(stddev_init),
            )
            self.yk = nn.Parameter(yk_init.contiguous())
        else:
            init_yk = torch.as_tensor(init_yk, dtype=torch.float32)
            if tuple(init_yk.shape) != (self.n_interp_knots, self.n_flt):
                raise ValueError(
                    f"Init values have incorrect shape: got {tuple(init_yk.shape)}, expected {(self.n_interp_knots, self.n_flt)}"
                )
            self.yk = nn.Parameter(init_yk.clone().contiguous())

    @property
    def minx(self):
        return -self.flt_response_var

    @property
    def maxx(self):
        return self.flt_response_var

    @property
    def w(self):
        return (self.maxx - self.minx) / (self.n_interp_knots - 1)

    @torch.no_grad()
    def register_input(self, xin: torch.Tensor):
        cur_max = xin.abs().max()
        self.flt_response_max.mul_(0.95).add_(cur_max * 0.05)
        return self.flt_response_max

    def forward(self, xin: torch.Tensor):
        if self.backend == "triton":
            return AdaptiveInterpolatorTritonFunction.apply(
                xin,
                self.yk,
                self.flt_response_var,
                self.n_flt,
                self.n_interp_knots,
                self.output_residual,
            )

        shp_in = xin.shape
        x = xin.reshape(-1, self.n_flt)

        xS = (x - self.minx) / self.w
        xS = torch.clamp(xS, 0.00001, self.n_interp_knots - 1.00001)

        xF = torch.floor(xS)
        k = xS - xF
        idx_f = xF.to(torch.long)
        idx_c = idx_f + 1
        idx_f = idx_f.clamp(0, self.n_interp_knots - 2)
        idx_c = idx_f + 1
        if self.lowmem:
            y_all = []
            for i in range(self.n_flt):
                yK_i = self.yk[:, i]
                nd_idx1 = idx_f[:, i]
                nd_idx2 = idx_c[:, i]
                k_ = k[:, i]
                y_f_ = yK_i.gather(0, nd_idx1)
                y_c_ = yK_i.gather(0, nd_idx2)
                y_all.append(y_f_ * (1 - k_) + k_ * y_c_)
            y = torch.stack(y_all, dim=1)
        else:
            yk_T = self.yk.transpose(0, 1)
            yf = torch.gather(yk_T, 1, idx_f.transpose(0, 1)).transpose(0, 1)
            yc = torch.gather(yk_T, 1, idx_c.transpose(0, 1)).transpose(0, 1)
            y = yf * (1 - k) + yc * k

        y = y.reshape(shp_in)
        return (xin + y) if self.output_residual else y

    @torch.no_grad()
    def readjust_response_range(self):
        rng = float(self.flt_response_var.item())
        rng_new = float(self.flt_response_max.item())

        if (rng < rng_new * 1.45) and (rng > rng_new * 0.95):
            return False

        rng_new = max(rng_new, 1e-6)
        rng_new = rng * 0.3 + rng_new * 0.7

        device = self.yk.device
        dtype = self.yk.dtype

        x_old = torch.linspace(
            -rng, +rng, self.n_interp_knots, device=device, dtype=dtype
        )
        x_new = torch.linspace(
            -rng_new, +rng_new, self.n_interp_knots, device=device, dtype=dtype
        )

        idx = torch.searchsorted(x_old, x_new, right=False)
        idx = idx.clamp(1, self.n_interp_knots - 1)

        x0 = x_old[idx - 1]
        x1 = x_old[idx]
        t = (x_new - x0) / (x1 - x0 + 1e-12)

        y0 = self.yk[idx - 1, :]
        y1 = self.yk[idx, :]
        y_new = y0 * (1 - t.unsqueeze(1)) + y1 * t.unsqueeze(1)

        left_mask = x_new <= x_old[0]
        right_mask = x_new >= x_old[-1]
        if left_mask.any():
            y_new[left_mask, :] = self.yk[0, :].unsqueeze(0)
        if right_mask.any():
            y_new[right_mask, :] = self.yk[-1, :].unsqueeze(0)

        self.yk.copy_(y_new)
        self.flt_response_var.fill_(float(rng_new))
        return True

    def get_knots_variable(self):
        return self.yk

    def get_response_vars(self):
        return self.flt_response_var, self.flt_response_max


class LinearActivationFlowVN(nn.Module):
    def __init__(self, num_activations, **kwargs):
        super().__init__()
        n_interp_knots = kwargs.get("num_act_weights", 91)
        activation_range = kwargs.get("vmax", 3.0)
        stddev_init = kwargs.get("stddev_init", 0.05)
        lowmem = kwargs.get("flowvn_activation_lowmem", True)
        backend = kwargs.get("flowvn_activation_backend", "torch")

        self.interpolator = AdaptiveInterpolatorTorchTF(
            n_flt=num_activations,
            maxx=activation_range,
            n_interp_knots=n_interp_knots,
            stddev_init=stddev_init,
            lowmem=lowmem,
            backend=backend,
        )

    def forward(self, x):
        return self.interpolator(x)


class LinearActivationFlowVN_DC(nn.Module):
    def __init__(self, num_activations, **kwargs):
        super().__init__()
        n_interp_knots = kwargs.get("num_act_weights", 91)
        data_activation_range = kwargs.get("vmax_dc", 7.0)
        stddev_init = kwargs.get("stddev_init", 0.05)
        lowmem = kwargs.get("flowvn_activation_lowmem", True)
        backend = kwargs.get("flowvn_activation_backend", "torch")

        self.interpolator = AdaptiveInterpolatorTorchTF(
            n_flt=num_activations,
            maxx=data_activation_range,
            n_interp_knots=n_interp_knots,
            stddev_init=stddev_init,
            lowmem=lowmem,
            backend=backend,
        )

    def forward(self, x):
        return self.interpolator(x)


class USRateModulation(nn.Module):
    def __init__(
        self,
        n_outputs=1,
        minx=0.0,
        maxx=1.0,
        n_interp_knots=11,
        stddev_init=0.1,
        init_yk=None,
    ):
        super().__init__()
        self.n_interp_knots = int(n_interp_knots)
        self.n_outputs = int(n_outputs)
        self.minx = float(minx)
        self.maxx = float(maxx)
        self.w = (self.maxx - self.minx) / (self.n_interp_knots - 1)

        if init_yk is None:
            init_values = torch.ones(self.n_interp_knots, self.n_outputs) + torch.randn(
                self.n_interp_knots, self.n_outputs
            ) * float(stddev_init)
            self.yK = nn.Parameter(init_values.contiguous())
        else:
            init_yk = torch.as_tensor(init_yk, dtype=torch.float32)
            if tuple(init_yk.shape) != (self.n_interp_knots, self.n_outputs):
                raise AssertionError("Init values have incorrect shape")
            self.yk = nn.Parameter(init_yk.contiguous())

    def forward(self, usrate):
        scalar_input = False
        if usrate.dim() == 0:
            usrate = usrate.unsqueeze(0).unsqueeze(1)
            scalar_input = True
        elif usrate.dim() == 1:
            usrate = usrate.unsqueeze(1)

        batch_size = usrate.shape[0]
        xin = usrate.expand(-1, self.n_outputs)

        xS = (xin - self.minx) / self.w
        xS = torch.clamp(xS, 0.00001, self.n_interp_knots - 1.00001)

        xF = torch.floor(xS)
        k = xS - xF

        idx_f = xF.to(torch.long)
        idx_c = idx_f + 1
        idx_f = idx_f.clamp(0, self.n_interp_knots - 2)
        idx_c = idx_f + 1
        yf = self.yK.gather(0, idx_f)
        yc = self.yK.gather(0, idx_c)
        y = yf * (1 - k) + yc * k
        y = F.softplus(y)

        if scalar_input:
            y = y.squeeze(0)

        return y


class VnMriReconCell(nn.Module):
    def __init__(self, block_id, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options

        conv_kernel_xyz = torch.normal(
            mean=0,
            std=1,
            size=(
                options["features_out"],
                options["features_in"],
                options["kernel_size"],
                options["kernel_size"],
                options["kernel_size"],
            ),
        )
        conv_kernel_xyz /= np.sqrt(options["kernel_size"] ** 2 * options["features_in"])
        conv_kernel_xyz -= torch.mean(conv_kernel_xyz, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel_xyz = zero_mean_norm_ball(conv_kernel_xyz, axis=(1, 2, 3, 4))
        self.conv_kernel_xyz = torch.nn.Parameter(conv_kernel_xyz)

        conv_kernel_xyt = torch.normal(
            mean=0,
            std=1,
            size=(
                options["features_out"],
                options["features_in"],
                options["kernel_size"],
                options["kernel_size"],
                options["kernel_size"],
            ),
        )
        conv_kernel_xyt /= np.sqrt(options["kernel_size"] ** 2 * options["features_in"])
        conv_kernel_xyt -= torch.mean(conv_kernel_xyt, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel_xyt = zero_mean_norm_ball(conv_kernel_xyt, axis=(1, 2, 3, 4))
        self.conv_kernel_xyt = torch.nn.Parameter(conv_kernel_xyt)

        conv_kernel_yzt = torch.normal(
            mean=0,
            std=1,
            size=(
                options["features_out"],
                options["features_in"],
                options["kernel_size"],
                options["kernel_size"],
                options["kernel_size"],
            ),
        )
        conv_kernel_yzt /= np.sqrt(options["kernel_size"] ** 2 * options["features_in"])
        conv_kernel_yzt -= torch.mean(conv_kernel_yzt, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel_yzt = zero_mean_norm_ball(conv_kernel_yzt, axis=(1, 2, 3, 4))
        self.conv_kernel_yzt = torch.nn.Parameter(conv_kernel_yzt)

        conv_kernel_xzt = torch.normal(
            mean=0,
            std=1,
            size=(
                options["features_out"],
                options["features_in"],
                options["kernel_size"],
                options["kernel_size"],
                options["kernel_size"],
            ),
        )
        conv_kernel_xzt /= np.sqrt(options["kernel_size"] ** 2 * options["features_in"])
        conv_kernel_xzt -= torch.mean(conv_kernel_xzt, axis=(1, 2, 3, 4), keepdims=True)
        conv_kernel_xzt = zero_mean_norm_ball(conv_kernel_xzt, axis=(1, 2, 3, 4))
        self.conv_kernel_xzt = torch.nn.Parameter(conv_kernel_xzt)

        if options["act"] == "rbf":
            self.activation1 = RBFActivation(feat=options["features_out"], **options)
            self.activation2 = RBFActivation(feat=options["features_out"], **options)
            self.activation3 = RBFActivation(feat=options["features_out"], **options)
            self.activation4 = RBFActivation(feat=options["features_out"], **options)
            self.activation5 = RBFActivation(feat=options["features_in"], **options)
        elif options["act"] == "linear":
            self.activation1 = LinearActivation(
                num_activations=options["features_out"], **options
            )
            self.activation2 = LinearActivation(
                num_activations=options["features_out"], **options
            )
            self.activation3 = LinearActivation(
                num_activations=options["features_out"], **options
            )
            self.activation4 = LinearActivation(
                num_activations=options["features_out"], **options
            )
            self.activation5 = LinearActivation(
                num_activations=options["features_in"], **options
            )
        elif options["act"] == "linear_flowvn":
            self.activation1 = LinearActivationFlowVN(
                num_activations=options["features_out"],
                n_interp_knots=options.get("n_interp_knots", 91),
                activation_range=options.get("activation_range", 3.0),
                stddev_init=options.get("stddev_init", 0.05),
                **options,
            )
            self.activation2 = LinearActivationFlowVN(
                num_activations=options["features_out"],
                n_interp_knots=options.get("n_interp_knots", 91),
                activation_range=options.get("activation_range", 3.0),
                stddev_init=options.get("stddev_init", 0.05),
                **options,
            )
            self.activation3 = LinearActivationFlowVN(
                num_activations=options["features_out"],
                n_interp_knots=options.get("n_interp_knots", 91),
                activation_range=options.get("activation_range", 3.0),
                stddev_init=options.get("stddev_init", 0.05),
                **options,
            )
            self.activation4 = LinearActivationFlowVN(
                num_activations=options["features_out"],
                n_interp_knots=options.get("n_interp_knots", 91),
                activation_range=options.get("activation_range", 3.0),
                stddev_init=options.get("stddev_init", 0.05),
                **options,
            )
            self.activation5 = LinearActivationFlowVN_DC(
                num_activations=options["features_in"],
                n_interp_knots=options.get("n_interp_knots", 91),
                data_activation_range=options.get("data_activation_range", 7.0),
                stddev_init=options.get("stddev_init", 0.05),
                **options,
            )
        else:
            raise ValueError("act should be 'rbf', 'linear', or 'linear_flowvn'")

        self.use_usrate_modulation = True

        if self.use_usrate_modulation:
            self.lamb_ru_modulation = USRateModulation(
                n_outputs=1,
                minx=9,
                maxx=51,
                n_interp_knots=options["num_act_weights"],
                stddev_init=0.1,
            )
            self.lamb_du_modulation = USRateModulation(
                n_outputs=1,
                minx=9,
                maxx=51,
                n_interp_knots=options["num_act_weights"],
                stddev_init=0.1,
            )
        else:
            self.lamb_ru = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.lamb_du = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.alpha = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.block_id = block_id
        self.pad = int((self.options["kernel_size"] - 1) / 2)
        # Depth padding must follow the actual 3D kernel depth, not D_size.
        # D_size can be -1 (use full depth in dataloader), which would create invalid negative padding.
        self.pad_d = int((self.options["kernel_size"] - 1) / 2)
        self.sgd_momentum = self.options["sgd_momentum"]
        self.flowvn_dc_centered_fft = str(
            self.options.get("flowvn_dc_centered_fft", "original")
        )
        if self.flowvn_dc_centered_fft not in ("original", "checkerboard"):
            raise ValueError(
                "flowvn_dc_centered_fft must be 'original' or 'checkerboard'"
            )
        self._regularizer_forward_impl = self._regularizer_forward_eager

        list = []

    def _regularizer_maybe_channels_last_3d(self, x):
        if self.options.get("flowvn_regularizer_channels_last_3d", False):
            return x.contiguous(memory_format=torch.channels_last_3d)
        return x

    def forward(self, u_t_1, f, c, usrate, S_prev=None, dc_cache=None):
        N, V, T, D, H, W, _ = u_t_1.shape

        Ru = self._regularizer_forward_impl(u_t_1)

        if self.flowvn_dc_centered_fft == "checkerboard":
            if dc_cache is None:
                dc_cache = make_flowvn_dc_checkerboard_cache(f, c)
            Au = mri_forward_op_checkerboard(torch.view_as_complex(u_t_1), dc_cache)
        else:
            Au = mri_forward_op(
                torch.view_as_complex(u_t_1), c, abs(f[:, :, 0, :, 0, :, :]) != 0
            )
        residual = Au - f
        C = residual.shape[2]

        residual = self.activation5(
            residual.real.reshape(N, V, C * T, D, H * W)
        ).reshape(N, V, C, T, D, H, W) + 1j * self.activation5(
            residual.imag.reshape(N, V, C * T, D, H * W)
        ).reshape(N, V, C, T, D, H, W)
        if self.flowvn_dc_centered_fft == "checkerboard":
            Du = torch.view_as_real(mri_adjoint_op_checkerboard(residual, dc_cache))
        else:
            Du = torch.view_as_real(mri_adjoint_op(residual, c))

        if self.use_usrate_modulation:
            if usrate.dim() == 0:
                usrate_input = usrate.unsqueeze(0)
            else:
                usrate_input = usrate

            lamb_ru = self.lamb_ru_modulation(usrate_input)
            lamb_du = self.lamb_du_modulation(usrate_input)

            if lamb_ru.dim() == 1:
                lamb_ru = lamb_ru.view(-1, 1, 1, 1, 1, 1, 1)
                lamb_du = lamb_du.view(-1, 1, 1, 1, 1, 1, 1)
            else:
                lamb_ru = lamb_ru.view(N, 1, 1, 1, 1, 1, 1)
                lamb_du = lamb_du.view(N, 1, 1, 1, 1, 1, 1)
        else:
            lamb_ru = self.lamb_ru
            lamb_du = self.lamb_du

        G = Ru * lamb_ru + Du * lamb_du

        if self.sgd_momentum:
            if S_prev is None:
                S = G
            else:
                S = G + self.alpha * S_prev
        else:
            S = G

        u_next = u_t_1 - S
        return u_next, S

    def _regularizer_forward_eager(self, u_t_1):
        N, V, T, D, H, W, _ = u_t_1.shape
        Ru = 0.0

        # ---- xyz branch ----
        u_k = self._regularizer_maybe_channels_last_3d(
            u_t_1.permute(0, 2, 6, 1, 3, 4, 5).contiguous().view(N * T * 2, V, D, H, W)
        )
        u_k = F.conv3d(
            u_k,
            self.conv_kernel_xyz,
            padding=(self.pad_d, self.pad, self.pad),
        )
        u_k = self.activation1(u_k)
        u_k = self._regularizer_maybe_channels_last_3d(u_k)
        u_kT = self._regularizer_transpose_conv(
            u_k, self.conv_kernel_xyz, padding=(self.pad_d, self.pad, self.pad)
        )
        Ru = Ru + u_kT.view(N, T, 2, V, D, H, W).permute(0, 3, 1, 4, 5, 6, 2)
        del u_k, u_kT

        # ---- xyt branch ----
        u_k = self._regularizer_maybe_channels_last_3d(
            u_t_1.permute(0, 5, 6, 1, 3, 4, 2).contiguous().view(N * W * 2, V, D, H, T)
        )
        u_k = F.conv3d(
            u_k,
            self.conv_kernel_xyt,
            padding=(self.pad_d, self.pad, self.pad),
        )
        u_k = self.activation2(u_k)
        u_k = self._regularizer_maybe_channels_last_3d(u_k)
        u_kT = self._regularizer_transpose_conv(
            u_k, self.conv_kernel_xyt, padding=(self.pad_d, self.pad, self.pad)
        )
        Ru = Ru + u_kT.view(N, W, 2, V, D, H, T).permute(0, 3, 6, 4, 5, 1, 2)
        del u_k, u_kT

        # ---- yzt branch ----
        u_k = self._regularizer_maybe_channels_last_3d(
            u_t_1.permute(0, 3, 6, 1, 4, 5, 2).contiguous().view(N * D * 2, V, H, W, T)
        )
        u_k = F.conv3d(
            u_k,
            self.conv_kernel_yzt,
            padding=(self.pad, self.pad, self.pad),
        )
        u_k = self.activation3(u_k)
        u_k = self._regularizer_maybe_channels_last_3d(u_k)
        u_kT = self._regularizer_transpose_conv(
            u_k, self.conv_kernel_yzt, padding=(self.pad, self.pad, self.pad)
        )
        Ru = Ru + u_kT.view(N, D, 2, V, H, W, T).permute(0, 3, 6, 1, 4, 5, 2)
        del u_k, u_kT

        # ---- xzt branch ----
        u_k = self._regularizer_maybe_channels_last_3d(
            u_t_1.permute(0, 4, 6, 1, 3, 5, 2).contiguous().view(N * H * 2, V, D, W, T)
        )
        u_k = F.conv3d(
            u_k,
            self.conv_kernel_xzt,
            padding=(self.pad_d, self.pad, self.pad),
        )
        u_k = self.activation4(u_k)
        u_k = self._regularizer_maybe_channels_last_3d(u_k)
        u_kT = self._regularizer_transpose_conv(
            u_k, self.conv_kernel_xzt, padding=(self.pad_d, self.pad, self.pad)
        )
        Ru = Ru + u_kT.view(N, H, 2, V, D, W, T).permute(0, 3, 6, 4, 1, 5, 2)
        del u_k, u_kT

        Ru = Ru / self.options["features_out"]
        return Ru

    def compile_regularizer(self, compile_mode="default"):
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile")
        self._regularizer_forward_impl = torch.compile(
            self._regularizer_forward_eager, mode=compile_mode
        )
        return self

    def _regularizer_transpose_conv(self, x, kernel, padding):
        if self.options.get("flowvn_transpose_conv_as_conv", False):
            kernel_t = torch.flip(kernel.transpose(0, 1), dims=(-3, -2, -1)).contiguous()
            return F.conv3d(x, kernel_t, padding=padding)
        return F.conv_transpose3d(x, kernel, padding=padding)


class FlowVN(nn.Module):
    def __init__(self, **kwargs):
        super(FlowVN, self).__init__()
        options = kwargs
        self.options = options
        self.nc = options["num_stages"]
        self.exp_loss = options["exp_loss"]

        cells = []
        for i in range(self.nc):
            cells.append(VnMriReconCell(block_id=i, **options))

        self.cell_list = nn.ModuleList(cells)
        self.input_norm = False
        self.norm_eps = float(options.get("norm_eps", 1e-6))

    def _parse_grad_checkpoint_policy(self):
        policy = str(self.options.get("flowvn_grad_checkpoint_policy", "all")).strip().lower()
        if policy in ("all", "none"):
            return policy, None
        for prefix in ("first:", "last:", "every:"):
            if policy.startswith(prefix):
                value = policy[len(prefix):]
                try:
                    count = int(value)
                except ValueError as exc:
                    raise ValueError(
                        "flowvn_grad_checkpoint_policy must be one of all, none, first:N, last:N, every:K"
                    ) from exc
                if count <= 0:
                    raise ValueError(
                        "flowvn_grad_checkpoint_policy count must be positive for first:N, last:N, every:K"
                    )
                return prefix[:-1], count
        raise ValueError(
            "flowvn_grad_checkpoint_policy must be one of all, none, first:N, last:N, every:K"
        )

    def _use_grad_checkpoint(self):
        return (
            bool(self.options.get("grad_check", False))
            and self.training
            and torch.is_grad_enabled()
        )

    def _should_checkpoint_stage(self, stage_idx):
        if not self._use_grad_checkpoint():
            return False
        policy, count = self._parse_grad_checkpoint_policy()
        if policy == "none":
            return False
        if policy == "all":
            return True
        if policy == "first":
            return int(stage_idx) < int(count)
        if policy == "last":
            return int(stage_idx) >= max(0, int(self.nc) - int(count))
        if policy == "every":
            return int(stage_idx) % int(count) == 0
        raise RuntimeError(f"Unhandled flowvn_grad_checkpoint_policy: {policy}")

    def forward(self, x, f, c, usrate, return_intermediates=False):
        x = torch.view_as_real(x)

        S_prev = None
        dc_cache = None
        if str(self.options.get("flowvn_dc_centered_fft", "original")) == "checkerboard":
            dc_cache = make_flowvn_dc_checkerboard_cache(f, c)
        collect_intermediates = bool(return_intermediates) or (
            self.exp_loss and self.options["mode"] == "train"
        )
        if collect_intermediates:
            x_layers = []

        for i in range(self.nc):
            block = self.cell_list[i]
            if self._should_checkpoint_stage(i):
                if S_prev is None:
                    def run_block_first(x_in, block=block, dc_cache=dc_cache):
                        return block(x_in, f, c, usrate, None, dc_cache=dc_cache)

                    x, S_prev = torch.utils.checkpoint.checkpoint(
                        run_block_first, x, use_reentrant=False
                    )
                else:
                    def run_block_with_state(x_in, s_in, block=block, dc_cache=dc_cache):
                        return block(x_in, f, c, usrate, s_in, dc_cache=dc_cache)

                    x, S_prev = torch.utils.checkpoint.checkpoint(
                        run_block_with_state, x, S_prev, use_reentrant=False
                    )
            else:
                x, S_prev = block(x, f, c, usrate, S_prev, dc_cache=dc_cache)

            if collect_intermediates:
                x_layers.append(x)

        if return_intermediates:
            complex_layers = tuple(torch.view_as_complex(layer) for layer in x_layers)
            return complex_layers[-1], complex_layers
        if self.exp_loss and self.options["mode"] == "train":
            x_layers = torch.stack(x_layers, dim=0)
            return torch.view_as_complex(x_layers)

        return torch.view_as_complex(x)
