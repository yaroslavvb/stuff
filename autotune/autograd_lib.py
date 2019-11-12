"""
Library for extracting interesting quantites from autograd.

Not thread-safe because of module-level variables affecting state of autograd

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
s: spatial dimension (Oh*Ow)
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Hi: per-example Hessian
    Linear layer: shape [do*di, do*di]
    Conv2d layer: shape [do*di*Kh*Kw, do*di*Kh*Kw]
Hi_bias: per-example Hessian of bias
H: mean Hessian of matmul
H_bias: mean Hessian of bias


Jo: batch output Jacobian of matmul, gradient of output for each output,example pair, [o, n, ....]
Jo_bias: output Jacobian of bias

A, activations: inputs into matmul
    Linear: [n, di]
    Conv2d: [n, di, Ih, Iw] -> (unfold) -> [n, di, Oh, Ow]
B, backprops: backprop values (aka Lop aka Jacobian-vector product) for current layer
    Linear: [n, do]
    Conv2d: [n, do, Oh, Ow]

weight: matmul part of layer, Linear [di, do], Conv [do, di, Kh, Kw]

H, hess  -- hessian
S, sigma -- noise
L, lyap -- lyapunov matrix

"""
import math
from typing import List, Optional, Callable, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import util as u
import globals as gl
from attrdict import AttrDefault, AttrDict

_supported_layers = ['Linear', 'Conv2d']  # Supported layer class types  TODO(y): make non-private
_supported_methods = ['exact', 'kron', 'mean_kron', 'experimental_kfac']  # supported approximation methods
_supported_losses = ['LeastSquares', 'CrossEntropy']

# module-level variables affecting state of autograd
_global_hooks_disabled: bool = False  # work-around for https://github.com/pytorch/pytorch/issues/25723
_global_enforce_fresh_backprop: bool = False  # global switch to catch double backprop errors on Hessian computation
_global_backprops_prefix = ''  # hooks save backprops to, ie param.{_backprops_prefix}backprops_list


class LayerStats:
    # Some notation/background from https://docs.google.com/document/d/19Jmh4spbSAnAGX_eq7WSFPgLzrpJEhiZRpjX1jSYObo/edit#heading=h.9fi55aowtmgy
    sparsity: torch.Tensor
    mean_activation: torch.Tensor
    mean_backprop: torch.Tensor

    sigma_l2: torch.Tensor  # l2 norm of centered gradient covariance (singlconnected noise covariant matrix)
    sigma_erank: torch.Tensor  # trace/l2 norm

    H_l2: torch.Tensor  # spectral norm of H (largest curvature)
    H_fro: torch.Tensor  # Frobenius norm of hessian
    grad_fro: torch.Tensor  # Frobenius norm of gradient update
    param_fro: torch.Tensor  # Frobenius norm of parameter tensor

    grad_curv: torch.Tensor  # curvature in direction of gradient
    newton_curv: torch.Tensor  # curvature in direction of Newton step

    step_openai: torch.Tensor  # optimal step length using gradient direction and Hessian curvature estimate
    step_div_inf: torch.Tensor  # divergent step size for infinite batch (2/spectral radius)
    step_div_1: torch.Tensor  # (2/trace, a bad attempt to approximate Jain divergent lr, should be 2/R^2)
    newton_fro: torch.Tensor  # Frobenius norm of newton step
    regret_gradient: torch.Tensor  # expected improvement if we took optimal step-size in gradient direction
    reget_newton: torch.Tensor  # expected improvement if we took Newton step
    batch_openai: torch.Tensor  # optimal batch size from gradient noise stat (loss change from noise part over loss change from deterministic part)
    batch_jain_simple: torch.Tensor  # optimal batch-size assuming well-specified model (trace/sigma)
    batch_jain_full: torch.Tensor  # optimal batch size using Jain/Kakade approach
    noise_variance_pinv: torch.Tensor  # asymptotic minimax rate (called noise variance in 1.1.4 of Jain/Kakade)

    # need: R^2 the largest Jacobian size
    # need: angle between gradient and newton step

    # 2 hessians, 2 covariance matrices, need l2, trace, rank, spectrum

    def __iter__(self):
        return iter(self.__dict__)

    def __init__(self):
        pass

    def __getitem__(self, item):
        return self.__dict__[item]

    def items(self):
        return self.__dict__.items()


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.

    The hooks will
    1. assign activations to layer.activations during forward pass
    2. assign layer output to layer.output during forward pass
    2. append backprops to layer.backprops_list during backward pass

    Call "clear_backprops" to clear backprops_list values for all parameters in the model
    Call "remove_hooks(model)" to undo this operation.


    Args:
        model:
    """

    global _global_hooks_disabled
    _global_hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_forward_hook(_capture_output))
            handles.append(layer.register_backward_hook(_capture_backprops))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)


def remove_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks. Provides Providesa hook for removing hooks.
    """

    assert model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"




    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle
        del model.autograd_hacks_hooks


def disable_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """

    global _global_hooks_disabled
    _global_hooks_disabled = True


def enable_hooks() -> None:
    """The opposite of disable_hooks()."""

    global _global_hooks_disabled
    _global_hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported."""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _global_hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "activations", input[0].detach())


def _capture_output(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _global_hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "output", output.detach())


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _global_enforce_fresh_backprop

    if _global_hooks_disabled:
        return

    backprops_list_attr = _global_backprops_prefix + 'backprops_list'
    if _global_enforce_fresh_backprop:
        assert not hasattr(layer,
                           backprops_list_attr), f"Seeing result of previous backprop in {backprops_list_attr}, use {_global_backprops_prefix}clear_backprops(model) to clear"
        _global_enforce_fresh_backprop = False

    if not hasattr(layer, backprops_list_attr):
        setattr(layer, backprops_list_attr, [])
    getattr(layer, backprops_list_attr).append(output[0].detach())


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list


def clear_hess_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'hess_backprops_list'):
            del layer.hess_backprops_list


def compute_grad1(model: nn.Module, loss_type: str = 'mean') -> None:
    """
    Compute per-example gradients and save them under 'param.grad1'. Must be called after loss.backprop()

    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
    """

    assert loss_type in ('sum', 'mean')
    for layer in model.modules():
        if hasattr(layer, 'expensive'):
            continue

        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        assert len(layer.backprops_list) == 1, "Multiple backprops detected, make sure to call clear_backprops(model)"

        A = layer.activations
        n = A.shape[0]
        if loss_type == 'mean':
            B = layer.backprops_list[0] * n
        else:  # loss_type == 'sum':
            B = layer.backprops_list[0]

        if layer_type == 'Linear':
            setattr(layer.weight, 'grad1', torch.einsum('ni,nj->nij', B, A))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', B)

        elif layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels
            Oh, Ow = layer.backprops_list[0].shape[2:]
            weight_shape = [n] + list(layer.weight.shape)  # n, do, di, Kh, Kw
            assert weight_shape == [n, do, di, Kh, Kw]
            A = torch.nn.functional.unfold(A, layer.kernel_size)  # n, di * Kh * Kw, Oh * Ow

            assert A.shape == (n, di * Kh * Kw, Oh * Ow)
            assert layer.backprops_list[0].shape == (n, do, Oh, Ow)

            # B = B.reshape(n, -1, A.shape[-1])
            B = B.reshape(n, do, Oh * Ow)
            # noinspection PyTypeChecker
            grad1 = torch.einsum('ijk,ilk->ijl', B, A)  # n, do, di * Kh * Kw
            assert grad1.shape == (n, do, di * Kh * Kw)

            setattr(layer.weight, 'grad1', grad1.reshape(weight_shape))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', torch.sum(B, dim=2))


def compute_hess(model: nn.Module, method='exact', attr_name=None, vecr_order=False, loss_aggregation='mean') -> None:
    """Compute Hessian (torch.Tensor) for each parameter and save it under 'param.hess' by default.
    Hessian can be a Tensor or a tensor-like object like KronFactored.

     If attr_name is specified, saves it under 'param.{attr_name}'

    Must be called after backprop_hess().

    Args:
        model:
        method: which method to use for computing the Hessian
            kron: kronecker product
            mean_kron: mean of kronecker products, one kronecker product per datapoint
            experimental_kfac: experimental method for Conv2d
        attr_name: which attribute of Parameter object to use for storing the Hessian.
        vecr_order: determines whether Hessian computed with respect to vectorized parameters (math notation default) or row-vectorized parameters (more efficient in PyTorch)
        loss_aggregation: 'mean' or 'sum', determines whether final loss is sum or mean of per-example losses
    """

    assert method in _supported_methods
    assert loss_aggregation in ['mean', 'sum']

    # TODO: get rid of hess_factored logic

    # legacy specification for factored version, remove
    if attr_name is None:
        hess_attr = 'hess' if (method == 'exact' or method == 'autograd') else 'hess_factored'
    else:
        hess_attr = attr_name

    li = 0
    for layer in model.modules():

        if hasattr(layer, 'expensive'):
            continue

        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'hess_backprops_list'), "No backprops detected, run hess_backprop"

        A = layer.activations
        n = A.shape[0]

        if layer_type == 'Linear':
            B = torch.stack(layer.hess_backprops_list)

            di = A.shape[1]
            do = layer.hess_backprops_list[0].shape[1]
            o = B.shape[0]

            original_A = A

            A = torch.stack([A] * o)

            if method == 'exact':
                Jo = torch.einsum("oni,onj->onij", B, A).reshape(n * o, -1)
                H = torch.einsum('ni,nj->ij', Jo, Jo) / n

                # Alternative way
                # Jo = torch.einsum("oni,onj->onij", B, A)
                # H = torch.einsum('onij,onkl->ijkl', Jo, Jo) / n
                # H = H.reshape(do*di, do*di)

                H_bias = torch.einsum('oni,onj->ij', B, B) / n
            else:  # TODO(y): can optimize this case by not stacking A
                assert method == 'kron'
                # AA = torch.einsum("oni,onj->ij", A, A) / (o * n)  # # TODO(y): makes more sense to apply o factor to B
                # BB = torch.einsum("oni,onj->ij", B, B) / n
                #                H = u.Kron(AA, BB)
                H_bias = u.Kron(torch.eye(1), torch.einsum("oni,onj->ij", B, B) / n)  # TODO: reuse BB

                hess = u.KronFactoredCov(di, do)
                hess.add_samples(original_A, B)
                H = hess.value()


        elif layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels
            n, do, Oh, Ow = layer.hess_backprops_list[0].shape
            o = len(layer.hess_backprops_list)

            A = torch.nn.functional.unfold(A, kernel_size=layer.kernel_size,
                                           stride=layer.stride,
                                           padding=layer.padding,
                                           dilation=layer.dilation)  # n, di * Kh * Kw, Oh * Ow
            assert A.shape == (n, di * Kh * Kw, Oh * Ow)
            B = torch.stack([Bh.reshape(n, do, -1) for Bh in layer.hess_backprops_list])  # o, n, do, Oh*Ow

            A = torch.stack([A] * o)  # o, n, di * Kh * Kw, Oh*Ow
            if gl.debug_dump_stats:
                print(f'layerA {li}', A)
                print(f'layerB {li}', B)

            if method == 'exact':
                Jo = torch.einsum('onis,onks->onik', B, A)  # o, n, do, di * Kh * Kw
                Jo_bias = torch.einsum('onis->oni', B)

                Hi = torch.einsum('onij,onkl->nijkl', Jo, Jo)  # n, do, di*Kh*Kw, do, di*Kh*Kw
                Hi = Hi.reshape(n, do * di * Kh * Kw, do * di * Kh * Kw)  # n, do*di*Kh*Kw, do*di*Kh*Kw
                Hi_bias = torch.einsum('oni,onj->nij', Jo_bias, Jo_bias)  # n, do, do
                H = Hi.mean(dim=0)
                H_bias = Hi_bias.mean(dim=0)
            elif method == 'kron':
                AA = torch.einsum("onis->oni", A) / (Oh * Ow)  # group input channels
                AA = torch.einsum("oni,onj->onij", AA, AA) / (o * n)  # remove factor of o because A is repeated o times

                AA = torch.einsum("onij->ij", AA)  # sum out outputs/classes

                BB = torch.einsum("onip->oni", B)  # group output channels
                BB = torch.einsum("oni,onj->ij", BB, BB) / n
            elif method == 'mean_kron':
                AA = torch.einsum("onis->oni", A) / (Oh * Ow)  # group input channels
                AA = torch.einsum("oni,onj->onij", AA, AA) / (o)  # remove factor of o because A is repeated o times

                AA = torch.einsum("onij->nij", AA)  # sum out outputs/classes

                BB = torch.einsum("onip->oni", B)  # group output channels
                BB = torch.einsum("oni,onj->nij", BB, BB)

            elif method == 'experimental_kfac':
                AA = torch.einsum("onis,onjs->onijs", A, A)
                AA = torch.einsum("onijs->onij", AA) / (Oh * Oh)
                AA = torch.einsum("onij->oij", AA) / n
                AA = torch.einsum("oij->ij", AA) / o

                BB = torch.einsum("onip,onjp->onijp", B, B) / n
                BB = torch.einsum("onijp->onij", BB)
                BB = torch.einsum("onij->nij", BB)
                BB = torch.einsum("nij->ij", BB)

            if method != 'exact':
                if method == 'mean_kron':
                    H = u.MeanKronFactored(AA, BB)
                    # H = u.KronFactored(AA[0,...], BB[0,...])
                else:
                    H = u.Kron(AA, BB)

                BB_bias = torch.einsum("onip->oni", B)  # group output channels
                BB_bias = torch.einsum("oni,onj->onij", BB_bias, BB_bias) / n  # covariance
                BB_bias = torch.einsum("onij->ij", BB_bias)  # sum out outputs + examples
                H_bias = u.Kron(torch.eye(1), BB_bias)

        if loss_aggregation == 'sum':
            H = n * H
            H_bias = H * n

        if vecr_order:
            H = H.commute()
            H_bias = H_bias.commute()

        setattr(layer.weight, hess_attr, H)
        if layer.bias is not None:
            setattr(layer.bias, hess_attr, H_bias)
        li += 1


def backprop_hess(output: torch.Tensor, hess_type: str, model: Optional[nn.Module] = None) -> None:
    """
    Call backprop 1 or more times to accumulate values needed for Hessian computation.

    Values are accumulated under .backprops_list attr of each layer and used by downstream functions like compute_hess

    Args:
        output: prediction of neural network (ie, input of nn.CrossEntropyLoss())
        hess_type: 'LeastSquares' or 'CrossEntropy'. Type of Hessian propagation, "CrossEntropy" results in exact Hessian for CrossEntropy
        model: optional model, used to freeze the parameters
    """

    global _global_enforce_fresh_backprop, _global_hooks_disabled, _global_backprops_prefix

    assert not _global_hooks_disabled
    _global_enforce_fresh_backprop = True  # enforce empty backprops_list on first backprop

    old_backprops_prefix = _global_backprops_prefix
    _global_backprops_prefix = 'hess_'  # backprops go into hess_backprops_list

    valid_hess_types = ('LeastSquares', 'CrossEntropy', 'DebugLeastSquares')
    assert hess_type in valid_hess_types, f"Unexpected hessian type: {hess_type}, valid types are {valid_hess_types}"
    n, o = output.shape

    if hess_type == 'CrossEntropy':
        batch = F.softmax(output, dim=1)

        mask = torch.eye(o).to(gl.device).expand(n, o, o)
        diag_part = batch.unsqueeze(2).expand(n, o, o) * mask
        outer_prod_part = torch.einsum('ij,ik->ijk', batch, batch)
        hess = diag_part - outer_prod_part
        assert hess.shape == (n, o, o)

        with u.timeit("xent-symsqrt"):
            for i in range(n):
                if torch.get_default_dtype() == torch.float64:
                    hess[i, :, :] = u.symsqrt_svd(
                        hess[i, :, :])  # more stable method since we don't care about speed with float64
                    print('warning, slow method for cross-entropy')
                else:
                    hess[i, :, :] = u.symsqrt(hess[i, :, :])
                u.nan_check(hess[i, :, :])
            hess = hess.transpose(0, 1)

    elif hess_type == 'LeastSquares':
        hess = []
        assert len(output.shape) == 2
        batch_size, output_size = output.shape

        id_mat = torch.eye(output_size).to(output.device).type(output.dtype)
        for out_idx in range(output_size):
            hess.append(torch.stack([id_mat[out_idx]] * batch_size))

    elif hess_type == 'DebugLeastSquares':
        hess = []
        assert len(output.shape) == 2
        batch_size, output_size = output.shape

        id_mat = torch.eye(output_size)
        id_mat[0, 0] = 10
        for out_idx in range(output_size):
            hess.append(torch.stack([id_mat[out_idx]] * batch_size))

    for o in range(o):
        output.backward(hess[o], retain_graph=True)

    # side-effect of Hessian backprop is that .grad buffers are updated.
    # Zero out those buffers to prevent accidental use
    if model is not None:
        model.zero_grad()

    _global_backprops_prefix = old_backprops_prefix


class LayerCov:
    """Class representing second-order information associated with a layer."""

    S: u.KronFactoredCov  # expected gradient outer product: E[gg'] where g=gradient of loss
    J: u.KronFactoredCov  # expected Jacobian outer product: E[hh'] where h=gradient of network output
    H: u.KronFactoredCov  # expected hessian: E[H] where H is per-example hessian

    def __init__(self):
        self.S = None
        self.J = None
        self.H = None


def compute_cov(model: nn.Module, loss_fn: Callable, stats_iter, batch_size, steps, loss_type='CrossEntropy'):
    """

    Augments model layers with their associated covariance matrices. At the end of the run, every layer will have an
    attribute 'cov' of type LayerCov

    Args:
        stats_iter: data iterator
        loss_type:
        model:
        loss_fn:
        batch_size: size of batch to use for estimation
        steps: number of steps to use to aggregate stats

    Returns:

    """

    assert loss_type == 'CrossEntropy', 'only cross entropy is implemented'

    enable_hooks()

    clear_backprops(model)
    clear_hess_backprops(model)
    model.zero_grad()

    for i in range(steps):
        data, targets = next(stats_iter)
        output = model(data)
        loss = loss_fn(output, targets)
        assert len(data) == batch_size

        with u.timeit("backprop_J"):
            backprop_hess(output, hess_type='LeastSquares')
            update_cov(model, 'activations', 'hess_backprops_list', 'J')
            clear_hess_backprops(model)

        with u.timeit("backprop_G"):
            loss.backward(retain_graph=True)
            update_cov(model, 'activations', 'backprops_list', 'S')
            clear_backprops(model)
            model.zero_grad()

        # disable because super-slow
        if not gl.hacks_disable_hess:
            with u.timeit("backprop_H"):
                backprop_hess(output, hess_type='CrossEntropy')
                update_cov(model, 'activations', 'hess_backprops_list', 'H')
                clear_hess_backprops(model)

    disable_hooks()


def update_cov(model, a_attr, b_attr, target_attr):
    """Update Kronecker-factored layer covariance of a,b values.
    For every layer in the model, will perform
        a = layer.{a_attr}
        b = layer.{b_attr}
        layer.cov.{target_attr}.add_samples(a, b)
    """

    for layer in model.modules():
        if not is_supported(layer):
            continue

        layer_type = _layer_type(layer)

        if hasattr(layer, 'cov'):
            layer_cov = layer.cov
        else:
            layer_cov = LayerCov()
            setattr(layer, 'cov', layer_cov)

        a_vals = getattr(layer, a_attr)
        a_dim = a_vals.shape[-1]
        b_vals = getattr(layer, b_attr)

        if layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels
            n, do, Oh, Ow = a_vals.shape
            o = len(layer.hess_backprops_list)

            a_vals = torch.nn.functional.unfold(a_vals, kernel_size=layer.kernel_size,
                                                stride=layer.stride,
                                                padding=layer.padding,
                                                dilation=layer.dilation)  # n, di * Kh * Kw, Oh * Ow
            assert a_vals.shape == (n, di * Kh * Kw, Oh * Ow)
            a_vals = torch.einsum("onis->oni", a_vals) / (Oh * Ow)  # group input channels
            b_vals = [b.reshape(n, do, -1) for b in b_vals]

        # backward vals are in a list
        # special handling for backprops, multiple vals are stacked into rank 3 tensor

        assert type(b_vals) == list
        if len(b_vals) > 1:
            b_vals = torch.stack(b_vals)

        else:
            b_vals = b_vals[0]

        b_dim = b_vals.shape[-1]

        covmat = getattr(layer_cov, target_attr)
        if covmat is None:
            covmat = u.KronFactoredCov(a_dim, b_dim)
            setattr(layer_cov, target_attr, covmat)
        covmat.add_samples(a_vals, b_vals)


def compute_stats(model, attr_name='stats', factored=False, sigma_centering=True):
    """

    Combines activations and backprops to compute statistics for a model.
    Args:
        model:
        attr_name: stats are saved under this attribute name on corresponding Parameter

    """

    # obtain n
    n = 0
    for param in model.modules():
        if hasattr(param, 'activations'):
            n = param.activations.shape[0]
            break
    assert n, "Couldn't figure out size of activations"

    for (i, layer) in enumerate(model.layers):

        if hasattr(layer, 'expensive'):
            continue

        param_names = {layer.weight: "weight", layer.bias: "bias"}
        for param in [layer.weight, layer.bias]:

            if param is None:
                continue

            s = LayerStats()  # dictionary-like object for layer stats

            #############################
            # Gradient stats
            #############################
            A_t = layer.activations
            B_t = layer.backprops_list[0] * n

            s.sparsity = torch.sum(
                layer.output <= 0).float() / layer.output.numel()  # proportion of activations that are zero
            s.mean_activation = torch.mean(A_t)
            s.mean_backprop = torch.mean(B_t)

            # empirical Fisher
            G = param.grad1.reshape((n, -1))
            g = G.mean(dim=0, keepdim=True)

            print(G)
            u.nan_check(G)
            with u.timeit(f'sigma-{i}'):
                efisher = G.t() @ G / n
                if sigma_centering:
                    sigma = efisher - g.t() @ g
                else:
                    sigma = efisher

                s.sigma_l2 = u.sym_l2_norm(sigma)
                s.sigma_erank = torch.trace(sigma) / s.sigma_l2

            H = param.hess

            u.nan_check(H)

            with u.timeit(f"H_l2-{i}"):
                s.H_l2 = u.sym_l2_norm(H)

            with u.timeit(f"norms-{i}"):
                s.H_fro = H.flatten().norm()
                s.grad_fro = g.flatten().norm()
                s.param_fro = param.data.flatten().norm()

            # TODO(y): col vs row fix
            def loss_direction(dd: torch.Tensor, eps):
                """

                Args:
                    dd: direction, as a n-by-1 matrix
                    eps: scalar length

                Returns:
                   loss improvement if we take step eps in direction dd.
                """
                assert u.is_row_matrix(dd)
                return (eps * (dd @ g.t()) - 0.5 * eps ** 2 * dd @ H @ dd.t()).squeeze()

            def curv_direction(dd: torch.Tensor):
                """Curvature in direction dd (directional eigenvalue). """
                assert u.is_row_matrix(dd)
                return (dd @ H @ dd.t() / (dd.flatten().norm() ** 2)).squeeze()

            with u.timeit(f"pinvH-{i}"):
                pinvH = u.pinv(H)

            with u.timeit(f'curv-{i}'):
                s.grad_curv = curv_direction(g)  # curvature (eigenvalue) in direction g
                ndir = g @ pinvH  # newton direction (TODO(y): replace with lstsqsolve)
                s.newton_curv = curv_direction(ndir)

                setattr(layer.weight, 'pre', pinvH)  # save Newton preconditioner
                s.step_openai = 1 / s.grad_curv if s.grad_curv else 1234567
                s.step_div_inf = 2 / s.H_l2  # divegent step size for batch_size=infinity
                s.step_div_1 = torch.tensor(2) / torch.trace(H)  # divergent step for batch_size=1

                s.newton_fro = ndir.flatten().norm()  # frobenius norm of Newton update
                s.regret_newton = u.to_python_scalar(g @ pinvH @ g.t() / 2)  # replace with "quadratic_form"
                s.regret_gradient = loss_direction(g, s.step_openai)

            # todo: Lyapunov has to be redone
            with u.timeit(f'rho-{i}'):
                s.rho, lyap_erank, L_evals = u.truncated_lyapunov_rho(H, sigma)
                s.step_div_1_adjusted = s.step_div_1 / s.rho

            with u.timeit(f"batch-{i}"):
                # s.batch_openai = torch.trace(H @ sigma) / (g @ H @ g.t()).squeeze()
                s.batch_openai = torch.trace(H @ sigma) / (g @ H @ g.t())
                print('original sigma: ', torch.trace(H @ sigma) / (g @ H @ g.t()))
                denom = (g @ H @ g.t())
                print('subtracted1:', torch.trace(H @ (sigma - g.t() @ g)) / denom)
                print('subtracted2:', torch.trace(H @ sigma) / denom - torch.trace(H @ g.t() @ g) / denom)
                print("left term: ", torch.trace(H @ sigma))
                print("right term: ", torch.trace(H @ g.t() @ g))
                print('denom: ', denom)

                s.diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2 / n  # Gradient diversity / n
                s.noise_variance_pinv = torch.trace(pinvH @ sigma)  # todo(y): replace with lsqtsolve
                s.H_erank = torch.trace(H) / s.H_l2
                s.batch_jain_simple = 1 + s.H_erank
                s.batch_jain_full = 1 + s.rho * s.H_erank

            param_name = f"{layer.name}={param_names[param]}"
            u.log_scalars(u.nest_stats(f"{param_name}", s))

            H_evals = u.symeig_pos_evals(H)
            S_evals = u.symeig_pos_evals(sigma)

            # s.H_evals = H_evals
            # s.S_evals = S_evals
            # s.L_evals = L_evals

            setattr(param, attr_name, s)

            # u.log_spectrum(f'{param_name}/hess', H_evals)
            # u.log_spectrum(f'{param_name}/sigma', S_evals)
            # u.log_spectrum(f'{param_name}/lyap', L_evals)

    return None


def compute_stats_factored(model, attr_name='stats', sigma_centering=True):
    """Combines activations and backprops to compute statistics for a model. Assumes factored hessian was saved into 'hess2' of each param"""

    ein = torch.einsum
    n = 0
    for param in model.modules():
        if hasattr(param, 'activations'):
            n = param.activations.shape[0]
            break
    assert n, "Couldn't figure out size of activations"

    for (i, layer) in enumerate(model.layers):

        if hasattr(layer, 'expensive'):
            continue

        param_names = {layer.weight: "weight", layer.bias: "bias"}
        for param in [layer.weight, layer.bias]:

            if param is None:
                continue

            s = AttrDefault(str, {})  # dictionary-like object for layer stats

            do, di = layer.weight.shape
            H: u.Kron = param.hess2

            if param is layer.weight:
                assert H.shape == ((di, di), (do, do))
            else:
                assert H.shape == ((1, 1), (do, do))

            # TODO(y): fix stats for bias
            if param is layer.bias:
                continue

            G = param.grad1.reshape((n, -1))
            g = G.mean(dim=0, keepdim=True)

            if param is layer.weight:
                vecG = u.Vec(g, shape=(do, di))
            else:  # bias
                vecG = u.Vec(g, shape=(do, 1))

            u.nan_check(G)

            A = layer.activations
            B = layer.backprops_list[0] * n

            AA = ein('ni,nj->ij', A, A)
            BB = ein('ni,nj->ij', B, B)
            Bc = B - torch.mean(B, dim=0)
            BBc = ein('ni,nj->ij', Bc, Bc)

            # subtracting mean breaks Kronecker factoring, so this is approximate
            if sigma_centering:
                sigma_k = u.Kron(AA, BBc) / n  # only center backprops, centering both leads to underestimate of cov
            else:
                sigma_k = u.Kron(AA, BB) / n
            sigma_k = sigma_k / n  # extra factor to average A's as well
            s.sparsity = torch.sum(
                layer.output <= 0).float() / layer.output.numel()  # proportion of activations that are zero
            s.mean_activation = torch.mean(A)
            s.mean_backprop = torch.mean(B)

            with u.timeit(f'sigma-{i}'):
                s.sigma_l2 = sigma_k.sym_l2_norm()
                s.sigma_erank = sigma_k.trace() / s.sigma_l2

            # u.nan_check(param.hess)

            with u.timeit(f"H_l2-{i}"):
                s.H_l2 = H.sym_l2_norm()

            with u.timeit(f"norms-{i}"):
                s.H_fro = H.frobenius_norm()
                s.grad_fro = g.flatten().norm()
                s.param_fro = param.data.flatten().norm()

            with u.timeit(f"pinvH-{i}"):
                pinvH = H.pinv()

            with u.timeit(f'curv-{i}'):
                s.grad_curv = u.matmul(vecG @ H, vecG) / (vecG @ vecG)
                newton_dir = vecG @ pinvH
                s.newton_curv = u.matmul(newton_dir @ H, newton_dir) / (newton_dir @ newton_dir)

                setattr(layer.weight, 'pre', pinvH)  # save Newton preconditioner
                s.step_openai = 1 / s.grad_curv if s.grad_curv else 1234567
                s.step_div_inf = 2 / s.H_l2  # divegent step size for batch_size=infinity
                s.step_div_1 = torch.tensor(2) / H.trace()  # divergent step for batch_size=1
                s.newton_fro = newton_dir.norm()  # frobenius norm of Newton update,

                def loss_direction(d: u.Vec, step):  # improvement in loss if we go eps units in direction dir
                    return step * (d @ vecG) - 0.5 * step ** 2 * (d @ H @ vecG)

                s.regret_gradient = loss_direction(vecG, s.step_openai)
                # can compute newton regret more efficiently by doing row-vectorized instead of col-vectorized
                vecG2 = u.Vecr(g, shape=(do, di))
                pinvH_rowvec = pinvH.commute()  # original H was for col-vectorized order
                s.regret_newton = vecG2 @ pinvH_rowvec @ vecG2 / 2

            with u.timeit(f'rho-{i}'):
                # lyapunov matrix
                Xk = u.lyapunov_spectral(H.RR, sigma_k.RR)  # compare backprops
                s.rho = u.erank(u.eye_like(Xk)) / u.erank(Xk)
                s.step_div_1_adjusted = s.step_div_1 / s.rho

            with u.timeit(f"batch-{i}"):
                s.batch_openai = (H @ sigma_k).trace() / (vecG @ H @ vecG)
                if not sigma_centering:
                    s.batch_openai -= 1

                expected_grad_norm_sq = torch.norm(G, "fro") ** 2 / n  # expected gradient norm squared
                s.diversity = expected_grad_norm_sq / torch.norm(g) ** 2  # Gradient diversity / n
                s.noise_variance_pinv = (pinvH @ sigma_k).trace()
                s.H_erank = H.trace() / s.H_l2
                s.batch_jain_simple = 1 + s.H_erank
                s.batch_jain_full = 1 + s.rho * s.H_erank

            param_name = f"{layer.name}={param_names[param]}"
            u.log_scalars(u.nest_stats(f"{param_name}", s))

            setattr(param, attr_name, s)


##########################################################################################
### post-refactoring
##########################################################################################

import torch.utils


# second order covariance for two variables X,Y
class SecondOrderCov:

    # todo: add "symmetric" flag
    def __init__(self):
        self.mean_x = None
        self.mean_y = None
        self.cov_xy = None
        self.d_x = -1
        self.d_y = -1
        self.initialized = False
        self.n = 0

    def accumulate(self, data_x, data_y):
        assert u.is_matrix(data_x)
        assert u.is_matrix(data_y)
        if not self.initialized:
            self.d_x = data_x.shape[1]
            self.d_y = data_y.shape[1]
            self.cov_xy = torch.zeros(self.d_x, self.d_y).type(data_x.dtype).to(data_x.device)
            self.mean_x = torch.zeros(self.d_x).type(data_x.dtype).to(data_x.device)
            self.mean_y = torch.zeros(self.d_y).type(data_y.dtype).to(data_y.device)
            self.initialized = True
        n = data_x.shape[0]
        assert n == data_y.shape[0]
        self.cov_xy += torch.einsum("ni,nj->ij", data_x, data_y)
        self.mean_x += torch.einsum("ni->i", data_x)
        self.mean_y += torch.einsum("ni->i", data_y)
        self.n += n

    def zero_(self):
        self.n = 0
        self.cov_xy.zero_()
        self.mean_x.zero_()
        self.mean_y.zero_()


class SymmetricFourthOrderCov:
    """Fourth order generalized covariance.
    rank=4 gives exact stats
    rank=3 uses Isserlis Theorem to for compact storage
    rank=2 is equivalent to Kronecker factoring
    rank=1 (not implemented) analogous to batch normalization
    """
    xx: SecondOrderCov
    yy: SecondOrderCov
    xy: SecondOrderCov
    xxyy: SecondOrderCov

    def __init__(self, rank=3):
        self.xx = SecondOrderCov()    # rank2
        self.yy = SecondOrderCov()    # rank2
        self.xy = SecondOrderCov()    # rank3
        self.xxyy = SecondOrderCov()  # rank4
        self.rank = rank

    def accumulate(self, data_x: torch.Tensor, data_y: torch.Tensor, cached_xx=None):
        assert u.is_matrix(data_x)
        assert u.is_matrix(data_y[0])
        n = data_x.shape[0]
        assert data_y.shape[0] == n

        if self.rank == 4:
            Jo = torch.einsum("oni,onj->onij", data_y, data_x).reshape(n, -1)
            self.xxyy.accumulate(Jo, Jo)

            # Alternative way
            # Jo = torch.einsum("oni,onj->onij", B, A)
            # H = torch.einsum('onij,onkl->ijkl', Jo, Jo) / n
            # H = H.reshape(do*di, do*di)

        else:
            if cached_xx is not None:
                self.xx.accumulate(data_x, data_x)
            self.yy.accumulate(data_y, data_y)

            if self.rank == 3:
                self.xy.accumulate(data_x, data_y)

    def zero_(self):
        self.xx.zero_()
        self.yy.zero_()
        self.xy.zero_()


class ModuleDict(dict):

    def __init__(self, defaultcreator=None, defaultvalue=None):
        assert (defaultcreator is None) or (defaultvalue is None), "only one of defaultcreator/defaultvalue must be set"
        self.defaultvalue = None
        self.defaultcreator = None
        if defaultcreator is not None:
            self.defaultcreator = defaultcreator
        elif defaultvalue is not None:
            self.defaultvalue = defaultvalue

    def __getitem__(self, item):
        if item not in self:
            if self.defaultcreator:
                self[item] = self.defaultcreator()
            elif self.defaultvalue:
                self[item] = self.defaultvalue
            else:
                assert False, f"Requested value {item} which doesn't exist in ModuleDict and defaultcreator nor default value is set"
        return self[item]


# Namespace of global settings used by the library internally
# Using module-level variable for settings means this library is not thread-safe.

global_settings_initialized = False


class Settings(object):
    forward_hooks: List[Callable]   # forward subhooks called by the global hook
    backward_hooks: List[Callable]  # backward subhooks
    model: Optional[nn.Module]
    hook_handles: List[torch.utils.hooks.RemovableHandle]    # removal handles of global hooks registered with PyTorch
    default_activations: Optional[ModuleDict]


    def __init__(self):
        assert global_settings_initialized is False, "Reinitializing Settings seems like a bug."
        self.model = None
        self.hook_handles = []
        self.forward_hooks = []
        self.backward_hooks = []
        self.default_activations = None
        self.default_Acov = None

        # temporary settings to aggregate gradient norm squared globally
        self._hack_gradient_norm_sum = ModuleDict(defaultvalue=torch.zeros(()))
        self._hack_gradient_norm_count = ModuleDict(defaultvalue=torch.zeros(()))
        self._hack_activations_squared = None   # initialized in set_default_activations

        # TODO(y): maybe remove all of this
        # store last activations captured for each layer. While this breaks encapsulation, this considerably simplifies the common use
        # case where the same set of activations is needed for several backward aggregation calls
        self.last_captured_activations = ModuleDict()

        # To prevent a mix of saved activations from different forward calls, keep counter which indicates in which
        # context each activation value was saved. This can be used to enforce that all activations were captured in the same context
        self.last_captured_activations_contextid = ModuleDict()
        self.activations_contextid = 0   # this gets incremented for each save_activations context





global_settings = Settings()


def layer_cov_dict(model):
    """Returns a dictionary of layer->KronFactoredCov for all supported layers in model."""
    return {layer: u.KronFactoredCov() for layer in model.layers() if is_supported(layer)}


def _forward_hook(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    for hook in global_settings.forward_hooks:
        hook(layer, input, output)


# TODO(y): fix signature
def _backward_hook(layer: nn.Module, _input: torch.Tensor, output: torch.Tensor):
    for hook in global_settings.backward_hooks:
        hook(layer, _input, output)


def register(model: nn.Module):
    """
    Registers given model with autograd_lib. This allows user to use decorators like save_activations(A) and module_hook
    """
    global global_settings
    _global_hooks_disabled = False

    # TODO(y): make it work for multiple models and test. This needs check that hook list remains a singleton
    assert 'handles' not in vars(global_settings), "Already called register in this thread"

    global_settings.model = model

    layer: nn.Module
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            global_settings.hook_handles.append(layer.register_forward_hook(_forward_hook))
            layer.register_backward_hook(_backward_hook)   # don't save handle, https://github.com/pytorch/pytorch/issues/25723


def _hack_zero_gradient_norms_squared():
    for layer in global_settings._hack_gradient_norm_count:
        global_settings._hack_gradient_norm_count[layer] = 0
        global_settings._hack_gradient_norm_sum[layer] = 0
        global_settings._hack_activations_squared[layer] = None


def _hack_update_gradient_norms_squared(layer: nn.Module, backprops: torch.Tensor):
    """Trick from Ian Goodfellow https://arxiv.org/abs/1510.01799

    Add up gradient norm squared of gradients
    """
    n = backprops.shape[0]
    A2 = global_settings._hack_activations_squared[layer]   # initialized in "set_default_activations"
    assert n == A2.shape[0]
    sq_components = A2 * backprops * backprops
    global_settings._hack_gradient_norm_sum[layer] += torch.sum(sq_components)
    global_settings._hack_gradient_norm_count[layer] += n


def unregister():
    # TODO(y): switch to tensor backward hooks
    for handle in global_settings.hook_handles:
        handle.remove()


from contextlib import contextmanager


@contextmanager
def save_activations(storage: ModuleDict):
    """Save activations to layer storage: storage[layer] = activations
    """

    assert global_settings.hook_handles, "No hooks have been registered."
    hook_called = [False]

    global_settings.activations_contextid += 1

    def hook(layer: nn.Module, input: List[torch.Tensor], _output: torch.Tensor):
        if layer in storage:
            print("warning, overwriting existing activation for layer ", layer)
        hook_called[0] = True
        activations = input[0].detach()
        storage[layer] = activations
        global_settings.last_captured_activations[layer] = activations
        global_settings.last_captured_activations_contextid[layer] = global_settings.activations_contextid
        assert len(global_settings.last_captured_activations) < 200, "warning, possibly leaking activations, got more than 200"

    global_settings.forward_hooks.append(hook)
    yield
    assert hook_called[0], "Forward hook was never called."
    global_settings.forward_hooks.pop()


@contextmanager
def extend_backprops(storage: ModuleDict):
    """Extends list of backprops in storage with current backprops. storage[layer].extend([backprops])
    """

    assert global_settings.hook_handles, "No hooks have been registered."
    hook_called = [False]

    def hook(layer: nn.Module, _input, output):
        storage.setdefault(layer, []).extend([output[0].detach()])
        hook_called[0] = True

    global_settings.backward_hooks.append(hook)
    yield
    assert hook_called[0], "Backward hook was never called."
    global_settings.backward_hooks.pop()


@contextmanager
def module_hook(hook: Callable):
    """Context manager for running given hook on forward or backward."""

    # TODO(y): maybe add checking for arg types on hook to catch forward/backward hook mismatches
    # TODO(y): use weak ref for the hook handles so they are removed when model goes out of scope
    assert global_settings.hook_handles, "Global hooks have not been registered. Make sure to call .register(model) on your model"
    forward_hook_called = [False]
    backward_hook_called = [False]

    def forward_hook(layer: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        assert len(input) == 1, "Only support single input modules on forward."
        assert type(output) == torch.Tensor, "Only support single output modules on forward."
        activations = input[0].detach()
        hook(layer, activations, output)
        forward_hook_called[0] = True

    def backward_hook(layer: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]):
        assert len(output) == 1, "Only support single output modules on backward."
        backprops = output[0].detach()
        hook(layer, input, backprops)
        backward_hook_called[0] = True

    global_settings.forward_hooks.append(forward_hook)
    global_settings.backward_hooks.append(backward_hook)
    yield
    assert forward_hook_called[0] or backward_hook_called[0], "Hook was called neither on forward nor backward pass, did you register your model?"
    assert not (forward_hook_called[0] and backward_hook_called[0]), "Hook was called both on forward and backward pass, did you register your model?"
    global_settings.forward_hooks.pop()
    global_settings.backward_hooks.pop()


@contextmanager
def save_activations2():
    """Save activations to layer storage: storage[layer] = activations
    """

    activations = {}
    def saveit(layer, A, _):
        activations[layer] = A
    with module_hook(saveit):
        yield activations


""""
Concept: backprop_func

Given output tensor, create a batch of matrices used for backprop.

Can be used to compute per-example Hessians if this function returns X where XiXi'=Hi
Examples: identity -- torch.eye for each example
Examples: cross_entropy_hessian -- Hessian of cross-entropy function
Examples: cross_entropy_rank1 -- Hessian of cross-entropy function, rank-1 approximation per example
Examples: cross_entropy_average -- average Hessian, low rank approximation
"""


def backward(tensor, backward_func, retain_graph=False):
    """Custom backprop.
    """

    vals = backward_func(tensor)
    o = len(vals)
    for idx, hess in enumerate(vals):
        tensor.backward(hess, retain_graph=(retain_graph or idx < o - 1))


def backward_accum(tensor, backward_func, storage: ModuleDict, retain_graph=False, update_gradient_norm=False):
    """
    Backpropagates from given tensor and updates FourthOrderCov for each layer in storage.


    Args:
        tensor:
        backward_func: function used to generate backward values. See "backward functions" below. Special value of 1 indicates 1 backpropagation
        storage: layer->FourthOrderCov ModuleDict storage
        retain_graph: whether to release activations at the end of forward prop
        update_gradient_norm: whether to update gradient norm squared estimates

    """

    if backward_func == 1:
        backward_func = backward_ones

    elif backward_func == 'identity':
        backward_func = backward_identity
    elif backward_func == 'xent':
        backward_func = backward_xent

    assert global_settings.hook_handles, "No hooks have been registered."
    hook_called = [False]

    def hook(layer: nn.Module, _input, output):
        backprops = output[0].detach()
        A = global_settings.default_activations
        Acov = global_settings.default_Acov
        if update_gradient_norm:
            _hack_update_gradient_norms_squared(layer, backprops)
        storage[layer].accumulate(data_x=A[layer], data_y=backprops, cached_xx=Acov)
        hook_called[0] = True

    global_settings.backward_hooks.append(hook)
    backward(tensor, backward_func, retain_graph)
    assert hook_called[0], "Backward hook was never called."
    global_settings.backward_hooks.pop()


# def backward_kron(target, tensor, A, A_cov, gradient):
#     """Calls backward, and aggrates covariance of backward values. If activations are provided, also updates cross covariance"""
#
#     def hook(module: nn.Module, _input, output):
#         """Appends all backprops (Jacobian Lops from upstream) to layer.backprops_list.
#         Using list in order to capture multiple backprop values for a single batch. Use util.clear_backprops(model)
#         to clear all saved values.
#         """
#         backprops = output[0].detach()
#         buffer[module].cov = KronFactoredCov
#         if activations:
#             activations = activations[module]
#
#         # compute covariance matrix
#
#     with backward_hook(hook):
#         tensor.backwards(gradient)


def set_default_activations(A):
    global_settings.default_activations = A

    global_settings._hack_activations_squared = ModuleDict()
    for layer in A:
        global_settings._hack_activations_squared[layer] = A[layer] * A[layer]


def set_default_Acov(Acov):
    global_settings.default_Acov = Acov


# Backward functions.
# These accept output tensor and produce a list of matrices [...,mi,...] suitable for output.backward(mi)
def backward_xent(output):
    pass


def backward_identity(tensor):
    assert u.is_matrix(tensor), "Only support rank-2 outputs."""
    n, o = tensor.shape

    hess = []
    batch_size, output_size = tensor.shape
    id_mat = u.eye(output_size)
    for out_idx in range(output_size):
        hess.append(torch.stack([id_mat[out_idx]] * batch_size))

    return hess


def backprop_identity(output, retain_graph=False) -> None:
    """
    Helper to find Jacobian with respect to given tensor. Backpropagates a row of identity matrix
    for each output of tensor. Rows are replicated across batch dimension.

    Args:
        output: target of backward
        retain_graph: same meaning as PyTorch retain_graph
    """

    assert u.is_matrix(output), "Only support rank-2 outputs."""

    n, o = output.shape
    id_mat = u.eye(o)
    for idx in range(o):
        output.backward(torch.stack([id_mat[idx]] * n), retain_graph=(retain_graph or idx < o - 1))


# TODO(y): rename to backward or backward_jacobian


def backward_ones(output):
    return [torch.ones_like(output)]

# backward_jacobian(strategy='exact')
# backward_jacobian(strategy='sampled')
# backward_hessian(loss='cross_entropy', strategy='exact')
# backward_hessian(loss='cross_entropy', strategy='sampled')


def backward_jacobian(output, sampled=False, retain_graph=False) -> None:
    """
    Helper to find Jacobian with respect to given tensor. Backpropagates a row of identity matrix
    for each output of tensor. Rows are replicated across batch dimension.

    Args:
        output: target of backward
        retain_graph: same meaning as PyTorch retain_graph
        sampled:
    """

    assert u.is_matrix(output), "Only support rank-2 outputs."""
    # assert strategy in ('exact', 'sampled')

    n, o = output.shape
    if not sampled:
        id_mat = torch.eye(o).to(gl.device)
        for idx in range(o):
            output.backward(torch.stack([id_mat[idx]] * n), retain_graph=(retain_graph or idx < o - 1))
    else:
        vals = torch.LongTensor(n, o).to(gl.device).random_(0, 2) * 2 - 1
        vals = vals.type(torch.get_default_dtype())
        vals /= o  # factor to preserve magnitudes from exact case.
        # switching to subsampling, kfac_fro became 1000x smaller, diversity became 300x larger, kfac_l2 unaffected
        output.backward(vals, retain_graph=retain_graph)


def backward_hessian(output, loss='CrossEntropy', sampled=False, retain_graph=False) -> None:
    assert loss in ('CrossEntropy',), f"Only CrossEntropy loss is supported, got {loss}"
    assert u.is_matrix(output)

    # use Cholesky-like decomposition from https://www.wolframcloud.com/obj/yaroslavvb/newton/square-root-formulas.nb
    n, o = output.shape
    p = F.softmax(output, dim=1)

    mask = torch.eye(o).to(gl.device).expand(n, o, o)
    diag_part = p.sqrt().unsqueeze(2).expand(n, o, o) * mask
    hess_sqrt = diag_part - torch.einsum('ij,ik->ijk', p.sqrt(), p)   # n, o, o

    if not sampled:
        for out_idx in range(o):
            output.backward(hess_sqrt[:, out_idx, :], retain_graph=(retain_graph or out_idx < o - 1))
    else:
        vals = torch.LongTensor(n, o).to(gl.device).random_(0, 2) * 2 - 1
        vals = vals.type(torch.get_default_dtype())/o
        mixed_vector = torch.einsum('nop,no->np', hess_sqrt, vals)
        output.backward(mixed_vector, retain_graph=retain_graph)


def grad_norms(A, B, m=None, approx='zero_order'):
    """
    Compute gradient norms squared with respect to given metric.

    "zero_order" approximation uses Euclidian metric computation (standard gradient norms), otherwise try to recover
    metric tensor from one or more of the moments defined as follows on the population activation/backprop values A, B
        m.a = einsum('ni->i', A)
        m.b = einsum('nk->k', B)
        m.AA = einsum('nij->ij', A, A)
        m.BB = einsum('nkl->kl', B, B)
        m.BA = einsum('nik->ik', B, A)
        m.BABA = einsum('nkilj->kilj', B, A, B, A)

    Args:
        A: n, d1 tensor of activationsn
        B: n, d2 tensor of backprops
        m: expected moments of covariance/curvature tensor
        approx: which approximation to use to reconstruct metric tensor out of moments.
            "zero_order":  ignore moments and use Euclidian metric
            "kfac": use m.AA, m.BB as Mijkl=Mij*Mkl
            "isserlis: use m.AA, m.BB, m.BA, m.a, m.b as Mijkl=Mij*Mkl+Mik*Mjl+Mil*Mjk-2Mi*Mj*Mk*Ml
            "full": full 4th order moment, use m.BABA

    Returns:
        (n,) tensor of per-example gradient norms squared
    """
    if approx == 'zero_order':
        norms = (A * A).sum(dim=1) * (B * B).sum(dim=1)
    elif approx == 'kfac':
        Am, Bm = A @ m.AA, B @ m.BB
        norms = (Am * A).sum(dim=1) * (Bm * B).sum(dim=1)
        # equivalent to torch.einsum('nk,ni,lk,ij,nl,nj->n', B, A, BB, AA, B, A)

    elif approx == 'isserlis':
        kfac = torch.einsum('nk,ni,lk,ij,nl,nj->n', B, A, m.BB, m.AA, B, A)
        cross1 = torch.einsum('nk,ni,ki,lj,nl,nj->n', B, A, m.BA, m.BA, B, A)
        cross2 = torch.einsum('nk,ni,li,kj,nl,nj->n', B, A, m.BA, m.BA, B, A)
        first_order = torch.einsum('nk,ni,i,j,k,l,nl,nj->n', B, A, m.a, m.a, m.b, m.b, B, A)
        norms = kfac + cross1 + cross2 - 2*first_order
    else:
        assert approx == 'full'
        norms = torch.einsum('ni,nk,nj,nl,likj->n', A, B, A, B, m.BABA)
    return norms


def offset_losses(A, B, alpha, offset, m, approx='zero_order'):
    """
    Evaluates expected improvement in loss on example i after taking gradient step from loss on example i+offset

    If alpha is None, uses optimal learning rate for minimizing i example loss by using direction of i+offset gradient

    Returns:
        (n,) tensor of improvements
    """
    if approx == 'zero_order':
        norms = (A * A).sum(dim=1) * (B * B).sum(dim=1)
    elif approx == 'kfac':
        Am, Bm = A @ m.AA, B @ m.BB
        norms = (Am * A).sum(dim=1) * (Bm * B).sum(dim=1)
        # equivalent to torch.einsum('nk,ni,lk,ij,nl,nj->n', B, A, BB, AA, B, A)
    elif approx == 'isserlis':  # TODO(y): currently this runs out of memory, optimize the einsums below 
        kfac = torch.einsum('nk,ni,lk,ij,nl,nj->n', B, A, m.BB, m.AA, B, A)
        cross1 = torch.einsum('nk,ni,ki,lj,nl,nj->n', B, A, m.BA, m.BA, B, A)
        cross2 = torch.einsum('nk,ni,li,kj,nl,nj->n', B, A, m.BA, m.BA, B, A)
        first_order = torch.einsum('nk,ni,i,j,k,l,nl,nj->n', B, A, m.a, m.a, m.b, m.b, B, A)
        norms = kfac + cross1 + cross2 - 2*first_order
    else:
        assert approx == 'full'
        norms = torch.einsum('ni,nk,nj,nl,likj->n', A, B, A, B, m.BABA)

    Ad = torch.roll(A, offset, 0)
    Bd = torch.roll(B, offset, 0)
    dot_prods = (A * Ad).sum(dim=1) * (B * Bd).sum(dim=1)
    if alpha is None:  # use optimal step for given direction
        improvements = 1/2 * dot_prods*dot_prods / norms
    else:
        improvements = alpha*dot_prods - 1/2 * alpha**2 * norms

    return improvements


def offset_cosines(A, B, offset=1):
    """
    Evaluates cosines between gradients

    If alpha is None, uses optimal learning rate for minimizing i example loss by using direction of i+offset gradient

    Returns:
        (n,) tensor of improvements
    """
    assert offset != 0

    Ad = torch.roll(A, offset, 0)
    Bd = torch.roll(B, offset, 0)
    dot_products = (A * Ad).sum(dim=1) * (B * Bd).sum(dim=1)
    norms1 = (A * A).sum(dim=1) * (B * B).sum(dim=1)
    norms2 = (Ad * Ad).sum(dim=1) * (Bd * Bd).sum(dim=1)
    cosines_squared = dot_products*dot_products/(norms1 * norms2)
    cosines = torch.sqrt(cosines_squared)
    # print('max cosine float32', max(abs(cosines)).item())
    # dot_products = dot_products.type(torch.float64)  # division by small numbers unstable, use higher precision
    # norms1 = norms1.type(torch.float64)  # division by small numbers unstable, use higher precision
    # norms2 = norms2.type(torch.float64)  # division by small numbers unstable, use higher precision
    # print('max cosine float64', max(abs(cosines)).item())
    return cosines


def offset_dotprod(A, B, offset=1):
    """
    Evaluates cosines between gradients

    If alpha is None, uses optimal learning rate for minimizing i example loss by using direction of i+offset gradient

    Returns:
        (n,) tensor of improvements
    """
    assert offset != 0

    Ad = torch.roll(A, offset, 0)
    Bd = torch.roll(B, offset, 0)
    dot_products = (A * Ad).sum(dim=1) * (B * Bd).sum(dim=1)
    return dot_products


def grad_curvs(A, B, metric):
    Am, Bm = A @ metric.AA, B @ metric.BB
    norms_before = (A * A).sum(dim=1) * (B * B).sum(dim=1)
    norms_after = (Am * A).sum(dim=1) * (Bm * B).sum(dim=1)
    return norms_after / norms_before

