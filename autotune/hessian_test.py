# Test exact Hessian computation

# import torch
import sys
from typing import Callable

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor):
        result = self.w(x)
        return result


def test_simple_hessian():
    # Compare against manual calculations in
    # https://www.wolframcloud.com/obj/yaroslavvb/newton/linear-jacobians-and-hessians.nb
    torch.set_default_dtype(torch.float32)

    d = [2, 3, 4, 2]
    n = d[0]
    c = d[-1]
    As = torch.tensor([[3, 1, -1], [1, -3, -2]]).float()
    Bs = torch.tensor([[[3, -3], [-3, -1], [-3, 3], [-3, 0]], [[2, -1], [-3, 0], [1, 1], [-2, 0]]]).float()

    # output Jacobian for first example
    Jo1 = u.kron(u.v2r(As[0]), Bs[0].t())
    u.check_equal(Jo1, [[9, -9, -9, -9, 3, -3, -3, -3, -3, 3, 3, 3], [-9, -3, 9, 0, -3, -1, 3, 0, 3, 1, -3, 0]])

    # batch output Jacobian
    Jb = torch.cat([u.kron(u.v2r(As[i]), Bs[i].t()) for i in range(n)])
    u.check_equal(Jb, [[9, -9, -9, -9, 3, -3, -3, -3, -3, 3, 3, 3], [-9, -3, 9, 0, -3, -1, 3, 0, 3, 1, -3, 0],
                       [2, -3, 1, -2, -6, 9, -3, 6, -4, 6, -2, 4], [-1, 0, 1, 0, 3, 0, -3, 0, 2, 0, -2, 0]])

    W = torch.nn.Parameter(torch.ones((d[2], d[1])))

    def loss(i):
        residuals = Bs[i].t() @ W @ u.v2c(As[i])
        return 0.5 * torch.sum(residuals * residuals)

    u.check_equal(loss(0), 333 / 2)

    # check against PyTorch autograd
    i = 0
    outputs = Bs[i].t() @ W @ u.v2c(As[i])
    jac = u.jacobian(outputs, W)

    u.check_equal(Jo1, jac.transpose(0, 1).transpose(2, 3).reshape((c, -1)))

    Jb = torch.cat([u.kron(u.v2r(As[i]), Bs[i].t()) for i in range(n)])
    manualHess = Jb.t() @ Jb
    u.check_equal(manualHess, [[167, -60, -161, -85, 39, 0, -57, -15, -64, 30, 52, 35],
                               [-60, 99, 51, 87, 0, 3, 27, 9, 30, -48, -12, -39],
                               [-161, 51, 164, 79, -57, 27, 48, 33, 52, -12, -58, -23],
                               [-85, 87, 79, 85, -15, 9, 33, 15, 35, -39, -23, -35],
                               [39, 0, -57, -15, 63, -60, -9, -45, 12, -30, 24, -15],
                               [0, 3, 27, 9, -60, 91, -21, 63, -30, 44, -24, 27],
                               [-57, 27, 48, 33, -9, -21, 36, -9, 24, -24, -6, -21],
                               [-15, 9, 33, 15, -45, 63, -9, 45, -15, 27, -21, 15],
                               [-64, 30, 52, 35, 12, -30, 24, -15, 38, -30, -14, -25],
                               [30, -48, -12, -39, -30, 44, -24, 27, -30, 46, -6, 33],
                               [52, -12, -58, -23, 24, -24, -6, -21, -14, -6, 26, 1],
                               [35, -39, -23, -35, -15, 27, -21, 15, -25, 33, 1, 25]])

    total_loss = torch.add(*[loss(i) for i in range(n)])
    u.check_equal(total_loss, 397 / 2)

    automaticHess = u.hessian(total_loss, W)
    automaticHess = automaticHess.transpose(0, 1).transpose(2, 3).reshape((d[1] * d[2], d[1] * d[2]))
    u.check_equal(automaticHess, manualHess)

    # Note: layers have dimensions (in, out), but the matrices have shape (out, in)
    layer = nn.Linear(d[1], d[2], bias=False)
    Blayer = nn.Linear(d[2], d[3], bias=False)
    model = torch.nn.Sequential(layer, nn.ReLU(), Blayer)
    layer.weight.data.copy_(torch.ones((d[2], d[1])))
    Blayer.weight.data.copy_(Bs[0].t())
    u.check_close(model(As[0]), [-18., -3.])


import autograd_lib
import globals as gl
# import torch
import torch
import util as u
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter


def test_explicit_hessian():
    """Check computation of hessian of loss(B'WA) from https://github.com/yaroslavvb/kfac_pytorch/blob/master/derivation.pdf


    """

    torch.set_default_dtype(torch.float64)
    A = torch.tensor([[-1., 4], [3, 0]])
    B = torch.tensor([[-4., 3], [2, 6]])
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)

    Y = B.t() @ X @ A
    u.check_equal(Y, [[-52, 64], [-81, -108]])
    loss = torch.sum(Y * Y) / 2
    hess0 = u.hessian(loss, X).reshape([4, 4])
    hess1 = u.Kron(A @ A.t(), B @ B.t())

    u.check_equal(loss, 12512.5)

    # PyTorch autograd computes Hessian with respect to row-vectorized parameters, whereas
    # autograd_lib uses math convention and does column-vectorized.
    # Commuting order of Kronecker product switches between two representations
    u.check_equal(hess1.commute(), hess0)

    # Do a test using Linear layers instead of matrix multiplies
    model: u.SimpleFullyConnected2 = u.SimpleFullyConnected2([2, 2, 2], bias=False)
    model.layers[0].weight.data.copy_(X)

    # Transpose to match previous results, layers treat dim0 as batch dimension
    u.check_equal(model.layers[0](A.t()).t(), [[5, -20], [-16, -8]])  # XA = (A'X0)'

    model.layers[1].weight.data.copy_(B.t())
    u.check_equal(model(A.t()).t(), Y)

    Y = model(A.t()).t()    # transpose to data-dimension=columns
    loss = torch.sum(Y * Y) / 2
    loss.backward()

    u.check_equal(model.layers[0].weight.grad, [[-2285, -105], [-1490, -1770]])
    G = B @ Y @ A.t()
    u.check_equal(model.layers[0].weight.grad, G)

    u.check_equal(hess0, u.Kron(B @ B.t(), A @ A.t()))

    # compute newton step
    u.check_equal(u.Kron(A@A.t(), B@B.t()).pinv() @ u.vec(G), u.v2c([-5, -2, 0, -6]))

    # compute Newton step using factored representation
    autograd_lib.add_hooks(model)

    Y = model(A.t())
    n = 2
    loss = torch.sum(Y * Y) / 2
    autograd_lib.backprop_hess(Y, hess_type='LeastSquares')
    autograd_lib.compute_hess(model, method='kron', attr_name='hess_kron', vecr_order=False, loss_aggregation='sum')
    param = model.layers[0].weight

    hess2 = param.hess_kron
    print(hess2)

    u.check_equal(hess2, [[425, 170, -75, -30], [170, 680, -30, -120], [-75, -30, 225, 90], [-30, -120, 90, 360]])

    # Gradient test
    model.zero_grad()
    loss.backward()
    u.check_close(u.vec(G).flatten(), u.Vec(param.grad))

    # Newton step test
    # Method 0: PyTorch native autograd
    newton_step0 = param.grad.flatten() @ torch.pinverse(hess0)
    newton_step0 = newton_step0.reshape(param.shape)
    u.check_equal(newton_step0, [[-5, 0], [-2, -6]])

    # Method 1: colummn major order
    ihess2 = hess2.pinv()
    u.check_equal(ihess2.LL, [[1/16, 1/48], [1/48, 17/144]])
    u.check_equal(ihess2.RR, [[2/45, -(1/90)], [-(1/90), 1/36]])
    u.check_equal(torch.flatten(hess2.pinv() @ u.vec(G)), [-5, -2, 0, -6])
    newton_step1 = (ihess2 @ u.Vec(param.grad)).matrix_form()

    # Method2: row major order
    ihess2_rowmajor = ihess2.commute()
    newton_step2 = ihess2_rowmajor @ u.Vecr(param.grad)
    newton_step2 = newton_step2.matrix_form()

    u.check_equal(newton_step0, newton_step1)
    u.check_equal(newton_step0, newton_step2)


def test_factored_hessian():
    """"Simple test to ensure Hessian computation is working.

    In a linear neural network with squared loss, Newton step will converge in one step.
    Compute stats after minimizing, pass sanity checks.
    """

    u.seed_random(1)
    loss_type = 'LeastSquares'

    data_width = 2
    n = 5
    d1 = data_width ** 2
    o = 10
    d = [d1, o]

    model = u.SimpleFullyConnected2(d, bias=False, nonlin=False)
    model = model.to(gl.device)
    print(model)

    dataset = u.TinyMNIST(data_width=data_width, dataset_size=n, loss_type=loss_type)
    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    stats_iter = u.infinite_iter(stats_loader)
    stats_data, stats_targets = next(stats_iter)

    if loss_type == 'LeastSquares':
        loss_fn = u.least_squares
    else:  # loss_type == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()

    autograd_lib.add_hooks(model)
    gl.reset_global_step()
    last_outer = 0

    data, targets = stats_data, stats_targets

    # Capture Hessian and gradient stats
    autograd_lib.enable_hooks()
    autograd_lib.clear_backprops(model)

    output = model(data)
    loss = loss_fn(output, targets)
    print(loss)
    loss.backward(retain_graph=True)
    layer = model.layers[0]

    autograd_lib.clear_hess_backprops(model)
    autograd_lib.backprop_hess(output, hess_type=loss_type)
    autograd_lib.disable_hooks()

    # compute Hessian using direct method, compare against PyTorch autograd
    hess0 = u.hessian(loss, layer.weight)
    autograd_lib.compute_hess(model)
    hess1 = layer.weight.hess
    print(hess1)
    u.check_close(hess0.reshape(hess1.shape), hess1, atol=1e-9, rtol=1e-6)

    # compute Hessian using factored method
    autograd_lib.compute_hess(model, method='kron', attr_name='hess2', vecr_order=True)
    # s.regret_newton = vecG.t() @ pinvH.commute() @ vecG.t() / 2  # TODO(y): figure out why needed transposes

    hess2 = layer.weight.hess2
    u.check_close(hess1, hess2, atol=1e-9, rtol=1e-6)

    # Newton step in regular notation
    g1 = layer.weight.grad.flatten()
    newton1 = hess1 @ g1

    g2 = u.Vecr(layer.weight.grad)
    newton2 = g2 @ hess2

    u.check_close(newton1, newton2, atol=1e-9, rtol=1e-6)

    # compute regret in factored notation, compare against actual drop in loss
    regret1 = g1 @ hess1.pinverse() @ g1 / 2
    regret2 = g2 @ hess2.pinv() @ g2 / 2
    u.check_close(regret1, regret2)

    current_weight = layer.weight.detach().clone()
    param: torch.nn.Parameter = layer.weight
    # param.data.sub_((hess1.pinverse() @ g1).reshape(param.shape))
    # output = model(data)
    # loss = loss_fn(output, targets)
    # print("result 1", loss)

    # param.data.sub_((hess1.pinverse() @ u.vec(layer.weight.grad)).reshape(param.shape))
    # output = model(data)
    # loss = loss_fn(output, targets)
    # print("result 2", loss)

    # param.data.sub_((u.vec(layer.weight.grad).t() @ hess1.pinverse()).reshape(param.shape))
    # output = model(data)
    # loss = loss_fn(output, targets)
    # print("result 3", loss)
    #

    del layer.weight.grad
    output = model(data)
    loss = loss_fn(output, targets)
    loss.backward()
    param.data.sub_(u.unvec(hess1.pinverse() @ u.vec(layer.weight.grad), layer.weight.shape[0]))
    output = model(data)
    loss = loss_fn(output, targets)
    print("result 4", loss)

    # param.data.sub_((g1 @ hess1.pinverse() @ g1).reshape(param.shape))

    print(loss)


def test_hessian_multibatch():
    """Test that Kronecker-factored computations still work when splitting work over batches."""

    u.seed_random(1)

    # torch.set_default_dtype(torch.float64)

    gl.project_name = 'test'
    gl.logdir_base = '/tmp/runs'
    run_name = 'test_hessian_multibatch'
    u.setup_logdir_and_event_writer(run_name=run_name)

    loss_type = 'CrossEntropy'
    data_width = 2
    n = 4
    d1 = data_width ** 2
    o = 10
    d = [d1, o]

    model = u.SimpleFullyConnected2(d, bias=False, nonlin=False)
    model = model.to(gl.device)

    dataset = u.TinyMNIST(data_width=data_width, dataset_size=n, loss_type=loss_type)
    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    stats_iter = u.infinite_iter(stats_loader)

    if loss_type == 'LeastSquares':
        loss_fn = u.least_squares
    else:  # loss_type == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()

    autograd_lib.add_hooks(model)
    gl.reset_global_step()
    last_outer = 0

    stats_iter = u.infinite_iter(stats_loader)
    stats_data, stats_targets = next(stats_iter)
    data, targets = stats_data, stats_targets

    # Capture Hessian and gradient stats
    autograd_lib.enable_hooks()
    autograd_lib.clear_backprops(model)

    output = model(data)
    loss = loss_fn(output, targets)
    loss.backward(retain_graph=True)
    layer = model.layers[0]

    autograd_lib.clear_hess_backprops(model)
    autograd_lib.backprop_hess(output, hess_type=loss_type)
    autograd_lib.disable_hooks()

    # compute Hessian using direct method, compare against PyTorch autograd
    hess0 = u.hessian(loss, layer.weight)
    autograd_lib.compute_hess(model)
    hess1 = layer.weight.hess
    u.check_close(hess0.reshape(hess1.shape), hess1, atol=1e-8, rtol=1e-6)

    # compute Hessian using factored method. Because Hessian depends on examples for cross entropy, factoring is not exact, raise tolerance
    autograd_lib.compute_hess(model, method='kron', attr_name='hess2', vecr_order=True)
    hess2 = layer.weight.hess2
    u.check_close(hess1, hess2, atol=1e-3, rtol=1e-1)

    # compute Hessian using multibatch
    # restart iterators
    dataset = u.TinyMNIST(data_width=data_width, dataset_size=n, loss_type=loss_type)
    assert n % 2 == 0
    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=n//2, shuffle=False)
    stats_iter = u.infinite_iter(stats_loader)
    autograd_lib.compute_cov(model, loss_fn, stats_iter, batch_size=n//2, steps=2)

    cov: autograd_lib.LayerCov = layer.cov
    hess2: u.Kron = hess2.commute()    # get back into AA x BB order
    u.check_close(cov.H.value(), hess2)


def _test_refactored_stats():
    gl.project_name = 'test'
    gl.logdir_base = '/tmp/runs'
    run_name = 'test_hessian_multibatch'
    u.setup_logdir_and_event_writer(run_name=run_name)

    loss_type = 'CrossEntropy'
    data_width = 2
    n = 4
    d1 = data_width ** 2
    o = 10
    d = [d1, o]

    model = u.SimpleFullyConnected2(d, bias=False, nonlin=False)
    model = model.to(gl.device)

    dataset = u.TinyMNIST(data_width=data_width, dataset_size=n, loss_type=loss_type)
    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    stats_iter = u.infinite_iter(stats_loader)

    if loss_type == 'LeastSquares':
        loss_fn = u.least_squares
    else:  # loss_type == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()

    autograd_lib.add_hooks(model)
    gl.reset_global_step()
    last_outer = 0

    stats_iter = u.infinite_iter(stats_loader)
    stats_data, stats_targets = next(stats_iter)
    data, targets = stats_data, stats_targets

    covG = autograd_lib.layer_cov_dict()
    covH = autograd_lib.layer_cov_dict()
    covJ = autograd_lib.layer_cov_dict()

    autograd_lib.register(model)

    A = {}
    with autograd_lib.save_activations(A):
        output = model(data)
        loss = loss_fn(output, targets)

    Acov = autograd_lib.ModuleDict(autograd_lib.SecondOrder)
    for layer, activations in A.items():
        Acov[layer].accumulate(activations)

    autograd_lib.set_default_activations(A)   # set activations to use by default when constructing cov matrices
    autograd_lib.set_default_Acov(Acov)

    # saves backprop covariances
    autograd_lib.backward_accum(loss, 1, covG)
    autograd_lib.backward_accum(output, autograd_lib.xent_bwd, covH)
    autograd_lib.backward_accum(output, autograd_lib.identity_bwd, covJ)

    #grad_cov = KronFactored(covA, covG.cov, covG.cross)
    #hess = KronFactored(covA, covH.cov, covH.cross)
    #grad_cov = KronFactored(covA, covJ.cov, covJ.cross)


def test_hessian_conv():
    """Test conv hessian computation using factored and regular method."""

    u.seed_random(1)
    unfold = torch.nn.functional.unfold
    fold = torch.nn.functional.fold

    import numpy as np

    u.seed_random(1)
    N, Xc, Xh, Xw = 3, 2, 3, 7
    dd = [Xc, 2]

    Kh, Kw = 2, 3
    Oh, Ow = Xh - Kh + 1, Xw - Kw + 1
    model = u.SimpleConvolutional(dd, kernel_size=(Kh, Kw), bias=True).double()

    weight_buffer = model.layers[0].weight.data

    # output channels, input channels, height, width
    assert weight_buffer.shape == (dd[1], dd[0], Kh, Kw)

    input_dims = N, Xc, Xh, Xw
    size = int(np.prod(input_dims))
    X = torch.arange(0, size).reshape(*input_dims).double()

    def loss_fn(data):
        err = data.reshape(len(data), -1)
        return torch.sum(err * err) / 2 / len(data)

    layer = model.layers[0]
    output = model(X)
    loss = loss_fn(output)
    loss.backward()

    u.check_equal(layer.activations, X)

    assert layer.backprops_list[0].shape == layer.output.shape
    assert layer.output.shape == (N, dd[1], Oh, Ow)

    out_unf = layer.weight.view(layer.weight.size(0), -1) @ unfold(layer.activations, (Kh, Kw))
    assert out_unf.shape == (N, dd[1], Oh * Ow)
    reshaped_bias = layer.bias.reshape(1, dd[1], 1)  # (Co,) -> (1, Co, 1)
    out_unf = out_unf + reshaped_bias

    u.check_equal(fold(out_unf, (Oh, Ow), (1, 1)), output)  # two alternative ways of reshaping
    u.check_equal(out_unf.view(N, dd[1], Oh, Ow), output)

    # Unfold produces patches with output dimension merged, while in backprop they are not merged
    # Hence merge the output (width/height) dimension
    assert unfold(layer.activations, (Kh, Kw)).shape == (N, Xc * Kh * Kw, Oh * Ow)
    assert layer.backprops_list[0].shape == (N, dd[1], Oh, Ow)

    grads_bias = layer.backprops_list[0].sum(dim=(2, 3)) * N
    mean_grad_bias = grads_bias.sum(dim=0) / N
    u.check_equal(mean_grad_bias, layer.bias.grad)

    Bt = layer.backprops_list[0] * N   # remove factor of N applied during loss batch averaging
    assert Bt.shape == (N, dd[1], Oh, Ow)
    Bt = Bt.reshape(N, dd[1], Oh*Ow)
    At = unfold(layer.activations, (Kh, Kw))
    assert At.shape == (N, dd[0] * Kh * Kw, Oh*Ow)

    grad_unf = torch.einsum('ijk,ilk->ijl', Bt, At)
    assert grad_unf.shape == (N, dd[1], dd[0] * Kh * Kw)

    grads = grad_unf.reshape((N, dd[1], dd[0], Kh, Kw))
    u.check_equal(grads.mean(dim=0), layer.weight.grad)

    # compute per-example gradients using autograd, compare against manual computation
    for i in range(N):
        u.clear_backprops(model)
        output = model(X[i:i + 1, ...])
        loss = loss_fn(output)
        loss.backward()
        u.check_equal(grads[i], layer.weight.grad)
        u.check_equal(grads_bias[i], layer.bias.grad)


def _test_explicit_hessian_refactored():

    """Check computation of hessian of loss(B'WA) from https://github.com/yaroslavvb/kfac_pytorch/blob/master/derivation.pdf


    """

    torch.set_default_dtype(torch.float64)
    A = torch.tensor([[-1., 4], [3, 0]])
    B = torch.tensor([[-4., 3], [2, 6]])
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)

    Y = B.t() @ X @ A
    u.check_equal(Y, [[-52, 64], [-81, -108]])
    loss = torch.sum(Y * Y) / 2
    hess0 = u.hessian(loss, X).reshape([4, 4])
    hess1 = u.Kron(A @ A.t(), B @ B.t())

    u.check_equal(loss, 12512.5)

    # Do a test using Linear layers instead of matrix multiplies
    model: u.SimpleFullyConnected2 = u.SimpleFullyConnected2([2, 2, 2], bias=False)
    model.layers[0].weight.data.copy_(X)

    # Transpose to match previous results, layers treat dim0 as batch dimension
    u.check_equal(model.layers[0](A.t()).t(), [[5, -20], [-16, -8]])  # XA = (A'X0)'

    model.layers[1].weight.data.copy_(B.t())
    u.check_equal(model(A.t()).t(), Y)

    Y = model(A.t()).t()    # transpose to data-dimension=columns
    loss = torch.sum(Y * Y) / 2
    loss.backward()

    u.check_equal(model.layers[0].weight.grad, [[-2285, -105], [-1490, -1770]])
    G = B @ Y @ A.t()
    u.check_equal(model.layers[0].weight.grad, G)

    autograd_lib.register(model)
    activations_dict = autograd_lib.ModuleDict()  # todo(y): make save_activations ctx manager automatically create A
    with autograd_lib.save_activations(activations_dict):
        Y = model(A.t())

    Acov = autograd_lib.ModuleDict(autograd_lib.SecondOrderCov)
    for layer, activations in activations_dict.items():
        print(layer, activations)
        Acov[layer].accumulate(activations, activations)
    autograd_lib.set_default_activations(activations_dict)
    autograd_lib.set_default_Acov(Acov)

    B = autograd_lib.ModuleDict(autograd_lib.SymmetricFourthOrderCov)
    autograd_lib.backward_accum(Y, "identity", B, retain_graph=False)

    print(B[model.layers[0]])

    autograd_lib.backprop_hess(Y, hess_type='LeastSquares')
    autograd_lib.compute_hess(model, method='kron', attr_name='hess_kron', vecr_order=False, loss_aggregation='sum')
    param = model.layers[0].weight

    hess2 = param.hess_kron
    print(hess2)

    u.check_equal(hess2, [[425, 170, -75, -30], [170, 680, -30, -120], [-75, -30, 225, 90], [-30, -120, 90, 360]])

    # Gradient test
    model.zero_grad()
    loss.backward()
    u.check_close(u.vec(G).flatten(), u.Vec(param.grad))

    # Newton step test
    # Method 0: PyTorch native autograd
    newton_step0 = param.grad.flatten() @ torch.pinverse(hess0)
    newton_step0 = newton_step0.reshape(param.shape)
    u.check_equal(newton_step0, [[-5, 0], [-2, -6]])

    # Method 1: colummn major order
    ihess2 = hess2.pinv()
    u.check_equal(ihess2.LL, [[1/16, 1/48], [1/48, 17/144]])
    u.check_equal(ihess2.RR, [[2/45, -(1/90)], [-(1/90), 1/36]])
    u.check_equal(torch.flatten(hess2.pinv() @ u.vec(G)), [-5, -2, 0, -6])
    newton_step1 = (ihess2 @ u.Vec(param.grad)).matrix_form()

    # Method2: row major order
    ihess2_rowmajor = ihess2.commute()
    newton_step2 = ihess2_rowmajor @ u.Vecr(param.grad)
    newton_step2 = newton_step2.matrix_form()

    u.check_equal(newton_step0, newton_step1)
    u.check_equal(newton_step0, newton_step2)


def _test_new_setup():
    torch.set_default_dtype(torch.float64)
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)

    model: u.SimpleFullyConnected2 = u.SimpleFullyConnected2([2, 2, 2], bias=False)
    model.layers[0].weight.data.copy_(X)

    A = torch.tensor([[-1., 4], [3, 0]])
    B = torch.tensor([[-4., 3], [2, 6]])
    X = torch.tensor([[-5., 0], [-2, -6]], requires_grad=True)

    #########
    # computing per-example gradients
    #########
    activations = {}
    def save_activations(layer, a, _): activations[layer] = a
    with autograd_lib.module_hooks(save_activations):
        Y = model(A.t())
        loss = torch.sum(Y * Y) / 2

    norms = {}
    def compute_norms(layer, _, b):
        a = activations[layer]
        del activations[layer]
        norms[layer] = a*a.sum(dim=0)*(b*b).sum(dim=0)

    with autograd_lib.module_hooks(compute_norms):
        loss.backward()

    #########
    # Computing higher rank Hessian approximation
    #########
    activations = {}
    with autograd_lib.module_hooks(save_activations):
        Y = model(A.t())
        loss = torch.sum(Y * Y) / 2

    # kfac moments: 'ij', 'kl'
    # isserlis moments: 'ij', 'kl', 'il', 'ik'
    # first moments: 'i', 'j', 'k', 'l'
    # third moments: ...
    # forth moments: ...
    # moment_dict = {'Ai': Buffer, 'Bk': Buffer, 'AiAj': Buffer, 'BkBl': Buffer, 'AiBk': Buffer}
    # moments_dict = MomentsDict('A', 'AA', 'B', 'BB', 'AB', 'diag')
    # moments_dict = MomentsDict('i', 'ij', 'k', 'kl', 'jk', 'iikk')
    moment_dict = MomentsDict(['Ai', 'Bk', 'AiAj', 'BkBl', 'AiBk', 'AiAiBkBk', 'AiAjBkBl'])
    def accumulate_moments(layer, _, b):
        a = activations[layer]
        util.accumulate_moments(moment_dict, a, b)

    with autograd_lib.module_hooks(accumulate_moments):
        pass




if __name__ == '__main__':
    #  _test_factored_hessian()
    # test_hessian_multibatch()
    # test_hessian_conv()
    # test_explicit_hessian_refactored()
    #    u.run_all_tests(sys.modules[__name__])

    u.run_all_tests(sys.modules[__name__])
