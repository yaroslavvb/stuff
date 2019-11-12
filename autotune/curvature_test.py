# Prototype batch-size quantities from
# Batch size formulas (https://docs.google.com/document/d/19Jmh4spbSAnAGX_eq7WSFPgLzrpJEhiZRpjX1jSYObo/edit)
import os
import sys
from typing import Optional, Tuple, Callable

# import torch
import torch.nn as nn
from torchcurv.optim import SecondOrderOptimizer

from util import *


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor):
        result = self.w(x)
        return result


def test_singlelayer():
    # Reproduce Linear Regression example
    # https://www.wolframcloud.com/obj/yaroslavvb/newton/curvature-unit-tests.nb

    torch.set_default_dtype(torch.float32)

    d = 2
    n = 3
    model = Net(d)

    w0 = torch.tensor([[1, 2]]).float()
    assert w0.shape[1] == d
    model.w.weight.data.copy_(w0)

    X = torch.tensor([[-2, 0, 2], [-1, 1, 3]]).float()
    assert X.shape[0] == d
    assert X.shape[1] == n

    Y = torch.tensor([[0, 1, 2]]).float()
    assert Y.shape[1] == X.shape[1]

    data = X.t()  # PyTorch expects batch dimension first
    target = Y.t()
    assert data.shape[0] == n

    output = model(data)
    # residuals, aka e
    residuals = output - Y.t()

    def compute_loss(residuals_):
        return torch.sum(residuals_ * residuals_) / (2 * n)

    loss = compute_loss(residuals)

    assert loss - 8.83333 < 1e-5, torch.norm(loss) - 8.83333

    # use learning rate 0 to avoid changing parameter vector
    optim_kwargs = dict(lr=0, momentum=0, weight_decay=0, l2_reg=0,
                        bias_correction=False, acc_steps=1,
                        curv_type="Cov", curv_shapes={"Linear": "Kron"},
                        momentum_type="preconditioned", )
    curv_args = dict(damping=1, ema_decay=1)  # todo: damping
    optimizer = SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=curv_args)

    def backward(last_layer: str) -> Callable:
        """Creates closure that backpropagates either from output layer or from loss layer"""

        def closure() -> Tuple[Optional[torch.Tensor], torch.Tensor]:
            optimizer.zero_grad()
            output = model(data)
            if last_layer == "output":
                output.backward(torch.ones_like(target))
                return None, output
            elif last_layer == 'loss':
                loss = compute_loss(output - target)
                loss.backward()
                return loss, output
            else:
                assert False, 'last layer must be "output" or "loss"'

        return closure

    #    loss = compute_loss(output - Y.t())
    #    loss.backward()

    loss, output = optimizer.step(closure=backward('loss'))
    check_equal(output.t(), [[-4, 2, 8]])
    check_equal(residuals.t(), [[-4, 1, 6]])
    check_equal(loss, 8.833333)

    # batch output Jacobian
    J = X.t()
    check_close(J, [[-2, -1], [0, 1], [2, 3]])

    # matrix of activations, (n, d)
    A = model.w.data_input
    check_close(A, J)

    # matrix of backprops, add factor n to remove dependence on batch-size
    B = model.w.grad_output * n
    check_close(B, residuals)

    # gradients, n,d
    # method 1, manual computation
    G = residuals.repeat(1, d) * J
    check_close(G, [[8., 4.], [0., 1.], [12., 18.]])

    # method 2, get them of activation + backprop values
    check_close(G, khatri_rao_t(A, B))

    # method 3, PyTorch autograd
    # (n,) losses vector
    losses = torch.stack([compute_loss(r) for r in residuals])
    # batch-loss jacobian
    G2 = jacobian(losses, model.w.weight) * n
    # per-example gradients are row-matrices, squeeze to stack them into a single matrix
    G2 = G2.squeeze(1)
    check_close(G2, G)

    # mean gradient
    g = G.sum(dim=0) / n
    check_close(g, [6.66667, 7.66667])

    # empirical Fisher
    efisher = G.t() @ G / n
    check_close(efisher, [[69.3333, 82.6667], [82.6667, 113.667]])

    # centered empirical Fisher (Sigma in OpenAI paper, estimate of Sigma in Jain paper)
    sigma = efisher - outer(g, g)
    check_close(sigma, [[24.8889, 31.5556], [31.5556, 54.8889]])

    # loss
    loss2 = (residuals * residuals).sum() / (2 * n)
    check_close(to_python_scalar(loss2), 8.83333)

    ####################################################################
    # Hessian
    ####################################################################

    # method 1, manual computation
    H = J.t() @ J / n
    check_close(H, [[2.66667, 2.66667], [2.66667, 3.66667]])

    # method 2, using activation + backprop values
    check_close(A.t() @ torch.eye(n) @ A / n, H)

    # method 3, PyTorch backprop
    hess = hessian(compute_loss(residuals), model.w.weight)
    hess = hess.squeeze(2)   # TODO(y): replace with transpose like in multilayer test
    hess = hess.squeeze(0)
    check_close(hess, H)

    sigma_norm = torch.norm(sigma)
    g_norm = torch.norm(g)

    g_ = g.unsqueeze(0)  # turn g into row matrix

    # predicted drop in loss if we take a Newton step
    excess = to_python_scalar(g_ @ H.inverse() @ g_.t() / 2)
    check_close(excess, 8.83333)

    def loss_direction(direction, eps):
        """loss improvement if we take step eps in direction dir"""
        return to_python_scalar(eps * (direction @ g.t()) - 0.5 * eps ** 2 * direction @ H @ direction.t())

    newtonImprovement = loss_direction(g_ @ H.inverse(), 1)
    check_close(newtonImprovement, 8.83333)

    ############################
    # OpenAI quantities
    grad_curvature = to_python_scalar(g_ @ H @ g_.t())  # curvature in direction of g
    stepOpenAI = to_python_scalar(g.norm() ** 2 / grad_curvature) if g_norm else 999
    check_close(stepOpenAI, 0.170157)
    batchOpenAI = to_python_scalar(torch.trace(H @ sigma) / grad_curvature) if g_norm else 999
    check_close(batchOpenAI, 0.718603)

    # improvement in loss when we take gradient step with optimal learning rate
    gradientImprovement = loss_direction(g_, stepOpenAI)
    assert newtonImprovement > gradientImprovement
    check_close(gradientImprovement, 8.78199)

    ############################
    # Gradient diversity  quantities
    diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2
    check_close(diversity, 5.31862)

    ############################
    # Jain/Kakade quantities

    # noise scale (Jain, minimax rate of estimator)
    noise_variance = torch.trace(H.inverse() @ sigma)
    check_close(noise_variance, 26.)

    isqrtH = pinv_square_root(H)
    # measure of misspecification between model and actual noise (Jain, \rho)
    # formula (3) of "Parallelizing Stochastic Gradient Descent"
    p_sigma = (kron(H, torch.eye(d)) + kron(torch.eye(d), H)).inverse() @ vec(sigma)
    p_sigma = unvec(p_sigma, d)
    rho = d / erank(p_sigma) if sigma_norm > 0 else 1
    check_close(rho, 1.21987)

    # use new method with Lyapunov factoring
    p_sigma2 = lyapunov_svd(H, sigma)
    rho2 = d / erank(p_sigma2)
    check_close(rho2, 1.21987)

    rhoSimple = (d / erank(isqrtH @ sigma @ isqrtH)) if sigma_norm > 0 else 1
    check_close(rhoSimple, 1.4221)
    assert 1 <= rho <= d, rho

    # divergent learning rate for batch-size 1 (Jain). Approximates max||x_i|| with avg.
    # For more accurate results may want to add stddev of ||x_i||
    # noinspection PyTypeChecker
    stepMin = 2 / torch.trace(H)
    check_close(stepMin, 0.315789)

    # divergent learning rate for batch-size infinity
    stepMax = 2 / l2_norm(H)
    check_close(stepMax, 0.340147)

    # divergent learning rate for batch-size 1, adjusted for misspecification
    check_close(stepMin / rhoSimple, 0.222058)
    check_close(stepMin / rho, 0.258871)

    # batch size that gives provides lr halfway between stepMin and stepMax
    batchJain = 1 + erank(H)
    check_close(batchJain, 2.07713)

    # batch size that provides halfway point after adjusting for misspecification
    check_close(1 + erank(H) * rhoSimple, 2.5318)
    check_close(1 + erank(H) * rho, 2.31397)


class Net2(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        self.W = nn.Linear(d1, d2, bias=False)
        self.X2t = nn.Linear(d2, 1, bias=False)

    def forward(self, X1: torch.Tensor):
        result = self.W(X1)
        result = self.X2t(result)
        return result


def test_multilayer():
    # Reproduce multilayer example
    # https://www.wolframcloud.com/obj/yaroslavvb/newton/curvature-unit-tests.nb

    torch.set_default_dtype(torch.float64)

    d1 = 2
    d2 = 4
    n = 3
    model = Net2(d1, d2)

    W0 = u.to_pytorch([[3, 3], [0, 3], [1, -1], [-3, 1]])
    model.W.weight.data.copy_(W0)
    X2 = u.to_pytorch([[1], [-2], [-1], [3]])
    assert X2.shape == (d2, 1)
    model.X2t.weight.data.copy_(X2.t())

    X1 = u.to_pytorch([[2, -2, 3], [-3, 1, -3]])
    assert X1.shape == (d1, n)

    Y = u.to_pytorch([[-2, -3, 0]])
    assert Y.shape == (1, n)

    data = X1.t()  # PyTorch expects batch dimension first
    target = Y.t()
    assert data.shape[0] == n

    output = model(data)
    # residuals, aka e
    residuals = output - Y.t()

    def compute_loss(residuals_):
        return torch.sum(residuals_ * residuals_) / (2 * n)

    loss = compute_loss(residuals)
    assert loss - 187.5 < 1e-5, torch.norm(loss) - 8.83333

    # use learning rate 0 to avoid changing parameter vector
    optim_kwargs = dict(lr=0, momentum=0, weight_decay=0, l2_reg=0,
                        bias_correction=False, acc_steps=1,
                        curv_type="Cov", curv_shapes={"Linear": "Kron"},
                        momentum_type="preconditioned", update_inv=False, precondition_grad=False)
    curv_args = dict(damping=0, ema_decay=1)
    optimizer = SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=curv_args)

    # def set_requires_grad(v):
    #     for p in model.parameters():
    #         p.requires_grad = False
    #
    def backward(last_layer: str) -> Callable:
        """Creates closure that backpropagates either from output layer or from loss layer"""

        def closure() -> Tuple[Optional[torch.Tensor], torch.Tensor]:
            optimizer.zero_grad()
            output = model(data)
            if last_layer == "output":
                output.backward(torch.ones_like(target))
                return None, output
            elif last_layer == 'loss':
                loss = compute_loss(output - target)
                loss.backward()
                return loss, output
            else:
                assert False, 'last layer must be "output" or "loss"'

        return closure

    #    loss = compute_loss(output - Y.t())
    #    loss.backward()

    loss, output = optimizer.step(closure=backward('loss'))
    check_close(output.t(), [[-17, 15, -24]])
    check_close(residuals.t(), [[-15, 18, -24]])
    check_close(loss, 187.5)

    # batch output Jacobian, n rows, i'th row gives sensitivity of i'th output example to parameters
    J = kron(X1, X2).t()
    assert J.shape == (n, d1 * d2)
    check_close(J, [[2, -4, -2, 6, -3, 6, 3, -9], [-2, 4, 2, -6, 1, -2, -1, 3], [3, -6, -3, 9, -3, 6, 3, -9]])

    # matrix of activations, (n, d1)
    At = model.W.data_input
    A = At.t()
    check_close(At, X1.t())

    # matrix of backprops, add factor n to remove dependence on batch-size
    Bt = model.W.grad_output * n
    check_close(Bt, [[-15, 30, 15, -45], [18, -36, -18, 54], [-24, 48, 24, -72]])

    # gradients, n,d
    # mean gradient, 1, d
    # method 1, manual computation
    G = khatri_rao_t(At, Bt)
    assert G.shape == (n, d1 * d2)
    check_close(G, [[-30, 60, 30, -90, 45, -90, -45, 135], [-36, 72, 36, -108, 18, -36, -18, 54],
                    [-72, 144, 72, -216, 72, -144, -72, 216]])

    g = G.sum(dim=0, keepdim=True) / n
    check_close(g, [[-46, 92, 46, -138, 45, -90, -45, 135]])
    check_close(g, vec(get_param(model.W).grad).t())

    # method 2, explicit PyTorch autograd
    # (n,) losses vector
    losses = torch.stack([compute_loss(r) for r in residuals])
    # batch-loss jacobian
    G2 = jacobian(losses, model.W.weight) * n
    # per-example gradients are row-matrices, squeeze to stack them into a single matrix
    # each element of G2 is a matrix, vectorize+transpose to turn it into a row
    G2 = G2.transpose(1, 2).reshape(n, d1 * d2)
    check_close(G2, G)

    # Hessian
    # method 1, manual computation
    H = J.t() @ J / n
    check_close(H * n,
                [[17, -34, -17, 51, -17, 34, 17, -51],
                 [-34, 68, 34, -102, 34, -68, -34, 102],
                 [-17, 34, 17, -51, 17, -34, -17, 51],
                 [51, -102, -51, 153, -51, 102, 51, -153],
                 [-17, 34, 17, -51, 19, -38, -19, 57],
                 [34, -68, -34, 102, -38, 76, 38, -114],
                 [17, -34, -17, 51, -19, 38, 19, -57],
                 [-51, 102, 51, -153, 57, -114, -57, 171]])

    # method 2, using activation + upstream matrices
    check_close(kron(A @ A.t(), X2 @ X2.t()) / n, H)

    # method 3, PyTorch autograd
    hess = hessian(compute_loss(residuals), model.W.weight)
    # Fix shape: vectorization flattens in column-major order, but PyTorch is row-major order
    # for reshape to flatten things correctly, transpose H_{ijkl} -> H_{jilk}
    hess = hess.transpose(2, 3).transpose(0, 1)
    hess = hess.reshape(d1 * d2, d1 * d2)
    check_close(hess, H)

    # method 4, get Jacobian + Hessian using backprop
    _loss, _output = optimizer.step(closure=backward('output'))
    B2t = model.W.grad_output

    # alternative way of getting batch Jacobian
    J2 = khatri_rao_t(At, B2t)
    check_close(J2, J)
    H2 = J2.t() @ J2 / n
    check_close(H2, H)

    # empirical Fisher
    efisher = G.t() @ G / n
    check_close(efisher, [[2460, -4920, -2460, 7380, -2394, 4788, 2394, -7182],
                          [-4920, 9840, 4920, -14760, 4788, -9576, -4788, 14364],
                          [-2460, 4920, 2460, -7380, 2394, -4788, -2394, 7182],
                          [7380, -14760, -7380, 22140, -7182, 14364, 7182, -21546],
                          [-2394, 4788, 2394, -7182, 2511, -5022, -2511, 7533],
                          [4788, -9576, -4788, 14364, -5022, 10044, 5022, -15066],
                          [2394, -4788, -2394, 7182, -2511, 5022, 2511, -7533],
                          [-7182, 14364, 7182, -21546, 7533, -15066, -7533, 22599]])

    # centered empirical Fisher (Sigma in OpenAI paper, estimate of Sigma in Jain paper)
    sigma = efisher - g.t() @ g
    check_close(sigma, [[344, -688, -344, 1032, -324, 648, 324, -972], [-688, 1376, 688, -2064, 648, -1296, -648, 1944],
                        [-344, 688, 344, -1032, 324, -648, -324, 972],
                        [1032, -2064, -1032, 3096, -972, 1944, 972, -2916],
                        [-324, 648, 324, -972, 486, -972, -486, 1458], [648, -1296, -648, 1944, -972, 1944, 972, -2916],
                        [324, -648, -324, 972, -486, 972, 486, -1458],
                        [-972, 1944, 972, -2916, 1458, -2916, -1458, 4374]])

    # loss
    loss2 = (residuals * residuals).sum() / (2 * n)
    check_close(to_python_scalar(loss2), 187.5)

    sigma_norm = torch.norm(sigma)
    g_norm = torch.norm(g)

    # predicted drop in loss if we take a Newton step
    excess = to_python_scalar(g @ H.pinverse() @ g.t() / 2)
    check_close(excess, 12747 / 68)

    def loss_direction(direction, eps):
        """loss improvement if we take step eps in direction dir"""
        return to_python_scalar(eps * (direction @ g.t()) - 0.5 * eps ** 2 * direction @ H @ direction.t())

    newtonImprovement = loss_direction(g @ H.pinverse(), 1)
    check_close(newtonImprovement, 12747/68)

    ############################
    # OpenAI quantities
    ############################
    grad_curvature = to_python_scalar(g @ H @ g.t())  # curvature in direction of g
    stepOpenAI = to_python_scalar(g.flatten().norm() ** 2 / grad_curvature) if g_norm else 999
    check_close(stepOpenAI, 0.00571855)
    batchOpenAI = to_python_scalar(torch.trace(H @ sigma) / grad_curvature) if g_norm else 999
    check_close(batchOpenAI, 0.180201)

    # improvement in loss when we take gradient step with optimal learning rate
    gradientImprovement = loss_direction(g, stepOpenAI)
    assert newtonImprovement > gradientImprovement
    check_close(gradientImprovement, 177.604)

    ############################
    # Gradient diversity  quantities
    ############################
    diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2
    check_close(diversity, 3.6013)

    ############################
    # Jain/Kakade quantities
    ############################

    # noise scale (Jain, minimax rate of estimator)
    noise_variance = torch.trace(H.pinverse() @ sigma)
    check_close(noise_variance, 333.706)

    isqrtH = pinv_square_root(H)
    #    isqrtH = torch.tensor(isqrtH)
    # measure of misspecification between model and actual noise (Jain, \rho)
    # formula (3) of "Parallelizing Stochastic Gradient Descent"
    p_sigma = torch.pinverse(kron(H, torch.eye(d1 * d2)) + kron(torch.eye(d1 * d2), H)) @ vec(sigma)
    p_sigma = unvec(p_sigma, d1 * d2)
    rho = d1 * d2 / erank(p_sigma) if sigma_norm > 0 else 1
    check_close(rho, 6.48399)

    rhoSimple = (d1 * d2 / erank(isqrtH @ sigma @ isqrtH)) if sigma_norm > 0 else 1
    check_close(rhoSimple, 6.55661)

    # divergent learning rate for batch-size 1 (Jain). Approximates max||x_i|| with avg.
    # For more accurate results may want to add stddev of ||x_i||
    # noinspection PyTypeChecker
    stepMin = 2 / torch.trace(H)
    check_close(stepMin, 0.0111111)

    # divergent learning rate for batch-size infinity
    stepMax = 2 / l2_norm(H)
    check_close(stepMax, 0.011419)

    # divergent learning rate for batch-size 1, adjusted for misspecification
    check_close(stepMin / rhoSimple, 0.00169464)
    check_close(stepMin / rho, 0.00171362)

    # batch size that gives provides lr halfway between stepMin and stepMax
    batchJain = 1 + erank(H)
    check_close(batchJain, 2.02771)

    # batch size that provides halfway point after adjusting for misspecification
    check_close(1 + erank(H) * rhoSimple, 7.73829)
    check_close(1 + erank(H) * rho, 7.66365)


if __name__ == '__main__':
    run_all_tests(sys.modules[__name__])
