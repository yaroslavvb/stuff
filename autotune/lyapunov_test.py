import os
import sys
import time
from typing import Optional, Tuple, Callable

# import torch
import scipy
import torch
from torchcurv.optim import SecondOrderOptimizer

import torch.nn as nn

import util as u

import numpy as np


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor):
        result = self.w(x)
        return result


# Backward via iterative Lyapunov solver
# from https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
def lyap_newton_schulz(z, dldz, numIters, dtype):
    batchSize = z.shape[0]
    dim = z.shape[1]
    normz = z.mul(z).sum(dim=1).sum(dim=1).sqrt()
    a = z.div(normz.view(batchSize, 1, 1).expand_as(z))
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
    q = dldz.div(normz.view(batchSize, 1, 1).expand_as(z))
    for i in range(numIters):
        q = 0.5 * (q.bmm(3.0 * I - a.bmm(a)) - a.transpose(1, 2).bmm(a.transpose(1, 2).bmm(q) - q.bmm(a)))
        a = 0.5 * a.bmm(3.0 * I - a.bmm(a))
    dlda = 0.5 * q
    return dlda


def test_lyapunov():
    """Test that scipy lyapunov solver works correctly."""
    d = 2
    n = 3
    torch.set_default_dtype(torch.float32)

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
    J = X.t()
    A = model.w.data_input
    B = model.w.grad_output * n
    G = residuals.repeat(1, d) * J
    losses = torch.stack([compute_loss(r) for r in residuals])
    g = G.sum(dim=0) / n
    efisher = G.t() @ G / n
    sigma = efisher - u.outer(g, g)
    loss2 = (residuals * residuals).sum() / (2 * n)
    H = J.t() @ J / n
    noise_variance = torch.trace(H.inverse() @ sigma)

    # H is not quite symmetric, make it so
    H = H + H.t()

    # Slow way
    p_sigma = u.lyapunov_lstsq(H, sigma)

    sigma0 = u.to_numpy(sigma)
    H0 = u.to_numpy(H)

    # Alternative faster way
    p_sigma2 = scipy.linalg.solve_lyapunov(H0, sigma0)
    print(f"Error 1: {np.max(abs(H0 @ p_sigma2 + p_sigma2 @ H0 - sigma0))}")
    u.check_close(p_sigma, p_sigma2)

    # alternative through SVD
    p_sigma3 = lyapunov_svd(torch.tensor(H0), torch.tensor(sigma0))
    u.check_close(p_sigma2, p_sigma3)

    # alternative through evals
    p_sigma4 = u.lyapunov_spectral(torch.tensor(H0), torch.tensor(sigma0))
    u.check_close(p_sigma2, p_sigma4)


def test_stability():
    bad_sigmas = torch.load('test/bad_sigmas.pt')
    H = bad_sigmas['H']
    sigma = bad_sigmas['sigma']

    X = u.lyapunov_spectral(H, sigma)
    discrepancy = torch.max(abs(X - X.t()) / X)
    assert discrepancy < 0.01


def compare_impl():
    """
    lstsq : error 2.23668351395645e-07 erank 4.248441219329834 discrepancy 4760.62841796875
    scipy : error 4.3144709138687176e-07 erank 5.015336036682129 discrepancy 0.38698798418045044
    svd : error 2.6252257612213725e-07 erank 5.567287921905518 discrepancy 11.784327507019043
    spectral : error 3.1967988434189465e-07 erank 4.978618144989014 discrepancy 0.0020115238148719072
    """

    bad_sigmas = torch.load('test/bad_sigmas.pt')
    H = bad_sigmas['H']
    sigma = bad_sigmas['sigma']

    # X = u.lyapunov_spectral(H, sigma)

    def print_stats(tag, X):
        error = torch.norm(H @ X + X @ H - sigma)
        discrepancy = torch.max(abs(X - X.t()) / X)
        print(tag, ": error", error.item(), "erank", u.erank(X).item(), 'discrepancy', discrepancy.item())

    H0, sigma0 = u.to_numpys(H, sigma)
    X0 = scipy.linalg.solve_lyapunov(H0, sigma0)

    print_stats('lstsq', u.lyapunov_lstsq(H, sigma))
    print_stats('scipy', torch.tensor(X0))
    print_stats('svd', u.lyapunov_svd(H, sigma))
    print_stats('spectral', u.lyapunov_spectral(H, sigma))


def lyapunov_svd(A, C, rtol=1e-4, use_svd=False):
    """Solve AX+XA=C"""

    assert A.shape[0] == A.shape[1]
    assert len(A.shape) == 2
    if use_svd:
        U, S, V = torch.svd(A)
    else:
        S, U = torch.symeig(A, eigenvectors=True)
    S = S.diag() @ torch.ones(A.shape)
    X = U @ ((U.t() @ C @ U) / (S + S.t())) @ U.t()
    error = A @ X + X @ A - C
    relative_error = torch.max(torch.abs(error)) / torch.max(torch.abs(A))
    if relative_error > rtol:
        print(f"Warning, error {relative_error} encountered in lyapunov_svd")

    return X


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
    it to TensorBoard."""

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        print(f"{interval_ms:8.2f}   {self.tag}")


def get_mkl_version():
    import ctypes
    import numpy as np

    # this recipe only works on Linux
    try:
        ver = np.zeros(199, dtype=np.uint8)
        mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")
        mkl.MKL_Get_Version_String(ver.ctypes.data_as(ctypes.c_char_p), 198)
        return ver[ver != 0].tostring()
    except:
        return 'unknown'


def print_cpu_info():
    ver = 'unknown'
    try:
        for l in open("/proc/cpuinfo").read().split('\n'):
            if 'model name' in l:
                ver = l
                break
    except:
        pass


if __name__ == '__main__':
    test_stability()
    #    u.run_all_tests(sys.modules[__name__])
