import math
import os
import sys

# import torch
import pytest
import scipy
from scipy import linalg
import torch

import numpy as np
import util as u

import torch.nn.functional as F


def test_khatri_rao():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[5, 6], [7, 8]])
    C = torch.tensor([[5, 12], [7, 16],
                      [15, 24], [21, 32]])
    u.check_equal(u.khatri_rao(A, B), C)


def test_khatri_rao_t():
    A = torch.tensor([[-2., -1.],
                      [0., 1.],
                      [2., 3.]])
    B = torch.tensor([[-4.],
                      [1.],
                      [6.]])
    C = torch.tensor([[8., 4.],
                      [0., 1.],
                      [12., 18.]])
    u.check_equal(u.khatri_rao_t(A, B), C)


def test_to_logits():
    torch.set_default_dtype(torch.float32)

    p = torch.tensor([0.2, 0.5, 0.3])
    u.check_close(p, F.softmax(u.to_logits(p), dim=0))
    u.check_close(p.unsqueeze(0), F.softmax(u.to_logits(p.unsqueeze(0)), dim=1))


def test_cross_entropy_soft():
    torch.set_default_dtype(torch.float32)

    q = torch.tensor([0.4, 0.6]).unsqueeze(0).float()
    p = torch.tensor([0.7, 0.3]).unsqueeze(0).float()
    observed_logit = u.to_logits(p)

    # Compare against other loss functions
    # https://www.wolframcloud.com/obj/user-eac9ee2d-7714-42da-8f84-bec1603944d5/newton/logistic-hessian.nb

    loss1 = F.binary_cross_entropy(p[0], q[0])
    u.check_close(loss1, 0.865054)

    loss_fn = u.CrossEntropySoft()
    loss2 = loss_fn(observed_logit, q)
    u.check_close(loss2, loss1)

    loss3 = F.cross_entropy(observed_logit, torch.tensor([0]))
    u.check_close(loss3, loss_fn(observed_logit, torch.tensor([[1, 0.]])))

    # check gradient
    observed_logit.requires_grad = True
    grad = torch.autograd.grad(loss_fn(observed_logit, target=q), observed_logit)
    u.check_close(p - q, grad[0])

    # check Hessian
    observed_logit = u.to_logits(p)
    observed_logit.zero_()
    observed_logit.requires_grad = True
    hessian_autograd = u.hessian(loss_fn(observed_logit, target=q), observed_logit)
    hessian_autograd = hessian_autograd.reshape((p.numel(), p.numel()))
    p = F.softmax(observed_logit, dim=1)
    hessian_manual = torch.diag(p[0]) - p.t() @ p
    u.check_close(hessian_autograd, hessian_manual)


def test_symsqrt():
    u.seed_random(1)
    torch.set_default_dtype(torch.float32)

    mat = torch.reshape(torch.arange(9) + 1, (3, 3)).float() + torch.eye(3) * 5
    mat = mat + mat.t()  # make symmetric
    smat = u.symsqrt(mat)
    u.check_close(mat, smat @ smat.t())
    u.check_close(mat, smat @ smat)

    def randomly_rotate(X):
        """Randomly rotate d,n data matrix X"""
        d, n = X.shape
        z = torch.randn((d, d), dtype=X.dtype)
        q, r = torch.qr(z)
        d = torch.diag(r)
        ph = d / abs(d)
        rot_mat = q * ph
        return rot_mat @ X

    n = 20
    d = 10
    X = torch.randn((d, n))

    # embed in a larger space
    X = torch.cat([X, torch.zeros_like(X)])
    X = randomly_rotate(X)
    cov = X @ X.t() / n
    sqrt, rank = u.symsqrt(cov, return_rank=True)
    assert rank == d
    assert torch.allclose(sqrt @ sqrt, cov, atol=1e-5)

    Y = torch.randn((d, n))
    Y = torch.cat([Y, torch.zeros_like(X)])
    Y = randomly_rotate(X)
    cov = u.Kron(X @ X.t(), Y @ Y.t())
    sqrt, rank = cov.symsqrt(return_rank=True)
    assert rank == d * d
    u.check_close(sqrt @ sqrt, cov, rtol=1e-4)

    X = torch.tensor([[7., 0, 0, 0, 0]]).t()
    X = randomly_rotate(X)
    cov = X @ X.t()
    u.check_close(u.sym_l2_norm(cov), 7 * 7)

    Y = torch.tensor([[8., 0, 0, 0, 0]]).t()
    Y = randomly_rotate(Y)
    cov = u.Kron(X @ X.t(), Y @ Y.t())
    u.check_close(cov.sym_l2_norm(), 7 * 7 * 8 * 8)


@pytest.mark.skip(reason="fails, need to redo pinv implementation")
def atest_pinv():
    a = torch.tensor([[2., 7, 9], [1, 9, 8], [2, 7, 5]])
    b = torch.tensor([[6., 6, 1], [10, 7, 7], [7, 10, 10]])
    C = u.Kron(a, b)
    u.check_close(a.flatten().norm() * b.flatten().norm(), C.frobenius_norm())

    u.check_close(C.frobenius_norm(), 4 * math.sqrt(11635.))

    Ci = [[0, 5 / 102, -(7 / 204), 0, -(70 / 561), 49 / 561, 0, 125 / 1122, -(175 / 2244)],
          [1 / 20, -(53 / 1020), 8 / 255, -(7 / 55), 371 / 2805, -(224 / 2805), 5 / 44, -(265 / 2244), 40 / 561],
          [-(1 / 20), 3 / 170, 3 / 170, 7 / 55, -(42 / 935), -(42 / 935), -(5 / 44), 15 / 374, 15 / 374],
          [0, -(5 / 102), 7 / 204, 0, 20 / 561, -(14 / 561), 0, 35 / 1122, -(49 / 2244)],
          [-(1 / 20), 53 / 1020, -(8 / 255), 2 / 55, -(106 / 2805), 64 / 2805, 7 / 220, -(371 / 11220), 56 / 2805],
          [1 / 20, -(3 / 170), -(3 / 170), -(2 / 55), 12 / 935, 12 / 935, -(7 / 220), 21 / 1870, 21 / 1870],
          [0, 5 / 102, -(7 / 204), 0, 0, 0, 0, -(5 / 102), 7 / 204],
          [1 / 20, -(53 / 1020), 8 / 255, 0, 0, 0, -(1 / 20), 53 / 1020, -(8 / 255)],
          [-(1 / 20), 3 / 170, 3 / 170, 0, 0, 0, 1 / 20, -(3 / 170), -(3 / 170)]]
    C = C.expand_vec()
    C0 = u.to_numpy(C)
    Ci = torch.tensor(Ci)
    u.check_close(C @ Ci @ C, C)

    u.check_close(linalg.pinv(C0), Ci, rtol=1e-5, atol=1e-6)
    u.check_close(torch.pinverse(C), Ci, rtol=1e-5, atol=1e-6)
    u.check_close(u.pinv(C), Ci, rtol=1e-5, atol=1e-6)
    u.check_close(C.pinv(), Ci, rtol=1e-5, atol=1e-6)


def test_pinverse():

    def subtest(dtype):
        # {{11041, 13359, 15023, 18177}, {13359, 16165, 18177, 21995}, {15023, 18177, 20453, 24747}, {18177, 21995, 24747, 29945}}
        x = [[11041, 13359, 15023, 18177], [13359, 16165, 18177, 21995], [15023, 18177, 20453, 24747], [18177, 21995, 24747, 29945]]
        x = u.to_pytorch(x).type(dtype)
        # {{29945, -24747, -21995, 18177}, {-24747, 20453, 18177, -15023}, {-21995, 18177, 16165, -13359}, {18177, -15023, -13359, 11041}}
        y0 = [[29945, -24747, -21995, 18177], [-24747, 20453, 18177, -15023], [-21995, 18177, 16165, -13359], [18177, -15023, -13359, 11041]]
        y0 = u.to_pytorch(y0)/16  # ground-truth
        y0 = u.to_pytorch(y0).type(dtype)

        #  print('discrepancy1 truth', torch.norm(x @ y0 @ x - x))
        #  print('discrepancy2 truth', torch.norm(y0 @ x @ y0 - y0))
        y1 = torch.pinverse(x)
        # print('torch error', torch.norm(y0-y1))
        if dtype == torch.float64:
            assert torch.norm(y0-y1) < 2e-5  # 1.8943e-05
        y2 = scipy.linalg.pinv(x)
        # print('scipy error', torch.norm(y0-u.from_numpy(y2)))  # 2.5631e-05

    subtest(torch.float64)
    subtest(torch.float32)


def test_l2_norm():
    mat = torch.tensor([[1, 1], [0, 1]]).float()
    u.check_equal(u.l2_norm(mat), 0.5 * (1 + math.sqrt(5)))
    ii = torch.eye(5)
    u.check_equal(u.l2_norm(ii), 1)


def test_symsqrt_neg():
    """Test robustness to small negative eigenvalues."""
    u.seed_random(1)
    torch.set_default_dtype(torch.float32)

    mat = torch.tensor([[1.704692840576171875e-05, -9.693153669044357601e-15, -4.637238930627063382e-07,
                         -5.784777457051859528e-08, -7.958237541183521557e-11, -9.898678399622440338e-06,
                         -2.152719247305867611e-07, -1.635662982835128787e-08, -6.400216989277396351e-06,
                         -1.906904145698717912e-08],
                        [-9.693153669044357601e-15, 9.693318840469072190e-15, -4.495100185314538545e-21,
                         -5.607465147510056466e-22, -7.714305304820877864e-25, -9.595268609986125189e-20,
                         -2.086734953246380139e-21, -1.585527368218132061e-22, -6.204040063550678436e-20,
                         -1.848454455492805625e-22],
                        [-4.637238930627063382e-07, -4.495100185314538545e-21, 4.637315669242525473e-07,
                         -2.682631101578927119e-14, -3.690551006200058401e-17, -4.590410516980281130e-12,
                         -9.983013764605641605e-14, -7.585219006177833234e-15, -2.968034672201635971e-12,
                         -8.843071403179941781e-15],
                        [-5.784777457051859528e-08, -5.607465147510056466e-22, -2.682631101578927119e-14,
                         5.784875867220762302e-08, -4.603820523722603687e-18, -5.726360683376563454e-13,
                         -1.245342652107404510e-14, -9.462269640040017228e-16, -3.702509490405986314e-13,
                         -1.103139288087268827e-15],
                        [-7.958237541183521557e-11, -7.714305304820877864e-25, -3.690551006200058401e-17,
                         -4.603820523722603687e-18, 7.958373543504038139e-11, -7.877872806981575052e-16,
                         -1.713243580702275093e-17, -1.301743849346897150e-18, -5.093618864028342207e-16,
                         -1.517611416046059107e-18],
                        [-9.898678399622440338e-06, -9.595268609986125189e-20, -4.590410516980281130e-12,
                         -5.726360683376563454e-13, -7.877872806981575052e-16, 9.898749340209178627e-06,
                         -2.130980288062023220e-12, -1.619145491510778911e-13, -6.335585528427500890e-11,
                         -1.887647481900456281e-13],
                        [-2.152719247305867611e-07, -2.086734953246380139e-21, -9.983013764605641605e-14,
                         -1.245342652107404510e-14, -1.713243580702275093e-17, -2.130980288062023220e-12,
                         2.152755484985391377e-07, -3.521243228436451468e-15, -1.377834044774539635e-12,
                         -4.105169319306963341e-15],
                        [-1.635662982835128787e-08, -1.585527368218132061e-22, -7.585219006177833234e-15,
                         -9.462269640040017228e-16, -1.301743849346897150e-18, -1.619145491510778911e-13,
                         -3.521243228436451468e-15, 1.635690871637507371e-08, -1.046895521119271810e-13,
                         -3.119158858896762436e-16],
                        [-6.400216989277396351e-06, -6.204040063550678436e-20, -2.968034672201635971e-12,
                         -3.702509490405986314e-13, -5.093618864028342207e-16, -6.335585528427500890e-11,
                         -1.377834044774539635e-12, -1.046895521119271810e-13, 6.400285201380029321e-06,
                         -1.220501699922618699e-13],
                        [-1.906904145698717912e-08, -1.848454455492805625e-22, -8.843071403179941781e-15,
                         -1.103139288087268827e-15, -1.517611416046059107e-18, -1.887647481900456281e-13,
                         -4.105169319306963341e-15, -3.119158858896762436e-16, -1.220501699922618699e-13,
                         1.906936653028878936e-08]])
    evals = torch.eig(mat).eigenvalues
    assert torch.min(evals) < 0
    smat = u.symsqrt(mat)
    u.check_close(mat, smat @ smat.t())
    u.check_close(mat, smat @ smat)


def test_truncated_lyapunov():
    d = 100
    n = 1000
    shared_rank = 2
    independent_rank = 1
    A, C = u.random_cov_pair(shared_rank=shared_rank, independent_rank=independent_rank, strength=0.1, d=d, n=n)
    X = u.lyapunov_truncated(A, C)

    # effective rank of X captures dimensionality of shared subspace
    u.check_close(u.rank(X), shared_rank + independent_rank, rtol=1e-4)
    u.check_close(u.erank(X), shared_rank, rtol=1e-2)


def test_lyapunov_lstsq():
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    # A = torch.tensor([1., 2, 3, 4]).reshape(2, 2)
    # C = torch.tensor([5., 6, 7, 8]).reshape(2, 2)
    # X = u.lyapunov_lstsq(A, C)
    # u.check_close(X, [[0.1, 1.1], [1.3, 0.1]])
    #
    # X = u.lyapunov_lstsq(A, A)
    # u.check_close(X, [[0.5, -.1], [.1, .5]])
    #
    A = u.random_cov(1, 3, n=100)
    # print('A=', u._to_mathematica(A))
    X = u.lyapunov_lstsq(A, 2 * A)
    # print('X=', u._to_mathematica(X))
    # print(torch.svd(X)[1])
    X = u.lyapunov_svd(A, 2 * A)
    # print(X)
    # print(torch.svd(X)[1])
    # torch.set_default_dtype(torch.float32)


def test_robust_svd():
    mat = np.genfromtxt('test/gesvd_crash.txt', delimiter=",")
    mat = torch.tensor(mat).type(torch.get_default_dtype())
    U, S, V = u.robust_svd(mat)
    mat2 = U @ torch.diag(S) @ V.T
    u.check_close(mat, mat2)


def test_misc():
    d = 3
    a = u.eye_like(torch.ones((d, d)))
    assert u.erank(a) == d


def test_kron():
    """Test kron, vec and vecr identities"""
    torch.set_default_dtype(torch.float64)
    a = torch.tensor([1, 2, 3, 4]).reshape(2, 2)
    b = torch.tensor([5, 6, 7, 8]).reshape(2, 2)
    u.check_close(u.Kron(a, b).trace(), 65)

    a = torch.tensor([[2., 7, 9], [1, 9, 8], [2, 7, 5]])
    b = torch.tensor([[6., 6, 1], [10, 7, 7], [7, 10, 10]])
    Ck = u.Kron(a, b)
    u.check_close(a.flatten().norm() * b.flatten().norm(), Ck.frobenius_norm())

    u.check_close(Ck.frobenius_norm(), 4 * math.sqrt(11635.))

    Ci = [[0, 5 / 102, -(7 / 204), 0, -(70 / 561), 49 / 561, 0, 125 / 1122, -(175 / 2244)],
          [1 / 20, -(53 / 1020), 8 / 255, -(7 / 55), 371 / 2805, -(224 / 2805), 5 / 44, -(265 / 2244), 40 / 561],
          [-(1 / 20), 3 / 170, 3 / 170, 7 / 55, -(42 / 935), -(42 / 935), -(5 / 44), 15 / 374, 15 / 374],
          [0, -(5 / 102), 7 / 204, 0, 20 / 561, -(14 / 561), 0, 35 / 1122, -(49 / 2244)],
          [-(1 / 20), 53 / 1020, -(8 / 255), 2 / 55, -(106 / 2805), 64 / 2805, 7 / 220, -(371 / 11220), 56 / 2805],
          [1 / 20, -(3 / 170), -(3 / 170), -(2 / 55), 12 / 935, 12 / 935, -(7 / 220), 21 / 1870, 21 / 1870],
          [0, 5 / 102, -(7 / 204), 0, 0, 0, 0, -(5 / 102), 7 / 204],
          [1 / 20, -(53 / 1020), 8 / 255, 0, 0, 0, -(1 / 20), 53 / 1020, -(8 / 255)],
          [-(1 / 20), 3 / 170, 3 / 170, 0, 0, 0, 1 / 20, -(3 / 170), -(3 / 170)]]
    C = Ck.expand()
    C0 = u.to_numpy(C)
    Ci = torch.tensor(Ci)
    u.check_close(C @ Ci @ C, C)

    u.check_close(Ck.inv().expand(), torch.inverse(Ck.expand()))
    u.check_close(Ck.inv().expand_vec(), torch.inverse(Ck.expand_vec()))
    u.check_close(Ck.pinv().expand(), torch.pinverse(Ck.expand()))

    u.check_close(linalg.pinv(C0), Ci, rtol=1e-5, atol=1e-6)
    u.check_close(torch.pinverse(C), Ci, rtol=1e-5, atol=1e-6)
    u.check_close(Ck.inv().expand(), Ci, rtol=1e-5, atol=1e-6)
    u.check_close(Ck.pinv().expand(), Ci, rtol=1e-5, atol=1e-6)

    Ck2 = u.Kron(b, 2 * a)
    u.check_close((Ck @ Ck2).expand(), Ck.expand() @ Ck2.expand())
    u.check_close((Ck @ Ck2).expand_vec(), Ck.expand_vec() @ Ck2.expand_vec())

    d2 = 3
    d1 = 2
    G = torch.randn(d2, d1)
    g = u.vec(G)
    H = u.Kron(u.random_cov(d1), u.random_cov(d2))

    Gt = G.t()
    gt = g.reshape(1, -1)

    vecX = u.Vec([1, 2, 3, 4], shape=(2, 2))
    K = u.Kron([[5, 6], [7, 8]], [[9, 10], [11, 12]])

    u.check_equal(vecX @ K, [644, 706, 748, 820])
    u.check_equal(K @ vecX, [543, 655, 737, 889])

    u.check_equal(u.matmul(vecX @ K, vecX), 7538)
    u.check_equal(vecX @ (vecX @ K), 7538)
    u.check_equal(vecX @ vecX, 30)

    vecX = u.Vec([1, 2], shape=(1, 2))
    K = u.Kron([[5]], [[9, 10], [11, 12]])

    u.check_equal(vecX.norm()**2, 5)

    # check kronecker rules
    X = torch.tensor([[1., 2], [3, 4]])
    A = torch.tensor([[5., 6], [7, 8]])
    B = torch.tensor([[9., 10], [11, 12]])
    x = u.Vec(X)

    # kron/vec/vecr identities
    u.check_equal(u.Vec(A @ X @ B), x @ u.Kron(B, A.t()))
    u.check_equal(u.Vec(A @ X @ B), u.Kron(B.t(), A) @ x)
    u.check_equal(u.Vecr(A @ X @ B), u.Kron(A, B.t()) @ u.Vecr(X))
    u.check_equal(u.Vecr(A @ X @ B), u.Vecr(X) @ u.Kron(A.t(), B))

    def extra_checks(A, X, B):
        x = u.Vec(X)
        u.check_equal(u.Vec(A @ X @ B), x @ u.Kron(B, A.t()))
        u.check_equal(u.Vec(A @ X @ B), u.Kron(B.t(), A) @ x)
        u.check_equal(u.Vecr(A @ X @ B), u.Kron(A, B.t()) @ u.Vecr(X))
        u.check_equal(u.Vecr(A @ X @ B), u.Vecr(X) @ u.Kron(A.t(), B))
        u.check_equal(u.Vecr(A @ X @ B), u.Vecr(X) @ u.Kron(A.t(), B).normal_form())
        u.check_equal(u.Vecr(A @ X @ B), u.matmul(u.Kron(A, B.t()).normal_form(), u.Vecr(X)))
        u.check_equal(u.Vec(A @ X @ B), u.matmul(u.Kron(B.t(), A).normal_form(), x))
        u.check_equal(u.Vec(A @ X @ B), x @ u.Kron(B, A.t()).normal_form())
        u.check_equal(u.Vec(A @ X @ B), x.normal_form() @ u.Kron(B, A.t()).normal_form())
        u.check_equal(u.Vec(A @ X @ B), u.Kron(B.t(), A).normal_form() @ x.normal_form())
        u.check_equal(u.Vecr(A @ X @ B), u.Kron(A, B.t()).normal_form() @ u.Vecr(X).normal_form())
        u.check_equal(u.Vecr(A @ X @ B), u.Vecr(X).normal_form() @ u.Kron(A.t(), B).normal_form())

    # shape checks
    d1, d2 = 3, 4
    extra_checks(torch.ones((d1, d1)), torch.ones((d1, d2)), torch.ones((d2, d2)))

    A = torch.rand(d1, d1)
    B = torch.rand(d2, d2)
    #x = torch.rand((d1*d2))
    #X = x.t().reshape(d1, d2)
    # X = torch.rand((d1, d2))
    # x = u.vec(X)
    x = torch.rand((d1*d2))
    #    print((u.vec(A@X@B)-u.kron(B.t(), A) @ x).norm())


def test_contiguous():

    d = 5
    A = torch.rand((d, d))
    B = torch.rand((d, d))
    result = torch.einsum("ab,cd->acbd", A, B)
    assert(result.is_contiguous())
    result = torch.einsum("ab,cd->acbd", B, A)
    assert(result.is_contiguous())
    result = torch.einsum("ab,cd->acbd", B, A.t())
    assert(not result.is_contiguous())


if __name__ == '__main__':
    # test_truncated_lyapunov()
    # test_lyapunov_lstsq()
    # test_robust_svd()
    # test_contiguous()
    #    test_kron()
    test_pinverse()
#    u.run_all_tests(sys.modules[__name__])
