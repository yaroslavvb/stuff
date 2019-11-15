# Take simple example, plot per-layer stats over time
# This function allows you to visualize the statistics of a layer.
import inspect
import math
import os
import random
import re
import sys
import time
from typing import Any, Dict, Callable, Optional, Tuple, Union, Sequence, Iterable
from typing import List

import six
import wandb
from attrdict import AttrDict
from torch.utils import tensorboard

import globals as gl
import numpy as np
import scipy
import scipy.linalg as linalg
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from PIL import Image

import platform

import torch.nn.functional as F

# to enable referring to functions in its own module as u.func
u = sys.modules[__name__]


# numerical noise cutoff for eigenvalues, from scipy.linalg.pinv2
# max float32 condition number: 8.4k (8388.61)
# max float64 condition number: 4.5B

def get_condition(dtype):
    """Return number x such that values below max(eigenval)*x are indistinguishable from noise"""
    assert 'float' in str(dtype)
    if str(dtype).endswith('float32') or str(dtype).endswith('float16'):
        return 1e3 * 1.1920929e-07
    else:  # assume float64
        return 1e6 * 2.220446049250313e-16


def v2c(vec):
    """Convert vector to column matrix."""
    vec = to_pytorch(vec)
    assert len(vec.shape) == 1
    return torch.unsqueeze(vec, 1)


def v2c_np(vec):
    """Convert vector to column matrix."""
    assert len(vec.shape) == 1
    return np.expand_dims(vec, 1)


def v2r(vec: torch.Tensor) -> torch.Tensor:
    """Converts rank-1 tensor to row matrix"""
    vec = to_pytorch(vec)
    assert len(vec.shape) == 1
    return vec.unsqueeze(0)


def c2v(col: torch.Tensor) -> torch.Tensor:
    """Convert vector into row matrix."""
    vec = to_pytorch(col)
    assert len(col.shape) == 2
    assert col.shape[1] == 1
    return torch.reshape(col, [-1])


def vec(mat):
    """vec operator, stack columns of the matrix into single column matrix."""
    vec = to_pytorch(mat)
    assert len(mat.shape) == 2
    return mat.t().reshape(-1, 1)


def test_vec():
    mat = torch.tensor([[1, 3, 5], [2, 4, 6]])
    check_equal(c2v(vec(mat)), [1, 2, 3, 4, 5, 6])


def test_kron_trace():
    n = 5
    m = 4
    A = torch.rand((m, m))
    B = torch.rand((n, n))
    C = kron(A, B)
    u.check_close(torch.trace(C), u.kron_trace((A, B)))


def tvec(mat):
    """transposed vec operator concatenates rows into single row matrix"""
    assert len(mat.shape) == 2
    return mat.reshape(1, -1)


def test_tvec():
    mat = torch.tensor([[1, 3, 5], [2, 4, 6]])
    check_equal(tvec(mat), [[1, 3, 5, 2, 4, 6]])


def unvec(a, rows):
    """reverse of vec, rows specifies number of rows in the final matrix."""
    assert len(a.shape) == 2
    assert a.shape[1] == 1, f"argument expected to be a column matrix, instead got shape {a.shape}"
    assert a.shape[0] % rows == 0
    cols = a.shape[0] // rows
    return a.reshape(cols, -1).t()


def untvec(a, rows):
    """reverse of tvec, rows specifies number of rows in the final matrix."""
    assert len(a.shape) == 2
    assert a.shape[0] == 1
    assert a.shape[1] % rows == 0
    return a.reshape(rows, -1)


def kron(a: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], b: Optional[torch.Tensor] = None):
    """Kronecker product a otimes b."""

    if isinstance(a, Tuple):
        assert b is None
        a, b = a

    if is_vector(a) and is_vector(b):
        return torch.einsum('i,j->ij', a, b).flatten()

    # print('inside a', a)
    # print('inside b', b)
    result = torch.einsum("ab,cd->acbd", a, b)
    # print('kron', result)
    # TODO: use tensor.continuous

    if result.is_contiguous():
        return result.view(a.size(0) * b.size(0), a.size(1) * b.size(1))
    else:
        print("Warning kronecker product not contiguous, using reshape")
        return result.reshape(a.size(0) * b.size(0), a.size(1) * b.size(1))


def stable_kron(a, b):
    a_norm, b_norm = torch.max(a), torch.max(b)
    return kron(a / a_norm, b / b_norm) * a_norm * b_norm


class SpecialForm:
    def normal_form(self):
        raise NotImplemented


class Vec(SpecialForm):
    """Helper class representing x=Vec(X) and associated kronecker products

    x = vec(X)
    vec(AXB)' = x'(B*A')
    vec(AXB) = (B'*A)x

    xx = dot-product
    """

    mat: torch.Tensor
    shape: Tuple
    numel: int
    rank: int

    def __init__(self, mat, shape: Tuple = None):
        mat = to_pytorch(mat)
        if shape is not None:
            mat = mat.reshape(shape)
        else:
            shape = mat.shape
        self.mat = mat
        self.shape = shape

        assert np.prod(shape) == mat.numel()
        self.rank = len(shape)
        self.numel = self.mat.numel()

        assert self.rank >= 0
        assert self.rank <= 2

    def vec_form(self):
        return u.vec(self.mat).flatten()

    def matrix_form(self):
        return self.mat

    def normal_form(self):
        return self.vec_form()

    def __matmul__(self, other):
        if type(other) == Vec:
            return torch.sum(self.mat * other.mat)
        elif type(other) != torch.Tensor:
            return NotImplemented

        return self.vec_form() @ other

    def __rmatmul__(self, other):
        if type(other) == Vec:
            return other.__matmul__(self)
        elif type(other) != torch.Tensor:
            return NotImplemented
        return other @ self.vec_form()

    def __truediv__(self, other):
        return Vec(self.mat / other)

    def norm(self):
        return self.mat.flatten().norm()

    def commute(self):
        """Transpose matrix inside of vec operation.
        Equivalent to left multiplication by the commutation matrix."""

        return Vecr(self.mat)

    def __str__(self):
        return str(to_numpy(self.normal_form()))


class Vecr(SpecialForm):
    """Helper class representing row vectorization Vecr(X)=Vec(X') and associated kronecker products

    x=vecr(X)
    (A*B')x = vecr(AXB)
     x'(A'*B) = vecr(AXB)'
     xx = dotproduct
    """

    mat: torch.Tensor
    shape: Tuple
    numel: int
    rank: int

    def __init__(self, mat, shape: Tuple = None):
        mat = to_pytorch(mat)
        if shape is not None:
            mat = mat.reshape(shape)
        else:
            shape = mat.shape
        self.mat = mat
        self.shape = shape

        assert np.prod(shape) == mat.numel()
        self.rank = len(shape)
        self.numel = self.mat.numel()

        assert self.rank >= 0
        assert self.rank <= 2

    def vec_form(self):
        return self.mat.flatten()

    def matrix_form(self):
        return self.mat

    def normal_form(self):
        return self.vec_form()

    def __matmul__(self, other):
        if type(other) == Vecr:
            return torch.sum(self.mat * other.mat)
        elif type(other) != torch.Tensor:
            return NotImplemented

        return self.vec_form() @ other

    def __rmatmul__(self, other):
        if type(other) == Vecr:
            return other.__matmul__(self)
        elif type(other) != torch.Tensor:
            return NotImplemented
        return other @ self.vec_form()

    def __truediv__(self, other):
        return Vecr(self.mat / other)

    def norm(self):
        return self.mat.flatten().norm()

    def commute(self):
        """Transpose matrix inside of vec operation.
        Equivalent to left multiplication by the commutation matrix."""

        return Vec(self.mat)

    def __str__(self):
        return str(to_numpy(self.normal_form()))


class Cov(SpecialForm):
    pass


class FactoredCov(SpecialForm):
    pass


class KronFactoredCov(SpecialForm):
    """Kronecker factored covariance matrix. Covariance matrix of random variable ba' constructed from paired
    samples of a and b. Each sample of a can correspond to multiple samples of b, reprented by an extra batch dimension in b sample matrix."""

    a_num: int  # number of a samples
    b_num: int  # number of b samples
    ab_num: int  # number of samples used for AB cross-covariance estimate
    a_dim: int  # dimension of a samples
    b_dim: int  # dimension of b samples
    AA: torch.Tensor  # sum of a covariances
    BB: torch.Tensor  # sum of b covariances
    AB: torch.Tensor  # sum of a,b cross covariances

    def __init__(self, a_dim, b_dim):
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.AA = torch.zeros(a_dim, a_dim).to(gl.device)
        self.BB = torch.zeros(b_dim, b_dim).to(gl.device)
        self.AB = torch.zeros(a_dim, b_dim).to(gl.device)

        self.a_num = 0
        self.b_num = 0
        self.ab_num = 0

    def add_samples(self, A: torch.Tensor, B: torch.Tensor):
        """

        Args:
            A: (*, d1) matrix of samples of a where * is zero or more batch dimensions
            B: (*, d2) matrix of samples of b where * is zero or more batch dimensions

        For  variable batch dimensions, currently only supports A having 1 batch dimension, and B having 1 or 2 batch dimensions
        """

        if is_matrix(A) and is_matrix(B):
            n = A.shape[0]
            assert B.shape[0] == n, f"Number of samples do not match, got {A.shape}, {B.shape}"
            assert A.shape[1] == self.a_dim
            assert B.shape[1] == self.b_dim

            self.AA += torch.einsum("ni,nj->ij", A, A)
            self.BB += torch.einsum("ni,nj->ij", B, B)
            self.AB += torch.einsum("ni,nj->ij", A, B)
            self.a_num += n
            self.b_num += n
            self.ab_num += n

        elif len(A.shape) == 2 and len(B.shape) == 3:
            # TODO(y): this can be done more efficiently without stacking A
            n = A.shape[0]
            assert n == B.shape[1], f"Number of samples do not match, got {A.shape}, {B.shape}"
            assert A.shape[1] == self.a_dim
            assert B.shape[2] == self.b_dim
            o = B.shape[0]
            A = torch.stack([A] * o)

            self.AA += torch.einsum("oni,onj->ij", A, A)
            self.BB += torch.einsum("oni,onj->ij", B, B)
            self.AB += torch.einsum("oni,onj->ij", A, B)
            # TODO(y): fix inconsistent counts, current version is what was needed to make Hessians match autograd
            self.a_num += (o * n)
            self.b_num += n
            self.ab_num += o * n
        else:
            assert False, f"Broadcasting not implemented for shapes {A.shape} and {B.shape}"

    def value(self) -> "Kron":
        return Kron(self.AA / self.a_num, self.BB / self.b_num)

    def cross(self) -> torch.Tensor:
        """Return cross covariance matrix AB'"""
        return self.AB / self.ab_num

    def wilks(self) -> torch.Tensor:
        """Returns Wilk's statistic for the test of independence of two terms"""

        covA, covB = self.value()
        covAB = self.cross()

        with u.timeit('wilks'):
            K = isymsqrt(covA) @ covAB @ isymsqrt(covB)
            U, S, V = robust_svd(K)
            vals = 1. - square(torch.diag(S))
            return torch.prod(vals)

    def bartlett(self):
        """Returns Bartlett statistic for the test of independence."""
        q = self.a_dim
        p = self.b_dim
        n = self.ab_num
        val = -(n - (p + q + 3.) / 2.) * torch.log(self.wilks())
        print('bartlett, ', val, 'mean', q * p)
        return val

    def prob_dep(self):
        """Returns probability of independence hypothesis holding."""
        from scipy.stats import chi2
        rv = chi2(self.a_dim * self.b_dim)
        return rv.sf(to_numpy(self.bartlett()))

    def sigmas_indep(self):
        """Returns number of standard deviations away from independence."""
        from scipy.stats import chi2
        df = self.a_dim * self.b_dim
        return (self.bartlett() - df) / (4 * np.sqrt(to_numpy(df)))

    def __str__(self):
        return f"KronFactoredCov(AA=\n{self.AA},\n BB={self.BB})"


def square(a: torch.Tensor):
    return a * a


# TODO(y): rename into sym-kron
class Kron(SpecialForm):
    """Represents kronecker product of two symmetric matrices.



    """
    LL: torch.Tensor  # left factor
    RR: torch.Tensor  # right factor

    def __init__(self, LL, RR):
        LL = to_pytorch(LL)
        RR = to_pytorch(RR)

        assert is_matrix(LL), f"shape check fail with {LL.shape}"
        assert is_matrix(RR), f"shape check fail with {RR.shape}"
        self.LL = LL
        self.RR = RR

        # todo(y): remove this check onces it's split into KronFactored and SymmetricKronFactored
        assert LL.shape[0] == LL.shape[1], f"shape check fail with {LL.shape}"
        assert RR.shape[0] == RR.shape[1], f"shape check fail with {RR.shape}"

        self.lsize = LL.shape[0]
        self.rsize = RR.shape[0]

    def commute(self):
        """Commutes order of operation: A kron B -> B kron A """

        return Kron(LL=self.RR, RR=self.LL)

    def normal_form(self):
        return self.expand()

    def expand(self):
        """Returns expanded representation (row-major form)"""
        return kron(self.LL, self.RR)

    def expand_vec(self):
        """Returns expanded representation (col-major form, to match vec order in literature)"""
        return kron(self.RR, self.LL)

    def sym_l2_norm(self):
        return sym_l2_norm(self.LL) * sym_l2_norm(self.RR)

    def symsqrt(self, cond=None, return_rank=False):
        a = symsqrt(self.LL, cond, return_rank)
        b = symsqrt(self.RR, cond, return_rank)
        if not return_rank:
            return Kron(a, b)
        else:
            a, rank_a = a
            b, rank_b = b
            return Kron(a, b), rank_a * rank_b

    def trace(self):
        return torch.trace(self.LL) * torch.trace(self.RR)

    def frobenius_norm(self):
        return torch.norm(self.LL.flatten()) * torch.norm(self.RR.flatten())

    def pinv(self):
        return Kron(torch.pinverse(self.LL), torch.pinverse(self.RR))

    def inv(self):
        return Kron(torch.inverse(self.LL), torch.inverse(self.RR))

    @property
    def shape(self):
        return self.LL.shape, self.RR.shape

    def qf(self, G):
        """Returns quadratic form g @ H @ g' """
        assert G.shape[1] == self.RR.shape[0]
        assert G.shape[0] == self.LL.shape[0]
        return torch.sum(G * (self.LL @ G @ self.RR))

    def qf_vec(self, G):
        """Returns quadratic form g' @ H @ g"""
        assert G.shape[0] == self.RR.shape[0]
        assert G.shape[1] == self.LL.shape[0]
        return torch.sum(G * (self.RR @ G @ self.LL))

    # TODO(y): implement in-place ops
    def __truediv__(self, other):
        return Kron(self.LL, self.RR / other)

    def __add__(self, other):
        other = to_python_scalar(other)
        return Kron(self.LL, self.RR + other)

    def __radd__(self, other):
        other = to_python_scalar(other)
        return Kron(self.LL + other, self.RR)

    def __mul__(self, other):
        other = to_python_scalar(other)
        return Kron(self.LL, self.RR * other)

    def __rmul__(self, other):
        other = to_python_scalar(other)
        return Kron(self.LL * other, self.RR)

    # remove, scalar addition doesn't make sense because it breaks factoring
    # def __add__(self, other):
    #     if u.is_scalar(other):
    #         return Kron(self.LL, self.RR+to_pytorch(other))
    #     else:
    #         return NotImplemented
    #
    # def __radd__(self, other):
    #     if u.is_scalar(other):
    #         return Kron(self.LL+to_pytorch(other), self.RR)
    #     else:
    #         return NotImplemented

    def __matmul__(self, x):
        if type(x) == Kron:
            return Kron(self.LL @ x.LL, self.RR @ x.RR)
        elif type(x) in [Vec, Vecr]:

            X = x.matrix_form()
            if type(x) == Vec:  # kron @ vec(mat)
                assert X.shape == (self.rsize, self.lsize), f"Dimension mismatch, {X.shape}, {self.lsize}, {self.rsize}"
                return Vec(self.RR @ X @ self.LL.t())
            elif type(x) == Vecr:
                assert X.shape == (self.lsize, self.rsize), f"Dimension mismatch, {X.shape}, {self.lsize}, {self.rsize}"
                return Vecr(self.LL @ X @ self.RR.t())
        elif type(x) is torch.Tensor:
            return self.normal_form() @ x
        else:
            return NotImplemented

    def __rmatmul__(self, x):
        if type(x) in [Vec, Vecr]:
            X = x.matrix_form()
            if type(x) == Vec:
                assert X.shape == (self.rsize, self.lsize), f"Dimension mismatch, {X.shape}, {self.lsize}, {self.rsize}"
                return Vec(self.RR.t() @ X @ self.LL)
            elif type(x) == Vecr:
                assert X.shape == (self.lsize, self.rsize), f"Dimension mismatch, {X.shape}, {self.lsize}, {self.rsize}"
                return Vecr(self.LL.t() @ X @ self.RR)
        elif type(x) is torch.Tensor:
            return x @ self.normal_form()
        else:
            return NotImplemented

    def __str__(self):
        return f"Kron(\n{self.LL},\n {self.RR})"

    def __iter__(self):
        return iter([self.LL, self.RR])


class MeanKronFactored(SpecialForm):
    """Factored representation as a mean of kronecker products"""
    AA: torch.Tensor  # stacked forward factors
    BB: torch.Tensor  # stacked backward factor

    def __init__(self, AA: torch.Tensor, BB: torch.Tensor):
        # AA: n, di, di
        # BB: n, do, do

        assert AA.shape[0] == BB.shape[0]
        assert AA.shape[1] == AA.shape[2]
        assert BB.shape[1] == BB.shape[2]
        n, di, _ = AA.shape
        n, do, _ = BB.shape
        self.AA = AA
        self.BB = BB
        self.n = n
        self.di = di
        self.do = do

    def expand(self):
        result = torch.einsum('nij,nkl->nikjl', self.BB, self.AA)
        # print(result)
        # result = kron(self.BB[0,...], self.AA[0,...]).unsqueeze(0)  # torch.einsum("ab,cd->acbd", a, b)
        # result = torch.einsum("ab,cd->acbd", a, b)
        # print('outside left', self.BB[0,...])
        # print('outside right', self.AA[0,...])
        # print('outside', torch.einsum('ab,cd->abcd', self.BB[0,...], self.AA[0,...]))
        # result = kron(self.BB[0,...], self.AA[0,...])  # torch.einsum("ab,cd->acbd", a, b)
        #        if not result.is_contiguous():
        #            print("Warning, using contiguous")
        #            result = result.contiguous()   # needed for .view

        result = result.sum(dim=0) / self.n
        return result.view(self.do * self.di, self.do * self.di)


def expand_hess(*v) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Expands Hessian represented in Kronecker factored form.

    Note: For consistency with PyTorch autograd, we use row-major order. This means the order of Kronecker
    multiplication needs to be reversed compared to literature which uses column-major order (implied by vec)."""
    result = [kron(a.RR, a.LL) for a in v]

    if len(result) == 1:
        return result[0]
    else:
        return result


def test_kron():
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[6, 7], [8, 9]])
    C = kron(A, B)
    Cnp = np.kron(to_numpy(A), to_numpy(B))
    check_equal(C, [[6, 7, 12, 14], [8, 9, 16, 18], [18, 21, 24, 28], [24, 27, 32, 36]])
    check_equal(C, Cnp)


def nan_check(mat):
    nan_mask = torch.isnan(mat).float()
    nans = torch.sum(nan_mask).item()
    not_nans = torch.sum(torch.tensor(1) - nan_mask).item()

    assert nans == 0, f"matrix of shape {mat.shape} has {nans}/{nans + not_nans} nans"


def has_nan(mat):
    return torch.sum(torch.isnan(mat)) > 0


def fro_norm(mat: torch.Tensor):
    return torch.norm(mat.flatten())


frobenius_norm = fro_norm


def l2_norm(mat: torch.Tensor):
    """Largest eigenvalue."""
    try:
        u, s, v = robust_svd(mat)
    except RuntimeError as e:
        if gl.debug_linalg_crashes:
            print(e)
            dump(mat, '/tmp/l2_norm.txt')
            assert False, f"svd failed with {e}"
        else:
            return -1
    return torch.max(s)


def sym_l2_norm(mat: torch.Tensor):
    """Largest eigenvalue assuming that matrix is symmetric."""

    u.check_symmetric(mat)

    if gl.debug_linalg_crashes:
        try:
            evals, _evecs = torch.symeig(mat)
        except RuntimeError as e:
            print(e)
            dump(mat, '/tmp/sym_l2_norm.txt')
            sys.exit()
    else:
        evals, _evecs = torch.symeig(mat)

    return torch.max(evals)


def inv_square_root_numpy(mat):
    assert type(mat) == np.ndarray
    return scipy.linalg.inv(scipy.linalg.sqrtm(mat))


def pinv_square_root_numpy(mat):
    assert type(mat) == np.ndarray
    result = scipy.linalg.inv(scipy.linalg.sqrtm(mat))
    return result


def erank(mat):
    """Effective rank of matrix."""
    return torch.trace(mat) / l2_norm(mat)


def rank(A):
    """Rank of a matrix"""
    U, S, V = torch.svd(A)
    cond = get_condition(A.dtype)
    cutoff = torch.max(S) * cond
    return torch.sum(S > cutoff).type(torch.get_default_dtype())


def sym_erank(mat):
    """Effective rank of symmetric matrix."""
    return torch.trace(mat) / sym_l2_norm(mat)


def lyapunov_spectral(A, B, cond=None):
    u.check_symmetric(A)
    u.check_symmetric(B)

    s, U = torch.symeig(A, eigenvectors=True)
    if cond is None:
        cond = get_condition(s.dtype)
    cutoff = cond * max(s)
    s = torch.where(s > cutoff, s, torch.tensor(0.).to(s.device))

    C = U.t() @ B @ U    # TODO(y): throw away eigenvectors corresponding to discarded evals. U=U[:num_eigs]
    s = s.unsqueeze(1) + s.unsqueeze(0)
    si = torch.where(s > 0, 1 / s, s)
    Y = C * si
    X = U @ Y @ U.t()

    # cancel small asymetries introduces by multiplication by small numbers
    X = (X + X.t()) / 2
    return X


def lyapunov_svd(A, C, rtol=1e-4, eps=1e-7, use_svd=False):
    """Solve AX+XA=C using SVD"""

    # This method doesn't work for singular matrices, so regularize it
    # TODO: can optimize performance by reusing eigenvalues from regularization computations
    A = regularize_mat(A, eps)
    C = regularize_mat(C, eps)

    assert A.shape[0] == A.shape[1]
    assert len(A.shape) == 2
    if use_svd:
        U, S, V = robust_svd(A)
    else:
        S, U = torch.symeig(A, eigenvectors=True)
    S = S.diag() @ torch.ones_like(A)
    X = U @ ((U.t() @ C @ U) / (S + S.t())) @ U.t()
    error = A @ X + X @ A - C
    relative_error = torch.max(torch.abs(error)) / torch.max(torch.abs(A))
    if relative_error > rtol:
        # TODO(y): currently spams with errors, implement another method based on Newton iteration
        pass
        print(f"Warning, error {relative_error} encountered in lyapunov_svd")

    return X


def deleteme():
    t = torch.ones(3, dtype=torch.float64)
    a = t / 2.


def lyapunov_svd2(A, C, rtol=1e-4, eps=1e-7, use_svd=False):
    """Solve AX+XA=C using SVD"""

    # This method doesn't work for singular matrices, so regularize it
    # TODO: can optimize performance by reusing eigenvalues from regularization computations
    # A = regularize_mat(A, eps)
    # C = regularize_mat(C, eps)

    assert A.shape[0] == A.shape[1]
    assert len(A.shape) == 2
    if use_svd:
        U, S, V = robust_svd(A)
    else:
        S, U = torch.symeig(A, eigenvectors=True)
    S = S.diag() @ torch.ones_like(A)
    factor = (S + S.t())
    cutoff = max(S) * get_condition(S)
    factor = torch.where(factor > cutoff, 1 / factor, factor)
    X = U @ ((U.t() @ C @ U) * factor) @ U.t()
    error = A @ X + X @ A - C
    relative_error = torch.max(torch.abs(error)) / torch.max(torch.abs(A))
    if relative_error > rtol:
        # TODO(y): currently spams with errors, implement another method based on Newton iteration
        pass
        print(f"Warning, error {relative_error} encountered in lyapunov_svd")

    return X


def lyapunov_truncated(A, C, use_svd=False, top_k=None, check_error=False):
    """Truncated solution to AX+XA=C. top_k specified how many dimensions of A to use. If None, use threshold for
    acceptable condition."""

    rankA = u.rank(A)
    rankC = u.rank(C)
    if rankA < rankC:
        print("Warning, losing precision, rank A: {rankA}, rank C: {rankC}")

    assert A.shape[0] == A.shape[1]
    assert len(A.shape) == 2
    cond = get_condition(A.dtype)

    if use_svd:
        U, S, V = torch.svd(A)
    else:
        # flip to be in decreasing order like for SVD
        # TODO: can optimize this case by skipping flip
        S, U = torch.symeig(A, eigenvectors=True)
        S = torch.flip(S, [0])
        U = torch.flip(U, [1])
    cutoff = torch.max(S) * cond
    rank = torch.sum(S > cutoff)
    if top_k is not None:
        top_k = rank

    S = torch.where(S > cutoff, S, torch.zeros_like(S))
    S = S.diag() @ torch.ones_like(A)
    U = U[:, :top_k]
    projected = U.t() @ C @ U
    divider = S + S.t()
    divider = divider[:top_k, :top_k]

    divided = torch.where(S > 0, projected / divider, torch.zeros_like(projected))
    X = U @ divided @ U.t()
    if check_error:
        error = A @ X + X @ A - C
        relative_error = torch.max(torch.abs(error)) / torch.max(torch.abs(A))
        if relative_error > 1e-3:
            print('rel error', relative_error)
    return X


def lyapunov_lstsq(A, C):
    """Slow explicit solution to least squares Lyapunov in kronecker expanded form."""
    n, n = A.shape
    ii = torch.eye(n)
    sol = torch.lstsq(u.vec(C), kron(ii, A) + kron(A.t(), ii))[0]
    return unvec(sol, n)


# TODO(y): reuse logic from above
def truncated_lyapunov_rho(A, C):
    """Returns quantities related to spectrum of solution to AX+XA=2C

    rho: measure of misfit
    erank: effective rank of X
    ."""

    C = 2 * C  # to center spectrum at 1
    assert A.shape[0] == A.shape[1]
    assert len(A.shape) == 2
    cond = get_condition(A.dtype)

    S, U = torch.symeig(A, eigenvectors=True)
    S = torch.flip(S, [0])
    U = torch.flip(U, [1])
    cutoff = torch.max(S) * cond
    rank = torch.sum(S > cutoff)
    top_k = rank

    S = S[:top_k].diag()
    S = S @ torch.ones_like(S)

    U = U[:, :top_k]
    projected = U.t() @ C @ U
    divider = S + S.t()
    divider = divider[:top_k, :top_k]

    divided = torch.where(S > 0, projected / divider, torch.zeros_like(projected))
    X = U @ divided @ U.t()
    nan_check(X)

    U, S, V = torch.svd(X)  # can we use symeig here?
    # S = torch.symeig(X).eigenvalues
    # S = u.filter_evals(S)

    erank = torch.sum(S) / torch.max(S)
    rho = A.shape[0] / erank
    spectrum = filter_evals(S)

    return rho, erank, spectrum


def outer(x, y=None):
    """Outer product of xy', treating x,y as column vectors. If y is not specified, compute xx'"""
    if y is None:
        y = x
    return x.unsqueeze(1) @ y.unsqueeze(0)


def to_python_scalar(x):
    """Convert object to Python scalar."""
    if hasattr(x, 'item'):
        return x.item()
    x = to_numpy(x).flatten()
    assert len(x) == 1
    return x[0]


def is_scalar(x):
    try:
        x = to_python_scalar(x)
    except:
        return False
    return True


def from_numpy(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.tensor(x)


_pytorch_floating_point_types = (torch.float16, torch.float32, torch.float64)

_numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def pytorch_dtype_to_floating_numpy_dtype(dtype):
    """Converts PyTorch dtype to numpy floating point dtype, defaulting to np.float32 for non-floating point types."""
    if dtype == torch.float64:
        dtype = np.float64
    elif dtype == torch.float32:
        dtype = np.float32
    elif dtype == torch.float16:
        dtype = np.float16
    else:
        dtype = np.float32
    return dtype


def to_normal_form(x):
    """Convert object to a normal expression, ie FactoredMatrix->Tensor, identity op for objects not in special form."""
    if hasattr(x, 'normal_form'):
        x = x.normal_form()
        assert not hasattr(x, 'normal_form'), 'infinite loop detected while expanding normal form'
    return x


def to_pytorch(x) -> torch.Tensor:
    """Convert numeric object to floating point PyTorch tensor."""
    x = to_normal_form(x)
    if type(x) == torch.Tensor:
        if x.dtype not in _pytorch_floating_point_types:
            x = x.type(torch.get_default_dtype())
        return x
    else:
        return from_numpy(to_numpy(x))


def to_pytorches(*xs) -> Tuple[torch.Tensor, ...]:
    return (to_pytorch(x) for x in xs)


def to_numpy(x, dtype: np.dtype = None) -> np.ndarray:
    """
    Convert numeric object to floating point numpy array. If dtype is not specified, use PyTorch default dtype.

    Args:
        x: numeric object
        dtype: numpy dtype, must be floating point

    Returns:
        floating point numpy array
    """

    assert np.issubdtype(dtype, np.floating), "dtype must be real-valued floating point"

    # Convert to normal_form expression from a special form (https://reference.wolfram.com/language/ref/Normal.html)
    if hasattr(x, 'normal_form'):
        x = x.normal_form()

    if type(x) == np.ndarray:
        assert np.issubdtype(x.dtype, np.floating), f"numpy type promotion not implemented for {x.dtype}"

    if type(x) == torch.Tensor:
        dtype = pytorch_dtype_to_floating_numpy_dtype(x.dtype)
        return x.detach().cpu().numpy().astype(dtype)

    # list or tuple, iterate inside to convert PyTorch arrrays
    if type(x) in [list, tuple]:
        x = [to_numpy(r) for r in x]

    # Some Python type, use numpy conversion
    result = np.array(x, dtype=dtype)
    assert np.issubdtype(result.dtype, np.number), f"Provided object ({result}) is not numeric, has type {result.dtype}"
    if dtype is None:
        return result.astype(pytorch_dtype_to_floating_numpy_dtype(torch.get_default_dtype()))
    return result


def to_numpys(*xs, dtype=np.float32):
    return (to_numpy(x, dtype) for x in xs)


def khatri_rao(A: torch.Tensor, B: torch.Tensor):
    """Khatri-Rao product.
     i'th column of result C_i is a Kronecker product of A_i and B_i

    Section 2.6 of Kolda, Tamara G., and Brett W. Bader. "Tensor decompositions and applications." SIAM review 51.3
    (2009): 455-500"""
    assert A.shape[1] == B.shape[1]
    # noinspection PyTypeChecker
    return torch.einsum("ik,jk->ijk", A, B).reshape(A.shape[0] * B.shape[0], A.shape[1])


def khatri_rao_t(A: torch.Tensor, B: torch.Tensor):
    """Transposed Khatri-Rao, inputs and outputs are transposed.

    i'th row of result C_i is a Kronecker product of corresponding rows of A and B"""

    assert A.shape[0] == B.shape[0]
    # noinspection PyTypeChecker
    return torch.einsum("ki,kj->kij", A, B).reshape(A.shape[0], A.shape[1] * B.shape[1])


# Autograd functions, from https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
# noinspection PyTypeChecker
def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y: torch.Tensor, x: torch.Tensor):
    return jacobian(jacobian(y, x, create_graph=True), x)


def pinv(mat: torch.Tensor, cond=None) -> torch.Tensor:
    """Computes pseudo-inverse of mat, treating eigenvalues below eps as 0.

        cond : float or None
        Cutoff for 'small' singular values. If omitted, singular values smaller
        than ``max(M,N)*largest_singular_value*eps`` are considered zero where
        ``eps`` is the machine precision.
        """

    # Take cut-off logic from scipy
    # https://github.com/ilayn/scipy/blob/0f4c793601ecdd74fc9826ac02c9b953de99403a/scipy/linalg/basic.py#L1307

    # assert False, "Disabled due to numerical instability, see test_pinverse"
    nan_check(mat)
    u, s, v = robust_svd(mat)
    if cond in [None, -1]:
        cond = torch.max(s) * max(mat.shape) * np.finfo(np.dtype('float32')).eps
    rank = torch.sum(s > cond)

    u = u[:, :rank]
    u /= s[:rank]
    return u @ v.t()[:rank]


def eig_real(mat: torch.Tensor) -> torch.Tensor:
    """Wrapper around torch.eig which discards imaginary values and returns result in descending order.

    Prints warning when non-zero imaginary parts detected, see "Criteria for the reality of matrix eigenvalues"
    Products of symmetric matrices are not symmetric but have eigenvalues for no imaginary parts.
    https://link.springer.com/article/10.1007%2FBF01195188
    """

    evals = torch.eig(mat).eigenvalues
    re_part = evals[:, 0]  # extract real part
    im_part = evals[:, 1]  # extract real part
    if im_part.sum() / evals.max() > 1e-7:
        print("Warning, eig_real is discarding non-zero imaginary parts")
    re_part = re_part.sort(descending=True).values
    return re_part


def pinv_square_root(mat: torch.Tensor, eps=1e-4) -> torch.Tensor:
    nan_check(mat)
    u, s, v = robust_svd(mat)
    one = torch.from_numpy(np.array(1))
    ivals: torch.Tensor = one / torch.sqrt(s)
    si = torch.where(s > eps, ivals, s)
    return u @ torch.diag(si) @ v.t()


def symeig_pos_evals(mat: torch.Tensor) -> torch.Tensor:
    """Returns positive eigenvalues from symeig in decreasing order (to match order of .svd())"""

    s, u = torch.symeig(mat, eigenvectors=False)
    return torch.flip(filter_evals(s, remove_negative=True), dims=[0])


def svd_pos_svals(mat):
    """Returns positive singular values of a matrix."""

    U, S, V = robust_svd(mat)
    return filter_evals(S)


def filter_evals(vals, cond=None, remove_small=True, remove_negative=True):
    """Given list of eigenvalues or singular values, remove values indistinguishable from noise and/or small values."""
    orig_vals = vals
    if cond is None:
        cond = get_condition(vals.dtype)
    above_cutoff = (abs(vals) > cond * torch.max(abs(vals)))
    if remove_small:
        vals = vals[above_cutoff]
    if remove_negative:
        vals = vals[vals > 0]
    #    if len(vals) == 0:
    #        print("Warning, got empty eigenvalue list")
    #        return orig_vals
    return vals


def isymsqrt(mat, *args):
    return symsqrt(mat, inverse=True, *args)


def symsqrt(mat, cond=None, return_rank=False, inverse=False):
    """Computes the symmetric square root of a symmetric matrix. Throws away small and negative eigenvalues."""

    nan_check(mat)
    s, u = torch.symeig(mat, eigenvectors=True)

    # check_symmetric(mat)

    # todo(y): dedupe with getcond
    cond_dict = {torch.float32: 1e3 * 1.1920929e-07, torch.float64: 1E6 * 2.220446049250313e-16}

    if cond in [None, -1]:
        cond = cond_dict[mat.dtype]

    # Note, this can include negative values, see https://github.com/pytorch/pytorch/issues/25972
    above_cutoff = (s > cond * torch.max(abs(s)))

    if torch.sum(above_cutoff) == 0:
        return torch.zeros_like(mat)

    sigma_diag = torch.sqrt(s[above_cutoff])
    if inverse:
        sigma_diag = 1 / sigma_diag
    u = u[:, above_cutoff]

    B = u @ torch.diag(sigma_diag) @ u.t()

    if torch.sum(torch.isnan(B)) > 0:
        if gl.debug_linalg_crashes:
            dump(mat, '/tmp/symsqrt.txt')
            assert False

    if return_rank:
        return B, len(sigma_diag)
    else:
        return B


def symsqrt_svd(mat: torch.Tensor):
    """Like symsqrt, but uses SVD."""

    u, s, v = robust_svd(mat)
    svals: torch.Tensor = torch.sqrt(s)
    eps = get_condition(mat.dtype) * torch.max(abs(s))
    si = torch.where(s > eps, svals, s)
    if len(si) == 0:
        return torch.zeros_like(mat)
    return u @ torch.diag(si) @ v.t()


def robust_svd(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Try to perform SVD and handle errors.
    """

    assert is_matrix(mat), f"shape {mat.shape}"
    try:
        U, S, V = torch.svd(mat)
    except Exception as e:  # this can fail, see https://github.com/pytorch/pytorch/issues/25978
        if is_square_matrix(mat):
            s = torch.symeig(mat).eigenvalues
            eps = get_condition(mat.dtype) * torch.max(abs(s))
        else:
            eps = get_condition(mat.dtype) * u.frobenius_norm(mat) / mat.shape[0]
        print(f"Warning, SVD diverged with {e}, regularizing with {eps}")
        mat = regularize_mat2(mat, eps * 2)
        U, S, V = torch.svd(mat)
    return U, S, V


def regularize_mat(mat, eps):
    rtol = l2_norm(mat) * eps
    atol = 1e-12
    return mat + torch.eye(mat.shape[0]) * (rtol + atol)


def regularize_mat2(mat, eps):
    """Adds a multiple of identity to matrix."""
    assert is_matrix(mat), f"{mat.shape}"
    if mat.shape[0] == mat.shape[1]:
        return mat + torch.eye(mat.shape[0]).to(gl.device) * eps
    if mat.shape[0] > mat.shape[1]:
        transpose = True
        mat = mat.T
    else:
        transpose = False
    reg = torch.cat([torch.eye(mat.shape[0]), torch.zeros(mat.shape[0], mat.shape[1] - mat.shape[0])], dim=1)
    mat = mat + reg.to(gl.device) * eps
    if transpose:
        return mat.T
    else:
        return mat


def symsqrt_dist(cov1: torch.Tensor, cov2: torch.Tensor) -> float:
    """Distance between square roots of matrices"""

    cov1 = to_pytorch(cov1)
    cov2 = to_pytorch(cov2)
    cov1 = symsqrt_svd(cov1)
    cov2 = symsqrt_svd(cov2)
    return torch.norm(cov1 - cov2).item()


def check_symmetric(mat):
    try:
        u.check_close(mat, mat.t())
    except:
        discrepancy = torch.max(abs(mat - mat.t()) / mat)
        print(f"warning, matrix not symmetric: {discrepancy}")


def check_close(a0, b0, rtol=1e-5, atol=1e-8, label: str = '') -> None:
    """Convenience method for check_equal with tolerances defaulting to typical errors observed in neural network
    ops in float32 precision."""
    return check_equal(a0, b0, rtol=rtol, atol=atol, label=label)


def check_equal(observed, truth, rtol=1e-9, atol=1e-12, label: str = '') -> None:
    """
    Assert fail any entries in two arrays are not close to each to desired tolerance. See np.allclose for meaning of rtol, atol

    """

    # special handling for lists, which could contain
    # if type(observed) == List and type(truth) == List:
    #    for a, b in zip(observed, truth):
    #        check_equal(a, b)

    truth = to_numpy(truth)
    observed = to_numpy(observed)

    # broadcast to match shapes if necessary
    if observed.shape != truth.shape:
        #        common_shape = (np.zeros_like(observed) + np.zeros_like(truth)).shape
        truth = truth + np.zeros_like(observed)
        observed = observed + np.zeros_like(truth)

    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    # run np.testing.assert_allclose for extra info on discrepancies
    if not np.allclose(observed, truth, rtol=rtol, atol=atol, equal_nan=True):
        print(f'Numerical testing failed for {label}')
        np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol, equal_nan=True)


def get_param(layer):  # TODO(y): deprecate?
    """Extract parameter out of layer, assumes there's just one parameter in a layer."""
    named_params = [(name, param) for (name, param) in layer.named_parameters()]
    assert len(named_params) == 1, named_params
    return named_params[0][1]


global_timeit_dict = {}


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
        global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
        # print(f"{interval_ms:8.2f}   {self.tag}")
        log_scalars({'time/' + self.tag: interval_ms})


def run_all_tests(module: nn.Module):
    class local_timeit:
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
            global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
            print(f"{interval_ms:8.2f}   {self.tag}")

    all_functions = inspect.getmembers(module, inspect.isfunction)
    for name, func in all_functions:
        if name.startswith("test_"):
            with local_timeit(name):
                func()
    print(module.__name__ + " tests passed.")


def freeze(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False
    setattr(layer, "frozen", True)


def unfreeze(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True
    setattr(layer, "frozen", False)


def mark_expensive(layer: nn.Module):
    setattr(layer, 'expensive', True)


def nest_stats(tag: str, stats) -> Dict:
    """Nest given dict of stats under tag using TensorBoard syntax /nest1/tag"""
    result = {}
    for key, value in stats.items():
        result[f"{tag}/{key}"] = value
    return result


def seed_random(seed: int) -> None:
    """Manually set seed to seed for configurable random number generators in current process."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TinyMNIST(datasets.MNIST):
    """Custom-size MNIST autoencoder dataset for debugging. Generates data/target images with reduced resolution and 0
    channels. When provided with original 28, 28 resolution, generates standard 1 channel MNIST dataset.

    Use original_targets kwarg to get original MNIST labels instead of autoencoder targets.


    """

    def __init__(self, dataset_root='/tmp/data', data_width=4, targets_width=4, dataset_size=0,
                 train=True, original_targets=None, loss_type=None):
        """

        Args:
            data_width: dimension of input images
            targets_width: dimension of target images
            dataset_size: number of examples, use for smaller subsets and running locally
            original_targets: if False, replaces original classification targets with image reconstruction targets
            loss_type: if LeastSquares, then convert classes to one-hot format
        """
        super().__init__(dataset_root, download=True, train=train)

        assert loss_type is None or original_targets is None  # can't specify both loss type and original targets
        assert loss_type in [None, 'LeastSquares', 'CrossEntropy']

        if loss_type is None and original_targets is None:
            original_targets = False  # default to LeastSquares targets
        if loss_type is not None:
            original_targets = True

        if dataset_size > 0:
            # assert dataset_size <= self.data.shape[0]
            self.data = self.data[:dataset_size, :, :]
            self.targets = self.targets[:dataset_size]

        if data_width != 28 or targets_width != 28:
            new_data = np.zeros((self.data.shape[0], data_width, data_width))
            new_targets = np.zeros((self.data.shape[0], targets_width, targets_width))
            for i in range(self.data.shape[0]):
                arr = self.data[i, :].numpy().astype(np.uint8)
                im = Image.fromarray(arr)
                im.thumbnail((data_width, data_width), Image.ANTIALIAS)
                new_data[i, :, :] = np.array(im) / 255
                im = Image.fromarray(arr)
                im.thumbnail((targets_width, targets_width), Image.ANTIALIAS)
                new_targets[i, :, :] = np.array(im) / 255
            self.data = torch.from_numpy(new_data).type(torch.get_default_dtype())
            if not original_targets:
                self.targets = torch.from_numpy(new_targets).type(torch.get_default_dtype())
        else:
            self.data = self.data.type(torch.get_default_dtype()).unsqueeze(1)
            if not original_targets:
                self.targets = self.data

        # self.data = self.data.type(torch.get_default_dtype())
        # if not original_targets:  # don't cast original int labels
        #    self.targets = self.targets.type(u.dtype)
        if loss_type == 'LeastSquares':  # convert to one-hot format
            new_targets = torch.zeros((self.targets.shape[0], 10))
            new_targets.scatter(1, self.targets.unsqueeze(1), 1)
            self.targets = new_targets

        self.data, self.targets = self.data.to(gl.device), self.targets.to(gl.device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


model_layer_map = {}
model_param_map = {}


class SimpleModel(nn.Module):
    """Simple sequential model. Adds layers[] attribute, flags to turn on/off hooks, and lookup mechanism from layer to parent
    model."""

    layers: List[nn.Module]
    all_layers: List[nn.Module]
    skip_forward_hooks: bool
    skip_backward_hooks: bool

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.skip_backward_hooks = False
        self.skip_forward_hooks = False

    def disable_hooks(self):
        self.skip_forward_hooks = True
        self.skip_backward_hooks = True

    def enable_hooks(self):
        self.skip_forward_hooks = False
        self.skip_backward_hooks = False

    # TODO(y): make public method
    def _finalize(self):
        """Extra logic shared across all SimpleModel instances."""
        # self.type(u.dtype)

        global model_layer_map
        for module in self.modules():
            model_layer_map[module] = self
        for param in self.parameters():
            model_layer_map[param] = self

        u.register_hooks(self)


# TODO(y): rename to LeastSquaresLoss
def least_squares(data, targets=None, aggregation='mean'):
    """Least squares loss (like MSELoss, but an extra 1/2 factor."""
    assert is_matrix(data), f"Expected matrix, got {data.shape}"
    assert aggregation in ('mean', 'sum')
    if targets is None:
        targets = torch.zeros_like(data)
    err = data - targets.view(-1, data.shape[1])
    normalizer = len(data) if aggregation == 'mean' else 1
    return torch.sum(err * err) / 2 / normalizer


def debug_least_squares(data, targets=None):
    """Least squares loss which weights one of the coordinates (for testing)."""
    if targets is None:
        targets = torch.zeros_like(data)
    err = data - targets.view(-1, data.shape[1])

    err[:, 0] *= 10

    return torch.sum(err * err) / 2 / len(data)


# Fork of SimpleModel that doesn't automatically register hooks, for autograd_lib.py refactoring
class SimpleModel2(nn.Module):
    """Simple sequential model. Adds layers[] attribute, flags to turn on/off hooks, and lookup mechanism from layer to parent
    model."""

    layers: List[nn.Module]
    all_layers: List[nn.Module]

    def __init__(self, *args, **kwargs):
        super().__init__()

    # TODO(y): make public method
    def _finalize(self):
        """Extra logic shared across all SimpleModel instances."""
        # self.type(u.dtype)

        global model_layer_map
        for module in self.modules():
            model_layer_map[module] = self
        for param in self.parameters():
            model_layer_map[param] = self


def get_parent_model(module_or_param) -> Optional[nn.Module]:
    """Returns root model for given parameter."""
    global model_layer_map
    global model_param_map
    if module_or_param in model_layer_map:
        assert module_or_param not in model_param_map
        return model_layer_map[module_or_param]
    if module_or_param in model_param_map:
        return model_param_map[module_or_param]


# Functions to capture backprops/activations and save them on the layer

# layer.register_forward_hook(capture_activations) -> saves activations/output as layer.activations/layer.output
# layer.register_backward_hook(capture_backprops)  -> appends each backprop to layer.backprops_list
# layer.weight.register_hook(save_grad(layer.weight)) -> saves grad under layer.weight.saved_grad
# util.clear_backprops(model) -> delete all values above

def capture_activations(module: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Saves activations (layer input) into layer.activations. """

    model = get_parent_model(module)
    if getattr(model, 'skip_forward_hooks', False):
        return
    assert not hasattr(module,
                       'activations'), "Seeing results of previous forward, call util.clear_backprops(model) to clear or do 'model.disable_hooks()'"
    assert len(input) == 1, "this was tested for single input layers only"
    setattr(module, "activations", input[0].detach())
    setattr(module, "output", output.detach())


def capture_backprops(module: nn.Module, _input, output):
    """Appends all backprops (Jacobian Lops from upstream) to layer.backprops_list.
    Using list in order to capture multiple backprop values for a single batch. Use util.clear_backprops(model)
    to clear all saved values.
    """
    model = get_parent_model(module)
    if getattr(model, 'skip_backward_hooks', False):
        return
    assert len(output) == 1, "this works for single variable layers only"
    if not hasattr(module, 'backprops_list'):
        setattr(module, 'backprops_list', [])
    assert len(
        module.backprops_list) < 100, "Possible memory leak, captured more than 100 backprops, comment this assert " \
                                      "out if this is intended."""

    module.backprops_list.append(output[0].detach())


def save_grad(param: nn.Parameter) -> Callable[[torch.Tensor], None]:
    """Hook to save gradient into 'param.saved_grad', so it can be accessed after model.zero_grad(). Only stores gradient
    if the value has not been set, call util.clear_backprops to clear it."""

    def save_grad_fn(grad):
        if not hasattr(param, 'saved_grad'):
            setattr(param, 'saved_grad', grad)

    return save_grad_fn


def clear_backprops(model: nn.Module) -> None:
    """model.zero_grad + delete all backprops/activations/saved_grad values"""
    model.zero_grad()
    for m in model.modules():
        if hasattr(m, 'backprops_list'):
            del m.backprops_list
        if hasattr(m, 'activations'):
            del m.activations
    for p in model.parameters():
        if hasattr(p, 'saved_grad'):
            del p.saved_grad


# TODO: remove?
def register_hooks(model: SimpleModel):
    # TODO(y): remove hardcoding of parameter name
    for layer in model.layers:
        assert not layer._forward_hooks, f"Some hooks already registered, bug? {layer._forward_hooks}"
        assert not layer._backward_hooks, f"Some hooks already registered, bug? {layer._backward_hooks}"

        layer.register_forward_hook(u.capture_activations)
        layer.register_backward_hook(u.capture_backprops)

    for param in model.parameters():
        assert not param._backward_hooks, f"Some param hooks already registered, bug? {param._backward_hooks}"
        param.register_hook(u.save_grad(param))


class SimpleFullyConnected(SimpleModel):
    """Simple feedforward network that works on images."""

    def __init__(self, d: List[int], nonlin=False, bias=False, dropout=False):
        """
        Feedfoward network of linear layers with optional ReLU nonlinearity. Stores layers in "layers" attr, ie
        model.layers[0] refers to first linear layer.

        Args:
            d: list of layer dimensions, ie [768, 20, 10] for MNIST 10-output with hidden layer of 20
            nonlin: whether to include ReLU nonlinearity
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):
            linear = nn.Linear(d[i], d[i + 1], bias=bias)
            setattr(linear, 'name', f'{i:02d}-linear')
            self.layers.append(linear)
            self.all_layers.append(linear)
            if nonlin:
                self.all_layers.append(nn.ReLU())
            if i <= len(d) - 3 and dropout:
                self.all_layers.append(nn.Dropout(p=0.5))
        self.predict = torch.nn.Sequential(*self.all_layers)

        super()._finalize()

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        return self.predict(x)


class SimpleFullyConnected2(SimpleModel2):
    """Simple feedforward network that works on images."""

    def __init__(self, d: List[int], nonlin=False, bias=False, last_layer_linear=False, dropout=False):
        """
        Feedfoward network of linear layers with optional ReLU nonlinearity. Stores layers in "layers" attr, ie
        model.layers[0] refers to first linear layer.

        Args:
            d: list of layer dimensions, ie [768, 20, 10] for MNIST 10-output with hidden layer of 20
            nonlin: whether to include ReLU nonlinearity
            last_layer_linear: don't apply nonlinearity to loast layer
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):
            linear = nn.Linear(d[i], d[i + 1], bias=bias)
            setattr(linear, 'name', f'{i:02d}-linear')
            self.layers.append(linear)
            self.all_layers.append(linear)
            if nonlin:
                if not last_layer_linear or i < len(d) - 2:
                    self.all_layers.append(nn.ReLU())
            if i <= len(d) - 3 and dropout:
                self.all_layers.append(nn.Dropout(p=0.5))
        self.predict = torch.nn.Sequential(*self.all_layers)

        super()._finalize()

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        return self.predict(x)


class SimpleMLP(nn.Module):
    """Simple feedforward network that works on images."""

    layers: List[nn.Module]
    all_layers: List[nn.Module]

    def __init__(self, d: List[int], nonlin=False, bias=False):
        """
        Feedfoward network of linear layers with optional ReLU nonlinearity. Stores layers in "layers" attr, ie
        model.layers[0] refers to first linear layer.

        Args:
            d: list of layer dimensions, ie [768, 20, 10] for MNIST 10-output with hidden layer of 20
            nonlin: whether to include ReLU nonlinearity
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):
            linear = nn.Linear(d[i], d[i + 1], bias=bias)
            setattr(linear, 'name', f'{i:02d}-linear')
            self.layers.append(linear)
            self.all_layers.append(linear)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        self.predict = torch.nn.Sequential(*self.all_layers)

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        return self.predict(x)


class RedundantFullyConnected2(SimpleModel2):
    """Simple feedforward network that works on images."""

    def __init__(self, d: List[int], nonlin=False, bias=False, last_layer_linear=False, dropout=False, redundancy=1):
        """
        Feedfoward network of linear layers with optional ReLU nonlinearity. Stores layers in "layers" attr, ie
        model.layers[0] refers to first linear layer.

        Args:
            d: list of layer dimensions, ie [768, 20, 10] for MNIST 10-output with hidden layer of 20
            nonlin: whether to include ReLU nonlinearity
            last_layer_linear: don't apply nonlinearity to loast layer
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        self.linear_groups = []
        self.dropout = dropout
        for i in range(len(d) - 1):
            group = []
            for l in range(redundancy):
                layer = nn.Linear(d[i], d[i + 1], bias=bias)
                group.append(layer)
                layer.weight.data.copy_(layer.weight.data / redundancy)
                if hasattr(layer, 'bias'):
                    layer.bias.data.copy_(layer.bias.data / redundancy)
                layer_name = f'layer%02d' % (i * redundancy + l,)
                setattr(self, layer_name, group[-1])  # needed to make params discoverable by optimizers
                #                print("adding layer ", layer_name, getattr(self, layer_name))
            self.linear_groups.append(group)
            if nonlin:
                if not last_layer_linear or i < len(d) - 2:
                    self.all_layers.append(nn.ReLU())
            if i <= len(d) - 3 and dropout:
                self.all_layers.append(nn.Dropout(p=0.5))
        super()._finalize()

    def forward(self, x: torch.Tensor):
        x = x.reshape((-1, self.d[0]))
        for layer_group in self.linear_groups:
            y = 0.
            for layer in layer_group:
                y0 = layer(x)
                if self.dropout:
                    y0 = F.dropout(y0, 0.01)
                y += y0

            y = F.relu(y)
            x = y
        return x


class SimpleConvolutional(SimpleModel):
    """Simple conv network."""

    def __init__(self, d: List[int], kernel_size=(2, 2), nonlin=False, bias=False):
        """

        Args:log
            d: list of channels, ie [2, 2] to have 2 conv layers with 2 channels
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):
            conv = nn.Conv2d(d[i], d[i + 1], kernel_size, bias=bias)
            setattr(conv, 'name', f'{i:02d}-conv')
            self.layers.append(conv)
            self.all_layers.append(conv)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        self.predict = torch.nn.Sequential(*self.all_layers)

        self._finalize()

    def forward(self, x: torch.Tensor):
        return self.predict(x)


class SimpleConvolutional2(SimpleModel2):
    """Simple conv network."""

    def __init__(self, d: List[int], kernel_size=(2, 2), nonlin=False, bias=False):
        """

        Args:
            d: list of channels, ie [2, 2] to have 2 conv layers with 2 channels
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        assert len(d) >= 2
        for di in d:
            assert di > 0

        for i in range(len(d) - 1):
            conv = nn.Conv2d(d[i], d[i + 1], kernel_size, bias=bias)
            setattr(conv, 'name', f'{i:02d}-conv')

            self.layers.append(conv)
            self.all_layers.append(conv)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        self.predict = torch.nn.Sequential(*self.all_layers)

        self._finalize()

    def forward(self, x: torch.Tensor):
        return self.predict(x)


class ReshapedConvolutional2(SimpleConvolutional2):
    """Simple conv network, output is flattened"""

    def __init__(self, *args, **kwargs):
        """

        Args:
            d: list of channels, ie [2, 2] to have 2 conv layers with 2 channels
        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        output = self.predict(x)
        return output.reshape(output.shape[0], -1)


class PooledConvolutional2(SimpleConvolutional2):
    """Simple conv network, output is pooled across spatial dimension. Num-channels = num_outputs"""

    def __init__(self, *args, **kwargs):
        """

        Args:
            d: list of channels, ie [2, 2, 2] to have 2 conv layers with 2 channels
        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        x = self.predict(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        return x.reshape(x.shape[0], -1)


class StridedConvolutional2(SimpleModel2):
    """Convolutional net without overlapping, single output, squeezes singleton spatial dimensions, rank-2 result"""

    def __init__(self, d: List[int], kernel_size=(2, 2), nonlin=False, bias=False):
        """

        Args:
            d: list of channels, ie [2, 2] to have 2 conv layers with 2 channels in each group
            o: number of output classes used in final classification layer
            input channels of input must by d[0]*o
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        for i in range(len(d) - 1):

            # each group considers o filters independently
            conv = nn.Conv2d(d[i], d[i + 1], kernel_size=kernel_size, stride=kernel_size, bias=bias)
            setattr(conv, 'name', f'{i:02d}-conv')
            self.layers.append(conv)
            self.all_layers.append(conv)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        # average out all groups of o filters

        self.final_chan = d[-1]
        self.predict = torch.nn.Sequential(*self.all_layers)

        self._finalize()

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.all_layers):
            x = layer(x)

        assert x.shape[2] == 1 and x.shape[3] == 1
        return x.reshape(x.shape[0], 1)


class GroupedConvolutional2(SimpleModel2):
    """Conv network without mixing of output dimension, applies convolution to o independent groups of d channels
    Each group is only affected by 1 output
    """

    def __init__(self, d: List[int], kernel_size=(2, 2), o=None, nonlin=False, bias=False):
        """

        Args:
            d: list of channels, ie [2, 2] to have 2 conv layers with 2 channels in each group
            o: number of output classes used in final classification layer
            input channels of input must by d[0]*o
        """
        super().__init__()
        self.layers: List[nn.Module] = []
        self.all_layers: List[nn.Module] = []
        self.d: List[int] = d
        self.o = o
        for i in range(len(d) - 1):

            # each group considers o filters independently
            conv = nn.Conv2d(d[i] * o, d[i + 1] * o, kernel_size, bias=bias, groups=o)
            setattr(conv, 'name', f'{i:02d}-conv')
            self.layers.append(conv)
            self.all_layers.append(conv)
            if nonlin:
                self.all_layers.append(nn.ReLU())
        # average out all groups of o filters

        self.final_chan = d[-1]
        self.predict = torch.nn.Sequential(*self.all_layers)

        self._finalize()

    def forward(self, x: torch.Tensor):
        x = self.predict(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # n, o*do, 1, 1
        n, out_dim, Oh, Ow = x.shape
        assert (Oh, Ow) == (1, 1)
        assert out_dim == self.final_chan * self.o
        x = x.reshape(n, self.o, self.final_chan)
        x = torch.einsum('noc->no', x)  # average across groups
        assert x.shape == (n, self.o)
        return x


class ReshapedConvolutional(SimpleConvolutional):
    """Simple conv network, output is flattened"""

    def __init__(self, *args, **kwargs):
        """

        Args:
            d: list of channels, ie [2, 2] to have 2 conv layers with 2 channels
        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        output = self.predict(x)
        return output.reshape(output.shape[0], -1)


def log_scalars(metrics: Dict[str, Any]) -> None:
    assert gl.event_writer is not None, "initialize event_writer as gl.event_writer = SummaryWriter(logdir)"
    for tag in metrics:
        gl.event_writer.add_scalar(tag=tag, scalar_value=metrics[tag], global_step=gl.get_global_step())
        # gl.event_writer.add_s
    if 'epoch' in metrics:
        print('logging at ', gl.get_global_step(), metrics.get('epoch', -1))


def log_scalar(**metrics) -> None:
    for tag in metrics:
        gl.event_writer.add_scalar(tag=tag, scalar_value=metrics[tag], global_step=gl.get_global_step())


# for line profiling
try:
    # noinspection PyUnboundLocalVariable
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator.


@profile
def log_spectrum(tag, vals: torch.Tensor, loglog=True, discard_tiny=False):
    """Given eigenvalues or singular values in decreasing order, log this plg."""

    if 'darwin' in platform.system().lower():
        import matplotlib
        matplotlib.use('PS')

    import matplotlib.pyplot as plt

    if discard_tiny:
        vals = filter_evals(vals)

    y = vals
    x = torch.arange(len(vals), dtype=vals.dtype) + 1.
    if loglog:
        y = torch.log10(y)
        x = torch.log10(x)
    fig, ax = plt.subplots()
    x, y = to_numpys(x, y)
    markerline, stemlines, baseline = ax.stem(x, y, markerfmt='bo', basefmt='r-', bottom=min(y))
    plt.setp(baseline, color='r', linewidth=2)
    gl.event_writer.add_figure(tag=tag, figure=fig, global_step=gl.get_global_step())


def get_events(fname, x_axis='step'):
    """Returns event dictionary for given run, has form
    {tag1: {step1: val1}, tag2: ..}

    If x_axis is set to "time", step is replaced by timestamp
    """

    from tensorflow.python.summary import summary_iterator  # local import because TF is heavy dep and only used here

    result = {}

    events = summary_iterator.summary_iterator(fname)

    try:
        for event in events:
            if x_axis == 'step':
                x_val = event.step
            elif x_axis == 'time':
                x_val = event.wall_time
            else:
                assert False, f"Unknown x_axis ({x_axis})"

            vals = {val.tag: val.simple_value for val in event.summary.value}
            # step_time: valuelayer
            for tag in vals:
                event_dict = result.setdefault(tag, {})
                if x_val in event_dict:
                    print(f"Warning, overwriting {tag} for {x_axis}={x_val}")
                    print(f"old val={event_dict[x_val]}")
                    print(f"new val={vals[tag]}")

                event_dict[x_val] = vals[tag]
    except Exception as e:
        print(e)
        pass

    return result


def infinite_iter(obj):
    """Wraps iterable object to restart on last iteration."""
    while True:
        for result in iter(obj):
            yield result


# noinspection PyTypeChecker
def dump(result, fname):
    """Save result to file. Load as np.genfromtxt(fname). """
    result = to_numpy(result)
    if result.shape == ():  # savetxt has problems with scalars
        result = np.expand_dims(result, 0)
    location = fname
    # special handling for integer datatypes
    if (
            result.dtype == np.uint8 or result.dtype == np.int8 or
            result.dtype == np.uint16 or result.dtype == np.int16 or
            result.dtype == np.uint32 or result.dtype == np.int32 or
            result.dtype == np.uint64 or result.dtype == np.int64
    ):
        np.savetxt(location, X=result, fmt="%d", delimiter=',')
    else:
        np.savetxt(location, X=result, delimiter=',')
    print("Dumping to", location)


def print_version_info():
    """Print version numbers of numerical packages in current env."""

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

    if np.__config__.get_info("lapack_mkl_info"):
        print("MKL version", get_mkl_version())
    else:
        print("not using MKL")

    print("PyTorch version", torch.version.__version__)

    print("Scipy version: ", scipy.version.full_version)
    print("Numpy version: ", np.version.full_version)
    print("Python version: ", sys.version, sys.platform)
    print_cpu_info()


def print_cpu_info():
    ver = 'unknown'
    try:
        for l in open("/proc/cpuinfo").read().split('\n'):
            if 'model name' in l:
                ver = l
                break
    except:
        pass

    # core counts from https://stackoverflow.com/a/23378780/419116
    print("CPU version: ", ver)
    sys.stdout.write("CPU logical cores: ")
    sys.stdout.flush()
    os.system(
        "echo $([ $(uname) = 'Darwin' ] && sysctl -n hw.logicalcpu_max || lscpu -p | egrep -v '^#' | wc -l)")
    sys.stdout.write("CPU physical cores: ")
    sys.stdout.flush()
    os.system(
        "echo $([ $(uname) = 'Darwin' ] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)")

    # get mapping of logical cores to physical sockets
    import re
    socket_re = re.compile(
        """.*?processor.*?(?P<cpu>\\d+).*?physical id.*?(?P<socket>\\d+).*?power""",
        flags=re.S)
    from collections import defaultdict
    socket_dict = defaultdict(list)
    try:
        for cpu, socket in socket_re.findall(open('/proc/cpuinfo').read()):
            socket_dict[socket].append(cpu)
    except FileNotFoundError:
        pass
    print("CPU physical sockets: ", len(socket_dict))


def move_to_gpu(tensors):
    return [tensor.cuda() for tensor in tensors]


def fmt(a):
    """Helper function for converting copy-pasted Mathematica matrices into Python."""

    a = a.replace('\n', '')
    print(a.replace("{", "[").replace("}", "]"))


def to_logits(p: torch.Tensor) -> torch.Tensor:
    """Inverse of F.softmax"""
    if len(p.shape) == 1:
        batch = torch.unsqueeze(p, 0)
    else:
        assert len(p.shape) == 2
        batch = p

    batch = torch.log(batch) - torch.log(batch[:, -1])
    return batch.reshape(p.shape)


class CrossEntropySoft(nn.Module):
    """Like torch.nn.CrossEntropyLoss but instead of class index it accepts a
    probability distribution.

    The `input` is expected to contain raw, unnormalized scores for each class.
    The `target` is expected to contain empirical probabilities for each class (positive and adding up to 1)

    """

    def __init__(self):
        super(CrossEntropySoft, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """

        # check that targets are positive and add up to 1
        assert (target >= 0).sum() == target.numel()
        sums = target.sum(dim=1)
        assert np.allclose(sums, torch.ones_like(sums))

        assert len(target.shape) == 2
        n = target.shape[0]
        log_likelihood = -F.log_softmax(inputs, dim=1)
        loss = torch.sum(torch.mul(target, log_likelihood)) / n

        return loss


def get_unique_logdir(root_logdir: str) -> str:
    """Increments suffix at the end of root_logdir until getting directory that doesn't exist locally, return that."""
    count = 0
    while os.path.exists(f"{root_logdir}{count:02d}"):
        count += 1

    return f"{root_logdir}{count:02d}"


from torch.utils.tensorboard import SummaryWriter


def setup_logdir_and_event_writer(run_name: str, init_wandb=False):
    """Creates unique logdir like project/runname02, sets up wandb if necessary"""
    assert gl.project_name is not None

    gl.logdir = u.get_unique_logdir(f'{gl.logdir_base}/{gl.project_name}/{run_name}')
    gl.run_name = os.path.basename(gl.logdir)
    gl.event_writer = SummaryWriter(gl.logdir)

    if init_wandb or (gl.args and hasattr(gl.args, 'wandb') and gl.args.wandb):
        wandb.init(project=gl.project_name, name=gl.run_name)
        wandb.tensorboard.patch(tensorboardX=False)
        wandb.config.update(vars(gl.args))


######################################################
# Hessian backward samplers
#
# A sampler provides a representation of hessian of the loss layer
#
# For a batch of size n,o, Hessian backward sampler will produce k backward values
# (k between 1 and o) where each value can be fed as model.backward(value)
#
# The covariance of gradients corresponding to these k backward will be summed up to form an estimate of Hessian of the network
# sum_i gg' \approx H
#
#   sampler = HessianSamplerMyLoss
#   for bval in sampler(model(batch)):
#       model.zero_grad()
#       model.backward(bval)

class HessianBackprop:
    num_samples: int  # number of samples


class HessianExactSqrLoss(HessianBackprop):
    """Sampler for loss err*err/2/len(batch), produces exact Hessian."""

    def __init__(self):
        super().__init__()

    def __call__(self, output: torch.Tensor):
        assert len(output.shape) == 2
        batch_size, output_size = output.shape
        self.num_samples = output_size

        id_mat = torch.eye(output_size)
        for out_idx in range(output_size):
            yield torch.stack([id_mat[out_idx]] * batch_size)


class HessianSampledSqrLoss(HessianBackprop):
    """Sampler for loss err*err/2/len(batch), produces exact Hessian."""

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __call__(self, output: torch.Tensor):
        assert len(output.shape) == 2
        batch_size, output_size = output.shape
        assert self.num_samples <= output_size, f"Requesting more samples than needed for exact Hessian computation " \
                                                f"({self.num_samples}>{output_size})"

        # exact sampler provides n samples whose outer products add up to Identity
        # here the sum is num_samples*identity in expectation
        # therefore must divide by sqrt(num_samples)

        for out_idx in range(self.num_samples):
            # sample random vectors of +1/-1's
            bval = torch.LongTensor(batch_size, output_size).to(gl.device).random_(0, 2) * 2 - 1
            yield bval.float() / math.sqrt(self.num_samples)


class HessianExactCrossEntropyLoss(HessianBackprop):
    """Sampler for nn.CrossEntropyLoss, produces exact Hessian."""

    def __init__(self):
        super().__init__()

    def __call__(self, logits: torch.Tensor):
        assert len(logits.shape) == 2

        n, d = logits.shape
        batch = F.softmax(logits, dim=1)

        mask = torch.eye(d).expand(n, d, d)
        diag_part = batch.unsqueeze(2).expand(n, d, d) * mask
        outer_prod_part = torch.einsum('ij,ik->ijk', batch, batch)
        hess = diag_part - outer_prod_part
        assert hess.shape == (n, d, d)

        for i in range(n):
            hess[i, :, :] = u.symsqrt(hess[i, :, :])

        for out_idx in range(d):
            sample = hess[:, out_idx, :]
            assert sample.shape == (n, d)
            yield sample


def hessian_from_backprops(A_t, Bh_t, bias=False):
    """Computes Hessian from a batch of forward and backward values.

    See documentation on HessianSampler for assumptions on how backprop values are generated

    For batch size n
    Forward values have shape n,layer_inputs
    Backward values is a list of length c of tensors of shape n,layer_outputs

    For exact Hessian computation, c is number of classes.

    Args:
      bias: if True, also return Hessian of the bias parameter
    """
    n = A_t.shape[0]
    Amat_t = torch.cat([A_t] * len(Bh_t), dim=0)  # todo: can instead replace with a khatri-rao loop
    Bmat_t = torch.cat(Bh_t, dim=0)
    Jb = u.khatri_rao_t(Bmat_t, Amat_t)  # batch Jacobian
    H = Jb.t() @ Jb / n

    if not bias:
        return H
    else:
        Hbias = Bmat_t.t() @ Bmat_t / n
        return H, Hbias


# TODO: rename to "mean_hess"
def per_example_hess(A_t, Bh_t, bias=False):
    """Computes Hessian from a batch of forward and backward values.


    Args:
      bias: if True, also return Hessian of the bias parameter
    """
    n = A_t.shape[0]
    in_dim = A_t.shape[1]
    out_dim = Bh_t[0].shape[1]
    o = len(Bh_t)
    assert Bh_t[0].shape[0] == n

    Amat_t = torch.stack([A_t] * len(Bh_t), dim=0)
    Bmat_t = torch.stack(Bh_t, dim=0)
    assert Amat_t.shape == (o, n, in_dim)
    assert Bmat_t.shape == (o, n, out_dim)

    # sum out output classes, get batch of per-example jacobians
    Ji = torch.einsum('oni,onj->nij', Bmat_t, Amat_t)
    assert Ji.shape == (n, out_dim, in_dim)
    Ji = Ji.reshape((n, out_dim * in_dim))  # individual jacobians

    # original Hessian computation
    Jb = u.khatri_rao_t(Bmat_t.reshape(o * n, -1), Amat_t.reshape(o * n, -1))
    Hmean = Jb.t() @ Jb / n

    # method 2: einsum-only version for mean hessian
    # o,n -> o,n,i,j
    Jb2 = torch.einsum('oni,onj->onij', Bmat_t, Amat_t)
    check_close(Jb2.reshape((o * n, out_dim * in_dim)), Jb)
    Hmean2 = torch.einsum('onij,onkl->ijkl', Jb2, Jb2).reshape((out_dim * in_dim,
                                                                out_dim * in_dim)) / n
    check_close(Hmean, Hmean2)

    # method 3: einsum-only for individual hessians
    # sum over classes, 
    Hi = torch.einsum('onij,onkl->nijkl', Jb2, Jb2)
    Hmean3 = Hi.mean(dim=0)
    Hmean3 = Hmean3.reshape((out_dim * in_dim, out_dim * in_dim))
    check_close(Hmean, Hmean3)

    # flatten last two pairs of dimensions for form d^2/dvec dvec
    Hi = Hi.reshape(n, out_dim * in_dim, out_dim * in_dim)
    if not bias:
        return Hi
    else:
        #        assert False, "not tested"
        Hb_i = torch.einsum('oni,onj->nij', Bmat_t, Bmat_t)
        #        Hbias = Bmat_t.t() @ Bmat_t / n
        return Hi, Hb_i


def kl_div_cov(mat1, mat2, eps=1e-3):
    """KL divergence between two zero centered Gaussian's with given covariance matrices."""

    evals1 = torch.symeig(mat1).eigenvalues
    evals2 = torch.symeig(mat2).eigenvalues
    k = mat1.shape[0]
    # scale regularizer in proportion to achievable numerical precision (taken from scipy.pinv2)
    l1 = torch.max(evals1) * k
    l2 = torch.max(evals2) * k
    l = max(l1, l2)
    reg = torch.eye(mat1.shape[0]) * l * eps
    mat1 = mat1 + reg
    mat2 = mat2 + reg

    div = torch.trace(mat1 @ torch.inverse(mat2)) - (torch.logdet(mat1) - torch.logdet(mat2)) - k
    return div


# Functions for kronecker factored representation. Matrix is given as tuple of two matrices
def kron_quadratic_form(H, dd):
    """dd @ H @ dd.t(),"""
    pass


def kron_trace(H: Tuple[torch.Tensor, torch.Tensor]):
    """trace(H)"""
    A, B = H
    return torch.trace(A) * torch.trace(B)


def kron_trace_matmul(H, sigma):
    """
    tr(H@sigma)
    """
    H = u.kron(H)
    sigma = u.kron(sigma)
    return torch.trace(H @ sigma)


def kron_pinv(H: Tuple):
    A, B = H
    return u.pinv(A), u.pinv(B)


def kron_nan_check(H):
    u.nan_check(H[0])
    u.nan_check(H[1])


def kron_fro_norm(H):
    return H[0].norm() * H[1].norm()


def kron_sym_l2_norm(H):
    return u.sym_l2_norm(H[0]) * u.sym_l2_norm(H[1])


def kron_inv(H):
    return torch.inverse(H[0]), torch.inverse(H[1])


def kron_sigma(G):
    Bt, At = G
    grad = torch.einsum('nij,nkl->ijkl', Bt, At)
    cov = torch.einsum('ij,kl->ijkl', grad, grad)


def kron_batch_sum(G: Tuple):
    """The format of gradient is G={Bt, At} where Bt is (n,do) and At is (n,di)"""
    Bt, At = G
    return torch.einsum('ni,nj->ij', Bt, At)


def chop(mat: torch.Tensor, eps=1e-10) -> torch.Tensor:
    """Set values below max(mat)*eps to zero"""

    mat = to_normal_form(mat)
    zeros = torch.zeros(mat.shape)
    return torch.where(abs(mat) < eps, zeros, mat)


def format_list(ll: List) -> str:
    formatted = ["%.2f" % (d,) for d in ll]
    return ', '.join(formatted)


def create_local_logdir(logdir) -> str:
    """Dedupes logdir by appending a number to avoid conflict with existing folder at logdir, ie logdir, logdir01"""
    attemp_count = 0
    while os.path.exists(f"{logdir}{attemp_count:02d}"):
        attemp_count += 1
    logdir = f"{logdir}{attemp_count:02d}"
    return logdir


class NoOp:
    """Dummy callable that accepts every signature"""

    def __getattr__(self, *_args, **_kwargs):
        def no_op(*_args, **_kwargs): pass

        return no_op


def install_pdb_handler():
    """Signals to automatically start pdb:
      1. CTRL+\\ breaks into pdb.
      2. pdb gets launched on exception.
    """

    import signal
    import pdb

    def handler(_signum, _frame):
        pdb.set_trace()

    signal.signal(signal.SIGQUIT, handler)

    # Drop into PDB on exception
    # from https://stackoverflow.com/questions/13174412
    def info(type_, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type_, value, tb)
        else:
            import traceback
            import pdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type_, value, tb)
            print()
            # ...then start the debugger in post-mortem mode.
            pdb.pm()

    sys.excepthook = info


def randomly_rotate(X: torch.Tensor) -> torch.Tensor:
    """Randomly rotate d,n data matrix X"""

    d, n = X.shape
    z = torch.randn((d, d), dtype=X.dtype)
    q, r = torch.qr(z)
    d = torch.diag(r)
    ph = d / abs(d)
    rot_mat = q * ph
    return rot_mat @ X


def random_cov(rank, d=None, n=20) -> torch.Tensor:
    """

    Args:
        rank: dimensionality of space spanned by data
        d: embedding dimension
        n: number of examples to generate covariance matrix

    Returns:
        covariance matrix of size d and rank rank
    """
    if d is None:
        d = rank
    assert d >= rank
    X = torch.randn((rank, n))
    X = torch.cat([X, torch.zeros(d - rank, n)])
    X = randomly_rotate(X)
    return X @ X.t() / n


def _to_mathematica(x):
    x = to_numpy(x)
    x = np.array2string(x, separator=',')
    x = x.replace('[', '{')
    x = x.replace(']', '}')
    x = x.replace('(', '')
    x = x.replace(')', '')
    x = x.replace('tensor', '')
    return x


def _from_mathematica(x):
    x = x.replace('\n', '')
    x = x.replace('{', '[')
    x = x.replace('}', ']')
    return x


def _dim_check(d, rank=0):
    assert d > 0
    assert d < 1e6
    assert 0 <= rank <= d


def random_cov_pair(shared_rank, independent_rank, d, n=20, strength=1):
    """Generate pair of covariance matrices which share covariance in subspace of dimension shared_rank,
    and have independent covariances of dimension independent_rank. Strength determines relative scale of covariance
    in the independent subspace."""
    _dim_check(d, shared_rank + independent_rank)
    _dim_check(d, shared_rank)
    _dim_check(d, independent_rank)
    print(shared_rank, independent_rank, d)
    shared = random_cov(shared_rank, d, n)
    if independent_rank == 0 or strength == 0:
        return shared, shared
    A = shared + strength * random_cov(independent_rank, d, n)
    B = shared + strength * random_cov(independent_rank, d, n)
    return A, B


def is_row_matrix(dd):
    return len(dd.shape) == 2 and dd.shape[0] == 1


def is_col_matrix(dd):
    return len(dd.shape) == 2 and dd.shape[1] == 1


def is_square_matrix(dd):
    return len(dd.shape) == 2 and dd.shape[0] == dd.shape[1] and dd.shape[0] >= 1


def is_vector(dd) -> bool:
    shape = dd.shape
    return len(shape) == 1 and shape[0] >= 1


def is_matrix(dd) -> bool:
    shape = dd.shape
    return len(shape) == 2 and shape[0] >= 1 and shape[1] >= 1


def eye(d: int) -> torch.Tensor:
    return torch.eye(d).to(gl.device)


def eye_like(X: torch.Tensor) -> torch.Tensor:
    """Create identity matrix of same shape as X."""
    # TODO(y): dedup with regularize to support rectangular matrices
    assert is_square_matrix(X)
    d = X.shape[0]

    return torch.eye(d).type(X.dtype).to(X.device)


def rmul(a: torch.Tensor, b):
    # https://github.com/pytorch/pytorch/issues/26333
    return b.__rmul__(a)


def matmul(a, b):
    try:
        return a @ b
    except TypeError:
        return rmatmul(a, b)


def rmatmul(a: torch.Tensor, b):
    return b.__rmatmul__(a)


# helper util for norm squared, usual norm is slow https://discuss.pytorch.org/t/torch-norm-3-6x-slower-than-manually-calculating-sum-of-squares/14684
def norm_squared(param):
    return (param * param).sum()


def dot_product(A, B):
    return (A * B).sum()  # computes tr(AB')


if __name__ == '__main__':
    run_all_tests(sys.modules[__name__])


# import matplotlib.pyplot as plt
#
#
# def spectral_plot(vals: torch.Tensor, loglog=True):
#     fig, ax = plt.subplots()
#     y = vals
#     x = torch.arange(len(vals), dtype=y.dtype) + 1.
#     if loglog:
#         y = torch.log10(y)
#         x = torch.log10(x)
#
#     markerline, stemlines, baseline = ax.stem(x, y, markerfmt='bo', basefmt='r-', bottom=min(y))
#     plt.setp(baseline, color='r', linewidth=2)
#
#     plt.show()


def copy_stats(shared_stats, stats):
    for key in shared_stats:
        assert key not in stats, f"Trying to overwrite {key}"
        stats[key] = shared_stats[key]
    return None


def skip_nans(t): return t[torch.isfinite(t)]


# list replacement. Workaround for AttrDict automatically converting list objects to Tuple
class MyList:
    def __init__(self, *args, **kwargs):
        super(MyList, self).__init__(*args, **kwargs)
        self.storage = list()

    def __getattr__(self, *_args, **_kwargs):
        return self.storage.__getattribute__(*_args, **_kwargs)

    def normal_form(self):
        return self.value()

    def value(self):
        return self.storage


def divide_attributes(d, n):
    """Helper util to divide all tensor attributes of d by n, return result as new AttrDict"""

    result = AttrDict()
    for val in d:
        if type(d[val]) == torch.Tensor:
            result[val] = d[val] / n
    return result
