"""Evaluate approximation quality of factoring on conv2d layers.

Evaluates discrepancy in magnitude (l2 norm) and value (difference between normalized square roots), 0 means perfect agreement

For LeastSquares loss: kron and mean_kron are exact for all combinations with kernel_size=1
For CrossEntropy loss: kron is exact for all combinations with kernel_size=1, num_channels=1
                       mean_kron is exact for all combinations with kernel_size=1


Increasing kernel_size>1, all methods over-estimate the Hessian magnitude, even in 1 channel/LeastSquares loss case

========== mean_kron ========================================
image_size
   value: : [1.1446763892308809e-06, 9.212457143803476e-07, 1.4112242752162274e-06, 1.2591850691023865e-06, 1.194795800074644e-06]
   magnitude : 1.00, 1.00, 1.00, 1.00, 1.00
num_channels
   value: : [5.330160206540313e-07, 9.375009426548786e-07, 1.2972760714546894e-06, 1.1446763892308809e-06, 1.0141359780391213e-05, 9.359457180835307e-06, 6.479138392023742e-06, 6.426676463888725e-06, 8.470763532386627e-06, 5.907071681576781e-06]
   magnitude : 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00
kernel_size
   value: : [1.1446763892308809e-06, 0.8146935105323792, 0.3585125207901001, 0.6153226494789124, 0.2633444368839264]
   magnitude : 1.00, 1.12, 1.13, 1.02, 1.00

========== kron ========================================
image_size
   value: : [0.012090419419109821, 0.018052944913506508, 0.004046864341944456, 0.006037037819623947, 0.008428926579654217]
   magnitude : 1.00, 1.01, 1.00, 1.01, 1.01
num_channels
   value: : [0.004109012894332409, 0.029988422989845276, 0.0047996193170547485, 0.012090419419109821, 0.02191711962223053, 0.012329756282269955, 0.12856155633926392, 0.015723882243037224, 0.027835888788104057, 0.012887174263596535]
   magnitude : 1.00, 1.01, 1.00, 1.00, 1.03, 1.00, 1.02, 1.01, 1.00, 1.00
kernel_size
   value: : [0.012090419419109821, 0.8148165941238403, 0.358644038438797, 0.6153406500816345, 0.263365775346756]
   magnitude : 1.00, 1.12, 1.13, 1.02, 1.00
"""

# Tests that compare manual computation of quantities against PyTorch autograd

from typing import List

import autograd_lib
import torch
import util as u
from attrdict import AttrDict

unfold = torch.nn.functional.unfold
fold = torch.nn.functional.fold


def compute_hess(n: int = 1, image_size: int = 1, kernel_size: int = 1, num_channels: int = 1, num_layers: int = 1,
                 nonlin: bool = False,
                 loss: str = 'CrossEntropy', method='exact', param_name='weight') -> List[torch.Tensor]:
    """

    Compute Hessians for all layers for given architecture

    Args:
        param_name: which parameter to compute ('weight' or 'bias')
        n: number of examples
        image_size:  width of image (square image)
        kernel_size: kernel size
        num_channels:
        num_layers:
        nonlin
        loss: LeastSquares or CrossEntropy
        method: 'kron', 'mean_kron'
        num_layers: number of layers in the network

    Returns:
        list of num_layers Hessian matrices.
    """

    assert param_name in ['weight', 'bias']
    assert loss in autograd_lib._supported_losses
    assert method in autograd_lib._supported_methods

    u.seed_random(1)

    Xh, Xw = 1, image_size
    Kh, Kw = 1, kernel_size
    dd = [num_channels] * (num_layers + 1)

    model: u.SimpleModel2 = u.PooledConvolutional2(dd, kernel_size=(Kh, Kw), nonlin=nonlin, bias=True)
    # model: u.SimpleModel2 = u.StridedConvolutional2(dd, kernel_size=(Kh, Kw), nonlin=nonlin, bias=True)
    data = torch.randn((n, dd[0], Xh, Xw))

    autograd_lib.clear_backprops(model)
    autograd_lib.add_hooks(model)
    output = model(data)
    autograd_lib.backprop_hess(output, hess_type=loss)
    autograd_lib.compute_hess(model, method=method)
    autograd_lib.disable_hooks()

    result = []
    for i in range(len(model.layers)):
        param = getattr(model.layers[i], param_name)
        if method == 'exact' or method == 'autograd':
            result.append(param.hess)
        else:
            result.append(param.hess_factored.expand())
    return result


def main():
    # for kernel_size=1, mean kron factoring works for any image size
    main_vals = AttrDict(n=2, kernel_size=1, image_size=625, num_channels=5, num_layers=4, loss='CrossEntropy',
                         nonlin=False)

    hess_list1 = compute_hess(method='exact', **main_vals)
    hess_list2 = compute_hess(method='kron', **main_vals)
    value_error = max([u.symsqrt_dist(h1, h2) for h1, h2 in zip(hess_list1, hess_list2)])
    magnitude_error = max([u.l2_norm(h2) / u.l2_norm(h1) for h1, h2 in zip(hess_list1, hess_list2)])
    print(value_error)
    print(magnitude_error)

    dimension_vals=dict(image_size=[2, 3, 4, 5, 6], num_channels=range(2, 12), kernel_size=[1, 2, 3, 4, 5])
    for method in ['mean_kron', 'kron']:  # , 'experimental_kfac']:
        print()
        print('='*10, method, '='*40)
        for dimension in ['image_size', 'num_channels', 'kernel_size']:
            value_errors = []
            magnitude_errors = []
            for val in dimension_vals[dimension]:
                vals = AttrDict(main_vals.copy())
                vals.method = method
                vals[dimension] = val
                vals.image_size = max(vals.image_size, vals.kernel_size ** vals.num_layers)
                # print(vals)
                vals_exact = AttrDict(vals.copy())
                vals_exact.method = 'exact'
                hess_list1 = compute_hess(**vals_exact)
                hess_list2 = compute_hess(**vals)
                magnitude_error = max([u.l2_norm(h2) / u.l2_norm(h1) for h1, h2 in zip(hess_list1, hess_list2)])
                hess_list1 = [h/u.l2_norm(h) for h in hess_list1]
                hess_list2 = [h/u.l2_norm(h) for h in hess_list2]

                value_error = max([u.symsqrt_dist(h1, h2) for h1, h2 in zip(hess_list1, hess_list2)])
                value_errors.append(value_error)
                magnitude_errors.append(magnitude_error.item())
            print(dimension)
            print('   value: :', value_errors)
            print('   magnitude :', u.format_list(magnitude_errors))


if __name__ == '__main__':
    main()
