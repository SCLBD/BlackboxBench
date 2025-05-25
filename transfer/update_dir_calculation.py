import torch
from utils.registry import Registry
from utils.registry import parse_name
import difflib


def build_update_dir_calculator(update_dir_calculator_pipeline):
    """
    Transform an input string into an update direction calculator.

    The minilanguage is as follows:
        fn1|fn2(arg1, arg2, ...)|...
    which describes the successive update 'fn's to the gradient,
    each function can optionally have one or more args, which are either positional or key:value.

    The output update function expects a pipeline of update direction calculators.

    :param update_dir_calculator_pipeline: A string describing the methods for calculating update direction.
    :return: update_dir_calculator.
    :raises: ValueError: if the name of update_dir_calculator is unknown.
    """

    pp = []
    if update_dir_calculator_pipeline:
        for update_dir_calculator_name in update_dir_calculator_pipeline.split('|'):
            try:
                pp.append(Registry.lookup(f"update_dir_calculation.{update_dir_calculator_name}")())
            except SyntaxError as err:
                raise ValueError(f"Syntax error on: {update_dir_calculator_name}") from err

        # apply convolution after variance tuning
        args = Registry.global_registry()['args']
        if 'TI' in args.input_transformation:
            kerlen = parse_name(difflib.get_close_matches('TI(kerlen=)', args.input_transformation.split('|'), 1, cutoff=0.1)[0])[2]['kerlen'] if 'kerlen' in args.input_transformation else 5
            TI_func = Registry.global_registry()['gradient_calculation.convolved_grad'](kerlen=kerlen)
            if 'var_tuning' in args.update_dir_calculation:
                pp.insert(1, TI_func)
            else:
                pp.insert(0, TI_func)

    def _calculators(args, gradient, grad_accumulate, grad_var_last):
        """The update dir calculator that is returned."""

        # Apply all optimization methods in a sequence.
        for calculator in pp:
            gradient, grad_accumulate = calculator(args, gradient, grad_accumulate, grad_var_last)
        return gradient, grad_accumulate

    return _calculators


@Registry.register("update_dir_calculation.sgd")
def sgd():
    """
    When applying this function, the update direction on the adversarial example is exactly its gradient.
    """
    def _sgd(args, gradient, grad_accumulate, grad_var_last):
        update_dir = gradient
        return update_dir, grad_accumulate
    return _sgd


@Registry.register("update_dir_calculation.momentum")
def momentum():
    """
    This function is the core of MI-FGSM, modified based on a third-party repo:
    link:
        https://github.com/SCLBD/Transfer_attack_RAP
    citation:
        @inproceedings{dong2018boosting,
          title={Boosting Adversarial Attacks with Momentum},
          author={Dong, Yinpeng and Liao, Fangzhou and Pang, Tianyu and Su, Hang and Zhu, Jun and Hu, Xiaolin and Li, Jianguo},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          year={2018}
        }
    """
    def _momentum(args, gradient, grad_accumulate, grad_var_last):
        if args.loss_function == 'max_logit':
            grad_accumulate = gradient + args.decay_factor * grad_accumulate
        else:
            if 'cwa' in args.grad_calculation:
                grad_accumulate = gradient / torch.norm(gradient, p=1) + args.decay_factor * grad_accumulate
            else:
                grad_accumulate = gradient / torch.mean(torch.abs(gradient), (1, 2, 3),
                                                        keepdim=True) + args.decay_factor * grad_accumulate
        update_dir = grad_accumulate
        return update_dir, grad_accumulate
    return _momentum


@Registry.register("update_dir_calculation.var_tuning")
def var_tuning(step_size=1):
    """
    This function is the core of VT, modified based on the following source:
    link:
        https://github.com/JHL-HUST/VT
    citation:
        @inproceedings{wang2021enhancing,
          title={Enhancing the transferability of adversarial attacks through variance tuning},
          author={Wang, Xiaosen and He, Kun},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          pages={1924--1933},
          year={2021}
        }
    """
    def _var_tuning(args, gradient, grad_accumulate, grad_var_last):
        update_dir = gradient + step_size*grad_var_last
        return update_dir, grad_accumulate
    return _var_tuning

