"""
Implements handy numerical computational functions
"""
import numpy as np
import tensorflow as tf
import torch as ch
from torch.nn.modules import Upsample


def norm(t):
    """
    Return the norm of a tensor (or numpy) along all the dimensions except the first one
    :param t:
    :return:
    """
    _shape = t.shape
    batch_size = _shape[0]
    num_dims = len(_shape[1:])
    if ch.is_tensor(t):
        norm_t = ch.sqrt(t.pow(2).sum(dim=[_ for _ in range(1, len(_shape))])).view([batch_size] + [1] * num_dims)
        norm_t += (norm_t == 0).float() * np.finfo(np.float64).eps
        return norm_t
    else:
        _norm = np.linalg.norm(
            t.reshape([batch_size, -1]), axis=1, keepdims=1
        ).reshape([batch_size] + [1] * num_dims)
        return _norm + (_norm == 0) * np.finfo(np.float64).eps


def eg_step(x, g, lr):
    """
    Performs an exponentiated gradient step in the convex body [-1,1]
    :param x: batch_size x dim x .. tensor (or numpy) \in [-1,1]
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :return:
    """
    # from [-1,1] to [0,1]
    real_x = (x + 1.) / 2.
    if ch.is_tensor(x):
        pos = real_x * ch.exp(lr * g)
        neg = (1 - real_x) * ch.exp(-lr * g)
    else:
        pos = real_x * np.exp(lr * g)
        neg = (1 - real_x) * np.exp(-lr * g)
    new_x = pos / (pos + neg)
    return new_x * 2 - 1


def step(x, g, lr):
    """
    Performs a step with no lp-ball constraints
    :param x: batch_size x dim x .. tensor (or numpy)
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :return:
    """
    return x + lr * g


def lp_step(x, g, lr, p):
    """
    performs lp step of x in the direction of g, where the norm is computed
    across all the dimensions except the first one (assuming it's the batch_size)
    :param x: batch_size x dim x .. tensor (or numpy)
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :param p: 'inf' or '2'
    :return:
    """
    if p == 'inf':
        return linf_step(x, g, lr)
    elif p == '2':
        return l2_step(x, g, lr)
    else:
        raise Exception('Invalid p value')


def l2_step(x, g, lr):
    """
    performs l2 step of x in the direction of g, where the norm is computed
    across all the dimensions except the first one (assuming it's the batch_size)
    :param x: batch_size x dim x .. tensor (or numpy)
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :return:
    """
    # print(x.device)
    # print(g.device)
    # print(norm(g).device)
    return x + lr * g / norm(g)


def linf_step(x, g, lr):
    """
    performs linfinity step of x in the direction of g
    :param x: batch_size x dim x .. tensor (or numpy)
    :param g: batch_size x dim x .. tensor (or numpy)
    :param lr: learning rate (step size)
    :return:
    """
    if ch.is_tensor(x):
        return x + lr * ch.sign(g)
    else:
        return x + lr * np.sign(g)


def l2_proj_maker(xs, eps):
    """
    makes an l2 projection function such that new points
    are projected within the eps l2-balls centered around xs
    :param xs:
    :param eps:
    :return:
    """
    if ch.is_tensor(xs):
        orig_xs = xs.clone()

        def proj(new_xs):
            delta = new_xs - orig_xs
            norm_delta = norm(delta)
            if np.isinf(eps):  # unbounded projection
                return orig_xs + delta
            else:
                return orig_xs + (norm_delta <= eps).float() * delta + (
                        norm_delta > eps).float() * eps * delta / norm_delta
    else:
        orig_xs = xs.copy()

        def proj(new_xs):
            delta = new_xs - orig_xs
            norm_delta = norm(delta)
            if np.isinf(eps):  # unbounded projection
                return orig_xs + delta
            else:
                return orig_xs + (norm_delta <= eps) * delta + (norm_delta > eps) * eps * delta / norm_delta
    return proj


def linf_proj_maker(xs, eps):
    """
    makes an linf projection function such that new points
    are projected within the eps linf-balls centered around xs
    :param xs:
    :param eps:
    :return:
    """
    if ch.is_tensor(xs):
        orig_xs = xs.clone()

        def proj(new_xs):
            return orig_xs + ch.clamp(new_xs - orig_xs, - eps, eps)
    else:
        orig_xs = xs.copy()

        def proj(new_xs):
            return np.clip(new_xs, orig_xs - eps, orig_xs + eps)
    return proj


def upsample_maker(target_h, target_w):
    """
    makes an upsampler which takes a numpy tensor of the form
    minibatch x channels x h x w and casts to
    minibatch x channels x target_h x target_w
    :param target_h: int to specify the desired height
    :param target_w: int to specify the desired width
    :return:
    """
    _upsampler = Upsample(size=(target_h, target_w))

    def upsample_fct(xs):
        if ch.is_tensor(xs):
            return _upsampler(xs)
        else:
            return _upsampler(ch.from_numpy(xs)).numpy()

    return upsample_fct


def hamming_dist(a, b):
    """
    reurns the hamming distance of a to b
    assumes a and b are in {+1, -1}
    :param a:
    :param b:
    :return:
    """
    assert np.all(np.abs(a) == 1.), "a should be in {+1,-1}"
    assert np.all(np.abs(b) == 1.), "b should be in {+1,-1}"
    return sum([_a != _b for _a, _b in zip(a, b)])


def tf_nsign(t):
    """
    implements a custom non-standard sign operation in tensor flow
    where sing(t) = 1 if t == 0
    :param t:
    :return:
    """
    return tf.sign(tf.sign(t) + 0.5)


def sign(t, is_ns_sign=True):
    """
    Given a tensor t of `batch_size x dim` return the (non)standard sign of `t`
    based on the `is_ns_sign` flag
    :param t: tensor of `batch_size x dim`
    :param is_ns_sign: if True uses the non-standard sign function
    :return:
    """
    _sign_t = ch.sign(t) if ch.is_tensor(t) else np.sign(t)
    if is_ns_sign:
        _sign_t[_sign_t == 0.] = 1.
    return _sign_t


def noisy_sign(t, retain_p=1, crit='top', is_ns_sign=True):
    """
    returns a noisy version of the tensor `t` where
    only `retain_p` * 100 % of the coordinates retain their sign according
    to a `crit`.
    The noise is of the following effect
        sign(t) * x where x \in {+1, -1}
    Thus, if sign(t) = 0, sign(t) * x is always 0 (in case of `is_ns_sign=False`)
    :param t: tensor of `batch_size x dim`
    :param retain_p: fraction of coordinates
    :param is_ns_sign: if True uses  the non-standard sign function
    :return:
    """
    assert 0. <= retain_p <= 1., "retain_p value should be in [0,1]"

    _shape = t.shape
    t = t.reshape(_shape[0], -1)
    batch_size, dim = t.shape

    sign_t = sign(t, is_ns_sign=is_ns_sign)
    k = int(retain_p * dim)

    if k == 0:  # noise-ify all
        return (sign_t * np.sign((np.random.rand(batch_size, dim) < 0.5) - 0.5)).reshape(_shape)
    if k == dim:  # retain all
        return sign_t.reshape(_shape)

    # do topk otheriwise
    noisy_sign_t = sign_t * np.sign((np.random.rand(*t.shape) < 0.5) - 0.5)
    _rows = np.zeros((batch_size, k), dtype=np.intp) + np.arange(batch_size)[:, None]
    if crit == 'top':
        _temp = np.abs(t)
    elif crit == 'random':
        _temp = np.random.rand(*t.shape)
    else:
        raise Exception('Unknown criterion for topk')

    _cols = np.argpartition(_temp, -k, axis=1)[:, -k:]
    noisy_sign_t[_rows, _cols] = sign_t[_rows, _cols]
    return noisy_sign_t.reshape(_shape)
