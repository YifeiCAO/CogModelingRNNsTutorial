"""Gate analysis for hybrnn(.twostep).BiChannelRNN.

Extract the value-module gate trace from a trained BiChannelRNN and relate it
to the reward-prediction error (RPE). Use this to check whether a freely
*learned* gate (gate_mode='learnable') ends up tracking surprise -- the
interpretability comparison against the 'surprise'-driven and 'additive'
variants.

Typical use (run from the repo root, like cv_runner.py):

    import numpy as np
    import gate_analysis as ga

    rl_params  = {'w_h': 1.0, 'w_v': 1.0, 'fit_forget': False, 'forget': 0.0}
    net_params = {'n_actions': 3, 'hidden_size': 32}

    params = ga.fit_bichannel(xs, ys, rl_params, net_params, 'learnable')
    logits, gate, rpe = ga.unroll_gates(params, rl_params, net_params,
                                        'learnable', xs)
    print(ga.gate_rpe_correlation(gate, rpe, valid_mask=(xs[:, :, 0] >= 0)))
"""
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from CogModelingRNNsTutorial import hybrnn, hybrnn_twostep


def _module(twostep):
    return hybrnn_twostep if twostep else hybrnn


def fit_bichannel(xs, ys, rl_params, network_params, gate_mode,
                  twostep=False, n_steps=3000, lr=1e-3, seed=0, verbose=True):
    """Quick standalone fit of a BiChannelRNN (plain NLL on choices).

    Convenience trainer for gate analysis -- trains on the whole array with no
    train/val/test masking. This is NOT the nested-CV protocol used for model
    scoring (cv_runner.py); it just produces a trained model to inspect.

    Args:
      xs: (T, N, features) inputs -- [action, reward] or [action, reward, transition]
      ys: (T, N, 1) integer choices; negative entries are treated as padding
      rl_params, network_params: passed to BiChannelRNN
      gate_mode: 'additive' | 'learnable' | 'surprise'
      twostep: use the hybrnn_twostep variant
    Returns:
      trained Haiku params (compatible with unroll_gates)
    """
    mod = _module(twostep)

    def unroll(x):
        core = mod.BiChannelRNN(rl_params, network_params, gate_mode=gate_mode)
        state = core.initial_state(x.shape[1])
        out, _ = hk.dynamic_unroll(core, x, state)
        return out

    net = hk.transform(unroll)
    xs = jnp.asarray(xs, jnp.float32)
    y = jnp.asarray(ys)[:, :, 0].astype(jnp.int32)
    key = jax.random.PRNGKey(seed)
    params = net.init(key, xs)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(p, k):
        logits = net.apply(p, k, xs)
        logp = jax.nn.log_softmax(logits, axis=-1)
        n = logits.shape[-1]
        valid = (y >= 0) & (y < n)
        oh = jax.nn.one_hot(jnp.clip(y, 0, n - 1), n)
        ll = jnp.sum(oh * logp, axis=-1)
        return -jnp.sum(ll * valid) / jnp.maximum(jnp.sum(valid), 1)

    @jax.jit
    def step(p, opt_s, k):
        loss, grad = jax.value_and_grad(loss_fn)(p, k)
        upd, opt_s = opt.update(grad, opt_s)
        return optax.apply_updates(p, upd), opt_s, loss

    for i in range(n_steps):
        key, k = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, k)
        if verbose and (i + 1) % max(1, n_steps // 5) == 0:
            print(f"  step {i + 1:5d}/{n_steps}  train NLL = {float(loss):.4f}")
    return params


def unroll_gates(params, rl_params, network_params, gate_mode, xs, twostep=False):
    """Unroll a trained BiChannelRNN; return per-trial (logits, gate, rpe).

    Returns three numpy arrays:
      logits (T, N, n_actions)
      gate   (T, N, 1)  value-module gate in [0, 1]; all-NaN if gate_mode='additive'
      rpe    (T, N, 1)  reward - value[chosen] at each trial (the surprise signal)
    """
    mod = _module(twostep)

    def f(x):
        core = mod.BiChannelRNN(rl_params, network_params, gate_mode=gate_mode)
        state = core.initial_state(x.shape[1])

        def step(carry, inp):
            out, new_state = core.call_with_gate(inp, carry)
            return new_state, out

        _, (logits, gate, rpe) = hk.scan(step, state, x)
        return logits, gate, rpe

    net = hk.transform(f)
    logits, gate, rpe = net.apply(params, jax.random.PRNGKey(0),
                                  jnp.asarray(xs, jnp.float32))
    return np.asarray(logits), np.asarray(gate), np.asarray(rpe)


def gate_rpe_correlation(gate, rpe, valid_mask=None):
    """Pearson correlation between the gate and the RPE (and |RPE|).

    Args:
      gate, rpe: (T, N, 1) arrays from unroll_gates
      valid_mask: optional (T, N) bool -- e.g. (xs[:, :, 0] >= 0) to drop padding
    Returns:
      dict with n, r_gate_rpe, r_gate_absrpe, gate_mean, gate_std.
      r_* are NaN for additive mode (gate is all NaN) or a constant gate.
    """
    g = np.asarray(gate).reshape(-1)
    r = np.asarray(rpe).reshape(-1)
    m = np.isfinite(g) & np.isfinite(r)
    if valid_mask is not None:
        m &= np.asarray(valid_mask).reshape(-1).astype(bool)
    g, r = g[m], r[m]

    out = {"n": int(g.size)}
    if g.size < 2 or np.std(g) == 0 or np.std(r) == 0:
        out.update(r_gate_rpe=float("nan"), r_gate_absrpe=float("nan"),
                   gate_mean=float(g.mean()) if g.size else float("nan"),
                   gate_std=float(g.std()) if g.size else float("nan"))
        return out
    out.update(
        r_gate_rpe=float(np.corrcoef(g, r)[0, 1]),
        r_gate_absrpe=float(np.corrcoef(g, np.abs(r))[0, 1]),
        gate_mean=float(g.mean()),
        gate_std=float(g.std()),
    )
    return out
