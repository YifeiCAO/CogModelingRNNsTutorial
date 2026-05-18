"""
Nested cross-validation following Nature protocol:
  - Within-session trial split: train / val / test by trial index
  - Model sees full sequence as input; loss masked to relevant trial set
  - Test trial indices NEVER contribute gradients
  - Outer CV: session-level GroupKFold (optional group labels per dataset)
  - Inner search: 3-fold session CV × 2 seeds

Nature argument on data leakage:
  Training input contains test-trial (a,r) as context, but loss/gradient
  never backpropagates through test-trial predictions. Thus the model is
  never trained on the mapping tested at evaluation time.
"""
import os, sys, time, itertools, pickle, warnings
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from sklearn.model_selection import KFold, GroupKFold

from CogModelingRNNsTutorial import rnn_utils, bandits, hybrnn_twostep

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── trial-type constants ──
TRIAL_TRAIN = 0
TRIAL_VAL   = 1
TRIAL_TEST  = 2

# ──────────────────────────────────────────────
# 1. TRIAL MASK CONSTRUCTION
# ──────────────────────────────────────────────

def build_trial_type(ys, train_ratio=0.75, val_ratio=0.125):
    """
    Assign each trial within each session to train / val / test.
    Splits are applied to *valid* trials only (y >= 0).

    Returns: trial_type (T, N, 1) int array, values 0/1/2.
    """
    T, N, _ = ys.shape
    trial_type = np.full((T, N, 1), -1, dtype=int)

    for sess in range(N):
        y_col = ys[:, sess, 0]
        valid = y_col >= 0
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        n_train = max(1, int(np.round(n_valid * train_ratio)))
        n_val   = max(1, int(np.round(n_valid * val_ratio)))

        valid_idx = np.where(valid)[0]
        train_idx = valid_idx[:n_train]
        val_idx   = valid_idx[n_train:n_train + n_val]
        test_idx  = valid_idx[n_train + n_val:]

        trial_type[train_idx, sess, 0] = TRIAL_TRAIN
        trial_type[val_idx,   sess, 0] = TRIAL_VAL
        trial_type[test_idx,  sess, 0] = TRIAL_TEST

    return trial_type


# ──────────────────────────────────────────────
# 2. MASKED LOSS & TRAINING
# ──────────────────────────────────────────────

def train_with_trial_split(
    model_fun,
    xs,
    ys,
    trial_type,
    optimizer=None,
    batch_size=64,
    n_steps_max=100000,
    eval_every=200,
    early_stop_patience=500,
    seed=0,
    verbose=True,
):
    """
    Train using within-session trial split.

    xs, ys: full dataset (all sessions for this training round)
    trial_type: (T, N, 1) with values 0=train, 1=val, 2=test
    Loss is computed on train trials; validation on val trials.
    Test trials are NEVER included in loss/gradients during training.
    """
    if optimizer is None:
        optimizer = optax.chain(optax.add_decayed_weights(1e-5), optax.adam(1e-4))

    # Pack trial_type into ys so batching stays aligned
    ys_packed = np.concatenate([ys.astype(float), trial_type.astype(float)], axis=-1)  # (T,N,2)
    dataset = rnn_utils.DatasetRNN(xs, ys_packed, batch_size=batch_size)

    key = jax.random.PRNGKey(seed)
    sample_xs, _ = next(dataset)

    def unroll_network(xs_in):
        core = model_fun()
        bs = jnp.shape(xs_in)[1]
        state = core.initial_state(bs)
        ys_out, _ = hk.dynamic_unroll(core, xs_in, state)
        return ys_out

    model = hk.transform(unroll_network)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, sample_xs)
    opt_state = optimizer.init(params)

    # compute masked NLL for given trial_type subset
    @partial(jax.jit, static_argnames="mask_type")
    def _masked_nll(p, x, y_packed, rng, mask_type):
        logits = model.apply(p, rng, x)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        n_classes = logits.shape[-1]

        y_true = y_packed[:, :, 0].astype(jnp.int32)       # actual choice
        y_mask = y_packed[:, :, 1].astype(jnp.int32)       # trial_type

        active = (y_true >= 0) & (y_true < n_classes) & (y_mask == mask_type)
        one_hot = jax.nn.one_hot(y_true, num_classes=n_classes)
        log_lik = jnp.sum(one_hot * log_probs, axis=-1)
        masked_sum = jnp.sum(log_lik * active.astype(log_lik.dtype))
        n_valid = jnp.maximum(jnp.sum(active), 1)
        return -masked_sum / n_valid

    @jax.jit
    def _train_step(p, os_, x, y_packed, rng):
        loss, grads = jax.value_and_grad(_masked_nll, argnums=0)(
            p, x, y_packed, rng, TRIAL_TRAIN
        )
        updates, os_ = optimizer.update(grads, os_, params=p)
        p = optax.apply_updates(p, updates)
        return loss, p, os_

    best_val = float("inf")
    best_params = None
    steps_since_best = 0
    val_history = []
    t_start = time.time()

    for step in range(1, n_steps_max + 1):
        key, step_key = jax.random.split(key)
        xs_batch, ys_batch_packed = next(dataset)
        train_loss, params, opt_state = _train_step(
            params, opt_state, xs_batch, ys_batch_packed, step_key
        )

        if step % eval_every == 0:
            # validation: iterate ALL batches, compute loss on val trials
            val_log_lik = 0.0
            val_count = 0

            # build a fresh one-shot iterator to cover all batches once
            ds_val_iter = rnn_utils.DatasetRNN(xs, ys_packed, batch_size=batch_size)
            for _ in range(ds_val_iter.n_batches):
                xs_v, ys_vp = next(ds_val_iter)
                key, val_key = jax.random.split(key)
                logits = model.apply(params, val_key, xs_v)
                log_probs = np.array(jax.nn.log_softmax(logits, axis=-1))
                n_classes = logits.shape[-1]

                y_true = ys_vp[:, :, 0].astype(int)
                y_mask = ys_vp[:, :, 1].astype(int)
                active = (y_true >= 0) & (y_true < n_classes) & (y_mask == TRIAL_VAL)

                for s in range(y_true.shape[1]):
                    for t in range(y_true.shape[0]):
                        if active[t, s]:
                            val_log_lik += log_probs[t, s, y_true[t, s]]
                            val_count += 1

            val_nll = -val_log_lik / max(val_count, 1)
            val_history.append((step, float(val_nll)))

            if val_nll < best_val:
                best_val = val_nll
                best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
                steps_since_best = 0
            else:
                steps_since_best += eval_every

            if verbose and step % (eval_every * 5) == 0:
                print(
                    f"  step {step:6d}/{n_steps_max}  "
                    f"train={float(train_loss):.4f}  val={float(val_nll):.4f}  "
                    f"best={float(best_val):.4f}  patience={steps_since_best}/{early_stop_patience}"
                )

            if steps_since_best >= early_stop_patience:
                if verbose:
                    print(f"  Early stop at step {step}")
                break

    if best_params is None:
        best_params = params
        best_val = float(val_history[-1][1]) if val_history else float("inf")

    if verbose:
        print(f"  Done in {time.time() - t_start:.0f}s, best val NLL={best_val:.4f}")
    return best_params, best_val, val_history


# ──────────────────────────────────────────────
# 3. EVALUATION (masked to test trials only)
# ──────────────────────────────────────────────

def evaluate_test_nll(xs, ys, trial_type, model_fun, params, batch_size=64, seed=0):
    """Evaluate NLL on test trials ONLY, across all sessions."""
    ys_packed = np.concatenate([ys.astype(float), trial_type.astype(float)], axis=-1)
    ds = rnn_utils.DatasetRNN(xs, ys_packed, batch_size=batch_size)

    total_log_lik = 0.0
    total_valid = 0
    key = jax.random.PRNGKey(seed)

    for _ in range(ds.n_batches):
        xs_b, ys_bp = next(ds)
        key, ek = jax.random.split(key)
        model_outputs, _ = rnn_utils.eval_model(model_fun, params, xs_b)
        log_probs = np.array(jax.nn.log_softmax(model_outputs, axis=-1))
        n_classes = log_probs.shape[-1]

        y_true = ys_bp[:, :, 0].astype(int)
        y_mask = ys_bp[:, :, 1].astype(int)
        active = (y_true >= 0) & (y_true < n_classes) & (y_mask == TRIAL_TEST)

        for s in range(y_true.shape[1]):
            for t in range(y_true.shape[0]):
                if active[t, s]:
                    total_log_lik += log_probs[t, s, y_true[t, s]]
                    total_valid += 1

    if total_valid == 0:
        raise ValueError("No test trials found.")
    return -total_log_lik / total_valid


def evaluate_masked_nll(xs, ys, trial_type, mask_type, model_fun, params, batch_size=64, seed=0):
    """Evaluate NLL on trials with given mask_type (0=train, 1=val, 2=test)."""
    ys_packed = np.concatenate([ys.astype(float), trial_type.astype(float)], axis=-1)
    ds = rnn_utils.DatasetRNN(xs, ys_packed, batch_size=batch_size)

    total_log_lik = 0.0
    total_valid = 0
    key = jax.random.PRNGKey(seed)

    for _ in range(ds.n_batches):
        xs_b, ys_bp = next(ds)
        key, ek = jax.random.split(key)
        model_outputs, _ = rnn_utils.eval_model(model_fun, params, xs_b)
        log_probs = np.array(jax.nn.log_softmax(model_outputs, axis=-1))
        n_classes = log_probs.shape[-1]

        y_true = ys_bp[:, :, 0].astype(int)
        y_mask = ys_bp[:, :, 1].astype(int)
        active = (y_true >= 0) & (y_true < n_classes) & (y_mask == mask_type)

        for s in range(y_true.shape[1]):
            for t in range(y_true.shape[0]):
                if active[t, s]:
                    total_log_lik += log_probs[t, s, y_true[t, s]]
                    total_valid += 1

    return -total_log_lik / max(total_valid, 1)


# ──────────────────────────────────────────────
# 4. INNER CV SEARCH
# ──────────────────────────────────────────────

def inner_cv_search(
    model_builder,
    xs, ys, trial_type,
    train_val_idx,
    param_grid,
    n_inner=3,
    n_seeds=2,
    batch_size=64,
    n_steps_max=100000,
    early_stop_patience=500,
    random_state=42,
    verbose=True,
    checkpoint_path=None,
):
    """
    Inner session-level 3-fold × 2 seeds. Loss is masked within-session
    (train trials for gradient, val trials for early stopping).

    If checkpoint_path is given, every completed (config, seed) evaluation is
    written there immediately; a later call resumes by skipping the
    (config, seed) pairs already present. Iteration order is deterministic
    (config-major, seed-minor) and each evaluation is seeded independently, so
    the i-th saved result always matches the i-th pair — resuming is bit-exact.

    Returns: (best_hparams_dict, best_score, all_results)
    """
    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    combos = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())

    # --- resume: load completed (config, seed) evaluations, if any ---
    all_results = []
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            saved = pickle.load(f)
        if (saved.get("keys") == keys
                and saved.get("combos") == combos
                and saved.get("n_seeds") == n_seeds):
            all_results = saved["results"]
            if verbose:
                print(f"  [resume] loaded {len(all_results)} completed configs "
                      f"from {os.path.basename(checkpoint_path)}")
        elif verbose:
            print("  [resume] checkpoint grid mismatch — ignoring stale checkpoint")

    n_done = len(all_results)

    # running best, seeded from any resumed results so the ★ marker stays right
    best_score = float("inf")
    best_config = None
    for r in all_results:
        if r["combined_score"] < best_score:
            best_score = r["combined_score"]
            best_config = {**{k: r[k] for k in keys}, "seed": r["seed"]}

    def _save_ckpt():
        if checkpoint_path is None:
            return
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"keys": keys, "combos": combos,
                         "n_seeds": n_seeds, "results": all_results}, f)
        os.replace(tmp, checkpoint_path)  # atomic — survives a kill mid-write

    if verbose:
        msg = f"  Inner search: {len(combos)} configs × {n_seeds} seeds × {n_inner} folds"
        if n_done:
            msg += f"  ({n_done} done, {len(combos) * n_seeds - n_done} to go)"
        print(msg)

    idx = 0
    for combo in combos:
        hparams = dict(zip(keys, combo))

        for seed in range(n_seeds):
            if idx < n_done:          # already evaluated in a previous run
                idx += 1
                continue
            idx += 1

            fold_scores_train = []
            fold_scores_val = []

            for inner_train_idx, inner_val_idx in inner_cv.split(train_val_idx):
                train_sessions = train_val_idx[inner_train_idx]
                val_sessions = train_val_idx[inner_val_idx]

                bs = hparams.get("batch_size", batch_size)

                def make_model():
                    return model_builder(hparams, seed)

                # Use trial-split training: train on train_sessions,
                # loss masked to train trials, val on val trials
                inner_params, best_val_loss, _ = train_with_trial_split(
                    make_model,
                    xs[:, train_sessions],
                    ys[:, train_sessions],
                    trial_type[:, train_sessions],
                    optimizer=optax.chain(
                        optax.add_decayed_weights(hparams.get("weight_decay", 1e-5)),
                        optax.adam(hparams.get("lr", 1e-4)),
                    ),
                    batch_size=bs,
                    n_steps_max=n_steps_max,
                    eval_every=200,
                    early_stop_patience=early_stop_patience,
                    seed=seed,
                    verbose=False,
                )

                # Train NLL on val sessions' train trials (generalization check)
                train_nll = evaluate_masked_nll(
                    xs[:, val_sessions], ys[:, val_sessions],
                    trial_type[:, val_sessions], TRIAL_TRAIN,
                    make_model, inner_params, batch_size=bs, seed=seed,
                )

                fold_scores_train.append(train_nll)
                fold_scores_val.append(best_val_loss)

            avg_train = np.mean(fold_scores_train)
            avg_val = np.mean(fold_scores_val)
            combined_score = (avg_train + avg_val) / 2.0

            all_results.append({**hparams, "seed": seed,
                                "avg_train_nll": avg_train, "avg_val_nll": avg_val,
                                "combined_score": combined_score})
            _save_ckpt()  # checkpoint after every completed (config, seed)

            if verbose:
                print(
                    f"    {hparams} seed={seed}  "
                    f"train={avg_train:.4f} val={avg_val:.4f}  combined={combined_score:.4f}"
                    + (" ★" if combined_score < best_score else "")
                )

            if combined_score < best_score:
                best_score = combined_score
                best_config = {**hparams, "seed": seed}

    if verbose:
        print(f"  Best config: {best_config} (score={best_score:.4f})")
    return best_config, best_score, all_results


# ──────────────────────────────────────────────
# 5. OUTER CV
# ──────────────────────────────────────────────

def run_outer_cv(
    model_builder,
    xs,
    ys,
    param_grid,
    dataset_name="dataset",
    group_ids=None,
    n_outer=10,
    n_inner=3,
    n_seeds=2,
    batch_size=64,
    n_steps_max=100000,
    early_stop_patience=500,
    random_state=42,
    assigned_folds=None,
    train_ratio=0.75,
    val_ratio=0.125,
    resume=False,
):
    """
    Full nested CV with within-session trial masking.

    Args:
        model_builder: fn(hparams, seed) -> hk.RNNCore
        xs, ys: (T, N, D) / (T, N, 1)
        group_ids: optional (N,) int array for GroupKFold (prevents subject leakage)
        n_outer, n_inner, n_seeds: CV splits
        assigned_folds: subset of folds to run (for parallel jobs)
        train_ratio, val_ratio: trial split within each session
        resume: if True, skip outer folds already saved in the results pkl and
            checkpoint the inner search per (config, seed), so an interrupted
            run picks up where it stopped (even mid-fold)
    """
    # Build trial-type array once
    trial_type = build_trial_type(ys, train_ratio=train_ratio, val_ratio=val_ratio)
    n_sessions = ys.shape[1]
    session_indices = np.arange(n_sessions)

    if group_ids is not None:
        outer_cv = GroupKFold(n_splits=n_outer)
        splits = outer_cv.split(session_indices, groups=group_ids)
    else:
        outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=random_state)
        splits = outer_cv.split(session_indices)

    save_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_outer_results.pkl")

    test_nlls = []
    best_configs = []
    all_inner_results = []
    n_resumed = 0
    if resume and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            _prev = pickle.load(f)
        test_nlls = list(_prev.get("test_nlls", []))
        best_configs = list(_prev.get("best_configs", []))
        all_inner_results = list(_prev.get("all_inner_results", []))
        n_resumed = len(test_nlls)
        print(f"\n[resume] {dataset_name}: {n_resumed} outer fold(s) already "
              f"complete in {os.path.basename(save_path)} — will skip them")

    for outer_fold, (train_val_idx, test_idx) in enumerate(splits, start=1):
        if resume and outer_fold <= n_resumed:
            print(f"\n=== Outer Fold {outer_fold}/{n_outer} — ALREADY DONE (resumed) ===")
            continue
        if assigned_folds is not None and outer_fold not in assigned_folds:
            print(f"\n=== Outer Fold {outer_fold}/{n_outer} — SKIP ===")
            continue

        print(f"\n{'=' * 60}")
        print(f"=== Outer Fold {outer_fold}/{n_outer} ===")
        print(f"  train_val sessions: {len(train_val_idx)}, test sessions: {len(test_idx)}")
        print(f"{'=' * 60}")

        # --- INNER SEARCH (on train_val sessions only) ---
        t0 = time.time()
        inner_ckpt = (
            os.path.join(OUTPUT_DIR, f"{dataset_name}_fold{outer_fold}_inner_ckpt.pkl")
            if resume else None
        )
        best_config, best_score, inner_results = inner_cv_search(
            model_builder=model_builder,
            xs=xs, ys=ys, trial_type=trial_type,
            train_val_idx=train_val_idx,
            param_grid=param_grid,
            n_inner=n_inner,
            n_seeds=n_seeds,
            batch_size=batch_size,
            n_steps_max=n_steps_max,
            early_stop_patience=early_stop_patience,
            random_state=random_state + outer_fold,
            verbose=True,
            checkpoint_path=inner_ckpt,
        )

        # --- RETRAIN BEST on full train_val ---
        bs = best_config.get("batch_size", batch_size)

        def make_best_model():
            return model_builder(best_config, best_config.get("seed", 0))

        print(f"\n  Retraining best config {best_config} on full train_val...")
        final_params, _, _ = train_with_trial_split(
            make_best_model,
            xs[:, train_val_idx],
            ys[:, train_val_idx],
            trial_type[:, train_val_idx],
            optimizer=optax.chain(
                optax.add_decayed_weights(best_config.get("weight_decay", 1e-5)),
                optax.adam(best_config.get("lr", 1e-4)),
            ),
            batch_size=bs,
            n_steps_max=n_steps_max,
            eval_every=200,
            early_stop_patience=early_stop_patience,
            seed=best_config.get("seed", 0),
            verbose=True,
        )

        # --- FINAL TEST EVAL: only test-trial indices in held-out sessions ---
        test_nll = evaluate_test_nll(
            xs[:, test_idx], ys[:, test_idx],
            trial_type[:, test_idx],
            make_best_model, final_params,
            batch_size=bs, seed=best_config.get("seed", 0),
        )
        print(f"\n  >>> TEST NLL (fold {outer_fold}): {test_nll:.6f} <<<\n")

        test_nlls.append(test_nll)
        best_configs.append(best_config)
        all_inner_results.append(inner_results)

        with open(save_path, "wb") as f:
            pickle.dump({
                "test_nlls": test_nlls,
                "best_configs": best_configs,
                "all_inner_results": all_inner_results,
            }, f)

        # fold fully checkpointed to the outer pkl — drop its inner-search ckpt
        if inner_ckpt is not None and os.path.exists(inner_ckpt):
            os.remove(inner_ckpt)

    test_nlls = np.array(test_nlls)
    mean_nll = test_nlls.mean()
    se_nll = test_nlls.std(ddof=1) / np.sqrt(len(test_nlls))

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULTS for {dataset_name}")
    print(f"  Test NLLs: {test_nlls}")
    print(f"  Mean NLL:  {mean_nll:.6f}")
    print(f"  SE:        {se_nll:.6f}")
    print(f"{'=' * 60}")

    return {
        "test_nlls": test_nlls,
        "test_nll_mean": mean_nll,
        "test_nll_se": se_nll,
        "best_configs": best_configs,
        "all_inner_results": all_inner_results,
    }


# ──────────────────────────────────────────────
# 6. MODEL BUILDERS (unchanged)
# ──────────────────────────────────────────────

def make_rnn_builder(rnn_type="vanilla", n_actions=2):
    def build(hparams, seed):
        hs = hparams["hidden_size"]
        if rnn_type == "vanilla":
            core = hk.VanillaRNN(hs)
        elif rnn_type == "gru":
            core = hk.GRU(hs)
        elif rnn_type == "lstm":
            core = hk.LSTM(hs)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        return hk.DeepRNN([core, hk.Linear(output_size=n_actions)])
    return build


def make_hybrid_builder(use_hidden_state=False, use_previous_values=False,
                         fit_forget=False, forget=0.0, n_actions=2, use_twostep=True):
    def build(hparams, seed):
        hs = hparams["hidden_size"]
        rl_params = {
            "s": use_hidden_state, "o": use_previous_values,
            "fit_forget": fit_forget, "forget": forget,
            "w_h": 1.0, "w_v": 1.0,
        }
        network_params = {"n_actions": n_actions, "hidden_size": hs}
        if use_twostep:
            return hybrnn_twostep.BiRNN(rl_params=rl_params, network_params=network_params)
        else:
            from CogModelingRNNsTutorial import hybrnn
            return hybrnn.BiRNN(rl_params=rl_params, network_params=network_params)
    return build


def make_bichannel_builder(gate_mode, n_actions=2, fit_forget=False, forget=0.0,
                           use_twostep=True):
    """Builder for hybrnn(.twostep).BiChannelRNN -- the two-channel gated RNN.

    gate_mode in {'additive', 'learnable', 'surprise'} selects how the context
    and memory channels are combined. Use 3 instances (one per mode) to compare.
    """
    def build(hparams, seed):
        hs = hparams["hidden_size"]
        rl_params = {
            "fit_forget": fit_forget, "forget": forget,
            "w_h": 1.0, "w_v": 1.0,
        }
        network_params = {"n_actions": n_actions, "hidden_size": hs}
        if use_twostep:
            return hybrnn_twostep.BiChannelRNN(
                rl_params=rl_params, network_params=network_params,
                gate_mode=gate_mode)
        else:
            from CogModelingRNNsTutorial import hybrnn
            return hybrnn.BiChannelRNN(
                rl_params=rl_params, network_params=network_params,
                gate_mode=gate_mode)
    return build


def make_cognitive_rl_builder(model_class="Hk_PreserveConAgentQ", n_actions=2):
    def build(hparams, seed):
        cls = getattr(bandits, model_class)
        try:
            return cls(n_actions=n_actions)
        except TypeError:
            return cls()
    return build


# ──────────────────────────────────────────────
# 7. PARAMETER GRIDS
# ──────────────────────────────────────────────

RNN_GRID = {
    "hidden_size": [16, 32, 64, 128],  # 128 added: vanilla_rnn also kept selecting the 64 ceiling
    "lr": [1e-3, 1e-4],
    "weight_decay": [1e-4, 1e-3],
    "batch_size": [64],
}

HYBRID_GRID = {
    "hidden_size": [16, 32, 64, 128],  # 128 added: earlier suthana hybrid runs kept selecting the 64 ceiling
    "lr": [1e-3, 1e-4],
    "weight_decay": [1e-4, 1e-3],
    "batch_size": [64],
}

COGNITIVE_GRID = {
    "lr": [1e-3, 1e-4],
    "weight_decay": [1e-4, 1e-3],
    "batch_size": [64],
}
