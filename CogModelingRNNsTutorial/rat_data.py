"""Functions for loading rat data."""
import json
import os
import numpy as np
import pickle

from typing import List, Optional


JSON_PATH = "./CogModelingRNNsTutorial/data/miller2018_all_rats.json"  # Where raw rat data is stored
DATA_DIR = "./CogModelingRNNsTutorial/data/rat_data/"  # Where you will save out individual rat data
PREFIX = "miller2018"


def _get_single_rat_fname(rat_id):
  assert isinstance(rat_id, int)
  rat_id_padded = f'{rat_id}'.rjust(2, '0')
  return f"{PREFIX}_rat{rat_id_padded}.npy"


def load_data_for_one_rat(fname=None, data_dir=DATA_DIR):
  """Load data for a single rat.

  Args:
    fname: name of file (will likely be the name of a npy file you loaded
    data_dir: directory where file lives


  Returns:
    xs: n_trials x n_sessions x 2 array of choices and rewards
    ys: n_trials x n_sessions x 1 array of choices (shifted forward
      by one compared to xs[..., 0]).
    fname: name of file
  """
  if not os.path.exists(data_dir):
    raise ValueError(f'data_dir {data_dir} not found.')

  if fname is None:
    rat_files = [f for f in os.listdir(data_dir) if (f.startswith(f'{PREFIX}_rat') and f.endswith('.npy'))]
    fname = rat_files[np.random.randint(len(rat_files))]
    print(f'Loading data from {fname}.')
  else:
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
      raise ValueError(f'path {fpath} not found.')

  data = np.load(os.path.join(data_dir, fname))
  xs, ys = data[..., :2], data[..., 2:]
  assert ys.shape[-1] == 1
  assert ys.ndim == xs.ndim == 3
  return xs, ys, fname


import numpy as np  # 确保导入了numpy

import numpy as np

# def format_into_datasets(xs, ys, dataset_constructor, n_train_sessions, n_test_sessions, n_validate_sessions, random_seed=None):
#     """
#     Format inputs xs and outputs ys into randomly split datasets with optional reproducibility.

#     Args:
#         xs: n_trials x n_sessions x 2 array of choices and rewards
#         ys: n_trials x n_sessions x 1 array of next choice. choice value of -1 denotes
#             instructed trial or padding at end of session.
#         dataset_constructor: constructor that accepts xs and ys as arguments; probably
#             use rnn_utils.DatasetRNN
#         n_train_sessions: number of sessions for the training dataset
#         n_test_sessions: number of sessions for the test dataset
#         n_validate_sessions: number of sessions for the validation dataset
#         random_seed: optional int, for reproducible random splits

#     Returns:
#         dataset_train: a dataset containing randomly selected training sessions
#         dataset_test: a dataset containing randomly selected test sessions
#         dataset_validate: a dataset containing randomly selected validation sessions
#     """
#     total_sessions = xs.shape[1]
#     total_needed = n_train_sessions + n_test_sessions + n_validate_sessions

#     assert total_sessions >= total_needed, \
#         f"Not enough sessions: required {total_needed}, but got {total_sessions}."

#     # Set random seed if provided
#     if random_seed is not None:
#         np.random.seed(random_seed)

#     # Randomly shuffle session indices
#     shuffled_indices = np.random.permutation(total_sessions)

#     train_indices = shuffled_indices[:n_train_sessions]
#     validate_indices = shuffled_indices[n_train_sessions:n_train_sessions + n_validate_sessions]
#     test_indices = shuffled_indices[n_train_sessions + n_validate_sessions:total_needed]

#     dataset_train = dataset_constructor(xs[:, train_indices], ys[:, train_indices])
#     dataset_validate = dataset_constructor(xs[:, validate_indices], ys[:, validate_indices])
#     dataset_test = dataset_constructor(xs[:, test_indices], ys[:, test_indices])

#     return dataset_train, dataset_test, dataset_validate


# with batch size
def format_into_datasets(xs,
                         ys,
                         dataset_constructor,
                         n_train_sessions,
                         n_test_sessions,
                         n_validate_sessions,
                         batch_size=None,
                         random_seed=None):
    """
    Format inputs xs and outputs ys into randomly split datasets with optional reproducibility.

    Args:
        xs: n_trials x n_sessions x … array of inputs
        ys: n_trials x n_sessions x … array of targets
        dataset_constructor: constructor that accepts (xs, ys, batch_size)
        n_train_sessions: number of sessions for the training dataset
        n_test_sessions: number of sessions for the test dataset
        n_validate_sessions: number of sessions for the validation dataset
        batch_size: number of episodes per batch (passed to DatasetRNN)
        random_seed: optional int, for reproducible random splits

    Returns:
        dataset_train, dataset_test, dataset_validate
    """
    total_sessions = xs.shape[1]
    total_needed = n_train_sessions + n_test_sessions + n_validate_sessions
    assert total_sessions >= total_needed, \
        f"Not enough sessions: required {total_needed}, but got {total_sessions}."

    # reproducible shuffle
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffled = np.random.permutation(total_sessions)

    train_idx = shuffled[:n_train_sessions]
    val_idx   = shuffled[n_train_sessions:n_train_sessions + n_validate_sessions]
    test_idx  = shuffled[n_train_sessions + n_validate_sessions:total_needed]

    # 传入 batch_size
    ds_train = dataset_constructor(xs[:, train_idx],
                                   ys[:, train_idx],
                                   batch_size=batch_size)
    ds_val   = dataset_constructor(xs[:, val_idx],
                                   ys[:, val_idx],
                                   batch_size=batch_size)
    ds_test  = dataset_constructor(xs[:, test_idx],
                                   ys[:, test_idx],
                                   batch_size=batch_size)

    return ds_train, ds_test, ds_val



def format_into_datasets_10(xs,
                            ys,
                            dataset_constructor,
                            n_validate_sessions,
                            batch_size=None,
                            random_seed=None):
    """
    10-fold cross‐validation 数据切分，每次：
      - 随机打乱所有 sessions（可固定 random_seed）
      - 平均分成 10 份 folds
      - 每次循环：
          test = 当前 fold
          剩余 sessions 中前 n_validate_sessions 做 validation
          剩下的做 train
      - batch_size 会透传给 dataset_constructor

    Args:
        xs: np.ndarray, shape (n_trials, n_sessions, …)
        ys: np.ndarray, shape (n_trials, n_sessions, …)
        dataset_constructor: 接受 (xs_subset, ys_subset, batch_size) 的构造器
        n_validate_sessions: 每个 fold 中 validation sessions 数量
        batch_size: mini‐batch 大小（episodes）；None 则每个数据集用全部 episodes
        random_seed: int，可选；用于可复现的 shuffle

    Returns:
        List of 10 tuples (train_ds, validate_ds, test_ds)
    """
    total_sessions = xs.shape[1]
    # 可复现地打乱
    if random_seed is not None:
        np.random.seed(random_seed)
    all_idx = np.random.permutation(total_sessions)

    # 平均分成 10 份，即使不能整除也会自动前几份多一个
    folds = np.array_split(all_idx, 10)

    folds_datasets = []
    for i in range(10):
        # 选出第 i fold 作为测试集
        test_idx = folds[i]
        # 剩余所有 idx
        remaining = np.hstack([folds[j] for j in range(10) if j != i])
        if len(remaining) < n_validate_sessions:
            raise ValueError(f"剩余 sessions ({len(remaining)}) 少于 n_validate_sessions ({n_validate_sessions})")

        # 前 n_validate_sessions 做验证，其余做训练
        val_idx = remaining[:n_validate_sessions]
        train_idx = remaining[n_validate_sessions:]

        ds_train = dataset_constructor(xs[:, train_idx], ys[:, train_idx], batch_size=batch_size)
        ds_val   = dataset_constructor(xs[:, val_idx],   ys[:, val_idx],   batch_size=batch_size)
        ds_test  = dataset_constructor(xs[:, test_idx],  ys[:, test_idx],  batch_size=batch_size)

        folds_datasets.append((ds_train, ds_val, ds_test))

    return folds_datasets



def find(s, ch):
  """Find index of character within string."""
  return [i for i, ltr in enumerate(s) if ltr == ch]


def get_rat_bandit_datasets(data_file: Optional[str] = None):
  """Packages up the rat datasets.

  Requires downloading the dataset file "tab_dataset.json", which is available
  on Figshare.

  https://figshare.com/articles/dataset/From_predictive_models_to_cognitive_models_Separable_behavioral_processes_underlying_reward_learning_in_the_rat/20449356

  Args:
    data_file: Complete path to the dataset file, including the filename. If not
      specified, will look for data in the predictive_cognitive folder on CNS.

  Returns:
    A list of DatasetRNN objects. One element per rat.
    In each of these, each session will be an episode, padded with NaNs
    to match length. "ys" will be the choices on each trial
    (left=0, right=1) "xs" will be the choice and reward (0 or 1) from
    the previous trial. Invalid xs and ys will be -1
  """
  if data_file is None:
    data_file = '/cns/ej-d/home/kevinjmiller/predictive_cognitive/tab_dataset.json'

  with open(data_file, 'r') as f:
    dataset = json.load(f)

  n_rats = len(dataset)

  dataset_list = []
  # Each creates a DatasetRNN object for a single rat, adds it to the list
  for rat_i in range(n_rats):
    ratdata = dataset[rat_i]
    sides = ratdata['sides']
    n_trials = len(sides)

    # Left choices will be 0s, right choices will be 1s, viols will be removed
    rights = find(sides, 'r')
    choices = np.zeros(n_trials)
    choices[rights] = 1

    vs = find(sides, 'v')
    viols = np.zeros(n_trials, dtype=bool)
    viols[vs] = True

    # Free will be 0 and forced will be 1
    free = find(ratdata['trial_types'], 'f')
    instructed_choice = np.ones(n_trials)
    instructed_choice[free] = 0

    rewards = np.array(ratdata['rewards'])
    new_sess = np.array(ratdata['new_sess'])

    n_sess = np.sum(new_sess)
    sess_starts = np.nonzero(np.concatenate((new_sess, [1])))[0]
    max_session_length = np.max(np.diff(sess_starts, axis=0))

    # Populate matrices for rewards and choices. size (n_trials, n_sessions, 1)
    rewards_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    choices_by_session = -1 * np.ones((max_session_length, n_sess, 1))
    instructed_by_session = -1 * np.ones((max_session_length, n_sess, 1))

    # Each iteration processes one session
    for sess_i in np.arange(n_sess):
      sess_start = sess_starts[sess_i]
      sess_end = sess_starts[sess_i + 1]

      viols_sess = viols[sess_start:sess_end]
      rewards_sess = rewards[sess_start:sess_end]
      choices_sess = choices[sess_start:sess_end]
      instructed_choice_sess = instructed_choice[sess_start:sess_end]

      rewards_sess = np.delete(rewards_sess, viols_sess)
      choices_sess = np.delete(choices_sess, viols_sess)
      instructed_choice_sess = np.delete(instructed_choice_sess, viols_sess)

      sess_length_noviols = len(choices_sess)

      rewards_by_session[0:sess_length_noviols, sess_i, 0] = rewards_sess
      choices_by_session[0:sess_length_noviols, sess_i, 0] = choices_sess
      instructed_by_session[0:sess_length_noviols, sess_i, 0] = (
          instructed_choice_sess
      )

    # Inputs: choices and rewards, offset by one trial
    choice_and_reward = np.concatenate(
        (choices_by_session, rewards_by_session), axis=2
    )
    # Add a dummy input at the beginning. First step has a target but no input
    xs = np.concatenate(
        (0. * np.ones((1, n_sess, 2)), choice_and_reward), axis=0
    )
    # Targets: choices on each free-choice trial
    free_choices = choices_by_session
    free_choices[instructed_by_session == 1] = -1
    # Add a dummy target at the end -- last step has input but no target
    ys = np.concatenate((free_choices, -1*np.ones((1, n_sess, 1))), axis=0)

    dataset_list.append([xs, ys])

  return dataset_list


def save_out_rat_data_as_pickle(json_path=JSON_PATH, data_dir=DATA_DIR, verbose=True):
  """Load json with all rat data + save out individual RNNDatasets for each rat."""
  if not os.path.exists(json_path):
    raise ValueError(f'json_path {json_path} does not exist.')

  # Make destination directory if it does not already exists.
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if verbose:
    print(f'Loading data from {json_path}.')
  dataset = get_rat_bandit_datasets(json_path)

  if verbose:
    print(f'Saving out data to {data_dir}.')

  for rat_id in range(len(dataset)):
    fname = _get_single_rat_fname(rat_id)
    save_path = os.path.join(data_dir, fname)
    xs, ys = dataset[rat_id]
    np.save(save_path, np.concatenate([xs, ys], axis=-1))

