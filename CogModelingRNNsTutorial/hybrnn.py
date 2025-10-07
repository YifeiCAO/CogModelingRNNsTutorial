"""Define hybRNNs."""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

RNNState = jnp.array


class BiRNN(hk.RNNCore):
  """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._hs = rl_params['s']
    self._vs = rl_params['s']
    self._ho = rl_params['o']
    self._vo = rl_params['o']

    self.w_h = rl_params['w_h']
    self.w_v = rl_params['w_v']
    self.init_value = init_value

    self._n_actions = network_params['n_actions']
    self._hidden_size = network_params['hidden_size']

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']

  def _value_rnn(self, state, value, action, reward):

    pre_act_val = jnp.sum(value * action, axis=1)  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = value + action * update

    return next_value, next_state

  def _habit_rnn(self, state, habit, action):

    inputs = action
    if self._ho:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, habit], axis=-1)
    if self._hs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    next_habit = hk.Linear(self._n_actions)(next_state)

    return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    h_state, v_state, habit, value = prev_state
    action = inputs[:, 0]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    action_onehot = jax.nn.one_hot(action,2)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action_onehot, reward)

    # Habit module: update/create new habit
    next_habit, next_h_state = self._habit_rnn(h_state, habit, action_onehot)

    # Combine value and habit
    logits = self.w_v * next_value + self.w_h * next_habit  # (bs, n_a)

    return logits, (next_h_state, next_v_state, next_habit, next_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # h_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        0 * jnp.ones([batch_size, self._n_actions]),  # habit
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # value
        )

class BiControlRNN(hk.RNNCore):
  """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._hs = rl_params['s']
    self._vs = rl_params['s']
    self._ho = rl_params['o']
    self._vo = rl_params['o']

    self.w_h = rl_params['w_h']
    self.w_v = rl_params['w_v']
    self.init_value = init_value

    self._n_actions = network_params['n_actions']
    self._hidden_size = network_params['hidden_size']

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']

  def _value_rnn(self, state, value, action, reward):

    # value: (B, n_actions), action: (B, n_actions), reward: (B,)
    B, nA = value.shape

    # 1) 选中/未选中的价值 —— 保持 (B,1)
    pre_act_val = jnp.sum(value * action, axis=1, keepdims=True)  # (B,1)
    pre_nonact_val = jnp.sum(value * (jnp.ones((1, nA)) - action), axis=1, keepdims=True)  # (B,1)

    # 2) 归一化命中概率（数值稳）
    pre_act_val_norm = pre_act_val / (pre_act_val + pre_nonact_val + 1e-8)  # (B,1)

    # 3) RPE（形状也保持 (B,1)）
    rpe = reward[:, None] - pre_act_val_norm  # (B,1)
    
    context_weight = rpe
    memory_weight = 1.0 - rpe

    # 5) 做逐元素缩放，得到与 value/state 形状可广播的权重
    value_mod = value * context_weight        # (B, n_actions)
    state_mod = state * memory_weight         # (B, hidden_size)

    # 6) 现在拼接 —— 确保所有都是 2D：(B, D)
    inputs = jnp.concatenate(
        [pre_act_val, reward[:, jnp.newaxis], value_mod, state_mod],
        axis=-1
    )

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = value + action * update

    return next_value, next_state

  def _habit_rnn(self, state, habit, action, value, reward):

    # value: (B, n_actions), action: (B, n_actions), reward: (B,)
    B, nA = value.shape

    # 1) 选中/未选中的价值 —— 保持 (B,1)
    pre_act_val = jnp.sum(value * action, axis=1, keepdims=True)  # (B,1)
    pre_nonact_val = jnp.sum(value * (jnp.ones((1, nA)) - action), axis=1, keepdims=True)  # (B,1)

    # 2) 归一化命中概率（数值稳）
    pre_act_val_norm = pre_act_val / (pre_act_val + pre_nonact_val + 1e-8)  # (B,1)

    # 3) RPE（形状也保持 (B,1)）
    rpe = reward[:, None] - pre_act_val_norm  # (B,1)
    
    context_weight = rpe
    memory_weight = 1.0 - rpe

    # 5) 做逐元素缩放，得到与 value/state 形状可广播的权重
    habit_mod = habit * context_weight        # (B, n_actions)
    state_mod = state * memory_weight         # (B, hidden_size)

    # 6) 现在拼接 —— 确保所有都是 2D：(B, D)
    inputs = antion
    inputs = jnp.concatenate(
        [action, reward[:, jnp.newaxis], habit_mod, state_mod],
        axis=-1
    )

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    next_habit = hk.Linear(self._n_actions)(next_state)

    return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    h_state, v_state, habit, value = prev_state
    action = inputs[:, 0]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    action_onehot = jax.nn.one_hot(action,2)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action_onehot, reward)

    # Habit module: update/create new habit
    next_habit, next_h_state = self._habit_rnn(h_state, habit, action_onehot, value, reward)

    # Combine value and habit
    logits = self.w_v * next_value + self.w_h * next_habit  # (bs, n_a)

    return logits, (next_h_state, next_v_state, next_habit, next_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # h_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        0 * jnp.ones([batch_size, self._n_actions]),  # habit
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # value
        )


class BiRNN_oneDim(hk.RNNCore):
  """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._hs = rl_params['s']
    self._vs = rl_params['s']
    self._ho = rl_params['o']
    self._vo = rl_params['o']

    self.w_h = rl_params['w_h']
    self.w_v = rl_params['w_v']
    self.init_value = 0

    self._n_actions = network_params['n_actions']
    self._hidden_size = network_params['hidden_size']

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']

  def _value_rnn(self, state, value, action, reward):

    pre_act_val = value  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    next_value = hk.Linear(1)(next_state)
    #value = (1 - self.forget) * value + self.forget * self.init_value
    #next_value = value + update

    return next_value, next_state

  def _habit_rnn(self, state, habit, action):

    inputs = jnp.concatenate(
        [habit[:, jnp.newaxis], action[:, jnp.newaxis]], axis=-1)
    if self._ho:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, habit], axis=-1)
    if self._hs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    next_habit = hk.Linear(1)(next_state)

    return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    h_state, v_state, habit, value = prev_state
    action = inputs[:, 0]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    action_onehot = jax.nn.one_hot(action,2)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action, reward)

    # Habit module: update/create new habit
    next_habit, next_h_state = self._habit_rnn(h_state, habit, action)

    # Combine value and habit
    logits = self.w_v * next_value + self.w_h * next_habit  # (bs, n_a)

    return logits, (next_h_state, next_v_state, next_habit, next_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # h_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        0 * jnp.ones([batch_size, 1]),  # habit
        self.init_value * jnp.ones([batch_size, 1]),  # value
        )

class BiRNN_onValue(hk.RNNCore):
  """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._hs = rl_params['s']
    self._vs = rl_params['s']
    self._ho = rl_params['o']
    self._vo = rl_params['o']

    self.w_h = rl_params['w_h']
    self.w_v = rl_params['w_v']
    self.init_value = init_value

    self._n_actions = network_params['n_actions']
    self._hidden_size = network_params['hidden_size']

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']

  def _value_rnn(self, state, value, action, reward):

    pre_act_val = jnp.sum(value * action, axis=1)  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = value + action * update

    return next_value, next_state

  def _habit_rnn(self, state, habit, action):

    inputs = action

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    next_habit = hk.Linear(self._n_actions)(next_state)

    return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    h_state, v_state, habit, value = prev_state
    action = inputs[:, 0]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    action_onehot = jax.nn.one_hot(action,2)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action_onehot, reward)

    # Habit module: update/create new habit
    next_habit, next_h_state = self._habit_rnn(h_state, habit, action_onehot)

    # Combine value and habit
    logits = self.w_v * next_value + self.w_h * next_habit  # (bs, n_a)

    return logits, (next_h_state, next_v_state, next_habit, next_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # h_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        0 * jnp.ones([batch_size, self._n_actions]),  # habit
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # value
        )

class BiRNN_onHabit(hk.RNNCore):
  """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._hs = rl_params['s']
    self._vs = rl_params['s']
    self._ho = rl_params['o']
    self._vo = rl_params['o']

    self.w_h = rl_params['w_h']
    self.w_v = rl_params['w_v']
    self.init_value = init_value

    self._n_actions = network_params['n_actions']
    self._hidden_size = network_params['hidden_size']

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']

  def _value_rnn(self, state, value, action, reward):

    pre_act_val = jnp.sum(value * action, axis=1)  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = value + action * update

    return next_value, next_state

  def _habit_rnn(self, state, habit, action):

    inputs = action
    if self._ho:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, habit], axis=-1)
    if self._hs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    next_habit = hk.Linear(self._n_actions)(next_state)

    return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    h_state, v_state, habit, value = prev_state
    action = inputs[:, 0]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    action_onehot = jax.nn.one_hot(action,2)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action_onehot, reward)

    # Habit module: update/create new habit
    next_habit, next_h_state = self._habit_rnn(h_state, habit, action_onehot)

    # Combine value and habit
    logits = self.w_v * next_value + self.w_h * next_habit  # (bs, n_a)

    return logits, (next_h_state, next_v_state, next_habit, next_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # h_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        0 * jnp.ones([batch_size, self._n_actions]),  # habit
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # value
        )

class BiRNN_noHabit(hk.RNNCore):
  """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._hs = rl_params['s']
    self._vs = rl_params['s']
    self._ho = rl_params['o']
    self._vo = rl_params['o']

    self.w_h = rl_params['w_h']
    self.w_v = rl_params['w_v']
    self.init_value = init_value

    self._n_actions = network_params['n_actions']
    self._hidden_size = network_params['hidden_size']

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']

  def _value_rnn(self, state, value, action, reward):

    pre_act_val = jnp.sum(value * action, axis=1)  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = value + action * update

    return next_value, next_state

#   def _habit_rnn(self, state, habit, action):

#     inputs = action
#     if self._ho:  # "o" = output -> feed previous output back in
#       inputs = jnp.concatenate([inputs, habit], axis=-1)
#     if self._hs:  # "s" = state -> feed previous hidden state back in
#       inputs = jnp.concatenate([inputs, state], axis=-1)

#     next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
#     next_habit = hk.Linear(self._n_actions)(next_state)

#     return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    v_state, value = prev_state
    action = inputs[:, 0]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    action_onehot = jax.nn.one_hot(action,2)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action_onehot, reward)

    # Habit module: update/create new habit
    #next_habit, next_h_state = self._habit_rnn(h_state, habit, action_onehot)

    # Combine value and habit
    logits = self.w_v * next_value# + self.w_h * next_habit  # (bs, n_a)

    return logits, (next_v_state, next_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # value
        )

class BiConRNN(hk.RNNCore):
  """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._hs = rl_params['s']
    self._vs = rl_params['s']
    self._ho = rl_params['o']
    self._vo = rl_params['o']

    self.w_h = rl_params['w_h']
    self.w_v = rl_params['w_v']
    self.w_c = rl_params['w_c']
    self.init_value = init_value

    self._n_actions = network_params['n_actions']
    self._hidden_size = network_params['hidden_size']

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']

  def _value_rnn(self, state, value, action, reward):

    pre_act_val = jnp.sum(value * action, axis=1)  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = value + action * update

    return next_value, next_state

  def _value_con_rnn(self, state, value, action, reward):

    pre_act_val = jnp.sum(value * (np.ones(1,2) - action), axis=1)  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = value + (np.ones(1,2) - action) * update

    return next_con_value, next_state

  def _habit_rnn(self, state, habit, action):

    inputs = action
    if self._ho:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, habit], axis=-1)
    if self._hs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    next_habit = hk.Linear(self._n_actions)(next_state)

    return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    h_state, v_state, c_state, habit, value, c_value = prev_state
    action = inputs[:, 0]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    action_onehot = jax.nn.one_hot(action,2)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action_onehot, reward)
    
    next_c_value, next_c_state = self._value_con_rnn(c_state, value, action_onehot, reward)

    # Habit module: update/create new habit
    next_habit, next_h_state = self._habit_rnn(h_state, habit, action_onehot)

    # Combine value and habit
    logits = self.w_v * next_value + self.w_h * next_habit + self.w_c * next_c_value  # (bs, n_a)

    return logits, (next_h_state, next_v_state, next_c_state, next_habit, next_value, next_c_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # h_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # c_state
        0 * jnp.ones([batch_size, self._n_actions]),  # habit
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # value
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # c_value
        )