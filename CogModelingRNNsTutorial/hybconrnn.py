"""Define hybRNNs."""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

RNNState = jnp.array


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

    pre_act_val = jnp.sum(value * (np.ones([1,2]) - action), axis=1)  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = action * value + (np.ones([1,2]) - action) * update

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
    logits = self.w_v * next_value * action_onehot + self.w_h * next_habit + self.w_c * next_c_value * (np.ones([1,2])-action_onehot)  # (bs, n_a)

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

class BiConRNN_hoHabit(hk.RNNCore):
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

    pre_act_val = jnp.sum(value * (np.ones([1,2]) - action), axis=1)  # (batch_s, 1)

    inputs = jnp.concatenate(
        [pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis]], axis=-1)
    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)

    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))

    update = hk.Linear(1)(next_state)
    value = (1 - self.forget) * value + self.forget * self.init_value
    next_value = action * value + (np.ones([1,2]) - action) * update

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
    v_state, c_state, value, c_value = prev_state
    action = inputs[:, 0]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    action_onehot = jax.nn.one_hot(action,2)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action_onehot, reward)
    
    next_c_value, next_c_state = self._value_con_rnn(c_state, value, action_onehot, reward)

    # Habit module: update/create new habit
    #next_habit, next_h_state = self._habit_rnn(h_state, habit, action_onehot)

    # Combine value and habit
    logits = self.w_v * next_value * action_onehot + self.w_c * next_c_value * (np.ones([1,2])-action_onehot)  # (bs, n_a)

    return logits, (next_v_state, next_c_state, next_value, next_c_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # c_state
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # value
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # c_value
        )