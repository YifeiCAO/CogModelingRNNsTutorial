"""Define hybRNNs."""
from typing import Optional, Tuple

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
    """
    A Learnable Gated Hybrid RNN (Update-Gating Version).
    Explicitly arbitrates between a 'Context Update' and a 'Memory Update'.
    """

    def __init__(self, rl_params, network_params, init_value=0.5):
        super().__init__()
        self._hidden_size = network_params['hidden_size']
        self._n_actions = network_params['n_actions']
        self.w_h = rl_params['w_h']
        self.w_v = rl_params['w_v']
        self.init_value = init_value
        
        if rl_params.get('fit_forget', False):
            init = hk.initializers.RandomNormal(stddev=1, mean=0)
            self.forget = jax.nn.sigmoid(
                hk.get_parameter('unsigmoid_forget', (1,), init=init)
            )
        else:
            self.forget = rl_params.get('forget', 0.1)

    def _learnable_gate_mech(self, state, value, action_onehot, reward):
        """
        Separated Stream Processing:
        Returns (h_context, h_memory, gate_signal) so we can compute distinct updates.
        """
        # 1. Input Preparation
        reward_expanded = reward[:, None] 
        
        # =========================================================
        # 2. The Gate (Arbitrator)
        # =========================================================
        # 输入: [Reward, Action, Value, Prev_State]
        # 逻辑: 判断 "Surprise" -> 决定听谁的
        
        # 修复维度问题：将 value 和 state 展平以便拼接
        # value: (B, n_actions), state: (B, hidden_size)
        # 需要确保所有输入都是兼容的维度
        value_flat = value  # 保持原样，已经是 (B, n_actions)
        state_flat = state  # 保持原样，已经是 (B, hidden_size)
        
        # 拼接所有输入
        gate_inputs = jnp.concatenate([
            reward_expanded,    # (B, 1)
            action_onehot,      # (B, n_actions)
            value_flat,         # (B, n_actions)
            state_flat          # (B, hidden_size)
        ], axis=-1)

        # 计算门控信号
        gate_hidden = jax.nn.relu(hk.Linear(16, name='gate_hidden')(gate_inputs))
        # Init to 0 -> Sigmoid(0)=0.5 (Fair start)
        gate_logit = hk.Linear(1, name='gate_out',
                               w_init=hk.initializers.RandomNormal(stddev=0.01),
                               b_init=hk.initializers.Constant(0.0))(gate_hidden)
        gate_signal = jax.nn.sigmoid(gate_logit) 

        # =========================================================
        # 3. Stream A: Context Channel (Fast / Evidence)
        # =========================================================
        # 输入: [Reward, Action, Value]
        context_inputs = jnp.concatenate([
            reward_expanded,    # (B, 1)
            action_onehot,      # (B, n_actions)
            value_flat          # (B, n_actions)
        ], axis=-1)
        h_context = hk.Linear(self._hidden_size, name='ctx_stream')(context_inputs)

        # =========================================================
        # 4. Stream B: Memory Channel (Slow / History)
        # =========================================================
        # 输入: [Prev_State]
        h_memory = hk.Linear(self._hidden_size, name='mem_stream')(state_flat)
        
        return h_context, h_memory, gate_signal

    def _value_rnn(self, state, value, action_onehot, reward):
        with hk.name_scope('value_module'):
            # 1. 获取两路隐状态和门控信号
            h_ctx, h_mem, gate_val = self._learnable_gate_mech(state, value, action_onehot, reward)
            
            # =========================================================
            # 核心修改：Output Gating (对 Update 进行加权)
            # =========================================================
            
            # 2.1 Context 建议的更新量 (Based on immediate evidence)
            update_ctx = hk.Linear(1, name='update_ctx_proj')(jax.nn.tanh(h_ctx))
            
            # 2.2 Memory 建议的更新量 (Based on history)
            update_mem = hk.Linear(1, name='update_mem_proj')(jax.nn.tanh(h_mem))
            
            # 2.3 最终更新量：Gate 决定听谁的
            # Gate=1 -> 全听 Context (快速反转); Gate=0 -> 全听 Memory (保持)
            final_update = gate_val * update_ctx + (1.0 - gate_val) * update_mem
            
            # =========================================================
            # 3. Value Update
            # =========================================================
            # Apply standard decay
            decayed_value = (1 - self.forget) * value + self.forget * self.init_value
            
            # Apply the gated update to the chosen action
            next_value = decayed_value + action_onehot * final_update

            # =========================================================
            # 4. State Update (For Recurrence)
            # =========================================================
            # 为了让 RNN 能够传递状态，我们依然需要计算一个 unified next_state
            # 这里我们也用同样的 gate 逻辑融合隐状态，作为下一时刻的 Memory
            mixed_hidden = gate_val * h_ctx + (1.0 - gate_val) * h_mem
            next_state = jax.nn.tanh(mixed_hidden)

        return next_value, next_state, gate_val

    def _habit_rnn(self, state, habit, action_onehot, value, reward):
        """
        Habit module applying the same logic
        """
        with hk.name_scope('habit_module'):
            h_ctx, h_mem, gate_val = self._learnable_gate_mech(state, value, action_onehot, reward)
            
            # 分别计算 Habit Update
            update_ctx = hk.Linear(self._n_actions, name='habit_ctx_proj')(jax.nn.tanh(h_ctx))
            update_mem = hk.Linear(self._n_actions, name='habit_mem_proj')(jax.nn.tanh(h_mem))
            
            # 融合 Habit
            # 注意: Habit 通常是直接输出 logits，这里我们把它看作是对 habit state 的 update，或者直接作为 next habit
            # 既然是 RNN，我们假设它是直接输出 next_habit
            next_habit = gate_val * update_ctx + (1.0 - gate_val) * update_mem
            
            # 更新 hidden state
            mixed_hidden = gate_val * h_ctx + (1.0 - gate_val) * h_mem
            next_state = jax.nn.tanh(mixed_hidden)

        return next_habit, next_state, gate_val

    def __call__(self, inputs: jnp.ndarray, prev_state: Tuple):
        h_state, v_state, habit, value = prev_state
        action = inputs[:, 0]
        reward = inputs[:, -1]
        action_onehot = jax.nn.one_hot(action, self._n_actions)
        
        # Value Module
        next_value, next_v_state, v_gate = self._value_rnn(v_state, value, action_onehot, reward)

        # Habit Module
        next_habit, next_h_state, h_gate = self._habit_rnn(h_state, habit, action_onehot, value, reward)

        logits = self.w_v * next_value + self.w_h * next_habit
        
        next_state_tuple = (next_h_state, next_v_state, next_habit, next_value)
        
        # 依然返回 logits 和 tuple，Gate 值需要 Analysis 函数通过 unroll 获取
        return logits, next_state_tuple

    def initial_state(self, batch_size: Optional[int]):
        return (
            jnp.zeros([batch_size, self._hidden_size]),
            jnp.zeros([batch_size, self._hidden_size]),
            jnp.zeros([batch_size, self._n_actions]),
            self.init_value * jnp.ones([batch_size, self._n_actions]),
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