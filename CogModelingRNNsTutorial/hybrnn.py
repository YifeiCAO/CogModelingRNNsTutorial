import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional

class BiRNN(hk.RNNCore):
    """A hybrid RNN: 'habit' processes action choices; 'value' processes rewards."""

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
        self.forget = jax.nn.sigmoid(hk.get_parameter('forget', (1,), init=hk.initializers.Constant(0.5)))

        # Initialize LSTM layers
        self.value_lstm = hk.LSTM(self._hidden_size)
        self.habit_lstm = hk.LSTM(self._hidden_size)

    def _value_rnn(self, state, value, action, reward):
        pre_act_val = jnp.sum(value * action, axis=1)
        inputs = jnp.concatenate([pre_act_val[:, jnp.newaxis], reward[:, jnp.newaxis], value], axis=-1)

        # Update state with LSTM
        next_state = self.value_lstm(inputs, state)
        new_hidden, new_cell = next_state

        # Update value based on LSTM's hidden state
        update = hk.Linear(self._n_actions)(next_state[0])
        next_value = (1 - self.forget) * value + self.forget * update + action * self.init_value

        return next_value, (new_state, new_cell)

    def _habit_rnn(self, state, habit, action):
        inputs = jnp.concatenate([action, habit], axis=-1)
        
        # Update state with LSTM
        next_state = self.habit_lstm(inputs, state)
        new_hidden, new_cell = next_state
        
        # Determine new habit from the LSTM's hidden state
        next_habit = hk.Linear(self._n_actions)(next_state[0])

        return next_habit, (new_state, new_cell)

    def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
        h_state, v_state, habit, value = prev_state
        action = inputs[:, 0]
        reward = inputs[:, -1]
        action_onehot = jax.nn.one_hot(action, self._n_actions)

        # Update value and habit states
        next_value, next_v_state = self._value_rnn(v_state, value, action_onehot, reward)
        next_habit, next_h_state = self._habit_rnn(h_state, habit, action_onehot)

        # Combine updated values and habits into logits
        logits = self.w_v * next_value + self.w_h * next_habit

        return logits, (next_h_state, next_v_state, next_habit, next_value)

    def initial_state(self, batch_size: Optional[int]):
        # Initial state for both LSTM modules
        return (self.value_lstm.initial_state(batch_size), 
                self.habit_lstm.initial_state(batch_size),
                0 * jnp.ones([batch_size, self._n_actions]),
                self.init_value * jnp.ones([batch_size, self._n_actions]))
