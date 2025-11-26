from typing import Callable, Optional, Union
from reservoirpy import Node
from reservoirpy.activationsfunc import tanh
from reservoirpy.mat_gen import bernoulli, normal, uniform
import numpy as np
import scipy as sp
from scipy import stats
from reservoirpy.type import Timestep, Weights, State


class StandardTanhNeuron(Node):

    def __init__(
        self,
        size: int,
        lr: float,
        bias: Optional[np.ndarray] = None,
        input_scaling=1.0,
        input_connectivity=0.1,
        activation=tanh,
        Win: Union["Weights", Callable] = bernoulli,
        W: Union["Weights", Callable] = normal,
        dtype: type = np.float64,
        seed=42,
    ):

        if lr < 0 or lr > 1:
            raise ValueError("Leaking rate should be in the range of [0, 1]")

        self.size = size
        self.lr = lr
        self.bias = np.zeros(size) if bias is None else np.array(bias)
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.Win = Win
        self.W = W
        self.seed = seed
        self.dtype = dtype
        self.activation = activation
        self.output_dim = size

        if len(self.bias) != size:
            raise ValueError(f"Bias vector length {len(self.bias)} != {size}.")

    def initialize(self, x):

        self._set_input_dim(x)

        if callable(self.Win):
            self.Win = self.Win(
                self.size,
                self.input_dim,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=self.seed,
            )

        if callable(self.W):
            self.W = self.W(
                self.size,
                self.size,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=self.seed,
            )

        self.state = {"out": np.zeros((self.size,))}

        self.initialized = True

    def _step(self, state: State, x: Timestep):
        W = self.W
        Win = self.Win
        bias = self.bias
        f = self.activation
        lr = self.lr
        s = state["out"]

        next_state = f(W @ s + Win @ x + bias)
        next_state = (1 - lr) * s + lr * next_state

        return {"out": next_state}


class ELMNeuron(Node):
    """
    Expressive Leaky Memory (ELM) Neuron.

    Dynamics:
    h(t+1) = (1-β)h(t) + β * tanh(W_h * u(t) + U_h * h(t))
    m(t+1) = (1-γ)m(t) + γ * tanh(W_m * u(t) + U_m * m(t))
    x(t+1) = tanh(W_x * [h(t+1); m(t+1)] + b_x)

    Where h is fast hidden state, m is slow memory state.
    """

    def __init__(
        self,
        size: int,
        input_scaling=1.0,
        input_connectivity=0.1,
        activation=tanh,
        fast_decay=0.3,
        slow_decay=0.01,
        scale=0.1,
        W_h: Union["Weights", Callable] = uniform,
        U_h: Union["Weights", Callable] = uniform,
        W_m: Union["Weights", Callable] = uniform,
        U_m: Union["Weights", Callable] = uniform,
        W_x: Union["Weights", Callable] = uniform,
        bias: Union["Weights", Callable, float] = uniform,
        dtype: type = np.float64,
        seed=None,
    ):
        self.size = size
        self.bias = bias
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.activation = activation
        self.fast_decay = fast_decay
        self.slow_decay = slow_decay
        self.scale = scale
        self.W_h = W_h
        self.U_h = U_h
        self.W_m = W_m
        self.U_m = U_m
        self.W_x = W_x
        self.dtype = dtype
        rnd_gen = np.random.default_rng()

        if seed is not None:
            self.seed = seed
        else:
            self.seed = rnd_gen.integers(1, 50)
        self.output_dim = size

    def _normalize_recurrent_weights(self):
        """Normalize recurrent weight matrices to have spectral radius < 1."""
        target_radius = 0.95
        # Normalize U_h
        eigenvals, _ = sp.sparse.linalg.eigs(self.U_h, k=1, which="LM")
        current_radius = (np.abs(eigenvals)).max()
        if current_radius > 0:
            self.U_h = self.U_h * (target_radius / current_radius)

        # Normalize U_m (slower dynamics, smaller radius)
        eigenvals, _ = sp.sparse.linalg.eigs(self.U_m)
        current_radius = (np.abs(eigenvals)).max()
        if current_radius > 0:
            self.U_m = self.U_m * (target_radius * 0.5 / current_radius)

    def initialize(self, x):

        self._set_input_dim(x)

        if callable(self.W_h):
            self.W_h = self.W_h(
                self.size,
                self.input_dim,
                low=-self.scale,
                high=self.scale,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=self.seed,
            )

        if callable(self.U_h):
            self.U_h = self.U_h(
                self.size,
                self.size,
                low=-self.scale,
                high=self.scale,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=self.seed,
            )

        if callable(self.W_m):
            self.W_m = self.W_m(
                self.size,
                self.input_dim,
                low=-self.scale,
                high=self.scale,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=self.seed,
            )

        if callable(self.U_m):
            self.U_m = self.U_m(
                self.size,
                self.size,
                low=-self.scale,
                high=self.scale,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=self.seed,
            )

        if callable(self.W_x):
            self.W_x = self.W_x(
                self.size,
                self.size + self.size,
                low=-self.scale,
                high=self.scale,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=self.seed,
            )

        if callable(self.bias):
            self.bias = self.bias(
                self.size,
                low=-self.scale,
                high=self.scale,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=self.seed,
            )

        self.hidden_state = np.zeros(self.size)
        self.memory_state = np.zeros(self.size)
        self.state = {"out": np.zeros((self.size,))}
        self._normalize_recurrent_weights()
        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:

        if x.size != self.input_dim:
            # Pad or truncate to match expected input dimension
            if x.size > self.input_dim:
                x = x[: self.input_dim]
            else:
                padded_input = np.zeros(self.input_dim)
                padded_input[: x.size] = x
                x = padded_input

        # Reshape for matrix multiplication
        input_vec = x.reshape(-1, 1) if x.ndim == 1 else x
        # Update hidden state (fast dynamics)
        h_input = self.W_h.dot(input_vec).flatten()
        h_recurrent = self.U_h.dot(self.hidden_state)
        h_total = h_input + h_recurrent
        h_activated = np.tanh(h_total)
        self.hidden_state = (
            1 - self.fast_decay
        ) * self.hidden_state + self.fast_decay * h_activated

        # Update memory state (slow dynamics)
        m_input = self.W_m.dot(input_vec).flatten()
        m_recurrent = self.U_m.dot(self.memory_state)
        m_total = m_input + m_recurrent
        m_activated = np.tanh(m_total)
        self.memory_state = (
            1 - self.slow_decay
        ) * self.memory_state + self.slow_decay * m_activated
        #
        # Combine hidden and memory states
        combined_state = np.concatenate((self.hidden_state, self.memory_state))
        # Compute output state
        output = self.W_x.dot(combined_state) + self.bias
        next_state = np.tanh(output)
        # Add reservoir current (from other neurons in ESN)
        next_state = np.tanh(next_state + state["out"])
        return {"out": next_state}


class CalcitronNeuron(Node):
    def __init__(
        self,
        size: int,
        calcium_decay: float = 0.95,
        influx_gains: Optional[np.ndarray] = None,
        modulation_strength: float = 0.1,
        random_seed: Optional[int] = None,
    ):

        if not 0 < calcium_decay < 1:
            raise ValueError("Calcium decay must be in range (0, 1)")

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.calcium_decay = calcium_decay
        self.modulation_strength = modulation_strength
        self.size = size
        self.output_dim = size
        # Default influx gains: [local, hetero, spike, supervisor]
        if influx_gains is None:
            influx_gains = np.array([0.1, 0.05, 0.2, 0.0])
        self.influx_gains = np.array(influx_gains)

        if len(self.influx_gains) != 4:
            raise ValueError(
                "influx_gains must have 4 elements: [local, hetero, spike, supervisor]"
            )

    def initialize(
        self,
        x,
    ):
        self._set_input_dim(x)

        self.calcium = np.zeros(self.size)

        self.prev_state = np.zeros(self.size)
        self.state = {"out": np.zeros((self.size,))}

        self.calcium_weights = uniform(self.size, low=-0.1, high=0.1)
        # self.calcium_weights = uni

    def _compute_calcium_influx(
        self, x: np.ndarray, reservoir_current: np.ndarray
    ) -> np.ndarray:
        """Compute calcium influx from different sources."""
        # Local influx: proportional to input current
        I_local = self.influx_gains[0] * np.abs(x)
        # Heterosynaptic influx: proportional to reservoir activity
        I_hetero = self.influx_gains[1] * np.abs(reservoir_current)
        # Spike-dependent influx: proportional to state changes
        state_change = np.abs(self.state["out"] - self.prev_state["out"])
        I_spike = self.influx_gains[2] * state_change

        # Supervisor influx: currently zero (could be used for supervised learning)
        I_supervisor = self.influx_gains[3] * np.zeros(self.size)

        return I_local + I_hetero + I_spike + I_supervisor

    def _calcium_modulation(self) -> np.ndarray:
        """Compute calcium-dependent modulation term."""
        # Sigmoidal modulation based on calcium concentration
        modulation = (
            self.modulation_strength * self.calcium_weights * np.tanh(self.calcium)
        )
        return modulation

    def _step(self, state: State, x: Timestep) -> State:

        self.prev_state = state.copy()

        influx = self._compute_calcium_influx(x, state["out"])

        self.calcium = self.calcium_decay * self.calcium + influx

        modulation = self._calcium_modulation()

        total_current = x + state["out"] + modulation

        next_state = np.tanh(total_current)

        return {"out": next_state}


if __name__ == "__main__":
    # Basic test of all neuron models
    print("Testing neuron models...")

    size = 10
    lr = 0.1
    input_current = np.random.randn(size) * 0.1
    reservoir_current = {"out": np.random.randn(size) * 0.05}

    # Test Standard neuron
    print("\n1. Testing Standard Tanh Neuron:")
    standard = StandardTanhNeuron(size, lr)
    for i in range(5):
        state = standard.step(input_current)
        print(f"  Step {i+1}: mean={state.mean():.4f}, std={state.std():.4f}")

    print("\n2. Testing ELM Neuron:")
    elm = ELMNeuron(size)
    for i in range(5):
        state = elm.step(input_current)
        print(f"  Step {i+1}: mean={state.mean():.4f}, std={state.std():.4f}")

    # Test Calcitron neuron
    print("\n3. Testing Calcitron Neuron:")
    calcitron = CalcitronNeuron(size)
    print(input_current.shape)
    for i in range(5):
        state = calcitron.step(input_current)
        print("State Shape:", state.shape)
        print(f"  Step {i+1}: mean={state.mean():.4f}, std={state.std():.4f}")
        print(f"           calcium: mean={calcitron.calcium.mean():.4f}")

    print("\nAll neuron models initialized successfully!")
