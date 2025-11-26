"""
Neuron Models for Reservoir Computing Study

This module implements three different neuron models:
1. Standard Tanh Neuron (baseline)
2. Expressive Leaky Memory (ELM) Neuron
3. Calcitron Neuron

Each model implements the same interface for use in ESN architectures.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import warnings


class BaseNeuron(ABC):
    """Abstract base class for all neuron models."""

    def __init__(self, size: int, **kwargs):
        """
        Initialize neuron model.

        Args:
            size: Number of neurons in the population
            **kwargs: Model-specific parameters
        """
        self.size = size
        self.state = np.zeros(size)
        self.parameters = {}

    @abstractmethod
    def update(
        self, input_current: np.ndarray, reservoir_current: np.ndarray
    ) -> np.ndarray:
        """
        Update neuron states given input and reservoir currents.

        Args:
            input_current: Current from input connections [size]
            reservoir_current: Current from reservoir connections [size]

        Returns:
            Updated neuron states [size]
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset neuron states to initial conditions."""
        pass

    @abstractmethod
    def get_trainable_parameters(self) -> Dict[str, np.ndarray]:
        """Get dictionary of trainable parameters."""
        pass

    @abstractmethod
    def set_trainable_parameters(self, params: Dict[str, np.ndarray]):
        """Set trainable parameters from dictionary."""
        pass


class StandardTanhNeuron(BaseNeuron):
    """
    Standard leaky integrator neuron with tanh activation.

    Dynamics: x(t+1) = (1-α)x(t) + α * tanh(W_in * u(t) + W_res * x(t) + bias)
    """

    def __init__(
        self, size: int, leaking_rate: float = 0.3, bias: Optional[np.ndarray] = None
    ):
        """
        Initialize Standard Tanh neuron.

        Args:
            size: Number of neurons
            leaking_rate: Leaking rate α (0 < α ≤ 1)
            bias: Bias terms for each neuron [size] or None for zero bias
        """
        super().__init__(size)

        # Validate parameters
        if not 0 < leaking_rate <= 1:
            raise ValueError("Leaking rate must be in range (0, 1]")

        self.leaking_rate = leaking_rate
        self.bias = np.zeros(size) if bias is None else np.array(bias)

        if len(self.bias) != size:
            raise ValueError(f"Bias vector length {len(self.bias)} != size {size}")

        self.parameters = {"leaking_rate": leaking_rate, "bias": self.bias.copy()}

    def update(
        self, input_current: np.ndarray, reservoir_current: np.ndarray
    ) -> np.ndarray:
        """Update neuron states using leaky integration with tanh activation."""
        # Compute total current
        total_current = input_current + reservoir_current + self.bias

        # Apply tanh activation
        activated = np.tanh(total_current)

        # Leaky integration
        self.state = (
            1 - self.leaking_rate
        ) * self.state + self.leaking_rate * activated

        return self.state.copy()

    def reset(self):
        """Reset neuron states to zero."""
        self.state = np.zeros(self.size)

    def get_trainable_parameters(self) -> Dict[str, np.ndarray]:
        """Standard tanh neuron has no trainable parameters in this study."""
        return {}

    def set_trainable_parameters(self, params: Dict[str, np.ndarray]):
        """Standard tanh neuron has no trainable parameters in this study."""
        if params:
            warnings.warn("StandardTanhNeuron has no trainable parameters")


class ELMNeuron(BaseNeuron):
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
        fast_decay: float = 0.3,
        slow_decay: float = 0.01,
        input_dim: int = 1,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize ELM neuron.

        Args:
            size: Number of output neurons
            fast_decay: Fast decay rate β (0 < β ≤ 1)
            slow_decay: Slow decay rate γ (0 < γ << β)
            input_dim: Dimensionality of input
            random_seed: Random seed for weight initialization
        """
        super().__init__(size)

        # Validate parameters
        if not 0 < fast_decay <= 1:
            raise ValueError("Fast decay rate must be in range (0, 1]")
        if not 0 < slow_decay <= 1:
            raise ValueError("Slow decay rate must be in range (0, 1]")
        if slow_decay >= fast_decay:
            warnings.warn("Slow decay rate should be << fast decay rate")

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        self.fast_decay = fast_decay
        self.slow_decay = slow_decay
        self.input_dim = input_dim

        # Hidden and memory state sizes (equal to output size for simplicity)
        self.hidden_size = size
        self.memory_size = size

        # Initialize states
        self.hidden_state = np.zeros(self.hidden_size)
        self.memory_state = np.zeros(self.memory_size)
        # Initialize weight matrices with small random values
        scale = 0.1
        self.W_h = np.random.uniform(-scale, scale, (self.hidden_size, input_dim))
        self.U_h = np.random.uniform(
            -scale, scale, (self.hidden_size, self.hidden_size)
        )
        self.W_m = np.random.uniform(-scale, scale, (self.memory_size, input_dim))
        self.U_m = np.random.uniform(
            -scale, scale, (self.memory_size, self.memory_size)
        )

        # Output weights (combines hidden and memory states)
        combined_size = self.hidden_size + self.memory_size
        self.W_x = np.random.uniform(-scale, scale, (size, combined_size))
        self.b_x = np.random.uniform(-scale, scale, size)

        # Normalize recurrent matrices to have spectral radius < 1
        self._normalize_recurrent_weights()

        self.parameters = {
            "fast_decay": fast_decay,
            "slow_decay": slow_decay,
            "W_h": self.W_h.copy(),
            "U_h": self.U_h.copy(),
            "W_m": self.W_m.copy(),
            "U_m": self.U_m.copy(),
            "W_x": self.W_x.copy(),
            "b_x": self.b_x.copy(),
        }

    def _normalize_recurrent_weights(self):
        """Normalize recurrent weight matrices to have spectral radius < 1."""
        target_radius = 0.95

        print(self.U_h.shape, type(self.U_h))
        print(self.U_m.shape, type(self.U_m))
        # Normalize U_h
        eigenvals = np.linalg.eigvals(self.U_h)
        current_radius = np.max(np.abs(eigenvals))
        if current_radius > 0:
            self.U_h = self.U_h * (target_radius / current_radius)

        # Normalize U_m (slower dynamics, smaller radius)
        eigenvals = np.linalg.eigvals(self.U_m)
        current_radius = np.max(np.abs(eigenvals))
        if current_radius > 0:
            self.U_m = self.U_m * (target_radius * 0.5 / current_radius)

    def update(
        self, input_current: np.ndarray, reservoir_current: np.ndarray
    ) -> np.ndarray:
        """Update ELM neuron states."""
        # Ensure input_current has correct dimensions
        if input_current.size != self.input_dim:
            # Pad or truncate to match expected input dimension
            if input_current.size > self.input_dim:
                input_current = input_current[: self.input_dim]
            else:
                padded_input = np.zeros(self.input_dim)
                padded_input[: input_current.size] = input_current
                input_current = padded_input

        # Reshape for matrix multiplication
        input_vec = (
            input_current.reshape(-1, 1) if input_current.ndim == 1 else input_current
        )

        # Update hidden state (fast dynamics)
        print(self.hidden_state.shape)
        print(self.memory_state.shape)
        h_input = np.dot(self.W_h, input_vec).flatten()
        h_recurrent = np.dot(self.U_h, self.hidden_state)
        h_total = h_input + h_recurrent
        h_activated = np.tanh(h_total)
        self.hidden_state = (
            1 - self.fast_decay
        ) * self.hidden_state + self.fast_decay * h_activated

        # Update memory state (slow dynamics)
        m_input = np.dot(self.W_m, input_vec).flatten()
        m_recurrent = np.dot(self.U_m, self.memory_state)
        m_total = m_input + m_recurrent
        m_activated = np.tanh(m_total)
        self.memory_state = (
            1 - self.slow_decay
        ) * self.memory_state + self.slow_decay * m_activated
        # Combine hidden and memory states
        combined_state = np.concatenate([self.hidden_state, self.memory_state])
        # Compute output state
        output = np.dot(self.W_x, combined_state) + self.b_x
        self.state = np.tanh(output)

        # Add reservoir current (from other neurons in ESN)
        self.state = np.tanh(self.state + reservoir_current)

        return self.state.copy()

    def reset(self):
        """Reset all neuron states to zero."""
        self.state = np.zeros(self.size)
        self.hidden_state = np.zeros(self.hidden_size)
        self.memory_state = np.zeros(self.memory_size)

    def get_trainable_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters (decay rates and weights)."""
        return {
            "fast_decay": np.array([self.fast_decay]),
            "slow_decay": np.array([self.slow_decay]),
            "W_h": self.W_h.copy(),
            "U_h": self.U_h.copy(),
            "W_m": self.W_m.copy(),
            "U_m": self.U_m.copy(),
            "W_x": self.W_x.copy(),
            "b_x": self.b_x.copy(),
        }

    def set_trainable_parameters(self, params: Dict[str, np.ndarray]):
        """Set trainable parameters."""
        if "fast_decay" in params:
            self.fast_decay = float(params["fast_decay"][0])
            self.fast_decay = np.clip(self.fast_decay, 0.1, 0.9)

        if "slow_decay" in params:
            self.slow_decay = float(params["slow_decay"][0])
            self.slow_decay = np.clip(self.slow_decay, 0.001, 0.1)

        if "W_h" in params:
            self.W_h = params["W_h"].copy()
        if "U_h" in params:
            self.U_h = params["U_h"].copy()
        if "W_m" in params:
            self.W_m = params["W_m"].copy()
        if "U_m" in params:
            self.U_m = params["U_m"].copy()
        if "W_x" in params:
            self.W_x = params["W_x"].copy()
        if "b_x" in params:
            self.b_x = params["b_x"].copy()

        # Re-normalize recurrent weights
        self._normalize_recurrent_weights()


class CalcitronNeuron(BaseNeuron):
    """
    Calcitron Neuron based on calcium control hypothesis.

    Dynamics:
    C(t+1) = λ * C(t) + I_local + I_hetero + I_spike + I_supervisor
    x(t+1) = tanh(W_in * u(t) + W_res * x(t) + f(C(t+1)))

    Where C is calcium concentration and I_* are different calcium influx sources.
    """

    def __init__(
        self,
        size: int,
        calcium_decay: float = 0.95,
        influx_gains: Optional[np.ndarray] = None,
        modulation_strength: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Calcitron neuron.

        Args:
            size: Number of neurons
            calcium_decay: Calcium decay rate λ (0 < λ < 1)
            influx_gains: Gains for [local, hetero, spike, supervisor] influx
            modulation_strength: Strength of calcium modulation
            random_seed: Random seed for initialization
        """
        super().__init__(size)

        # Validate parameters
        if not 0 < calcium_decay < 1:
            raise ValueError("Calcium decay must be in range (0, 1)")

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.calcium_decay = calcium_decay
        self.modulation_strength = modulation_strength

        # Default influx gains: [local, hetero, spike, supervisor]
        if influx_gains is None:
            influx_gains = np.array([0.1, 0.05, 0.2, 0.0])
        self.influx_gains = np.array(influx_gains)

        if len(self.influx_gains) != 4:
            raise ValueError(
                "influx_gains must have 4 elements: [local, hetero, spike, supervisor]"
            )

        # Calcium concentration for each neuron
        self.calcium = np.zeros(size)

        # Previous state for spike detection
        self.prev_state = np.zeros(size)

        # Calcium modulation weights (random initialization)
        self.calcium_weights = np.random.uniform(-0.1, 0.1, size)

        self.parameters = {
            "calcium_decay": calcium_decay,
            "influx_gains": self.influx_gains.copy(),
            "modulation_strength": modulation_strength,
            "calcium_weights": self.calcium_weights.copy(),
        }

    def _compute_calcium_influx(
        self, input_current: np.ndarray, reservoir_current: np.ndarray
    ) -> np.ndarray:
        """Compute calcium influx from different sources."""
        # Local influx: proportional to input current
        I_local = self.influx_gains[0] * np.abs(input_current)

        # Heterosynaptic influx: proportional to reservoir activity
        I_hetero = self.influx_gains[1] * np.abs(reservoir_current)

        # Spike-dependent influx: proportional to state changes
        state_change = np.abs(self.state - self.prev_state)
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

    def update(
        self, input_current: np.ndarray, reservoir_current: np.ndarray
    ) -> np.ndarray:
        """Update Calcitron neuron states."""
        # Store previous state for spike detection
        self.prev_state = self.state.copy()

        # Compute calcium influx
        influx = self._compute_calcium_influx(input_current, reservoir_current)

        # Update calcium concentration
        self.calcium = self.calcium_decay * self.calcium + influx

        # Compute calcium modulation
        modulation = self._calcium_modulation()

        # Update neuron state with calcium modulation
        total_current = input_current + reservoir_current + modulation
        self.state = np.tanh(total_current)

        return self.state.copy()

    def reset(self):
        """Reset neuron and calcium states to zero."""
        self.state = np.zeros(self.size)
        self.calcium = np.zeros(self.size)
        self.prev_state = np.zeros(self.size)

    def get_trainable_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {
            "calcium_decay": np.array([self.calcium_decay]),
            "influx_gains": self.influx_gains.copy(),
            "modulation_strength": np.array([self.modulation_strength]),
            "calcium_weights": self.calcium_weights.copy(),
        }

    def set_trainable_parameters(self, params: Dict[str, np.ndarray]):
        """Set trainable parameters."""
        if "calcium_decay" in params:
            self.calcium_decay = float(params["calcium_decay"][0])
            self.calcium_decay = np.clip(self.calcium_decay, 0.8, 0.99)

        if "influx_gains" in params:
            self.influx_gains = params["influx_gains"].copy()
            self.influx_gains = np.clip(self.influx_gains, 0.01, 1.0)

        if "modulation_strength" in params:
            self.modulation_strength = float(params["modulation_strength"][0])
            self.modulation_strength = np.clip(self.modulation_strength, 0.01, 1.0)

        if "calcium_weights" in params:
            self.calcium_weights = params["calcium_weights"].copy()


def create_neuron_model(model_type: str, size: int, **kwargs) -> BaseNeuron:
    """
    Factory function to create neuron models.

    Args:
        model_type: Type of neuron ('standard', 'elm', 'calcitron')
        size: Number of neurons
        **kwargs: Model-specific parameters

    Returns:
        Initialized neuron model
    """
    model_type = model_type.lower()

    if model_type == "standard":
        return StandardTanhNeuron(size, **kwargs)
    elif model_type == "elm":
        return ELMNeuron(size, **kwargs)
    elif model_type == "calcitron":
        return CalcitronNeuron(size, **kwargs)
    else:
        raise ValueError(f"Unknown neuron model type: {model_type}")


if __name__ == "__main__":
    # Basic test of all neuron models
    print("Testing neuron models...")

    size = 10
    input_current = np.random.randn(size) * 0.1
    reservoir_current = np.random.randn(size) * 0.05

    # Test Standard neuron
    print("\n1. Testing Standard Tanh Neuron:")
    standard = StandardTanhNeuron(size)
    for i in range(5):
        state = standard.update(input_current, reservoir_current)
        print(f"  Step {i+1}: mean={state.mean():.4f}, std={state.std():.4f}")

    # Test ELM neuron
    print("\n2. Testing ELM Neuron:")
    elm = ELMNeuron(size, input_dim=size)
    for i in range(5):
        state = elm.update(input_current, reservoir_current)
        print(f"  Step {i+1}: mean={state.mean():.4f}, std={state.std():.4f}")
        print(state.shape)
    # Test Calcitron neuron
    print("\n3. Testing Calcitron Neuron:")
    calcitron = CalcitronNeuron(size + 2)
    for i in range(5):
        state = calcitron.update(input_current, reservoir_current)
        print(f"  Step {i+1}: mean={state.mean():.4f}, std={state.std():.4f}")
        print(f"           calcium: mean={calcitron.calcium.mean():.4f}")

    print("\nAll neuron models initialized successfully!")
