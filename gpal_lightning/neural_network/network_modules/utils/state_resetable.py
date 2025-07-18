import copy

from torch import Tensor


class StateResetable:
    """StateResetable allows user to initialize, set, and reset states. It's written as a
    mixin to allow subclasses of nn.Module to manage the recurrent state such as ConvLSTM.
    See here for an explanation of Mixin:
    https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # default_registry keeps track of the default value of each state (aka initial state).
        self.default_registry = {}
        # state_registry stores the states, each has a name. state_registry's key-value pairs have
        # one-to-one correspondence with default_registry.
        self.state_registry = {}

    def get_state(self, name: str, default: Tensor = None):
        if name in self.default_registry:
            return self.state_registry[name]
        if default is None:
            raise ValueError(f"state with {name} has not been registered and " "no default state is specified.")
        assert not default.requires_grad, "default value should not need gradient"
        
        self.default_registry[name] = copy.deepcopy(default)
        self.state_registry[name] = default
        return self.state_registry[name]

    def set_state(self, name: str, state: Tensor):
        """Set the named state with specified tensor."""
        if name not in self.default_registry:
            raise RuntimeError(f"buffer {name} hasn't been registered yet.")
        default = self.default_registry[name]
        assert default.shape == state.shape and default.dtype == state.dtype
        self.state_registry[name] = state

    def reset_state(self, name: str):
        """Reset the state with name to the default."""
        if name not in self.default_registry:
            raise RuntimeError(f"buffer {name} hasn't been registered yet.")
        self.state_registry[name] = self.default_registry[name]

    def reset_all_states(self):
        """Resets all the states."""
        for name, _ in self.default_registry.items():
            self.reset_state(name)
