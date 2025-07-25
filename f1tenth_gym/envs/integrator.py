from abc import abstractmethod
from enum import Enum

class IntegratorType(Enum):
    RK4 = 1
    Euler = 2

    @staticmethod
    def from_string(integrator: str, *args):
        if integrator == "rk4":
            return RK4Integrator(*args)
        elif integrator == "euler":
            return EulerIntegrator(*args)
        else:
            raise ValueError(f"Unknown integrator type {integrator}")


class Integrator:
    def __init__(self) -> None:
        self._integrator_type = None

    @abstractmethod
    def integrate(self, f, x, u, *args):
        raise NotImplementedError("integrate method not implemented")

    @property
    def type(self) -> str:
        return self._integrator_type


class RK4Integrator(Integrator):
    def __init__(self, dt, integrator_dt=0.01) -> None:
        super().__init__()
        self._integrator_type = "rk4"
        self.integrator_dt = integrator_dt
        self.dt = dt
        self.n_steps = int(dt / integrator_dt)
        if self.n_steps < 1:
            raise ValueError("Integrator dt must be smaller than the total dt.")

    def integrate(self, f, x, u, *args):
        for _ in range(self.n_steps):
            k1 = f(x, u, *args)

            k2_state = x + self.integrator_dt * (k1 / 2)
            k2 = f(k2_state, u, *args)

            k3_state = x + self.integrator_dt * (k2 / 2)
            k3 = f(k3_state, u, *args)

            k4_state = x + self.integrator_dt * k3
            k4 = f(k4_state, u, *args)

            # dynamics integration
            x = x + self.integrator_dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


class EulerIntegrator(Integrator):
    def __init__(self, dt, integrator_dt=0.01) -> None:
        super().__init__()
        self._integrator_type = "euler"
        self.integrator_dt = integrator_dt
        self.dt = dt
        self.n_steps = int(dt / integrator_dt)
        if self.n_steps < 1:
            raise ValueError("Integrator dt must be smaller than the total dt.")

    def integrate(self, f, x, u, *args):
        for _ in range(self.n_steps):
            x = x + self.integrator_dt * f(x, u, *args)
        return x
