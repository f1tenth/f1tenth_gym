from abc import abstractmethod
from enum import Enum


class IntegratorType(Enum):
    RK4 = 1
    Euler = 2

    @staticmethod
    def from_string(integrator: str):
        if integrator == "rk4":
            return RK4Integrator()
        elif integrator == "euler":
            return EulerIntegrator()
        else:
            raise ValueError(f"Unknown integrator type {integrator}")


class Integrator:
    def __init__(self) -> None:
        self._integrator_type = None

    @abstractmethod
    def integrate(self, f, x, u, dt, params):
        raise NotImplementedError("integrate method not implemented")

    @property
    def type(self) -> str:
        return self._integrator_type


class RK4Integrator(Integrator):
    def __init__(self) -> None:
        super().__init__()
        self._integrator_type = "rk4"

    def integrate(self, f, x, u, dt, params):
        k1 = f(x, u, params)

        k2_state = x + dt * (k1 / 2)
        k2 = f(k2_state, u, params)

        k3_state = x + dt * (k2 / 2)
        k3 = f(k3_state, u, params)

        k4_state = x + dt * k3
        k4 = f(k4_state, u, params)

        # dynamics integration
        x = x + dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


class EulerIntegrator(Integrator):
    def __init__(self) -> None:
        super().__init__()
        self._integrator_type = "euler"

    def integrate(self, f, x, u, dt, params):
        dstate = f(x, u, params)
        x = x + dt * dstate
        return x
