from abc import abstractmethod
from enum import Enum


class IntegratorType(Enum):
    """Integrator enum

    RK4: 1

    Euler: 2
    """

    RK4 = 1
    Euler = 2

    @staticmethod
    def from_string(integrator: str):
        """Set integrator by string

        Parameters
        ----------
        integrator : str
            integrator type

        Returns
        -------
        RK4Integrator | EulerIntegrator
            integrator object

        Raises
        ------
        ValueError
            Unknown integrator type
        """
        if integrator == "rk4":
            return RK4Integrator()
        elif integrator == "euler":
            return EulerIntegrator()
        else:
            raise ValueError(f"Unknown integrator type {integrator}")


class Integrator:
    """Integrator abstract class"""

    def __init__(self) -> None:
        self._integrator_type = None

    @abstractmethod
    def integrate(self, f, x, u, dt, params):
        """Integrate dynamics

        Parameters
        ----------
        f : np.ndarray
            RHS of ODE
        x : np.ndarray
            state
        u : np.ndarray
            control input
        dt : float
            sampling time
        params : dict
            parameter dictionary

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("integrate method not implemented")

    @property
    def type(self) -> str:
        """property, integrator type

        Returns
        -------
        str
            type
        """
        return self._integrator_type


class RK4Integrator(Integrator):
    """Runge Kutta fourth order integrator"""

    def __init__(self) -> None:
        super().__init__()
        self._integrator_type = "rk4"

    def integrate(self, f, x, u, dt, params):
        """Integrate dynamics

        Parameters
        ----------
        f : np.ndarray
            RHS of ODE
        x : np.ndarray
            state
        u : np.ndarray
            control input
        dt : float
            sampling time
        params : dict
            parameter dictionary

        Returns
        -------
        np.ndarray:
            integrated state
        """
        k1 = f(
            x,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )

        k2_state = x + dt * (k1 / 2)

        k2 = f(
            k2_state,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )

        k3_state = x + dt * (k2 / 2)

        k3 = f(
            k3_state,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )

        k4_state = x + dt * k3

        k4 = f(
            k4_state,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )

        # dynamics integration
        x = x + dt * (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x


class EulerIntegrator(Integrator):
    """Euler integrator"""

    def __init__(self) -> None:
        super().__init__()
        self._integrator_type = "euler"

    def integrate(self, f, x, u, dt, params):
        """Integrate dynamics

        Parameters
        ----------
        f : np.ndarray
            RHS of ODE
        x : np.ndarray
            state
        u : np.ndarray
            control input
        dt : float
            sampling time
        params : dict
            parameter dictionary

        Returns
        -------
        np.ndarray:
            integrated state
        """
        dstate = f(
            x,
            u,
            params["mu"],
            params["C_Sf"],
            params["C_Sr"],
            params["lf"],
            params["lr"],
            params["h"],
            params["m"],
            params["I"],
            params["s_min"],
            params["s_max"],
            params["sv_min"],
            params["sv_max"],
            params["v_switch"],
            params["a_max"],
            params["v_min"],
            params["v_max"],
        )
        x = x + dt * dstate
        return x
