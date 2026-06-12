"""Visualizer of circuit representation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from mqc3.circuit.state import BosonicState

logger = logging.getLogger(__name__)


def meshed_wigner(state: BosonicState, xvec: ArrayLike, pvec: ArrayLike) -> NDArray[np.floating[Any]]:
    r"""Calculates the discretized Wigner function of the specified mode.

    Args:
        state (BosonicState): The state to calculate the Wigner function for.
        xvec (ArrayLike): Array of discretized :math:`x` quadrature values.
        pvec (ArrayLike): Array of discretized :math:`p` quadrature values.

    Raises:
        ValueError: If the state is not a single mode state.

    Returns:
        NDArray[np.floating[Any]]: The discretized Wigner function. The shape is (len(xvec), len(pvec)).
    """
    if state.n_modes != 1:
        msg = (
            "Only single mode states are supported. "
            f"The state has {state.n_modes} modes. "
            "Use `BosonicState.extract_mode()` to extract a single mode."
        )
        logger.error(msg)
        raise ValueError(msg)

    x_grid, p_grid = np.meshgrid(xvec, pvec, sparse=True)

    wigner = 0
    for coeff, gauss in zip(state.coeffs, state.gaussian_states, strict=True):
        if x_grid.shape == p_grid.shape:
            arr = np.array([x_grid - gauss.mean[0], p_grid - gauss.mean[1]])
            arr = arr.squeeze()
        else:
            arr = np.array([x_grid - gauss.mean[0], p_grid - gauss.mean[1]], dtype=object)

        exp_arg = arr @ np.linalg.inv(gauss.cov) @ arr
        prefactor = 1 / (np.sqrt(np.linalg.det(2 * np.pi * gauss.cov)))
        wigner += (coeff * prefactor) * np.exp(-0.5 * (exp_arg))

    eps = 1e-6
    if np.any(np.abs(wigner.imag) > eps):
        msg = "Imaginary components are non-zero."
        logger.warning(msg)
    return np.real_if_close(wigner)


def _auto_vec(
    state: BosonicState,
    cutoff: float = 1e-6,
    n_points: int = 200,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    x_max = 0.0
    p_max = 0.0
    for coeff, gauss in zip(state.coeffs, state.gaussian_states, strict=True):
        peak = abs(coeff) / np.sqrt(np.linalg.det(2 * np.pi * gauss.cov))
        if peak <= cutoff:
            continue
        n_sigma = np.sqrt(-2.0 * np.log(cutoff / peak))
        x_max = max(x_max, abs(float(gauss.mean[0])) + n_sigma * np.sqrt(float(gauss.cov[0, 0])))
        p_max = max(p_max, abs(float(gauss.mean[1])) + n_sigma * np.sqrt(float(gauss.cov[1, 1])))
    r = max(x_max, p_max)
    return np.linspace(-r, r, n_points), np.linspace(-r, r, n_points)


def make_wigner_figure(
    state: BosonicState,
    xvec: ArrayLike | None = None,
    pvec: ArrayLike | None = None,
    *,
    cutoff: float = 1e-6,
    n_points: int = 200,
    title: str | None = None,
) -> Figure:
    r"""Plots the discretized Wigner function of the specified mode.

    Args:
        state (BosonicState): The state to calculate the Wigner function for.
        xvec (ArrayLike | None, optional): Array of discretized :math:`x` quadrature values.
            If `None`, the range is determined automatically. Defaults to `None`.
        pvec (ArrayLike | None, optional): Array of discretized :math:`p` quadrature values.
            If `None`, the range is determined automatically. Defaults to `None`.
        cutoff (float, optional): Absolute threshold below which the Wigner function is
            considered zero when computing the auto range. Defaults to `1e-6`.
        n_points (int, optional): Number of points in each axis for the auto range.
            Defaults to `200`.
        title (str | None, optional): The title of the plot. Defaults to `None`.

    Returns:
        Figure: The matplotlib figure object.
    """
    if xvec is None or pvec is None:
        auto_x, auto_p = _auto_vec(state, cutoff=cutoff, n_points=n_points)
        if xvec is None:
            xvec = auto_x
        if pvec is None:
            pvec = auto_p
    wigner = meshed_wigner(state, xvec, pvec)
    wigner = np.round(wigner.real, 4)
    scale = float(np.max(wigner.real))
    nrm = colors.Normalize(-scale, scale)
    r = max(float(np.abs(np.asarray(xvec)).max()), float(np.abs(np.asarray(pvec)).max()))
    fig, ax = plt.subplots()
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    contour_plot = ax.contourf(xvec, pvec, wigner, 60, cmap="RdBu", norm=nrm)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    if title is not None:
        ax.set_title(title)
    fig.colorbar(contour_plot, label="Wigner function value")
    plt.close(fig)
    return fig


def save_wigner_plot(state: BosonicState, filename: str, **kwargs) -> None:  # noqa: ANN003
    r"""Plots the discretized Wigner function of the specified mode.

    Args:
        state (BosonicState): The state to calculate the Wigner function for.
        filename (str): The path to save the plot to.
        kwargs: Keyword arguments for visualizing configuration.
    """
    fig = make_wigner_figure(state, **kwargs)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
