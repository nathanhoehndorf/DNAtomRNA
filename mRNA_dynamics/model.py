import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def simulate(
    input_model,
    alpha=1.0,
    gamma=0.5,
    m0=0.0,
    t_span=(0.0, 20.0),
    n_points=1000,
    rtol=1e-8,
    atol=1e-10,
):
    """
    Solve m' = alpha*u(t) - gamma*m.

    Returns
    -------
    t : ndarray
    m : ndarray
    u : ndarray
    """
    if alpha <= 0 or gamma <= 0:
        raise ValueError("alpha and gamma must be positive.")

    if m0 < 0:
        raise ValueError("Initial abundance m0 must be nonnegative.")

    t_start, t_end = map(float, t_span)

    if t_end <= t_start:
        raise ValueError("t_span must have positive duration.")

    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    t_eval = np.linspace(t_start, t_end, n_points)

    # Split integration at known discontinuities.
    breaks = sorted({
        float(b)
        for b in input_model.breakpoints
        if t_start < b < t_end
    })
    edges = [t_start, *breaks, t_end]

    all_t = []
    all_m = []
    current_m = float(m0)

    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        # Include each requested output time in its segment.
        segment_times = t_eval[
            (t_eval >= left) & (t_eval <= right)
        ]

        # Ensure the solver evaluates the segment endpoints.
        segment_times = np.unique(
            np.concatenate(([left], segment_times, [right]))
        )

        def rhs(t, y):
            return [alpha * float(input_model(t)) - gamma * y[0]]

        sol = solve_ivp(
            rhs,
            (left, right),
            [current_m],
            t_eval=segment_times,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        # Avoid duplicate times at segment boundaries.
        start_index = 0 if i == 0 else 1
        all_t.extend(sol.t[start_index:])
        all_m.extend(sol.y[0, start_index:])

        current_m = float(sol.y[0, -1])

    t = np.asarray(all_t)
    m = np.asarray(all_m)
    u = np.asarray(input_model(t), dtype=float)

    return t, m, u


def plot_simulation(t, m, u, title="mRNA dynamics"):
    """Plot input and mRNA abundance against time."""
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True
    )

    axes[0].plot(t, u, color="tab:blue")
    axes[0].set_ylabel("Input u(t)")
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(t, m, color="tab:orange")
    axes[1].set_ylabel("mRNA m(t)")
    axes[1].set_xlabel("Time")

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def compare_simulations(
    models, alpha=1.0, gamma=0.5,
    m0=0.0, t_span=(0.0, 20.0), n_points=1000,
):
    """Compare the input and mRNA trajectories for multiple models."""
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True
    )

    for model in models:
        t, m, u = simulate(
            model,
            alpha=alpha,
            gamma=gamma,
            m0=m0,
            t_span=t_span,
            n_points=n_points,
        )

        axes[0].plot(t, u, label=model.name)
        axes[1].plot(t, m, label=model.name)

    axes[0].set_ylabel("Input u(t)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("mRNA m(t)")
    axes[1].set_xlabel("Time")

    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

    return fig, axes
