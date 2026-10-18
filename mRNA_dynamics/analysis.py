import numpy as np
import pandas as pd


def analyze_input(input_model, alpha=1.0, gamma=0.5):
    """
    Return the analytical properties known from the input type.

    A limiting steady state is distinguished from an actual
    equilibrium of the original time-dependent equation.
    """
    if alpha <= 0 or gamma <= 0:
        raise ValueError("alpha and gamma must be positive.")

    result = {
        "input_name": input_model.name,
        "alpha": alpha,
        "gamma": gamma,
        "perturbation_eigenvalue": -gamma,
        "asymptotically_stable_for_fixed_input": True,
        "equilibrium": None,
        "limiting_steady_state": None,
        "period": input_model.period,
        "interpretation": "",
    }

    if input_model.kind == "constant":
        equilibrium = alpha * input_model.params["c"] / gamma
        result["equilibrium"] = equilibrium
        result["limiting_steady_state"] = equilibrium
        result["interpretation"] = (
            "Constant-input equilibrium; asymptotically stable."
        )

    elif input_model.limit is not None:
        result["limiting_steady_state"] = (
            alpha * input_model.limit / gamma
        )
        result["interpretation"] = (
            "The input approaches a constant, so m(t) approaches "
            "the corresponding limiting steady state. This need "
            "not be an equilibrium at finite times."
        )

    elif input_model.kind == "periodic":
        result["interpretation"] = (
            "Unique attracting periodic response; no constant "
            "equilibrium is implied."
        )

    else:
        result["interpretation"] = (
            "Transient response is exponentially stable with "
            "respect to initial conditions. Further long-term "
            "classification depends on the input."
        )

    return result


def parameter_sweep(
    make_model,
    parameter_name,
    values,
    alpha=1.0,
    gamma=0.5,
    m0=0.0,
    t_span=(0.0, 20.0),
    n_points=1000,
):
    """
    Simulate a family of inputs created by make_model(**kwargs).

    Returns a list of dictionaries, one per parameter value.
    """
    from model import simulate

    results = []

    for value in values:
        input_model = make_model(**{parameter_name: float(value)})

        t, m, u = simulate(
            input_model,
            alpha=alpha,
            gamma=gamma,
            m0=m0,
            t_span=t_span,
            n_points=n_points,
        )

        results.append({
            "parameter": float(value),
            "t": t,
            "m": m,
            "u": u,
            "final_m": float(m[-1]),
            "max_m": float(np.max(m)),
            "time_of_max_m": float(t[np.argmax(m)]),
            "integrated_m": float(np.trapezoid(m, t)),
        })

    return results


def sweep_summary(results):
    """Convert sweep results into a compact pandas DataFrame."""
    return pd.DataFrame([
        {
            key: result[key]
            for key in (
                "parameter",
                "final_m",
                "max_m",
                "time_of_max_m",
                "integrated_m",
            )
        }
        for result in results
    ])


def plot_sweep(results, quantity="m"):
    """Plot trajectories from a parameter sweep."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for result in results:
        ax.plot(
            result["t"],
            result[quantity],
            label=f"parameter={result['parameter']:.3g}",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(quantity)
    ax.legend()
    fig.tight_layout()

    return fig, ax
