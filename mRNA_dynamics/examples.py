import numpy as np

from .inputs import (
    constant,
    heaviside,
    exponential_ramp,
    sinusoidal,
    sinusoid_sum,
)
from .model import compare_simulations, plot_sweep
from .analysis import (
    analyze_input,
    parameter_sweep,
    sweep_summary,
)


def main():
    alpha = 2.0
    gamma = 0.5
    m0 = 0.0

    # Define the inputs to investigate.
    models = [
        constant(c=0.6),
        heaviside(t0=2.0, level=0.8),
        exponential_ramp(k=0.7),
        sinusoidal(
            offset=0.5,
            amplitude=0.3,
            omega=2.0,
        ),
        sinusoid_sum(
            offset=0.5,
            components=[
                (0.2, 1.0, 0.0),
                (0.1, 2.0, 0.5),
            ],
        ),
    ]

    # Print analytical information for each input.
    for model in models:
        print(f"\n--- {model.name} ---")
        for key, value in analyze_input(
            model, alpha=alpha, gamma=gamma
        ).items():
            print(f"{key}: {value}")

    # Compare trajectories.
    compare_simulations(
        models,
        alpha=alpha,
        gamma=gamma,
        m0=m0,
        t_span=(0.0, 30.0),
    )

    # Explore how the exponential-ramp rate affects the response.
    k_values = np.linspace(0.1, 3.0, 25)

    results = parameter_sweep(
        exponential_ramp,
        parameter_name="k",
        values=k_values,
        alpha=alpha,
        gamma=gamma,
        m0=m0,
        t_span=(0.0, 30.0),
    )

    print("\n--- Exponential-ramp parameter sweep ---")
    print(sweep_summary(results).to_string(index=False))

    plot_sweep(results, quantity="m")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
