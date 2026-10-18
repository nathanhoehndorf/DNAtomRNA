from dataclasses import dataclass, field
from typing import Callable

import numpy as np

@dataclass
class InputModel:
  """A transcription input $u(t;\theta)$, constrained to [0,1]."""

name: str
u: Callable
params: dict = field(default_factory=dict)

kind: str = "general"
limit: float | None = None
period: float | None = None
breakpoints: tuple = ()

def __call__(self, t):
  values = np.asarray(self.u(t, self.params), dtype=float)
  if not np.all(np.isfinite(values)):
    raise ValueError(f"{self.name}: u returned non-finite values.")

  tol = 1e-10
  if np.any(values < -tol) or np.any(values > 1+tol):
    raise ValueError(f"{self.name}: u(t) must lie in [0,1].")

  return np.clip(values, 0.0, 1.0)

def constant(c=1.0):
  return InputModel(
    name = "constant",
    u=lambda t, p: np.zeros_like(np.asarray(t, dtype=float)) + p["c"],
    params={"c": float(c)},
    kind="constant",
    limit=float(c),
  )

def heaviside(t0=1.0, level=1.0):
  return InputModel(
    name="heaviside",
    u=lambda t, p: p["level"] * (
      np.asarray(t) >= p["t0"]
    ),
    params={"t0": float(t0), "level": float(level)},
    kind="step",
    limit=float(level),
    breakpoints=(float(t0),),
  )

def exponential_ramp(k=1.0):
  if k <= 0:
    raise ValueError("k must be postive.")

  return InputModel(
    name="exponential_ramp",
    u=lambda t, p: 1.0 - np.exp(-p["k"] * np.asarray(t)),
    params={"k": float(k)},
    kind="asymptotic",
    limit=1.0,
  )

def sinusoidal(offset=0.5, amplitude=0.3, omega=1.0, phase=0.0):
  if omega <= 0:
    raise ValueError("omega must be positive")

  if amplitude < 0 or offset - amplitude < 0 or offset+amplitude > 1:
    raise ValueError("Requires function to be bounded between 0 and 1").

  return InputModel(
    name="sinusoidal",
    u=lambda t,p: (p["offset"]+p["amplitude"]*np.sin(p["omega"]*np.asarray(t)+p["phase"])),
    params={"offset": float(offset), "amplitude": float(amplitude), "omega": float(omega), "phase": float(phase)},
    kind="periodic",
    limit=None,
    period=2*np.pi/omega
  )

def sinusoid_sum(offset, components):
    """
    Build u(t) = offset + sum(A_i * sin(omega_i*t + phase_i)).

    components is an iterable of (amplitude, omega, phase) tuples.
    All parameters are fixed when this function is called.
    """
    components = tuple(
        (float(a), float(w), float(phi))
        for a, w, phi in components
    )

    if any(a < 0 or w <= 0 for a, w, _ in components):
        raise ValueError("Amplitudes must be >= 0 and frequencies > 0.")

    # A conservative sufficient condition for u(t) to stay in [0, 1].
    total_amplitude = sum(a for a, _, _ in components)
    if offset - total_amplitude < 0 or offset + total_amplitude > 1:
        raise ValueError(
            "Require 0 <= offset - sum(amplitudes) and "
            "offset + sum(amplitudes) <= 1."
        )

    def evaluate(t, p):
        t = np.asarray(t, dtype=float)
        result = np.zeros_like(t) + p["offset"]

        for amplitude, omega, phase in p["components"]:
            result += amplitude * np.sin(omega * t + phase)

        return result

    # The sum is periodic only when its frequencies are commensurate.
    # Leave period unspecified unless the user establishes one.
    return InputModel(
        name="sinusoid_sum",
        u=evaluate,
        params={"offset": float(offset), "components": components},
        kind="periodic_or_quasiperiodic",
    )
