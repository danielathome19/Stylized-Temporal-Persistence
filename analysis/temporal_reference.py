"""Independent float64 temporal oracle; no real-time approximations or GPU timings."""

import math
import numpy as np
from fractions import Fraction


def retention(a_ref, dt, interval):
    if (
        not 0 <= a_ref < 1
        or not math.isfinite(dt)
        or dt < 0
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("Invalid retention/time")
    return 1.0 if dt == 0 else a_ref ** (dt / interval)


def target_ticks(start, end, rate, phase=Fraction(0)):
    """Exact targets in (start,end]; caller seeds at phase separately."""
    if rate <= 0 or end < start:
        raise ValueError("Invalid schedule")
    first = math.floor((start - phase) * rate) + 1
    last = math.floor((end - phase) * rate)
    return [phase + Fraction(k, rate) for k in range(max(0, first), last + 1)]


def recursive_oracle(images, retentions):
    """Direct expanded weighted sum, independent of a ping-pong recurrence."""
    images = np.asarray(images, dtype=np.float64)
    a = np.asarray(retentions, dtype=np.float64)
    if (
        len(images) == 0
        or len(a) != len(images) - 1
        or np.any(~np.isfinite(a))
        or np.any((a < 0) | (a > 1))
    ):
        raise ValueError("Invalid images/retentions")
    weights = [float(np.prod(a))]
    weights.extend((1 - a[j]) * float(np.prod(a[j + 1 :])) for j in range(len(a)))
    return np.tensordot(weights, images, axes=(0, 0))


def kernel_weights(ages, kind, support, scale):
    ages = np.asarray(ages, dtype=np.float64)
    if (
        not math.isfinite(support)
        or not math.isfinite(scale)
        or support <= 0
        or scale <= 0
        or np.any(~np.isfinite(ages))
        or np.any(ages < 0)
    ):
        raise ValueError("Invalid kernel parameters")
    if kind == "exponential":
        values = np.exp(-ages / scale)
    elif kind == "linear":
        values = np.maximum(0, 1 - ages / support)
    elif kind == "gaussian":
        values = np.exp(-0.5 * (ages / scale) ** 2)
    elif kind == "hyperbolic":
        values = 1 / (1 + ages / scale)
    else:
        raise ValueError("Unknown kernel")
    values[ages > support] = 0
    return values / values.sum() if values.sum() else values


def live_composite(
    current,
    history,
    timestamps,
    now,
    current_weight,
    kind="exponential",
    support=2.0,
    scale=0.25 / math.log(2),
):
    if not 0 <= current_weight <= 1 or len(history) != len(timestamps):
        raise ValueError("Invalid history/current weight")
    eligible = [i for i, t in enumerate(timestamps) if t < now and now - t <= support]
    current = np.asarray(current, dtype=np.float64)
    if not eligible:
        return current.copy()
    weights = kernel_weights(
        [float(now - timestamps[i]) for i in eligible], kind, support, scale
    )
    if not weights.sum():
        return current.copy()
    past = np.tensordot(
        weights,
        np.asarray([history[i] for i in eligible], dtype=np.float64),
        axes=(0, 0),
    )
    return current_weight * current + (1 - current_weight) * past
