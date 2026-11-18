"""Independent finite-lifetime merge model; no GPU or timing claims."""

import numpy as np


def scores(pool, queue, incoming, horizon, alpha):
    k = len(pool)
    count = len(queue) + 1
    # Future unknown images affect normalization but cancel from merge differences.
    ages = np.arange(len(queue), 0, -1)
    images = list(pool) + [incoming]
    result = []
    for a in range(k + 1):
        exposure = 0.0
        for future in range(horizon):
            normalizer = sum(alpha**age for age in range(min(count + future, horizon)))
            mass = (
                alpha**future
                if a == k
                else sum(
                    alpha ** (int(age) + future)
                    for age, version in zip(ages, queue)
                    if version == a and age + future < horizon
                )
            )
            exposure += (mass / normalizer) ** 2
        for b in range(k + 1):
            if a != b:
                result.append(
                    (
                        exposure
                        * np.mean((images[a][..., :3] - images[b][..., :3]) ** 2),
                        a,
                        b,
                    )
                )
    return result


def step(pool, queue, incoming, horizon, alpha, decision=None):
    previous = queue[-1] if queue else None
    if len(queue) == horizon:
        queue.pop(0)
    match = next(
        (
            v
            for v in ([previous] if previous is not None else []) + sorted(set(queue))
            if np.array_equal(pool[v], incoming)
        ),
        None,
    )
    candidates = []
    if match is not None:
        chosen = match
    else:
        free = next((v for v in range(len(pool)) if v not in queue), None)
        if free is not None:
            chosen = free
            pool[chosen] = incoming.copy()
        else:
            candidates = scores(pool, queue, incoming, horizon, alpha)
            _, a, b = (
                min(candidates, key=lambda v: v[0])
                if decision is None
                else next(c for c in candidates if c[1:] == decision)
            )
            if a == len(pool):
                chosen = b
            else:
                if b != len(pool):
                    queue[:] = [b if v == a else v for v in queue]
                chosen = a
                pool[a] = incoming.copy()
    queue.append(chosen)
    w = alpha ** np.arange(len(queue) - 1, -1, -1, dtype=float)
    w /= sum(w)
    return sum(weight * pool[v] for weight, v in zip(w, queue)), candidates
