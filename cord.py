import numpy as np


EPS_NUM = 1e-12
DELTA_STAB = 1e-10


class CORD:
    def fit(self, original: np.ndarray, calibrated: np.ndarray) -> "CORD":
        p0, q = _stabilize(original), _stabilize(calibrated)
        _, b, _, g, lower, upper = _quantities(p0, q)
        target = float(np.clip(b.mean(), lower.mean(), upper.mean()))
        self.eta_ = _solve_eta(g, lower, upper, target)
        return self

    def transform(self, original: np.ndarray, calibrated: np.ndarray) -> np.ndarray:
        p0, q = _stabilize(original), _stabilize(calibrated)
        a, _, tail, g, lower, upper = _quantities(p0, q)
        head = _response(self.eta_, g, lower, upper)
        repaired = (1.0 - head[:, None]) * tail
        repaired[np.arange(a.size), a] = head
        return repaired


def _stabilize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    boundary = np.any(p == 0.0, axis=1)

    if not np.any(boundary):
        return p

    stabilized = p.copy()
    stabilized[boundary] = (
        (1.0 - DELTA_STAB) * stabilized[boundary]
        + DELTA_STAB / p.shape[1]
    )
    return stabilized


def _quantities(p0: np.ndarray, q: np.ndarray):
    rows = np.arange(p0.shape[0])
    a = p0.argmax(axis=1)
    b = q[rows, a]
    tail = q.copy()
    tail[rows, a] = 0.0
    tail /= tail.sum(axis=1, keepdims=True)
    rho = tail.max(axis=1)
    lower = rho / (1.0 + rho) + EPS_NUM
    upper = np.full_like(lower, 1.0 - EPS_NUM)
    g = np.where(q.argmax(axis=1) == a, b, 0.5 * (b + p0[rows, a]))
    return a, b, tail, g, lower, upper


def _interior(eta: float, g: np.ndarray) -> np.ndarray:
    radius = np.hypot(
        eta + 2.0 * g - 1.0,
        2.0 * np.sqrt(g * (1.0 - g)),
    )

    if eta > 1.0:
        return (eta - 1.0 + radius) / (2.0 * eta)

    return 2.0 * g / (1.0 - eta + radius)


def _response(
    eta: float,
    g: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.clip(_interior(eta, g), lower, upper)


def _stationarity(s: np.ndarray, g: np.ndarray) -> np.ndarray:
    return (s - g) / (s * (1.0 - s))


def _solve_eta(
    g: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: float,
) -> float:
    mean_at_zero = float(_response(0.0, g, lower, upper).mean())
    if mean_at_zero == target:
        return 0.0

    if mean_at_zero < target:
        left, right = 0.0, float(_stationarity(upper, g).max())
        while True:
            midpoint = left + 0.5 * (right - left)
            if midpoint == left or midpoint == right:
                return right
            if _response(midpoint, g, lower, upper).mean() < target:
                left = midpoint
            else:
                right = midpoint

    left, right = float(_stationarity(lower, g).min()), 0.0
    while True:
        midpoint = left + 0.5 * (right - left)
        if midpoint == left or midpoint == right:
            return left
        if _response(midpoint, g, lower, upper).mean() <= target:
            left = midpoint
        else:
            right = midpoint
