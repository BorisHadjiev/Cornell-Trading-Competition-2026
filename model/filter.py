"""Class probabilities and a flip hazard.

Advisories and the track are features. The label is the age-dependent manifest
class, used only while fitting and while scoring log loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.config import CLASSES
from model.labels import label_at

ISLANDS = ("cascadia_prime", "coral_key", "azure_isle", "palm_atoll")
N_FEATURES = 31


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def first_turn_age(obs: np.ndarray, degrees: float) -> float | None:
    """First age at which velocity has turned `degrees` from the initial heading."""
    if len(obs) == 0:
        return None
    base = None
    for row in obs:
        vel = row[3:5]
        if np.linalg.norm(vel) >= 0.2:
            base = vel
            break
    if base is None:
        return None
    for row in obs:
        vel = row[3:5]
        if np.linalg.norm(vel) < 0.2:
            continue
        if _angle_deg(vel, base) >= degrees:
            return float(row[0])
    return None


def _latest(events: list[dict], source: str, t) -> dict | None:
    found = None
    for event in events:
        if event["time"] > t:
            break
        if event["type"] == "advisory" and event.get("source") == source:
            found = event
    return found


def features_from(obs_row: np.ndarray, events: list[dict], t, storm_seconds: float) -> np.ndarray:
    age, x, y, vx, vy, radius, intensity = [float(v) for v in obs_row[:7]]
    official = _latest(events, "official", t)
    wire = _latest(events, "wire", t)
    vec = np.zeros(N_FEATURES, dtype=float)
    if official is not None:
        outlook = official.get("outlook")
        if outlook in CLASSES:
            vec[CLASSES.index(outlook)] = 1.0
        probs = official.get("hurricane_conditions") or {}
        for i, island in enumerate(ISLANDS):
            vec[6 + i] = float(probs.get(island, 0.0))
        vec[17] = 1.0
        cone = official.get("cone") or []
        nxt = None
        for point in cone:
            if float(point["age"]) >= age:
                nxt = point
                break
        if nxt is None and cone:
            nxt = cone[-1]
        if nxt is not None:
            vec[23] = (float(nxt["x"]) - x) / 1000.0
            vec[24] = (float(nxt["y"]) - y) / 700.0
            vec[25] = float(nxt.get("spread", 0.0)) / 100.0
            vec[26] = 1.0
    if wire is not None:
        outlook = wire.get("outlook")
        if outlook in CLASSES:
            vec[10 + CLASSES.index(outlook)] = 1.0
        vec[16] = 1.0
    vec[18] = x / 1000.0
    vec[19] = y / 700.0
    vec[20] = vx
    vec[21] = vy
    vec[22] = radius / 100.0
    vec[27] = intensity
    vec[28] = age / storm_seconds
    speed = np.hypot(vx, vy)
    if speed > 1e-6:
        vec[29] = vx / speed
        vec[30] = vy / speed
    return vec


def iter_examples(storm, events: list[dict], storm_seconds: float, step: int = 10):
    """Yield (features, class_index, hazard, age) every `step` seconds of age."""
    labels = [label_at(storm.pattern, storm.flip, float(age)) for age in storm.obs[:, 0]]
    for row, when, label in zip(storm.obs, storm.times, labels):
        age = float(row[0])
        if int(round(age)) % step != 0:
            continue
        hazard = 1.0 if storm.flip is not None and age < storm.flip else 0.0
        yield features_from(row, events, when, storm_seconds), CLASSES.index(label), hazard, age


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class FilterFit:
    mean: np.ndarray
    std: np.ndarray
    weight: np.ndarray  # (F, C)
    bias: np.ndarray  # (C,)
    hazard_weight: np.ndarray  # (F,)
    hazard_bias: float
    prior: np.ndarray
    storm_seconds: float

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 1
        if single:
            x = x[None, :]
        prob = _softmax(self._standardize(x) @ self.weight + self.bias)
        return prob[0] if single else prob

    def hazard(self, x: np.ndarray) -> float:
        z = float(self._standardize(x) @ self.hazard_weight + self.hazard_bias)
        return float(1.0 / (1.0 + np.exp(-z)))

    def as_dict(self) -> dict:
        return {
            "estimated": True,
            "prior": {name: float(self.prior[i]) for i, name in enumerate(CLASSES)},
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
            "hazard_weight": self.hazard_weight.tolist(),
            "hazard_bias": self.hazard_bias,
            "feature_mean": self.mean.tolist(),
            "feature_std": self.std.tolist(),
        }


def _fit_softmax(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, f = x.shape
    c = len(CLASSES)
    weight = rng.normal(0, 0.01, size=(f, c))
    bias = np.zeros(c)
    lr = 0.15
    reg = 1e-3
    batch = 2048
    y_oh = np.eye(c)[y]
    for _ in range(50):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            take = order[start:start + batch]
            xb = x[take]
            prob = _softmax(xb @ weight + bias)
            grad = (prob - y_oh[take]) / len(take)
            weight -= lr * (xb.T @ grad + reg * weight)
            bias -= lr * grad.sum(axis=0)
    return weight, bias


def _fit_hazard(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    weight = np.zeros(x.shape[1])
    bias = 0.0
    lr = 0.2
    for _ in range(40):
        z = x @ weight + bias
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = (prob - y) / len(y)
        weight -= lr * (x.T @ grad + 1e-3 * weight)
        bias -= lr * float(grad.sum())
    return weight, float(bias)


def fit_filter(seasons: list, storm_seconds: float, rng: np.random.Generator) -> FilterFit:
    rows = []
    labels = []
    hazards = []
    for season in seasons:
        for storm in season.storms:
            for feat, label, hazard, _age in iter_examples(storm, season.events, storm_seconds):
                rows.append(feat)
                labels.append(label)
                hazards.append(hazard)
    x = np.vstack(rows)
    y = np.asarray(labels, dtype=int)
    h = np.asarray(hazards, dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xs = (x - mean) / std
    weight, bias = _fit_softmax(xs, y, rng)
    hazard_w, hazard_b = _fit_hazard(xs, h)
    prior = np.bincount(y, minlength=len(CLASSES)).astype(float)
    prior /= prior.sum()
    return FilterFit(mean, std, weight, bias, hazard_w, hazard_b, prior, storm_seconds)


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def frequency_log_loss(train_labels: np.ndarray, test_labels: np.ndarray) -> float:
    freq = np.bincount(train_labels, minlength=len(CLASSES)).astype(float)
    freq = np.clip(freq / max(freq.sum(), 1.0), 1e-12, 1.0)
    return float(-np.log(freq[test_labels]).mean()) if len(test_labels) else float("nan")


def bucket_masks(ages: np.ndarray, turn_ages: np.ndarray) -> dict[str, np.ndarray]:
    after = np.array([turn is not None and age >= turn for age, turn in zip(ages, turn_ages)])
    return {
        "0-30s": ages < 30,
        "30-90s": (ages >= 30) & (ages < 90),
        "after_first_turn": after,
    }


def collect_scored(seasons: list, model: FilterFit, turn_degrees: float, step: int = 10):
    feats = []
    labels = []
    ages = []
    turns = []
    for season in seasons:
        for storm in season.storms:
            turn = first_turn_age(storm.obs, turn_degrees)
            for feat, label, _hazard, age in iter_examples(storm, season.events, model.storm_seconds, step):
                feats.append(feat)
                labels.append(label)
                ages.append(age)
                turns.append(turn if turn is not None else np.nan)
    return np.vstack(feats), np.asarray(labels, dtype=int), np.asarray(ages), np.asarray(turns, dtype=float)
