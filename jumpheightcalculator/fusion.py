"""S3 - Fusion zum Kontaktsignal.

Die Signale beider Zweige werden auf [0,1] normiert und zu einem Kontaktsignal
c(n) zusammengefuehrt. Darueber laeuft ein Zwei-Zustands-HMM mit Viterbi statt
einer harten Schwelle: eine Schwelle flackert an verrauschten Frames und
zerlegt den Flug in Stuecke. Viterbi erzwingt eine global konsistente
Segmentierung.
"""

from __future__ import annotations

import numpy as np

FLIGHT, CONTACT = 0, 1


def normalise(x: np.ndarray, *, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """Robust auf [0,1]. Perzentile statt min/max, damit ein einzelner Ausreisser
    nicht die ganze Skala verschiebt."""
    x = np.asarray(x, float)
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def fuse(parts: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Gewichtetes Mittel der normierten Signale -> c(n) in [0,1].
    1 heisst Kontakt, 0 heisst Flug."""
    total = sum(weights.get(k, 0.0) for k in parts)
    if total <= 0:
        raise ValueError("Alle Fusionsgewichte sind null.")
    acc = np.zeros(len(next(iter(parts.values()))), float)
    for k, v in parts.items():
        acc += weights.get(k, 0.0) * normalise(v)
    return np.clip(acc / total, 0.0, 1.0)


def viterbi(c: np.ndarray, *, p_stay: float = 0.97, floor: float = 1e-4) -> np.ndarray:
    """Zwei-Zustands-HMM (Flug / Kontakt), Viterbi-Dekodierung.

    Emission direkt aus c(n), Uebergaenge mit hoher Selbsthaltewahrscheinlichkeit -
    das bestraft Zustandswechsel und verhindert das Flackern.
    """
    c = np.clip(np.asarray(c, float), floor, 1 - floor)
    n = c.size
    log_e = np.stack([np.log(1 - c), np.log(c)], axis=1)      # (n, 2)
    stay, switch = np.log(p_stay), np.log(1 - p_stay)

    dp = np.zeros((n, 2))
    back = np.zeros((n, 2), int)
    dp[0] = log_e[0]
    for i in range(1, n):
        for s in (FLIGHT, CONTACT):
            cand = (dp[i - 1, s] + stay, dp[i - 1, 1 - s] + switch)
            back[i, s] = s if cand[0] >= cand[1] else 1 - s
            dp[i, s] = max(cand) + log_e[i, s]
    path = np.zeros(n, int)
    path[-1] = int(np.argmax(dp[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def flight_window(states: np.ndarray) -> tuple[int, int]:
    """Laengster zusammenhaengender Flugblock als [n0, n1)."""
    best, i = (0, 0), 0
    while i < states.size:
        if states[i] == FLIGHT:
            j = i
            while j < states.size and states[j] == FLIGHT:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def crossing_time(t: np.ndarray, c: np.ndarray, n: int, *, level: float = 0.5) -> float:
    """Zeitpunkt, zu dem c(n) am Uebergang bei `n` die Schwelle schneidet.

    Linear zwischen den beiden Frames interpoliert - der Uebergang liegt fast nie
    genau auf einem Frame (R4).
    """
    if n <= 0 or n >= c.size:
        return float(t[min(max(n, 0), t.size - 1)])
    a, b = c[n - 1], c[n]
    if abs(b - a) < 1e-9:
        return float(t[n])
    f = np.clip((level - a) / (b - a), 0.0, 1.0)
    return float(t[n - 1] + f * (t[n] - t[n - 1]))
