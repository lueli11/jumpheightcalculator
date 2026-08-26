"""S4/S5 - Parabel-Fit, Selbstkalibrierung, Höhenschaetzer, Güte.

Alle Bildkoordinaten sind Pixel mit **y nach unten**. Ein steigender Körper hat
also fallendes y, und die Schwerkraft zeigt in +y: der quadratische Koeffizient
ist positiv.

R4: Der Absprung liegt immer zwischen zwei Frames. Frames zu zaehlen quantisiert
die Flugzeit; weil h quadratisch von t abhaengt, verdoppelt sich der relative
Fehler auf dem Weg zur Höhe. Darum ein Fit über alle sauberen Flugframes und
Ereigniszeitpunkte als Fliesskommazahlen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import G


class FitError(RuntimeError):
    """R7: konkrete Diagnose statt geratener Zahl."""


@dataclass(frozen=True)
class ParabolaFit:
    """y(t) = c0 + c1*(t-t_ref) + c2*(t-t_ref)^2, y in Pixeln nach unten."""
    c0: float
    c1: float
    c2: float
    t_ref: float
    t: np.ndarray            # alle angebotenen Zeiten
    y: np.ndarray            # alle angebotenen Höhen
    inliers: np.ndarray      # bool-Maske der verwendeten Punkte
    r2: float
    rms_resid: float         # [px]
    cubic_t: float           # t-Statistik des kubischen Terms -> B3

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())

    @property
    def a_px(self) -> float:
        """Krümmung [px/s^2], positiv nach unten."""
        return 2.0 * self.c2

    @property
    def scale(self) -> float:
        """R5 - Selbstkalibrierung: der Massstab kommt aus der Physik, nicht vom
        Nutzer. Die Krümmung der Flugparabel *muss* 9.81 m/s^2 entsprechen,
        also ist s = g / a_px [m/px]. Kein Massband, kein Referenzobjekt."""
        if self.a_px <= 0:
            raise FitError(
                "Die gefittete Bahn kruemmt sich nach oben statt nach unten. "
                "Das ist keine Wurfparabel - vermutlich wurde die Flugphase falsch "
                "abgegrenzt oder die Person ist nicht durchgehend erkannt."
            )
        return G / self.a_px

    def y_at(self, t: float | np.ndarray) -> np.ndarray:
        u = np.asarray(t, float) - self.t_ref
        return self.c0 + self.c1 * u + self.c2 * u * u

    def v_up_px(self, t: float | np.ndarray) -> np.ndarray:
        """Aufwaerts-Geschwindigkeit [px/s]; y zeigt nach unten, daher das Minus."""
        u = np.asarray(t, float) - self.t_ref
        return -(self.c1 + 2.0 * self.c2 * u)

    @property
    def t_apex(self) -> float:
        return float(self.t_ref - self.c1 / (2.0 * self.c2))

    @property
    def y_apex(self) -> float:
        return float(self.y_at(self.t_apex))

    def roots_at(self, y_level: float) -> tuple[float, float]:
        """Die beiden Zeitpunkte, zu denen die Bahn die Höhe `y_level` schneidet.

        Das sind die Ereigniszeitpunkte als Fliesskommazahlen (R4) - nicht der
        nächstgelegene Frame.
        """
        disc = self.c1 * self.c1 - 4.0 * self.c2 * (self.c0 - y_level)
        if disc <= 0:
            raise FitError(
                "Die gefittete Bahn erreicht das Absprungniveau nie - der Fit oder "
                "die Bodenlinie stimmt nicht."
            )
        r = np.sqrt(disc)
        u1 = (-self.c1 - r) / (2.0 * self.c2)
        u2 = (-self.c1 + r) / (2.0 * self.c2)
        return float(self.t_ref + min(u1, u2)), float(self.t_ref + max(u1, u2))


def fit_parabola(t: np.ndarray, y: np.ndarray, *, seed: int = 0,
                 min_points: int = 8) -> ParabolaFit:
    """RANSAC-Parabel mit anschliessendem Least-Squares auf den Inliern.

    Der RNG-Seed ist festgesetzt: ohne ihn wäre dasselbe Video nicht bitgleich
    reproduzierbar, und Gate P2.1 (API-Ergebnis identisch zur CLI) nicht erreichbar.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size != y.size:
        raise FitError("t und y haben verschiedene Längen.")
    if t.size < min_points:
        raise FitError(
            f"Nur {t.size} verwertbare Punkte erfasst (mindestens {min_points} noetig). "
            f"Bitte in Slo-Mo mit mindestens 120 fps aufnehmen."
        )

    # Zeitnullpunkt in die Mitte legen - sonst ist die Normalengleichung bei
    # grossen absoluten Zeitstempeln schlecht konditioniert.
    t_ref = float(np.mean(t))
    u = t - t_ref

    coef0 = np.polyfit(u, y, 2)
    sigma = 1.4826 * np.median(np.abs(y - np.polyval(coef0, u)))
    thresh = max(1.0, 2.5 * sigma)

    rng = np.random.default_rng(seed)
    best_mask, best_n = None, -1
    idx = np.arange(t.size)
    for _ in range(200):
        pick = rng.choice(idx, size=3, replace=False)
        if np.ptp(u[pick]) < 1e-9:
            continue
        try:
            c = np.polyfit(u[pick], y[pick], 2)
        except np.linalg.LinAlgError:
            continue
        mask = np.abs(y - np.polyval(c, u)) < thresh
        if mask.sum() > best_n:
            best_mask, best_n = mask, int(mask.sum())

    mask = best_mask if best_mask is not None and best_n >= min_points else np.ones_like(y, bool)

    # Nachziehen: zweimal auf den Inliern refitten und die Maske aktualisieren.
    for _ in range(2):
        if mask.sum() < min_points:
            break
        c = np.polyfit(u[mask], y[mask], 2)
        resid = y - np.polyval(c, u)
        s = 1.4826 * np.median(np.abs(resid[mask]))
        mask = np.abs(resid) < max(1.0, 3.0 * s)

    if mask.sum() < min_points:
        raise FitError(
            f"Nur {int(mask.sum())} Punkte passen auf eine gemeinsame Parabel "
            f"(mindestens {min_points} noetig). Die Bahn ist zu verrauscht oder "
            f"die Flugphase wurde falsch abgegrenzt."
        )

    c2, c1, c0 = np.polyfit(u[mask], y[mask], 2)
    resid = y[mask] - np.polyval([c2, c1, c0], u[mask])
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rms = float(np.sqrt(ss_res / resid.size))

    return ParabolaFit(
        c0=float(c0), c1=float(c1), c2=float(c2), t_ref=t_ref,
        t=t, y=y, inliers=mask, r2=r2, rms_resid=rms,
        cubic_t=_cubic_t_stat(u[mask], y[mask]),
    )


def _cubic_t_stat(u: np.ndarray, y: np.ndarray) -> float:
    """B3 - hat die Bahn eine kubische Komponente, die nicht vom Rauschen kommt?

    Eine echte Wurfparabel hat keine. Eine radiale Objektivverzeichnung macht aus
    ihr eine leicht verbogene Kurve - genau das passiert, wenn die Kamera tief
    steht und die Hüfte weit ausserhalb der Bildmitte sitzt. Das verfaelscht die
    Krümmung und über R5 direkt den Massstab, ohne dass R^2 nennenswert fällt.
    """
    if u.size < 6:
        return 0.0
    X = np.vstack([u ** 3, u ** 2, u, np.ones_like(u)]).T
    coef, res, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < 4:
        return 0.0
    dof = u.size - 4
    if dof <= 0:
        return 0.0
    resid = y - X @ coef
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0
    se = float(np.sqrt(max(cov[0, 0], 1e-300)))
    if se <= 0:
        return 0.0
    t = abs(coef[0]) / se

    # Korrektur für autokorrelierte Residuen.
    #
    # Eine gewoehnliche t-Statistik setzt unabhaengige Residuen voraus. Bei einer
    # Trajektorie sind sie das nie: benachbarte Frames zeigen dieselbe Szene, und
    # ein Pose-Modell traegt seinen Fehler über Frames hinweg mit. Gemessen liegt
    # die Autokorrelation bei Lag 1 auf echtem Material bei +0.75 - es gibt also
    # rund siebenmal weniger unabhaengige Punkte als Frames, und ein naiver Test
    # schlaegt an, wo nichts ist (gemessen t = 4.5 auf einer sauberen Bahn).
    #
    # n_eff = n * (1 - rho) / (1 + rho), und t skaliert mit sqrt(n_eff / n).
    r = resid - resid.mean()
    denom = float(r @ r)
    rho = float(r[:-1] @ r[1:] / denom) if denom > 0 else 0.0
    rho = min(max(rho, 0.0), 0.98)
    n_eff = max(4.0, u.size * (1.0 - rho) / (1.0 + rho))
    return float(t * np.sqrt(n_eff / u.size))


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Laengster zusammenhaengender True-Block als [start, ende) - das Flugfenster."""
    best = (0, 0)
    i = 0
    while i < mask.size:
        if mask[i]:
            j = i
            while j < mask.size and mask[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best
