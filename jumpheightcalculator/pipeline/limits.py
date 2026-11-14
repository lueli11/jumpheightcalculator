"""Alle Schwellen der Pipeline - jede mit ihrer gemessenen Begründung.

Die Zahlen stehen bewusst an **einem** Ort, nicht verteilt über Module: wer
eine Schwelle anfasst, soll die Befunde daneben sehen, auf denen sie beruhen.
"""

MIN_FLIGHT_S, MAX_FLIGHT_S = 0.15, 1.10
MIN_FLIGHT_FRAMES = 10
MIN_R2 = 0.98
# Schulter bis Knöchel, aus dem Massstab der Selbstkalibrierung zurückgerechnet.
# Weit gefasst - das ist eine Absurditätsprüfung gegen eine gestreckte
# Zeitachse, keine Grössenmessung.
BODY_LEN_BOUNDS_M = (0.5, 3.0)

# Das Gütemass ist |E2-E3|, nicht die Spannweite über alle vier Schätzer.
# E1 und E4 weichen von E2 um Delta ab - die Landeasymmetrie. Die ist gemessene
# Physiologie, kein Messfehler, und gehört darum nicht in ein Fehlermass.
# Nur E2 und E3 messen dieselbe Grösse auf verschiedenen Landmarks.
MAX_DISAGREE_M = 0.04
MAX_CUBIC_T = 4.0
MAX_SHIFT_RESID_PX = 0.5

# Perspektive: um wie viel darf die Person im Bild während des Flugs grösser
# oder kleiner werden? Gemessen an einem sauberen Video (Kamera seitlich, Sprung
# senkrecht zur Achse) bleiben die starren Segmente über den ganzen Flug ohne
# Trend, bei rund 3 % Streuung. Ein Video, auf dem die Person auf die Kamera zu
# springt, kam auf +33 % - ein Oberschenkelknochen wächst nicht, das ist reine
# Tiefenbewegung. Damit ändert sich der Massstab mitten im Sprung, die Krümmung
# der Parabel wird bedeutungslos und die Selbstkalibrierung (R5) ungültig.
MAX_SCALE_DRIFT = 0.12
# Eine echte Entfernungsänderung skaliert alle starren Segmente gleich. Streuen
# die vier Werte weiter auseinander, ist die Ursache Haltung oder Pose-Rauschen,
# und eine Verweigerung wäre ein Fehlalarm. Gemessen: saubere Referenz 5 Punkte,
# IMG_0366 (schräge Kamera, Hocke) 15 Punkte.
MAX_SCALE_SPREAD = 0.10

# Kontaktsignal (Fuss) und Körperbewegung (Knöchelhub) schätzen das Flugfenster
# unabhängig voneinander. Nach R3 ist die Knöchelversion systematisch zu lang -
# der Knöchel steigt beim Abdruck schon, während die Zehe noch aufliegt - aber
# sie müssen in derselben Grössenordnung liegen. Auf einem sauberen Video sind
# es 93 gegen 95 Frames. Bei einem Sprung mit Anlauf mass das Kontaktsignal 137
# gegen 76: der Streifen war leer, weil die Person weggelaufen war, nicht weil
# sie flog. Wo die beiden so weit auseinanderliegen, ist das Timing nicht zu
# gebrauchen - und eine Höhe daraus wäre geraten.
CONTACT_VS_BODY_RANGE = (0.6, 1.45)

# Ab wann ist die Person zu klein im Bild? Gemessen (Lochkamera, 2 px
# Keypoint-Rauschen, 120 fps) bleibt die Streuung noch bei 175 px Körpergrösse
# unter 0.4 cm - der Fit mittelt über rund hundert Frames. Weiter unten wird
# aber die Pose-Erkennung selbst unzuverlässig, und das fällt nicht mehr in die
# Zuständigkeit der Geometrie. Darum eine Warnung, keine Verweigerung.
MIN_BODY_PX = 140
