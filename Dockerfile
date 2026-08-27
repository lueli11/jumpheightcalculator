FROM python:3.12-slim

# ffmpeg/ffprobe sind Pflicht, nicht optional: die Zeitachse kommt aus den
# Container-Zeitstempeln (R1), und decodiert wird ueber die ffmpeg-Rohpipe.
#
# libegl1 und libgles2 braucht MediaPipe auch headless. Ohne sie stirbt der
# Worker beim Import mit "libEGL.so.1: cannot open shared object file" - das
# Image baut sauber durch, und erst die erste echte Analyse faellt um.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 libglib2.0-0 \
        libegl1 libgles2 libgomp1 \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete

WORKDIR /app
COPY pyproject.toml ./
COPY jumpheightcalculator ./jumpheightcalculator
# --root-user-action/--disable-pip-version-check: in einem Container ist root
# der Normalfall, beide Hinweise sind im Deploy-Log nur Rauschen.
RUN pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check .

# MediaPipe schon beim Bauen importieren, nicht erst zur Laufzeit: fehlt eine
# Systembibliothek, soll der Build scheitern und nicht der erste Job des Nutzers.
RUN python -c "import mediapipe, cv2; print('mediapipe', mediapipe.__version__, 'cv2', cv2.__version__)"

# Das Pose-Modell liegt im Image, nicht im Volume: ein Download zur Laufzeit
# machte den Start vom Netz abhaengig und die Ergebnisse von der Tagesform des
# Modell-Servers.
#
# Geprueft wird die **installierte** Stelle, nicht der Quellbaum. Die erste
# Fassung testete /app/jumpheightcalculator/models/... - das existiert immer,
# weil es gerade hineinkopiert wurde, und sagt nichts darueber, ob pip die Datei
# nach site-packages mitgenommen hat. Genau dort fehlte sie dann zur Laufzeit.
RUN python -c "import pathlib, jumpheightcalculator as j;     p = pathlib.Path(j.__file__).parent / 'models' / 'pose_landmarker_full.task';     assert p.exists(), f'Pose-Modell fehlt im Image: {p}';     print(f'Pose-Modell installiert: {p} ({p.stat().st_size/1e6:.1f} MB)')"

ENV JUMPHEIGHTCALCULATOR_DATA=/data
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "jumpheightcalculator.api:app", "--host", "0.0.0.0", "--port", "8000"]
