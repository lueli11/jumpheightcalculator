FROM python:3.12-slim

# ffmpeg/ffprobe sind Pflicht, nicht optional: die Zeitachse kommt aus den
# Container-Zeitstempeln (R1), und decodiert wird ueber die ffmpeg-Rohpipe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete

WORKDIR /app
COPY pyproject.toml ./
COPY jumpheightcalculator ./jumpheightcalculator
RUN pip install --no-cache-dir .

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
