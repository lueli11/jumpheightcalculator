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
RUN test -f jumpheightcalculator/models/pose_landmarker_full.task

ENV JUMPHEIGHTCALCULATOR_DATA=/data
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "jumpheightcalculator.api:app", "--host", "0.0.0.0", "--port", "8000"]
