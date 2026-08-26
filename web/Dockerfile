FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY . .
# NEXT_PUBLIC_* backt Next zur **Bauzeit** ein, nicht zur Laufzeit. Hinter einem
# Reverse Proxy ist Port 8000 von aussen nicht erreichbar, dort braucht die API
# eine eigene Adresse - die muss also hier hinein, nicht in die Umgebung des
# laufenden Containers. Leer gelassen faellt die App auf Port 8000 desselben
# Hosts zurueck, was fuer ein reines Heimnetz reicht.
ARG NEXT_PUBLIC_API_BASE=""
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
ENV JUMPHEIGHTCALCULATOR_API=http://api:8000
RUN npm run build

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production JUMPHEIGHTCALCULATOR_API=http://api:8000 PORT=3000
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
