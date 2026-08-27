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
# Der Container lauscht auf **3050**, derselben Zahl wie der Host-Port. Vorher
# war es 3000 innen und 3050 aussen - und weil Traefik die Container ueber das
# Docker-Netz anspricht, nicht ueber den Host, gehoerte in die Domain-Konfiguration
# der Container-Port. Das hat verlaesslich zu 502 gefuehrt. Eine Zahl fuer alles.
# HOSTNAME=0.0.0.0 ist Pflicht, nicht Kosmetik: ohne die Angabe bindet Next'
# standalone-server.js an localhost und ist damit nur *innerhalb* des Containers
# erreichbar. Traefik verbindet sich ueber das Docker-Netz auf die Container-IP
# und bekam "connection refused" - sichtbar als 502, obwohl Container, Netz,
# Router-Regel und Portangabe alle stimmten.
ENV NODE_ENV=production JUMPHEIGHTCALCULATOR_API=http://api:8000
ENV PORT=3050 HOSTNAME=0.0.0.0
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3050
CMD ["node", "server.js"]
