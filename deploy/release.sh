#!/bin/bash
# Release von JumpLab nach k3s: baut beide Images aus einem sauberen `git archive main` auf der
# Build-VM, legt sie in zot (Registry im Cluster), setzt sie per Digest und prüft den Rollout.
#
#   bash deploy/release.sh <tag>            z. B. 20261120
#   TROCKEN=1 bash deploy/release.sh <tag>  baut nur, ändert nichts im Cluster
#
# Rückweg (Deployments behalten 3 Revisionen):
#   ssh root@10.11.5.43 'KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl -n jumplab rollout undo deploy/api deploy/web'
#
# Bewusst kein GitHub-Actions-Deploy: Das Repo ist öffentlich, und ein Self-hosted Runner an einem
# öffentlichen Repo führt über Fork-PRs fremden Code auf der Build-VM aus.
set -euo pipefail

TAG=${1:?Aufruf: bash deploy/release.sh <tag>}
[[ $TAG =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo "Tag ungültig: $TAG"; exit 1; }
PUBLIC_URL=${PUBLIC_URL:-https://jumplab.jonasleuenberger.com}
NS=jumplab
REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-ubuntu@10.11.5.148}

NODE=${NODE:-}
if [ -z "$NODE" ]; then
  for ip in 10.11.5.43 10.11.5.45 10.11.5.41; do
    ssh -o BatchMode=yes -o ConnectTimeout=5 root@$ip true 2>/dev/null && { NODE=$ip; break; }
  done
fi
[ -n "$NODE" ] || { echo "Keine k3s-Node per SSH erreichbar."; exit 1; }
k() { ssh root@$NODE "KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl $(printf '%q ' "$@")"; }
ZOT=localhost:30500

ssh root@$NODE "curl -sf -o /dev/null -I http://$ZOT/v2/jumplab-api/manifests/$TAG" && { echo "Tag $TAG gibt es in zot schon."; exit 3; }
COMMIT=$(git -C "$REPO" rev-parse --short main)
DIR=/tmp/jumplab-$TAG
echo "Baue jumplab-api und jumplab-web:$TAG aus main $COMMIT auf $BUILD"
git -C "$REPO" archive main | ssh $BUILD "rm -rf $DIR && mkdir -p $DIR && tar -x -C $DIR && cd $DIR \
  && sudo docker build -q -t $NODE:30500/jumplab-api:$TAG . >/dev/null \
  && sudo docker build -q --build-arg NEXT_PUBLIC_API_BASE=$PUBLIC_URL -t $NODE:30500/jumplab-web:$TAG web >/dev/null"
[ -z "${TROCKEN:-}" ] || { ssh $BUILD "rm -rf $DIR"; echo "TROCKEN: gebaut, Cluster unverändert."; exit 0; }

declare -A IMG
for app in jumplab-api jumplab-web; do
  ssh $BUILD "sudo docker push -q $NODE:30500/$app:$TAG >/dev/null && sudo docker image rm $NODE:30500/$app:$TAG >/dev/null"
  d=$(ssh root@$NODE "curl -sI -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json' http://$ZOT/v2/$app/manifests/$TAG" | tr -d '\r' | awk 'tolower($1)=="docker-content-digest:" {print $2}')
  [[ $d == sha256:* ]] || { echo "Digest für $app aus zot nicht lesbar."; exit 1; }
  IMG[$app]=$ZOT/$app@$d
  echo "In zot: ${IMG[$app]}"
done
ssh $BUILD "rm -rf $DIR"

k -n $NS set image ds/image-cache api=${IMG[jumplab-api]} web=${IMG[jumplab-web]}
k -n $NS rollout status ds/image-cache --timeout=600s
k -n $NS set image deploy/api api=${IMG[jumplab-api]} worker=${IMG[jumplab-api]}
k -n $NS set image deploy/web web=${IMG[jumplab-web]}
for d in api web; do
  if ! k -n $NS rollout status deploy/$d --timeout=300s; then
    echo "Rollout $d gescheitert, nehme zurück."
    k -n $NS rollout undo deploy/$d
    exit 1
  fi
done

code() { curl -s -o /dev/null -m 20 -w '%{http_code}' "$@"; }
FAIL=0
chk() { # FALLE: Variable nicht PUBLIC nennen, Windows setzt sie auf C:/Users/Public
  local n=$1 want=$2 got i; shift 2
  for i in $(seq 12); do got=$(code "$@" || true); [ "$got" = "$want" ] && break; sleep 5; done
  if [ "$got" = "$want" ]; then echo "  ok   $n $got"; else echo "  FAIL $n: erwartet $want, ist $got"; FAIL=1; fi
}
chk "/api/health" 200 $PUBLIC_URL/api/health
chk "/" 200 $PUBLIC_URL/

sed -i -E "s#image: localhost:30500/jumplab-api(:[a-z0-9._-]+|@sha256:[0-9a-f]+)#image: ${IMG[jumplab-api]}#; s#image: localhost:30500/jumplab-web(:[a-z0-9._-]+|@sha256:[0-9a-f]+)#image: ${IMG[jumplab-web]}#" "$REPO/deploy/k3s/jumplab.yaml"
echo "deploy/k3s/jumplab.yaml auf die neuen Digests gesetzt (noch committen)."
[ $FAIL = 0 ] || { echo "Prüfung fehlgeschlagen. Zurück: kubectl -n $NS rollout undo deploy/api deploy/web"; exit 1; }
echo "Fertig: $TAG ($COMMIT)"
