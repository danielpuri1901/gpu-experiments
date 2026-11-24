#!/usr/bin/env bash
# Runs the six tests on a kind cluster that already has metrics-server and Istio.
# Every check prints PASS or FAIL with its measured number. Exit code 1 if any check fails.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT=$PWD/out
L="python3 $PWD/base/loadgen.py"
IMAGE=$(awk '/image:/{print $2; exit}' base/app.yaml)
FAIL=0

now() { date +%s.%N; }
j() { jq -r "$2" <<<"$1"; }                       # j JSON FILTER
stats() { $L summary "$@" 2>/dev/null || echo '{"n":0,"failed":-1,"p50_ms":null,"p95_ms":null,"versions":{},"pods":0}'; }
pick() { kubectl -n "$1" get pod -l "$2" -o json | jq -r '[.items[] | select(.metadata.deletionTimestamp == null) | .metadata.name][0]'; }
ready_n() { kubectl -n "$1" get deploy model -o jsonpath='{.status.readyReplicas}'; }
want_n() { kubectl -n "$1" get deploy model -o jsonpath='{.spec.replicas}'; }
poll() { local f=$1 v; shift; while :; do v=$("$@" 2>/dev/null); echo "$(now) ${v:-0}" >>"$f"; sleep 1; done; }
rollout() { kubectl -n "$1" rollout status deploy/"${2:-model}" --timeout="${3:-180s}" >/dev/null; }
reset() { kubectl -n "$1" apply -k base >/dev/null && rollout "$1"; }   # back to the manifest state
reset_fixed() { reset "$1"; kubectl -n "$1" delete hpa model >/dev/null; }  # no HPA, so the replica count stays what the test sets

verdict() {  # NAME AWK-CONDITION MEASURED
  local r=FAIL
  awk "BEGIN{exit !($2)}" && r=PASS
  echo "$r  $1: $3"
  jq -nc --arg t "$1" --arg r "$r" --arg m "$3" '{test:$t,result:$r,measured:$m}' >>"$OUT/results.jsonl"
  [ $r = PASS ] || FAIL=1
}

load_start() {  # NS NAME LOADGEN-ARGS... : start a Job that calls the Service from inside the cluster
  local ns=$1 name=$2 args i; shift 2
  args=$(printf ',"%s"' "$@")
  kubectl -n "$ns" apply -f - >/dev/null <<YAML
apiVersion: batch/v1
kind: Job
metadata: {name: $name}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: loadgen
          image: $IMAGE
          command: ["python", "-u", "/app/loadgen.py", "run", "http://model/v1/chat/completions"$args]
          volumeMounts: [{name: code, mountPath: /app}]
      volumes: [{name: code, configMap: {name: code}}]
YAML
  for i in $(seq 90); do kubectl -n "$ns" get pod -l job-name="$name" -o name | grep -q . && break; sleep 1; done
  kubectl -n "$ns" wait --for=jsonpath='{.status.phase}'=Running pod -l job-name="$name" --timeout=180s >/dev/null
}

load_end() {  # NS NAME : wait for the Job, keep its JSON in $OUT/NAME.json
  kubectl -n "$1" wait --for=condition=complete job/"$2" --timeout=400s >/dev/null || echo "job $2 did not complete"
  kubectl -n "$1" logs job/"$2" -c loadgen >"$OUT/$2.json"
  kubectl -n "$1" delete job "$2" --wait=false >/dev/null
}

t1() {
  echo "== 1 readiness gate"
  reset_fixed plain
  kubectl -n plain set env deploy/model LOAD_SECONDS=15 >/dev/null; rollout plain
  local v S n f p
  for v in probe noprobe; do
    if [ $v = noprobe ]; then   # control: same test, readiness probe removed
      kubectl -n plain scale deploy/model --replicas=3 >/dev/null; rollout plain
      kubectl -n plain patch deploy model --type=json -p '[{"op":"remove","path":"/spec/template/spec/containers/0/readinessProbe"}]' >/dev/null
      rollout plain; sleep 20   # let the new pods finish loading before the test starts
    fi
    load_start plain t1-$v --seconds 40 --workers 4 --rps 20
    sleep 5; kubectl -n plain scale deploy/model --replicas=6 >/dev/null
    load_end plain t1-$v; rollout plain
    S=$(stats "$OUT/t1-$v.json"); n=$(j "$S" .n); f=$(j "$S" .failed); p=$(j "$S" .pods)
    if [ $v = probe ]; then
      verdict "1a readiness probe on, scale 3 to 6, 15 s model load" "$n>0 && $f==0" "failed=$f of $n requests, pods that served=$p"
    else
      verdict "1b control without readiness probe" "$n>0 && $f>0" "failed=$f of $n requests (the count to expect without the gate)"
    fi
  done
}

t2() {
  echo "== 2 self-healing"
  reset_fixed plain
  local T0 T1 victim rec restart=-1 c S n f
  load_start plain t2 --seconds 80 --workers 4 --rps 20
  poll "$OUT/t2-ready.trace" ready_n plain & local P=$!
  sleep 5; victim=$(pick plain app=model)
  T0=$(now); kubectl -n plain delete pod "$victim" --wait=false >/dev/null
  sleep 3; rollout plain; sleep 2
  rec=$($L recover "$OUT/t2-ready.trace" "$T0" 3)
  victim=$(pick plain app=model)
  kubectl -n plain exec "$victim" -- python -c "import urllib.request as u;u.urlopen('http://127.0.0.1:8080/break')" >/dev/null
  T1=$(now)
  for _ in $(seq 120); do
    c=$(kubectl -n plain get pod "$victim" -o jsonpath='{.status.containerStatuses[0].restartCount}')
    if [ "${c:-0}" -ge 1 ]; then restart=$(awk "BEGIN{printf \"%.1f\", $(now)-$T1}"); break; fi
    sleep 0.5
  done
  rollout plain
  load_end plain t2; kill $P; wait $P 2>/dev/null
  S=$(stats "$OUT/t2.json"); n=$(j "$S" .n); f=$(j "$S" .failed)
  verdict "2a delete a pod under load: back to 3 ready" "$rec>=0 && $rec<=60" "time to 3 ready=${rec}s"
  verdict "2b requests lost during delete and restart (limit 1%)" "$n>0 && $f>=0 && $f<=$n*0.01" "failed=$f of $n requests"
  verdict "2c unhealthy pod (/break): liveness restarts it" "$restart>=0 && $restart<=60" "time to restart=${restart}s (probe: every 3 s, 3 failures)"
}

t3() {
  echo "== 3 rolling update and bad release"
  reset_fixed plain
  local S n f v1 v2 v3 ready stalled=yes undone=no
  load_start plain t3a --seconds 75 --workers 4 --rps 20
  sleep 5; kubectl -n plain set env deploy/model VERSION=v2 >/dev/null; rollout plain
  load_end plain t3a
  S=$(stats "$OUT/t3a.json"); n=$(j "$S" .n); f=$(j "$S" .failed); v1=$(j "$S" '.versions.v1 // 0'); v2=$(j "$S" '.versions.v2 // 0')
  $L timeline "$OUT/t3a.json" 5 | tee "$OUT/t3-timeline.txt"
  verdict "3a rolling update v1 to v2 under load" "$n>0 && $f==0 && $v1>0 && $v2>0" "failed=$f of $n requests, served v1=$v1 v2=$v2"
  load_start plain t3b --seconds 60 --workers 4 --rps 20
  sleep 3; kubectl -n plain set env deploy/model VERSION=v3 NEVER_READY=1 >/dev/null
  if rollout plain model 30s; then stalled=no; fi
  ready=$(ready_n plain)
  kubectl -n plain rollout undo deploy/model >/dev/null && rollout plain && undone=yes
  load_end plain t3b
  S=$(stats "$OUT/t3b.json"); n=$(j "$S" .n); f=$(j "$S" .failed); v3=$(j "$S" '.versions.v3 // 0'); v2=$(j "$S" '.versions.v2 // 0')
  verdict "3b bad release v3 stalls, then rollout undo" "$n>0 && $f==0 && $v3==0 && $v2>0 && \"$stalled\"==\"yes\" && \"$undone\"==\"yes\" && $ready==3" \
    "failed=$f of $n requests, stalled=$stalled, ready pods during stall=$ready, served v3=$v3, undo done=$undone"
}

t4() {
  echo "== 4 autoscaling"
  reset plain   # re-creates the HPA
  local T0 T1 up down max S B A i
  for i in $(seq 60); do kubectl -n plain top pod -l app=model >/dev/null 2>&1 && break; sleep 3; done
  poll "$OUT/t4-want.trace" want_n plain & local P=$!
  load_start plain t4 --seconds 150 --workers 12
  T0=$(now); load_end plain t4; T1=$(now)
  up=$($L until "$OUT/t4-want.trace" "$T0" gt 3)
  B=$(stats "$OUT/t4.json" "$T0" "$(awk "BEGIN{printf \"%.3f\", $T0+$up}")")
  A=$(stats "$OUT/t4.json" "$(awk "BEGIN{printf \"%.3f\", $T0+$up+45}")" "$T1")
  max=$(awk '$2>m{m=$2} END{print m+0}' "$OUT/t4-want.trace")
  verdict "4a HPA scales out under CPU load" "$up>=0 && $up<=150 && $max>3 && $max<=6" \
    "first scale-out after ${up}s, max replicas=$max, p50 latency before=$(j "$B" .p50_ms) ms after=$(j "$A" .p50_ms) ms, p95 before=$(j "$B" .p95_ms) ms after=$(j "$A" .p95_ms) ms, failed=$(j "$B" .failed)+$(j "$A" .failed)"
  for i in $(seq 150); do [ "$($L until "$OUT/t4-want.trace" "$T1" le 3)" != -1 ] && break; sleep 2; done
  down=$($L until "$OUT/t4-want.trace" "$T1" le 3); kill $P; wait $P 2>/dev/null
  verdict "4b scale-in waits for the 60 s stabilisation window" "$down>=30 && $down<=240" "back to 3 replicas ${down}s after the load stopped"
}

weights() {  # STABLE CANARY : shift the mesh weights
  kubectl -n mesh patch virtualservice model --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/http/0/route/0/weight\",\"value\":$1},{\"op\":\"replace\",\"path\":\"/spec/http/0/route/1/weight\",\"value\":$2}]" >/dev/null
  sleep 8   # let the sidecars get the new config
}

mesh_up() {
  kubectl delete ns plain --wait=false >/dev/null   # free CPU for the sidecars
  kubectl create ns mesh >/dev/null; kubectl label ns mesh istio-injection=enabled >/dev/null
  reset_fixed mesh
  kubectl -n mesh get deploy model -o json | jq '{apiVersion, kind, metadata: {name: "model-canary"}, spec}
    | .spec.replicas = 2 | .spec.selector.matchLabels.track = "canary" | .spec.template.metadata.labels.track = "canary"
    | (.spec.template.spec.containers[0].env[] | select(.name == "VERSION")).value = "v2"' | kubectl -n mesh apply -f - >/dev/null
  rollout mesh model-canary; kubectl -n mesh apply -f mesh/mesh.yaml >/dev/null; sleep 8
}

t5() {
  echo "== 5 canary by mesh weight"
  local w c S n f k r ok
  for w in "90 10" "50 50" "0 100"; do
    set -- $w; weights "$1" "$2"
    load_start mesh t5-$2 --count 500 --workers 4; load_end mesh t5-$2
    S=$(stats "$OUT/t5-$2.json"); n=$(j "$S" .n); f=$(j "$S" .failed); k=$(j "$S" '.versions.v2 // 0')
    r=$($L share "$k" "$n" "$(awk "BEGIN{print $2/100}")"); ok=$(j "$r" 'if .ok then 1 else 0 end')
    verdict "5 canary weight $1/$2" "$n>0 && $ok==1" "v2 share=$(j "$r" .share) (target $(awk "BEGIN{print $2/100}"), tolerance +-$(j "$r" .tol)) over $n requests, failed=$f"
  done
  echo "== 5b outlier detection (circuit breaker)"
  local pod fo fw
  weights 100 0; pod=$(pick mesh track=stable)
  kubectl -n mesh exec "$pod" -c server -- python -c "import urllib.request as u;u.urlopen('http://127.0.0.1:8080/poison')" >/dev/null
  load_start mesh t5b-off --count 300 --workers 4; load_end mesh t5b-off
  S=$(stats "$OUT/t5b-off.json"); fo=$(j "$S" .failed)
  kubectl -n mesh apply -f mesh/outlier.yaml >/dev/null; sleep 8
  load_start mesh t5b-on --count 300 --workers 4; load_end mesh t5b-on
  S=$(stats "$OUT/t5b-on.json"); n=$(j "$S" .n); fw=$(j "$S" .failed)
  verdict "5b outlier detection ejects the failing pod" "$n>0 && $fo>0 && $fw<=$n*0.05 && $fw<$fo/3" "failed without=$fo of 300, failed with ejection=$fw of $n"
}

t6() {
  echo "== 6 mTLS STRICT"
  local before after inside c=0
  call() { kubectl -n "$1" exec "$2" ${3:+-c $3} -- python -c "import urllib.request as u;print(u.urlopen('http://model.mesh.svc.cluster.local/health',timeout=5).status)" 2>&1 | grep -v '^command terminated' | tail -1; }
  kubectl create ns outside >/dev/null
  kubectl -n outside run client --image="$IMAGE" --restart=Never -- sleep 600 >/dev/null
  kubectl -n outside wait --for=condition=ready pod/client --timeout=180s >/dev/null
  before=$(call outside client)
  kubectl -n mesh apply -f mesh/strict.yaml >/dev/null; sleep 10
  after=$(call outside client); inside=$(call mesh "$(pick mesh app=model)" server)
  [ "$before" = 200 ] && [ "$inside" = 200 ] && [ "$after" != 200 ] && c=1
  verdict "6 mTLS STRICT: plain-text pod outside the mesh is refused" "$c==1" "outside before STRICT=$before, outside after STRICT='${after:0:90}', inside the mesh=$inside"
}

finish() {
  jq -s '.' "$OUT/results.jsonl" >"$OUT/results.json" 2>/dev/null
  {
    echo "### k8s-model-service results"; echo; echo "| Test | Result | Measured |"; echo "|---|---|---|"
    jq -r '"| \(.test) | \(.result) | \(.measured) |"' "$OUT/results.jsonl"
    if [ -s "$OUT/t3-timeline.txt" ]; then
      echo; echo "Version served during the v1 to v2 rollout (requests per 5 s):"; echo '```'; cat "$OUT/t3-timeline.txt"; echo '```'
    fi
  } >"$OUT/results.md"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$OUT/results.md" >>"$GITHUB_STEP_SUMMARY"; fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"; trap finish EXIT
  kubectl create ns plain >/dev/null
  t1; t2; t3; t4
  mesh_up; t5; t6
  exit $FAIL
fi
