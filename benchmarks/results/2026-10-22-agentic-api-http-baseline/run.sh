#!/usr/bin/env bash
# w1-agentic-api.sh: agentic-api's own response_provider benchmark against the frozen box, the HTTP baseline for
# vllm-project/agentic-api#367: agentic-api over WebSocket, HTTP/SSE and JSON vs direct vLLM, same model and hosts.
set -uo pipefail
R=$HOME/leaseholder/runs/w1-agentic-api
A=$HOME/leaseholder/bin/agentic-api-a7dc7b5
MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
mkdir -p "$R"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$R/campaign.log"; }
log "agentic-server $("$A/agentic-server" --version) at a7dc7b5; vLLM through the tunnel at 127.0.0.1:18000"
# The gateway binds loopback only (its default is 0.0.0.0), on its own cores, with a fresh SQLite store.
taskset -c 8-15 "$A/agentic-server" --llm-api-base http://127.0.0.1:18000 --gateway-host 127.0.0.1 \
  --gateway-port 9000 --db-url "sqlite://$R/agentic_api.db?mode=rwc" >"$R/server.log" 2>&1 &
srv=$!
for _ in $(seq 1 120); do curl -sf -o /dev/null http://127.0.0.1:9000/health && break; sleep 1; done
curl -sf -o /dev/null http://127.0.0.1:9000/health || { log "gateway not healthy"; tail -20 "$R/server.log"; kill $srv; exit 1; }
# The gateway's CPU time (clock ticks) and resident memory (kB), once a second.
( while kill -0 $srv 2>/dev/null; do
    echo "$(date +%s) $(awk '{print $14+$15}' /proc/$srv/stat) $(awk '/VmRSS/{print $2}' /proc/$srv/status)"
    sleep 1
  done ) >"$R/server-usage.txt" 2>/dev/null &
bench() {
  local name=$1; shift
  log "start $name"
  taskset -c 16-23 "$A/response_provider" --model "$MODEL" --agentic-url http://127.0.0.1:9000/v1 \
    --vllm-url http://127.0.0.1:18000/v1 --timeout-seconds 300 --seed 20261022 --output-dir "$R/$name" "$@" \
    >"$R/$name.log" 2>&1
  log "$name rc=$?"
}
bench history-s5 --workload history-rehydration --sessions 5 --depths 1,5,10,25 --provider all
bench transport-s1 --workload transport --sessions 1 --requests-per-session 10 --transport-rounds 4 --provider all
bench transport-s5 --workload transport --sessions 5 --requests-per-session 10 --transport-rounds 4 --provider all
bench transport-s10 --workload transport --sessions 10 --requests-per-session 10 --transport-rounds 4 --provider all
kill $srv; wait $srv 2>/dev/null
log "done"
