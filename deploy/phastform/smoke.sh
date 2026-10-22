#!/usr/bin/env bash
# smoke.sh [ENGINE] [STT] [TTS]: end-to-end check of the frozen box, from the GPU node or through a tunnel.
# Speaks a question with the TTS, transcribes it with the STT, asks the LLM, and speaks the reply; prints each
# stage's latency. On the GPU node it also prints GPU memory and fails if a service port listens beyond loopback.
set -euo pipefail
engine=${1:-http://127.0.0.1:8000}
stt=${2:-http://127.0.0.1:8100}
tts=${3:-http://127.0.0.1:8880}
model=Qwen/Qwen3-4B-Instruct-2507-FP8
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

field() { python3 -c "import json, sys; print(json.load(sys.stdin)$1)"; }
stage() { printf '%-26s %s\n' "$1" "$2"; }

for url in "$engine/health" "$stt/health" "$tts/health"; do curl -fsS -m 5 -o /dev/null "$url"; done
stage health ok

t=$(curl -fsS -o "$work/question.wav" -w '%{time_total}' -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"af_heart","response_format":"wav","input":"What is the capital of France?"}' \
  "$tts/v1/audio/speech")
stage "tts question (wav)" "${t}s, $(wc -c <"$work/question.wav") bytes"

curl -fsS -o "$work/stt.json" -w '%{time_total}\n' -F file=@"$work/question.wav" \
  -F model=openai/whisper-large-v3-turbo "$stt/v1/audio/transcriptions" >"$work/stt.time"
text=$(field '["text"]' <"$work/stt.json")
stage "stt" "$(cat "$work/stt.time")s: ${text}"
grep -qi france <<<"$text" || { echo "smoke: transcript does not mention France" >&2; exit 1; }

python3 -c 'import json, sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 32, "temperature": 0,
  "messages": [{"role": "user", "content": sys.argv[2] + " Answer in one short sentence."}]}))' \
  "$model" "$text" >"$work/chat.json"
t=$(curl -fsS -o "$work/reply.json" -w '%{time_total}' -H 'Content-Type: application/json' \
  -d @"$work/chat.json" "$engine/v1/chat/completions")
reply=$(field '["choices"][0]["message"]["content"]' <"$work/reply.json")
stage "llm" "${t}s: ${reply}"
grep -qi paris <<<"$reply" || { echo "smoke: reply does not mention Paris" >&2; exit 1; }

python3 -c 'import json, sys; print(json.dumps({"model": "kokoro", "voice": "af_heart",
  "response_format": "pcm", "input": sys.argv[1]}))' "$reply" >"$work/speak.json"
t=$(curl -fsS -o "$work/reply.pcm" -w '%{time_starttransfer}/%{time_total}' -H 'Content-Type: application/json' \
  -d @"$work/speak.json" "$tts/v1/audio/speech")
stage "tts reply (first/total)" "${t}s, $(wc -c <"$work/reply.pcm") bytes"

# Only on the GPU node: through a tunnel these ports belong to another host (on cyan, possibly another user's).
if command -v nvidia-smi >/dev/null; then
  stage "gpu memory" "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
  if command -v ss >/dev/null; then
    exposed=$(ss -Hlntu | awk '{print $5}' | grep -E ':(8000|8100|8880|5557|5558)$' |
      grep -v -E '^(127\.0\.0\.1|\[::1\]):' || true)
    [[ -z $exposed ]] || { echo "smoke: service ports listen beyond loopback: $exposed" >&2; exit 1; }
    stage "port audit" "loopback only"
  fi
fi
echo "smoke: ok"
