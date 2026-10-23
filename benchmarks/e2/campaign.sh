#!/usr/bin/env bash
# campaign.sh [--dry-run] [--pause] CAMPAIGN: measure the E2 points a campaign file lists, from a workstation with SSH
# to the hosts, like deploy/deploy.sh. CAMPAIGN names arm files, one per line, relative to itself. For each point
# whose run directory on cyan is not complete, in order, it
#   1. switches each GPU node to the arm's engine (deploy/phastform/variant.sh use, with what lh-bench e2 engine prints),
#   2. measures the point on cyan (lh-bench e2 run) into ~/leaseholder/runs/e2/CAMPAIGN/POINT, sampling each GPU's
#      clocks and temperature once a second meanwhile into that directory's gpu.csv (gpu-NODE.csv on several nodes),
#   3. fetches that run directory into $LH_E2_RESULTS/CAMPAIGN/POINT (default ~/leaseholder/runs/e2).
# A complete point is skipped, so running a stopped campaign again resumes it; the first failure stops it. At the end
# it writes the campaign's frontier (lh-bench e2 report, fetched into CAMPAIGN/report) and puts cyan's units back
# (lh-bench e2 reset). --pause then stops each GPU node's units. --dry-run copies and checks every arm file and lists
# the points with their engines, without switching or measuring anything. LH_SSH is your ssh command for the hosts,
# as for deploy.sh; LH_CYAN names cyan, and LH_GPU the GPU nodes every point of the campaign runs on: one host, or
# several separated by commas for the points on several GPU nodes. A node is named after its host's first label
# (gray for gray.cis.upenn.edu), as the arm files' [gateway.nodes] name it.
set -euo pipefail
usage() { echo "usage: campaign.sh [--dry-run] [--pause] CAMPAIGN" >&2; exit 2; }
dry_run=false pause=false
while [[ ${1:-} == --* ]]; do
  case $1 in
    --dry-run) dry_run=true ;;
    --pause) pause=true ;;
    *) usage ;;
  esac
  shift
done
(($# == 1)) || usage
[[ -f $1 ]] || { echo "campaign: no such file: $1" >&2; exit 2; }
ssh=${LH_SSH:-ssh}
cyan=${LH_CYAN:-cyan.cis.upenn.edu}
IFS=', ' read -r -a gpus <<<"${LH_GPU:-phastform.cis.upenn.edu}"
bad_gpus() { echo "campaign: LH_GPU names each GPU node's host once, separated by commas: '${LH_GPU-}'" >&2; exit 2; }
((${#gpus[@]})) || bad_gpus
named=" "
for gpu in "${gpus[@]}"; do
  [[ -n $gpu && $named != *" ${gpu%%.*} "* ]] || bad_gpus
  named+="${gpu%%.*} "
done
name=$(basename "$1" .campaign)
here=$(cd "$(dirname "$1")" && pwd)
results=${LH_E2_RESULTS:-$HOME/leaseholder/runs/e2}/$name
runs=leaseholder/runs/e2/$name # on cyan, under the home directory, as every path given to ssh here
bench=leaseholder/venvs/bench/bin/lh-bench
complete() { echo "grep -qs '\"status\": \"complete\"' $runs/$1/run.json"; } # run.json as e2 run writes it
mkdir -p "$results"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$results/campaign.log"; }
die() { log "campaign: $*"; exit 1; }

# The points in order, each named after its arm file.
arms=() points=()
while IFS= read -r line || [[ -n $line ]]; do
  line=${line%%#*}
  line=${line#"${line%%[![:space:]]*}"}
  line=${line%"${line##*[![:space:]]}"}
  [[ -n $line ]] || continue
  [[ -f $here/$line ]] || die "$1 names $line, which is not a file"
  point=$(basename "$line" .toml)
  for seen in ${points[@]+"${points[@]}"}; do
    [[ $seen != "$point" ]] || die "$1 names two points $point"
  done
  arms+=("$here/$line") points+=("$point")
done <"$1"
((${#points[@]})) || die "$1 names no point"
log "campaign $name: ${#points[@]} points, arm files at $(git -C "$here" describe --always --dirty 2>/dev/null || echo '?')"

# Copy the arm files to cyan, where lh-bench reads them, and check each one there.
$ssh "$cyan" mkdir -p "$runs/arms"
rsync -a -e "$ssh" "${arms[@]}" "$cyan:$runs/arms/"
engines=()
for point in "${points[@]}"; do
  state="{ $(complete "$point") && echo complete || echo pending; }"
  answer=$($ssh "$cyan" "$bench e2 engine $runs/arms/$point.toml && $state") ||
    die "$point: its arm file does not load (above)"
  engines+=("${answer%%$'\n'*}")
  log "$point: ${answer#*$'\n'}, engine ${answer%%$'\n'*}"
done
if $dry_run; then exit 0; fi

fetch() { # fetch POINT: mirror its run directory, without AIPerf's copy of the dataset
  rsync -a --delete -e "$ssh" --exclude aiperf/inputs.json "$cyan:$runs/$1/" "$results/$1/"
}
# A consumer card's boost clock follows its temperature, so each point keeps the clocks it ran at: nvidia-smi on each
# GPU node, once a second while the point runs (line-buffered, timestamps in UTC); it stops when its ssh does.
gpu_query=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active
samplers=()
trap 'for sampler in ${samplers[@]+"${samplers[@]}"}; do kill "$sampler" 2>/dev/null; done' EXIT
gpu_csv() { # gpu_csv HOST: the file of its GPU's samples in a run directory
  if ((${#gpus[@]} == 1)); then echo gpu.csv; else echo "gpu-${1%%.*}.csv"; fi
}
sample_gpus() { # sample_gpus POINT
  for gpu in "${gpus[@]}"; do
    $ssh "$gpu" "exec env TZ=UTC stdbuf -oL nvidia-smi --query-gpu=$gpu_query --format=csv,noheader -l 1" \
      >"$results/$1.$(gpu_csv "$gpu")" 2>>"$results/campaign.log" &
    samplers+=($!)
  done
}
stop_gpus() { # stop_gpus POINT: end the sampling and put the samples into the point's run directory on cyan
  for sampler in ${samplers[@]+"${samplers[@]}"}; do
    kill "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
  done
  samplers=()
  for gpu in "${gpus[@]}"; do
    file=$(gpu_csv "$gpu")
    if $ssh "$cyan" "cat >$runs/$1/$file" <"$results/$1.$file"; then
      rm "$results/$1.$file"
    else
      log "$1: the samples of ${gpu%%.*}'s GPU stay in $results/$1.$file"
    fi
  done
}
for i in "${!points[@]}"; do
  point=${points[$i]}
  if $ssh "$cyan" "$(complete "$point")"; then
    fetch "$point"
    continue
  fi
  # A point whose ssh session dropped may still be running on cyan; switching the engine would spoil it.
  $ssh "$cyan" flock -n leaseholder/runs/.lh-bench-e2.lock true ||
    die "a point is still running on cyan (its pid is in leaseholder/runs/.lh-bench-e2.lock); run again once it ends"
  for gpu in "${gpus[@]}"; do
    log "$point: ${gpu%%.*} runs ${engines[$i]}"
    $ssh "$gpu" "bash leaseholder/src/deploy/phastform/variant.sh use ${engines[$i]}" 2>&1 |
      tee -a "$results/campaign.log" || die "$point: ${gpu%%.*} did not start ${engines[$i]} (journalctl there)"
  done
  log "$point: measuring"
  sample_gpus "$point"
  status=0
  $ssh "$cyan" "$bench e2 run $runs/arms/$point.toml --out $runs/$point" 2>&1 | tee -a "$results/campaign.log" ||
    status=$?
  stop_gpus "$point"
  if ((status == 255)); then
    die "$point: lost the connection to cyan; the point may still be running there"
  elif ((status)); then
    fetch "$point" || true
    die "$point failed (exit status $status); its run directory is $results/$point"
  fi
  fetch "$point"
done

dirs=""
for point in "${points[@]}"; do dirs+=" $runs/$point"; done
log "the frontier of $name"
$ssh "$cyan" "$bench e2 report$dirs --out $runs/report" 2>&1 | tee -a "$results/campaign.log" ||
  die "lh-bench e2 report failed"
rsync -a --delete -e "$ssh" "$cyan:$runs/report/" "$results/report/"
$ssh "$cyan" "$bench e2 reset" 2>&1 | tee -a "$results/campaign.log" || die "lh-bench e2 reset failed"
if $pause; then
  for gpu in "${gpus[@]}"; do
    $ssh "$gpu" systemctl --user stop leaseholder.target || die "the units of ${gpu%%.*} did not stop"
    log "${gpu%%.*} is paused; start it again with systemctl --user start leaseholder.target"
  done
fi
log "done: $results/report/frontier.md"
