#!/usr/bin/env bash
# For each kernel with an m4.metal variant, bench the m4 version on M2 vs the
# current default. If m4 is at least as fast on M2, REPLACE the flat/default
# with m4 (delete directory). Otherwise keep dir structure.
set -uo pipefail
cd "$(dirname "$0")"

run_bench() {
  local name=$1
  SKIP_BUILD=1 ./bench "$name" --no-save -- --iters 30 2>&1 | grep -E "correctness|kernel  " | head -2
}

extract_min_ms() {
  awk '/^  kernel  / { gsub(/[^0-9.]/,"",$5); print $5; exit }' <<<"$1"
}

while IFS= read -r m4path; do
  dir=$(dirname "$m4path")
  k=$(basename "$dir")
  # Determine kernel set
  set=$(basename "$(dirname "$dir")")

  echo ""
  echo "=== $k (set=$set) ==="

  # The "default" for this kernel could be:
  # - $dir/default.metal (explicit)
  # - $set_dir/$k.metal (flat that coexists with the dir)
  flat="metal/kernels/$set/$k.metal"
  default="$dir/default.metal"

  if [ -f "$default" ]; then
    current="$default"
    layout="dir+default+m4"
  elif [ -f "$flat" ]; then
    current="$flat"
    layout="flat+m4_variant"
  else
    echo "  no current default found, skipping"
    continue
  fi

  echo "  layout: $layout"

  # 1. bench current (uses flat or default fallback on M2)
  out=$(run_bench "$k")
  cur_ms=$(echo "$out" | awk '/^  kernel  / { gsub(/ms/,"",$3); print $3; exit }')
  cur_ok=$(echo "$out" | grep -c "✓ correct")
  echo "  current M2: ${cur_ms}ms correct=$cur_ok"

  if [ "$cur_ok" -eq 0 ]; then
    echo "  current broken on M2; skipping"
    continue
  fi

  # 2. swap m4 into the M2-picked path
  backup=$(mktemp)
  cp "$current" "$backup"
  cp "$m4path" "$current"
  make --silent all >/dev/null 2>&1 || { echo "  build failed; reverting"; cp "$backup" "$current"; rm "$backup"; make --silent all >/dev/null 2>&1; continue; }

  out=$(run_bench "$k")
  m4_ms=$(echo "$out" | awk '/^  kernel  / { gsub(/ms/,"",$3); print $3; exit }')
  m4_ok=$(echo "$out" | grep -c "✓ correct")
  echo "  m4 on M2:   ${m4_ms}ms correct=$m4_ok"

  if [ "$m4_ok" -eq 0 ]; then
    echo "  → SPLIT: m4 broken on M2"
    cp "$backup" "$current"
    rm "$backup"
    make --silent all >/dev/null 2>&1
    continue
  fi

  # 3. compare. Promote if m4 is faster or within 5% slower.
  faster=$(python3 -c "print('1' if float('$m4_ms') <= float('$cur_ms') * 1.05 else '0')")
  if [ "$faster" = "1" ]; then
    echo "  → PROMOTE: m4 ($m4_ms ms) <= current ($cur_ms ms) × 1.05; flat-replace"
    # m4 content already in $current. Now collapse dir to flat.
    if [ "$layout" = "dir+default+m4" ]; then
      # Move "$current" (which is $dir/default.metal with m4 content) to flat
      mv "$current" "metal/kernels/$set/$k.metal"
      rm -f "$m4path"
      # Remove dir if empty
      rmdir "$dir" 2>/dev/null || ls "$dir"
    else
      # layout flat+m4_variant — just remove the m4.metal and dir
      rm -f "$m4path"
      rmdir "$dir" 2>/dev/null || ls "$dir"
    fi
  else
    echo "  → KEEP SPLIT: m4 ($m4_ms ms) is >5% slower than M2 current ($cur_ms ms)"
    # Restore current (it was overwritten with m4). Ensure default form is set up.
    cp "$backup" "$current"
    if [ "$layout" = "flat+m4_variant" ]; then
      # Convert flat to default in dir
      mv "metal/kernels/$set/$k.metal" "$dir/default.metal"
    fi
  fi
  rm "$backup"
  make --silent all >/dev/null 2>&1
done < <(find metal/kernels -name "m4.metal" -type f)

echo ""
echo "=== final structure ==="
find metal/kernels -type d -mindepth 3 | sort
