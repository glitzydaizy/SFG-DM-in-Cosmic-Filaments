#!/bin/bash
# Usage (inside or via the container): bash pipeline.sh STAGE DAYNAME
# Stages: load flag calprim split calsec check
set -u
ROOT=$HOME/Projects/FYP-StarFormation
STAGE=$1
source "$ROOT/scripts/$2.conf"
WORK=$ROOT/data/work/$DAY
LOG=$ROOT/notes/$DAY
mkdir -p "$WORK" "$LOG"
cd "$WORK"

targets() {
  sed -n '/following pointings/,/^-----/p' "$LOG/uvindex.txt" | awk '$2=="n"{print $1}'
}

case "$STAGE" in
  load)
    INLIST=""
    for f in $FILES; do INLIST="$INLIST,$ROOT/data/C2767_raw/$f.C2767"; done
    INLIST=${INLIST#,}
    atlod in="$INLIST" out=$DAY.uv ifsel=1 options=birdie,xycorr,rfiflag,noauto 2>&1 | tee "$LOG/atlod.txt"
    uvindex vis=$DAY.uv > "$LOG/uvindex.txt" 2>&1
    grep -E "Total observing time|Total number of records" "$LOG/uvindex.txt"
    cp -r $DAY.uv ${DAY}_unflagged.uv
    ;;
  flag)
    {
    for s in $PRIM $SEC $(targets); do
      echo "=== $s ==="
      pgflag vis=$DAY.uv select="source($s)" stokes=xx,yy,xy,yx flagpar=6,2,2,2,5,3 command='<b' options=nodisp
    done
    if [ -f "$ROOT/scripts/$DAY.handflags" ]; then
      echo "=== hand flags ==="
      bash "$ROOT/scripts/$DAY.handflags"
    fi
    } 2>&1 | tee "$LOG/flag.txt"
    grep -E "===|now flagged" "$LOG/flag.txt"
    ;;
  calprim)
    mfcal vis=$DAY.uv select="source($PRIM)" interval=0.1 2>&1 | tee "$LOG/mfcal_primary.txt"
    ;;
  split)
    mkdir -p split
    cd split
    uvsplit vis=../$DAY.uv select='-auto' maxwidth=0.128 options=mosaic 2>&1 | tee "$LOG/uvsplit.txt"
    ls -d * | head -60
    ;;
  calsec)
    cd split
    {
    for p in $SEC.*; do
      b=${p#$SEC.}
      echo "=== band $b ==="
      if mfcal vis=$SEC.$b interval=0.1 flux=$SECFLUX; then
        gpcopy vis=$SEC.$b out=$TARGET.$b
      else
        echo "FAILED band $b"
      fi
    done
    } 2>&1 | tee "$LOG/calsec.txt"
    grep -E "===|accepted|FAILED" "$LOG/calsec.txt"
    ;;
  check)
    cd split
    for p in $PRIM.* $SEC.*; do
      echo "== $p"
      uvflux vis=$p select='-auto' 2>&1 | grep -E " (XX|YY) "
    done | tee "$LOG/flux_check.txt"
    ;;
  *)
    echo "unknown stage: $STAGE"
    ;;
esac
