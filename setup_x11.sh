#!/usr/bin/env bash

export DISPLAY=host.docker.internal:0

echo "Checking XQuartz connection..."

if timeout 5 xdpyinfo >/dev/null 2>&1; then
    echo "SUCCESS: X11 is connected."
    echo "DISPLAY=$DISPLAY"
else
    echo "FAILED: Cannot connect to XQuartz."
    echo "Check that XQuartz is running on your Mac."
    return 1 2>/dev/null || exit 1
fi
