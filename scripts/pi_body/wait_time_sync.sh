#!/bin/bash
set -euo pipefail

timeout=${KIKI_SYNC_TIMEOUT:-120}
interval=${KIKI_SYNC_INTERVAL:-2}
deadline=$((SECONDS + timeout))

while true; do
    if [[ $(timedatectl show -p NTPSynchronized --value) == yes ]]; then
        exit 0
    fi
    if (( SECONDS >= deadline )); then
        printf '等待系统时钟网络对时超时，请检查树莓派网络和对时服务；身体服务将自动重试。\n' >&2
        exit 1
    fi
    sleep "$interval"
done
