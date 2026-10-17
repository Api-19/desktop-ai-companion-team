#!/bin/bash
set -euo pipefail

dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
if [[ $# != 1 || $1 == -* || -z $1 ]]; then
    printf '用法：scripts/pi_body/deploy.sh [--dry-run] <ssh目标>\n' >&2
    exit 2
fi
target=$1

# Resolve relative invocation paths without running utilities in dry-run mode.
script_dir=${BASH_SOURCE[0]%/*}
if [[ $script_dir != /* ]]; then
    script_dir="$PWD/$script_dir"
fi
repo_root="$script_dir/../.."
cert_dir="$HOME/Library/Application Support/kiki-companion/bodylink/body"
ssh_command=(ssh -i "$HOME/.ssh/id_ed25519" -o BatchMode=yes)

shell_quote() {
    local escaped_quote="'\\''"
    printf "'%s'" "${1//\'/$escaped_quote}"
}

run() {
    if "$dry_run"; then
        local arg
        for arg in "$@"; do
            shell_quote "$arg"
            printf ' '
        done
        printf '\n'
    else
        "$@"
    fi
}

# Embed only the three public templates in a single remote shell command.
file_command() {
    local line
    printf '%s' "printf '%s\\n'"
    while IFS= read -r line || [[ -n $line ]]; do
        printf ' '
        shell_quote "$line"
    done < "$1"
}

rsync_shell="$(shell_quote ssh) -i $(shell_quote "$HOME/.ssh/id_ed25519") -o BatchMode=yes"

# a. The remote rsync creates its destination before receiving the source tree.
run rsync -a --delete --exclude __pycache__ -e "$rsync_shell" \
    --rsync-path 'mkdir -p "$HOME/kiki-body/src/companion" && rsync' \
    "$repo_root/src/companion/" "$target:~/kiki-body/src/companion/"

# b. Restrict certificate permissions during transfer as well as afterwards.
run rsync -a --chmod=D700,F600 -e "$rsync_shell" \
    --rsync-path 'umask 077; mkdir -p "$HOME/.config/kiki-body/certs" && chmod 0700 "$HOME/.config/kiki-body/certs" && rsync' \
    "$cert_dir/" "$target:~/.config/kiki-body/certs/"
run "${ssh_command[@]}" "$target" \
    'chmod 0700 "$HOME/.config/kiki-body/certs" && find "$HOME/.config/kiki-body/certs" -type f -exec chmod 0600 {} \;'

# c. A missing sudo grant is actionable and has its own exit code.
run "${ssh_command[@]}" "$target" \
    'if ! sudo -n apt-get install -y libportaudio2; then printf "%s\n" "请在树莓派上手动执行：sudo -n apt-get install -y libportaudio2，然后重试部署。" >&2; exit 3; fi'

# d. Reuse Debian numpy through system site packages; pin the added packages.
run "${ssh_command[@]}" "$target" \
    'set -e; if [ ! -d "$HOME/kiki-body/venv" ]; then python3 -m venv --system-site-packages "$HOME/kiki-body/venv"; fi; "$HOME/kiki-body/venv/bin/python" -m pip install pydantic==2.13.5 websockets==15.0.1 sounddevice==0.5.5 samplerate==0.2.4'

# e. SSH carries public templates, so there are exactly two rsync sources.
install_command='set -e; umask 077; mkdir -p "$HOME/kiki-body" "$HOME/.config/systemd/user" "$HOME/.config/kiki-body"; '
install_command+="$(file_command "$script_dir/wait_time_sync.sh")"
install_command+=' > "$HOME/kiki-body/wait_time_sync.sh"; chmod 0755 "$HOME/kiki-body/wait_time_sync.sh"; '
install_command+="$(file_command "$script_dir/kiki-body.service")"
install_command+=' > "$HOME/.config/systemd/user/kiki-body.service"; if [ ! -e "$HOME/.config/kiki-body/kiki-body.env" ]; then '
install_command+="$(file_command "$script_dir/kiki-body.env.example")"
install_command+=' > "$HOME/.config/kiki-body/kiki-body.env"; fi; chmod 0600 "$HOME/.config/kiki-body/kiki-body.env"'
run "${ssh_command[@]}" "$target" "$install_command"

# f. Linger keeps the user manager available without an interactive login.
run "${ssh_command[@]}" "$target" \
    'if ! loginctl enable-linger "$USER"; then if ! sudo -n loginctl enable-linger "$USER"; then printf "%s\n" "请在树莓派上手动执行：sudo loginctl enable-linger \"\$USER\"，然后重试部署。" >&2; exit 4; fi; fi'

# g. Enable only; the owner must fill in the brain host and audio devices first.
run "${ssh_command[@]}" "$target" \
    'systemctl --user daemon-reload && systemctl --user enable kiki-body.service'

# h. List devices on the deployment SSH connection for the owner to choose.
run "${ssh_command[@]}" "$target" \
    '"$HOME/kiki-body/venv/bin/python" -c "import sounddevice; print(sounddevice.query_devices())"'
run printf '%s\n' \
    "部署配置完成，服务已启用开机自启。下一步：ssh ${target}，在 ~/.config/kiki-body/kiki-body.env 填写 KIKI_BRAIN_HOST、KIKI_INPUT_DEVICE、KIKI_OUTPUT_DEVICE（设备名参照上方列表），核对端口和 ALSA 后手动执行 systemctl --user start kiki-body.service；用 systemctl --user status kiki-body.service 查看状态。"
