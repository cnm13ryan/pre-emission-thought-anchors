#!/usr/bin/env bash
# Bootstrap a fresh remote GPU instance for preanchor with one command.
#
#   git clone <repo> ~/preanchor-repo && cd ~/preanchor-repo && bash scripts/bootstrap_remote.sh
#
# Every step is idempotent: rerunning verifies existing components and only
# installs what is missing. Nothing beyond the project's needs is touched:
# no apt upgrades, no driver/CUDA changes, no Docker daemon reconfiguration.
#
# Options:
#   --quick            Routine rerun: skip the CPU test suite and the production doctor
#                      (network + NFS probe). Dependency, storage, GPU and environment
#                      checks still run.
#   --cpu-only         Install the base + dev environment only (no torch/CUDA extra)
#                      and skip every GPU check. Default is GPU mode when nvidia-smi exists.
#   --allow-ephemeral  Downgrade a missing/unwritable persistent filesystem from a hard
#                      failure to a warning (development only; stages need PREANCHOR_ROOT).
#   --pull-vllm        Pull the pinned vLLM Docker image (~29 GB) if it is absent.
#   --skip-tests       Do not run the CPU unit/integration test suite.
#   --no-doctor        Skip the production `preanchor doctor` (the CPU-only doctor always runs).
#   -h, --help         Show this help.
#
# Environment overrides:
#   UV_INSTALL_VERSION   uv version to install when uv is absent (default: latest stable).
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Constants derived from the repository (keep in sync with pyproject/README/locks).
# ---------------------------------------------------------------------------
PYTHON_SERIES="3.11"                       # pyproject: requires-python >=3.11,<3.12
EXPECTED_TORCH="2.7.0"                     # pyproject [gpu] extra, uv.lock: 2.7.0+cu126
EXPECTED_CUDA="12.6"                       # tool.uv.index pytorch-cu126
VLLM_IMAGE="vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
# Required command -> Ubuntu/Debian apt package that provides it (verified with dpkg -S).
declare -A APT_PACKAGE_FOR=(
  [git]=git
  [curl]=curl
  [tmux]=tmux
  [rsync]=rsync
  [findmnt]=util-linux
)
REQUIRED_COMMANDS=(git curl tmux rsync findmnt)

CPU_ONLY=0
PULL_VLLM=0
SKIP_TESTS=0
RUN_DOCTOR=1
ALLOW_EPHEMERAL=0

usage() { sed -n '2,25p' "$0" | sed -e 's/^# \{0,1\}//'; }

for arg in "$@"; do
  case "$arg" in
    --quick) SKIP_TESTS=1; RUN_DOCTOR=0 ;;
    --cpu-only) CPU_ONLY=1 ;;
    --allow-ephemeral) ALLOW_EPHEMERAL=1 ;;
    --pull-vllm) PULL_VLLM=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --no-doctor) RUN_DOCTOR=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Logging and failure handling
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_STEP=$'\033[1;34m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'; C_ERR=$'\033[1;31m'; C_END=$'\033[0m'
else
  C_STEP=""; C_OK=""; C_WARN=""; C_ERR=""; C_END=""
fi
CURRENT_STEP="preflight"
WARNINGS=()

step() { CURRENT_STEP="$1"; printf '\n%s==> %s%s\n' "$C_STEP" "$1" "$C_END"; }
ok()   { printf '%s  ok%s  %s\n' "$C_OK" "$C_END" "$*"; }
info() { printf '      %s\n' "$*"; }
warn() { WARNINGS+=("$*"); printf '%s  warn%s %s\n' "$C_WARN" "$C_END" "$*"; }
die()  { printf '%s  FAIL%s [%s] %s\n' "$C_ERR" "$C_END" "$CURRENT_STEP" "$*" >&2; exit 1; }
on_error() {
  local line=$1 cmd=$2
  printf '\n%sBootstrap failed%s during step "%s" (line %s):\n  %s\n' \
    "$C_ERR" "$C_END" "$CURRENT_STEP" "$line" "$cmd" >&2
  printf 'Fix the problem above and rerun: bash scripts/bootstrap_remote.sh\n' >&2
  exit 1
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

# Summary state, filled in by the steps below.
SUMMARY_PYTHON="not verified"
SUMMARY_TORCH="not installed (cpu-only)"
SUMMARY_GPU="not checked (cpu-only)"
SUMMARY_STORAGE="not checked"
SUMMARY_TESTS="skipped"
SUMMARY_CPU_DOCTOR="not run"
SUMMARY_PROD_DOCTOR="skipped"
SUMMARY_VLLM="not checked (cpu-only)"

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------
step "Preflight"
[[ "$(uname -s)" == "Linux" ]] || die "This bootstrap supports Linux only (got $(uname -s))."
ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" ]]; then
  # AGENTS.md §11: arm64 (GH200) is not used unless uv.lock is first verified on arm64.
  die "Unsupported architecture $ARCH: uv.lock is verified for x86_64 only (AGENTS.md §11)."
fi
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
[[ -f pyproject.toml && -f uv.lock && -f .python-version ]] \
  || die "Run this from a preanchor checkout; missing pyproject.toml/uv.lock in $REPO_ROOT."
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$REPO_ROOT is not a git checkout."
PINNED_SERIES=$(<.python-version)
[[ "$PINNED_SERIES" == "$PYTHON_SERIES" ]] \
  || die ".python-version ($PINNED_SERIES) disagrees with expected $PYTHON_SERIES."
export PATH="$HOME/.local/bin:$PATH"

if [[ $CPU_ONLY -eq 0 ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found; falling back to --cpu-only. Pass --cpu-only explicitly to silence this."
  CPU_ONLY=1
fi
if [[ $CPU_ONLY -eq 1 ]]; then MODE="cpu-only"; else MODE="gpu"; fi
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_COMMIT=$(git rev-parse --short HEAD)
ok "repo=$REPO_ROOT branch=$GIT_BRANCH commit=$GIT_COMMIT mode=$MODE"

# ---------------------------------------------------------------------------
# 1. System packages (install only what is missing; never upgrade anything)
# ---------------------------------------------------------------------------
step "System packages"
missing_cmds=()
for cmd in "${REQUIRED_COMMANDS[@]}"; do
  command -v "$cmd" >/dev/null 2>&1 || missing_cmds+=("$cmd")
done
if [[ ${#missing_cmds[@]} -eq 0 ]]; then
  ok "all required system tools present: ${REQUIRED_COMMANDS[*]}"
else
  packages=()
  for cmd in "${missing_cmds[@]}"; do packages+=("${APT_PACKAGE_FOR[$cmd]}"); done
  info "missing commands: ${missing_cmds[*]} -> apt packages: ${packages[*]}"
  command -v apt-get >/dev/null 2>&1 \
    || die "Missing tools (${missing_cmds[*]}) and no apt-get available; install: ${packages[*]}"
  if [[ $EUID -eq 0 ]]; then
    SUDO=()
  elif sudo -n true 2>/dev/null; then
    SUDO=(sudo -n)
  else
    die "Missing tools (${missing_cmds[*]}) require passwordless sudo for apt-get. Install: ${packages[*]}"
  fi
  export DEBIAN_FRONTEND=noninteractive
  "${SUDO[@]}" apt-get update -qq
  "${SUDO[@]}" apt-get install -y -qq --no-install-recommends "${packages[@]}"
  for cmd in "${missing_cmds[@]}"; do
    command -v "$cmd" >/dev/null 2>&1 \
      || die "apt-get installed ${APT_PACKAGE_FOR[$cmd]} but '$cmd' is still missing."
  done
  ok "installed: ${packages[*]}"
fi

# ---------------------------------------------------------------------------
# 2. Git submodules pinned in configs/locks/upstream.yaml
# ---------------------------------------------------------------------------
step "Vendored submodules"
git submodule update --init --recursive --quiet
# Verify each submodule sits exactly at the locked revision (what `preanchor doctor` checks).
submodule_count=0
while IFS=' ' read -r path revision; do
  [[ -e "$path/.git" ]] || die "Submodule $path was not initialised."
  actual=$(git -C "$path" rev-parse HEAD)
  [[ "$actual" == "$revision" ]] \
    || die "Submodule $path is at $actual, lock expects $revision. Run: git submodule update --init"
  [[ -z "$(git -C "$path" status --porcelain --untracked-files=all)" ]] \
    || die "Submodule $path has local modifications; doctor will reject it."
  ok "$path @ ${revision:0:12}"
  submodule_count=$((submodule_count + 1))
done < <(awk '
  /^  vendor\//      { sub(/:$/, "", $1); path=$1 }
  /^    revision:/   { print path, $2 }
' configs/locks/upstream.yaml)
[[ $submodule_count -gt 0 ]] || die "No submodules parsed from configs/locks/upstream.yaml."

# ---------------------------------------------------------------------------
# 3. uv (user-local install, no system modification)
# ---------------------------------------------------------------------------
step "uv package manager"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}') at $(command -v uv)"
else
  info "installing uv into ~/.local/bin"
  installer_url="https://astral.sh/uv/install.sh"
  [[ -n "${UV_INSTALL_VERSION:-}" ]] && installer_url="https://astral.sh/uv/${UV_INSTALL_VERSION}/install.sh"
  curl -LsSf "$installer_url" | env UV_NO_MODIFY_PATH=1 sh
  command -v uv >/dev/null 2>&1 || die "uv installer finished but uv is not on PATH ($HOME/.local/bin)."
  ok "uv $(uv --version | awk '{print $2}') installed"
fi

# ---------------------------------------------------------------------------
# 4. Python 3.11 (uv-managed; skipped when already installed)
# ---------------------------------------------------------------------------
step "Python $PYTHON_SERIES"
if python_path=$(uv python find "$PYTHON_SERIES" 2>/dev/null); then
  ok "python $PYTHON_SERIES present: $python_path"
else
  uv python install "$PYTHON_SERIES"
  ok "python $PYTHON_SERIES installed: $(uv python find "$PYTHON_SERIES")"
fi

# ---------------------------------------------------------------------------
# 5. Locked project environment (.venv) from uv.lock
# ---------------------------------------------------------------------------
step "Project environment (uv sync --frozen)"
SYNC_ARGS=(--frozen --group dev)
if [[ $CPU_ONLY -eq 0 ]]; then SYNC_ARGS+=(--extra gpu); fi
info "uv sync ${SYNC_ARGS[*]}"
# uv sync is idempotent: an up-to-date .venv is verified, not rebuilt.
uv sync "${SYNC_ARGS[@]}"
VENV_PY="$REPO_ROOT/.venv/bin/python"
[[ -x "$VENV_PY" ]] || die ".venv/bin/python missing after uv sync."
venv_version=$("$VENV_PY" -c 'import platform; print(platform.python_version())')
[[ "$venv_version" == "$PYTHON_SERIES".* ]] \
  || die ".venv uses Python $venv_version, expected $PYTHON_SERIES.x (delete .venv and rerun)."
SUMMARY_PYTHON="$venv_version ($VENV_PY)"
ok ".venv ready with Python $venv_version"

# ---------------------------------------------------------------------------
# 6. Configuration file and persistent directories
# ---------------------------------------------------------------------------
step "Environment file (.env)"
if [[ -f .env ]]; then
  ok ".env exists (left untouched)"
else
  cp .env.example .env
  ok ".env created from .env.example; fill HF_TOKEN/WANDB_API_KEY/backup settings as needed"
fi
chmod 600 .env
# Warn about keys present in the template but absent from an older .env.
while IFS= read -r key; do
  grep -qE "^${key}=" .env || warn ".env lacks ${key} (present in .env.example)"
done < <(grep -oE '^[A-Z_]+=' .env.example | tr -d =)

# Read the persistent-root values from .env without executing it.
env_value() { grep -E "^$1=" .env | tail -n1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'; }
PREANCHOR_ROOT_VALUE=$(env_value PREANCHOR_ROOT)
HF_HOME_VALUE=$(env_value HF_HOME)
WANDB_DIR_VALUE=$(env_value WANDB_DIR)
[[ -n "$PREANCHOR_ROOT_VALUE" ]] || die "PREANCHOR_ROOT is empty in .env."

step "Persistent experiment root"
PERSISTENT_READY=0
case "$PREANCHOR_ROOT_VALUE" in
  /lambda/nfs/*) ;;
  *) die "PREANCHOR_ROOT must be beneath /lambda/nfs (got $PREANCHOR_ROOT_VALUE)." ;;
esac
# Fail hard unless --allow-ephemeral: every stage writes runs, caches and logs here.
storage_problem() {
  SUMMARY_STORAGE="UNAVAILABLE: $1"
  if [[ $ALLOW_EPHEMERAL -eq 1 ]]; then
    warn "$1 (continuing because --allow-ephemeral was given; stages cannot run)"
  else
    die "$1 Attach the Lambda persistent filesystem, or pass --allow-ephemeral for a code-only setup."
  fi
}
persistent_parent=$(dirname "$PREANCHOR_ROOT_VALUE")
mount_target=""
if [[ -d "$persistent_parent" ]]; then
  mount_target=$(findmnt --target "$persistent_parent" -no TARGET 2>/dev/null || true)
fi
if [[ -z "$mount_target" || "$mount_target" == "/" ]]; then
  storage_problem "Persistent filesystem for $PREANCHOR_ROOT_VALUE is not mounted (no dedicated mount at $persistent_parent)."
elif [[ ! -w "$persistent_parent" && ! -w "$PREANCHOR_ROOT_VALUE" ]]; then
  storage_problem "Persistent filesystem $persistent_parent is mounted but not writable by $(id -un)."
else
  for dir in "$PREANCHOR_ROOT_VALUE" "$PREANCHOR_ROOT_VALUE/artifacts" \
             "$PREANCHOR_ROOT_VALUE/data" "${HF_HOME_VALUE:-$PREANCHOR_ROOT_VALUE/cache/huggingface}" \
             "${WANDB_DIR_VALUE:-$PREANCHOR_ROOT_VALUE/wandb}"; do
    mkdir -p "$dir" || die "Could not create $dir on the persistent filesystem."
  done
  mount_desc=$(findmnt --target "$persistent_parent" -no FSTYPE,OPTIONS | cut -d, -f1-2)
  free_gb=$(df -BG --output=avail "$PREANCHOR_ROOT_VALUE" | tail -n1 | tr -dc '0-9')
  PERSISTENT_READY=1
  SUMMARY_STORAGE="$PREANCHOR_ROOT_VALUE ($mount_desc, ${free_gb} GB free)"
  ok "$PREANCHOR_ROOT_VALUE ($mount_desc, ${free_gb} GB free)"
fi

# ---------------------------------------------------------------------------
# 7. Verify key Python dependencies against the pins in pyproject.toml
# ---------------------------------------------------------------------------
step "Dependency verification"
"$VENV_PY" - "$CPU_ONLY" <<'PY'
import importlib, importlib.metadata as md, re, sys, tomllib
cpu_only = sys.argv[1] == "1"
with open("pyproject.toml", "rb") as fh:
    project = tomllib.load(fh)["project"]
specs = list(project["dependencies"])
if not cpu_only:
    specs += project["optional-dependencies"]["gpu"]
failures = []
for spec in specs:
    m = re.match(r"^([A-Za-z0-9_.-]+)(?:==([^\s;]+))?", spec)
    name, pin = m.group(1), m.group(2)
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        failures.append(f"{name}: not installed"); continue
    if pin and installed.split("+")[0] != pin:
        failures.append(f"{name}: installed {installed}, pyproject pins {pin}")
try:
    md.version("preanchor")
except md.PackageNotFoundError:
    failures.append("preanchor: project package not installed")
for module in ("numpy", "pandas", "sklearn", "pydantic", "yaml", "huggingface_hub", "preanchor.cli"):
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"import {module} failed: {type(exc).__name__}: {exc}")
if failures:
    print("\n".join("  FAIL " + f for f in failures)); sys.exit(1)
print(f"      {len(specs)} pinned/required packages match pyproject.toml")
PY
ok "python dependencies match the lock"

# ---------------------------------------------------------------------------
# 8. GPU / CUDA / PyTorch verification (mirrors preanchor.doctor.check_gpu)
# ---------------------------------------------------------------------------
if [[ $CPU_ONLY -eq 0 ]]; then
  step "GPU, driver and CUDA PyTorch"
  smi=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1) \
    || die "nvidia-smi failed: $smi"
  info "nvidia-smi: $smi"
  gpu_report=$("$VENV_PY" - "$EXPECTED_TORCH" "$EXPECTED_CUDA" <<'PY'
import sys
import torch
want_torch, want_cuda = sys.argv[1:3]
problems = []
if torch.__version__.split("+")[0] != want_torch or torch.version.cuda != want_cuda:
    problems.append(f"torch {torch.__version__} (cuda {torch.version.cuda}); expected {want_torch}+cu{want_cuda.replace('.', '')}")
if not torch.cuda.is_available():
    problems.append("torch.cuda.is_available() is False (driver/runtime mismatch or no GPU visible)")
else:
    count = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    print(f"      torch {torch.__version__}, cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()}", file=sys.stderr)
    print(f"      {count} visible GPU(s): {props.name}, {props.total_memory / 1e9:.1f} GB", file=sys.stderr)
    if not torch.cuda.is_bf16_supported():
        problems.append("GPU does not support BF16")
    else:
        with torch.inference_mode():
            t = torch.ones((8, 8), dtype=torch.bfloat16, device="cuda:0")
            value = (t @ t).float().sum().item()
            torch.cuda.synchronize()
        if value != 512:
            problems.append(f"BF16 matmul returned {value}, expected 512")
        else:
            print("      BF16 CUDA kernel executed correctly", file=sys.stderr)
    if count != 1:
        problems.append(f"preanchor doctor requires exactly one visible GPU (found {count}); set CUDA_VISIBLE_DEVICES")
    if props.total_memory < 38 * 10**9:
        problems.append(f"GPU has {props.total_memory / 1e9:.1f} GB; doctor requires >= 38 GB")
for mod in ("transformers", "accelerate", "sentence_transformers"):
    try:
        __import__(mod)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"import {mod} failed: {type(exc).__name__}: {exc}")
if problems:
    print("\n".join("  FAIL " + p for p in problems), file=sys.stderr); sys.exit(1)
# Machine-readable summary line on stdout.
print(f"torch {torch.__version__} cuda {torch.version.cuda}|{props.name}, {props.total_memory / 1e9:.1f} GB, BF16 ok")
PY
  ) || die "GPU verification failed (see above)."
  SUMMARY_TORCH="${gpu_report%%|*}"
  SUMMARY_GPU="${gpu_report#*|} (driver $(echo "$smi" | awk -F', ' '{print $2}'))"
  ok "CUDA $EXPECTED_CUDA PyTorch $EXPECTED_TORCH runs on this GPU"
fi

# ---------------------------------------------------------------------------
# 9. Docker + NVIDIA container hook for the pinned vLLM server (generation stages)
# ---------------------------------------------------------------------------
if [[ $CPU_ONLY -eq 0 ]]; then
  step "Docker / vLLM image"
  if ! command -v docker >/dev/null 2>&1; then
    SUMMARY_VLLM="docker not installed"
    warn "docker not installed; scripts/start_vllm.sh needs Docker + nvidia-container-toolkit (Lambda images ship both)."
  elif ! sudo -n docker info >/dev/null 2>&1; then
    SUMMARY_VLLM="docker present but 'sudo -n docker' fails"
    warn "docker is installed but 'sudo -n docker info' fails; start_vllm.sh needs passwordless sudo docker."
  else
    command -v nvidia-container-runtime-hook >/dev/null 2>&1 \
      || warn "nvidia-container-runtime-hook missing; 'docker run --gpus all' will fail."
    if sudo -n docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
      SUMMARY_VLLM="pinned image present"
      ok "pinned vLLM image present"
    elif [[ $PULL_VLLM -eq 1 ]]; then
      info "pulling $VLLM_IMAGE (~29 GB)"
      sudo -n docker pull "$VLLM_IMAGE"
      SUMMARY_VLLM="pinned image pulled"
      ok "pinned vLLM image pulled"
    else
      SUMMARY_VLLM="pinned image ABSENT (rerun with --pull-vllm)"
      warn "pinned vLLM image absent; rerun with --pull-vllm before generation stages."
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 10. Project-level checks: CPU doctor, tests, production doctor
# ---------------------------------------------------------------------------
step "preanchor doctor --cpu-only"
uv run --frozen preanchor doctor --config configs/mvp.yaml --cpu-only >/dev/null \
  || die "CPU doctor failed; rerun 'uv run --frozen preanchor doctor --config configs/mvp.yaml --cpu-only' for the JSON report."
SUMMARY_CPU_DOCTOR="passed"
ok "cpu bootstrap doctor passed"

if [[ $SKIP_TESTS -eq 0 ]]; then
  step "CPU test suite"
  CUDA_VISIBLE_DEVICES="" uv run --frozen pytest -q tests/unit tests/integration \
    || die "Tests failed; see output above."
  SUMMARY_TESTS="passed (tests/unit tests/integration)"
  ok "unit + integration tests passed"
fi

if [[ $CPU_ONLY -eq 0 && $RUN_DOCTOR -eq 1 ]]; then
  step "preanchor doctor (production)"
  if [[ $PERSISTENT_READY -eq 0 ]]; then
    SUMMARY_PROD_DOCTOR="skipped (persistent root unavailable)"
    warn "skipped production doctor: persistent root unavailable."
  else
    # Export .env non-secrets for this check only; secrets stay in the file.
    set -a
    # shellcheck disable=SC1091
    source ./.env
    set +a
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    if uv run --frozen preanchor doctor --config configs/mvp.yaml >/dev/null; then
      SUMMARY_PROD_DOCTOR="passed (Stage 6 measured fit test still required)"
      ok "production doctor passed (Stage 6 measured fit test still required)"
    else
      SUMMARY_PROD_DOCTOR="FAILED"
      warn "production doctor failed; run 'set -a; source .env; set +a; uv run --frozen preanchor doctor --config configs/mvp.yaml' for details."
    fi
  fi
elif [[ $CPU_ONLY -eq 1 ]]; then
  SUMMARY_PROD_DOCTOR="skipped (cpu-only)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
printf '\n%s==> Bootstrap complete (%s mode)%s\n' "$C_OK" "$MODE" "$C_END"
printf '  %-20s %s\n' \
  "repository"        "$REPO_ROOT ($GIT_BRANCH @ $GIT_COMMIT)" \
  "python"            "$SUMMARY_PYTHON" \
  "torch / cuda"      "$SUMMARY_TORCH" \
  "gpu"               "$SUMMARY_GPU" \
  "persistent root"   "$SUMMARY_STORAGE" \
  "cpu doctor"        "$SUMMARY_CPU_DOCTOR" \
  "tests"             "$SUMMARY_TESTS" \
  "production doctor" "$SUMMARY_PROD_DOCTOR" \
  "vllm image"        "$SUMMARY_VLLM"
if [[ ${#WARNINGS[@]} -gt 0 ]]; then
  printf '%sWarnings (%d):%s\n' "$C_WARN" "${#WARNINGS[@]}" "$C_END"
  for w in "${WARNINGS[@]}"; do printf '  - %s\n' "$w"; done
fi
cat <<EOF

Next steps in every new shell:
  cd $REPO_ROOT
  set -a; source .env; set +a          # export PREANCHOR_ROOT, HF_HOME, WANDB_DIR, tokens
  uv run --frozen preanchor --help
EOF
