#!/usr/bin/env bash
set -euo pipefail

API_URL="${1:-${VITE_API_URL:-${API_URL:-}}}"

if [[ -z "${API_URL}" ]]; then
  echo "Usage: scripts/verify_prod.sh <lambda-url>" >&2
  echo "Or set VITE_API_URL or API_URL in your environment." >&2
  exit 1
fi

payload='{"sensor_readings":[[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9],[0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9]],"window_size":10,"threshold":0.001}'

echo "POST ${API_URL}"
response="$(curl -sS -X POST "${API_URL}" -H 'Content-Type: application/json' -d "${payload}")"

RESPONSE="${response}" python3 - <<'PY'
import json
import os
import sys

raw = os.environ.get("RESPONSE")
if not raw:
    print("No response body received", file=sys.stderr)
    sys.exit(1)

try:
    data = json.loads(raw)
except json.JSONDecodeError as exc:
    print(f"Response is not valid JSON: {exc}", file=sys.stderr)
    print(raw)
    sys.exit(1)

is_anomaly = data.get("is_anomaly") is True
diagnosis = data.get("diagnosis") or ""

if not is_anomaly:
    print("Expected is_anomaly=true but got:", data.get("is_anomaly"), file=sys.stderr)
    sys.exit(1)

if not diagnosis.strip():
    print("Expected non-empty diagnosis but got empty/None", file=sys.stderr)
    sys.exit(1)

print("OK: is_anomaly=true and diagnosis returned")
print("diagnosis:", diagnosis)
PY
