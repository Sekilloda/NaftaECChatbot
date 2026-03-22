#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/Local/.venv/bin/python}"
PIP_BIN="${PIP_BIN:-$ROOT_DIR/Local/.venv/bin/pip}"
VALIDATION_WEBHOOK_SECRET="${VALIDATION_WEBHOOK_SECRET:-validate-secret}"

FAILED_STEPS=()

run_step() {
  local name="$1"
  shift
  echo
  echo "=== $name ==="
  if "$@"; then
    echo "[PASS] $name"
  else
    echo "[FAIL] $name"
    FAILED_STEPS+=("$name")
  fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found at: $PYTHON_BIN"
  echo "Set PYTHON_BIN=/path/to/python and rerun."
  exit 1
fi

if [[ ! -x "$PIP_BIN" ]]; then
  echo "Pip binary not found at: $PIP_BIN"
  echo "Set PIP_BIN=/path/to/pip and rerun."
  exit 1
fi

run_step "CompileAll" "$PYTHON_BIN" -m compileall -q "$ROOT_DIR/Local" "$ROOT_DIR/OptimizedBot"
run_step "PipCheck" "$PIP_BIN" check

run_step "LocalFlaskSmoke" bash -lc "cd '$ROOT_DIR/Local' && WEBHOOK_SECRET='$VALIDATION_WEBHOOK_SECRET' timeout 120s '$PYTHON_BIN' -c \"import app as m; c=m.app.test_client(); h={'X-Webhook-Secret':'$VALIDATION_WEBHOOK_SECRET'}; r1=c.get('/health'); assert r1.status_code==200, r1.status_code; r2=c.post('/webhook', headers=h, json={'event':'ping','data':{}}); assert r2.status_code==200, (r2.status_code, r2.get_json()); print('LOCAL_FLASK_OK')\""
run_step "LocalWebhookAuthEnforced" bash -lc "cd '$ROOT_DIR/Local' && WEBHOOK_SECRET='$VALIDATION_WEBHOOK_SECRET' timeout 120s '$PYTHON_BIN' -c \"import app as m; c=m.app.test_client(); r=c.post('/webhook', json={'event':'ping','data':{}}); assert r.status_code==401, (r.status_code, r.get_json()); print('LOCAL_WEBHOOK_AUTH_OK')\""
run_step "LocalWebhookFlowSmoke" bash -lc "cd '$ROOT_DIR/Local' && WEBHOOK_SECRET='$VALIDATION_WEBHOOK_SECRET' timeout 120s '$PYTHON_BIN' -c \"import app as m; m.client=None; m.send_whatsapp_message=lambda *_a, **_k: True; m.responder=lambda *_a, **_k: 'VALIDATION_OK'; c=m.app.test_client(); h={'X-Webhook-Secret':'$VALIDATION_WEBHOOK_SECRET'}; p={'event':'messages.upsert','data':{'messages':[{'key':{'fromMe':False,'remoteJid':'593900000001@s.whatsapp.net','id':'validate-local-flow'},'pushName':'Validator','message':{'conversation':'Hola, necesito informacion'}}]}}; r=c.post('/webhook', headers=h, json=p); body=r.get_json(); assert r.status_code==200, (r.status_code, body); assert body.get('status')=='ok', body; print('LOCAL_WEBHOOK_FLOW', r.status_code, body)\""

run_step "OptimizedFlaskSmoke" bash -lc "cd '$ROOT_DIR/OptimizedBot' && WEBHOOK_SECRET='$VALIDATION_WEBHOOK_SECRET' timeout 120s '$PYTHON_BIN' -c \"import app as m; c=m.app.test_client(); h={'X-Webhook-Secret':'$VALIDATION_WEBHOOK_SECRET'}; r1=c.get('/health'); assert r1.status_code==200, r1.status_code; r2=c.post('/webhook', headers=h, json={'event':'ping','data':{}}); assert r2.status_code==200, (r2.status_code, r2.get_json()); print('OPT_FLASK_OK')\""
run_step "OptimizedWebhookAuthEnforced" bash -lc "cd '$ROOT_DIR/OptimizedBot' && WEBHOOK_SECRET='$VALIDATION_WEBHOOK_SECRET' timeout 120s '$PYTHON_BIN' -c \"import app as m; c=m.app.test_client(); r=c.post('/webhook', json={'event':'ping','data':{}}); assert r.status_code==401, (r.status_code, r.get_json()); print('OPT_WEBHOOK_AUTH_OK')\""
run_step "OptimizedWebhookFlowSmoke" bash -lc "cd '$ROOT_DIR/OptimizedBot' && WEBHOOK_SECRET='$VALIDATION_WEBHOOK_SECRET' timeout 120s '$PYTHON_BIN' -c \"import app as m; m.client=None; m.send_whatsapp_message=lambda *_a, **_k: True; m.responder=lambda *_a, **_k: 'VALIDATION_OK'; c=m.app.test_client(); h={'X-Webhook-Secret':'$VALIDATION_WEBHOOK_SECRET'}; p={'event':'messages.upsert','data':{'messages':[{'key':{'fromMe':False,'remoteJid':'593900000001@s.whatsapp.net','id':'validate-opt-flow'},'pushName':'Validator','message':{'conversation':'Hola, necesito informacion'}}]}}; r=c.post('/webhook', headers=h, json=p); body=r.get_json(); assert r.status_code==200, (r.status_code, body); assert body.get('status')=='ok', body; print('OPT_WEBHOOK_FLOW', r.status_code, body)\""

run_step "LocalResponderSmoke" bash -lc "cd '$ROOT_DIR/Local' && timeout 180s '$PYTHON_BIN' -c \"from core.knowledge import responder; out=responder('Hola, soy Carlos Ruiz. Quiero confirmar mi inscripcion.', sender_jid='593900000001@s.whatsapp.net', history=[]); print('LOCAL_RESP', (out or '')[:120]); assert isinstance(out, str) and len(out) > 0 and 'ERROR_TECHNICAL' not in out\""

run_step "OptimizedResponderSmoke" bash -lc "cd '$ROOT_DIR/OptimizedBot' && timeout 180s '$PYTHON_BIN' -c \"from core.knowledge import responder; out=responder('Hola, soy Carlos Ruiz. Quiero confirmar mi inscripcion.', sender_jid='593900000001@s.whatsapp.net', history=[]); print('OPT_RESP', (out or '')[:120]); assert isinstance(out, str) and len(out) > 0 and 'ERROR_TECHNICAL' not in out\""

run_step "LocalSimRun" bash -lc "cd '$ROOT_DIR/Local' && timeout 240s '$PYTHON_BIN' simulation/run_simulation.py"
run_step "OptimizedSimRun" bash -lc "cd '$ROOT_DIR/OptimizedBot' && timeout 240s '$PYTHON_BIN' simulation/run_simulation.py"
run_step "OptimizedLocalSimRun" bash -lc "cd '$ROOT_DIR/OptimizedBot' && timeout 300s '$PYTHON_BIN' simulation/run_local_simulation.py"
run_step "LocalValidationEngineQuick" bash -lc "cd '$ROOT_DIR/Local' && timeout 240s '$PYTHON_BIN' -c \"from simulation.validation_engine import run_simulation; run_simulation(num_personas=1, convs_per_persona=1); print('VALIDATION_ENGINE_OK')\""

if command -v sqlite3 >/dev/null 2>&1; then
  run_step "LocalSimDbIntegrity" bash -lc "sqlite3 '$ROOT_DIR/Local/simulation/simulation_results.db' \"select case when (select count(*) from test_conversations where evaluation_score is null)=0 and (select coalesce(avg(evaluation_score),0) from test_conversations)>=5 then 'OK' else 'FAIL' end;\" | grep -q '^OK$'"
  run_step "OptimizedSimDbIntegrity" bash -lc "sqlite3 '$ROOT_DIR/OptimizedBot/simulation/simulation_results.db' \"select case when (select count(*) from test_conversations where evaluation_score is null)=0 and (select coalesce(avg(evaluation_score),0) from test_conversations)>=5 then 'OK' else 'FAIL' end;\" | grep -q '^OK$'"
  run_step "ValidationEngineDbIntegrity" bash -lc "sqlite3 '$ROOT_DIR/Local/simulation/test_dataset.db' \"select case when (select count(*) from conversations where evaluation_score is null)=0 and (select coalesce(avg(evaluation_score),0) from conversations)>=5 then 'OK' else 'FAIL' end;\" | grep -q '^OK$'"
else
  echo
  echo "sqlite3 not found, skipping DB integrity checks."
fi

echo
if (( ${#FAILED_STEPS[@]} > 0 )); then
  echo "Validation finished with failures (${#FAILED_STEPS[@]}):"
  for step in "${FAILED_STEPS[@]}"; do
    echo " - $step"
  done
  exit 1
fi

echo "Validation finished successfully."
exit 0
