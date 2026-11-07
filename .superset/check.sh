#!/bin/sh
# Is Ollama up with the models Buddy needs? Says what is missing; never fails the setup.
if ! curl -s http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Ollama is not running: 'ollama serve' (or 'brew services start ollama')"; exit 0
fi
for m in gemma3:4b gemma4; do
  ollama list 2>/dev/null | grep -q "^$m" || echo "model $m missing: 'ollama pull $m'"
done
.venv/bin/python -c "import mujoco, cv2; print('venv ok: mujoco', mujoco.__version__)" 2>/dev/null || echo "venv not ready"
exit 0
