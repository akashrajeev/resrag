from __future__ import annotations

import subprocess
import sys


def test_lightweight_runtime_does_not_import_sentence_transformers():
    code = "import sys; import src.lightweight_runtime; assert 'sentence_transformers' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lightweight_manager_does_not_schedule_background_work_by_default(monkeypatch):
    monkeypatch.delenv("RESRAG_BACKGROUND_FULL_INDEX", raising=False)
    from src.lightweight_runtime import LightweightProgressiveIndexManager

    manager = LightweightProgressiveIndexManager()
    assert manager.executor is None
    assert manager._background_enabled is False
