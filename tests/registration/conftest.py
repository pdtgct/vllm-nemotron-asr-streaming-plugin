# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the registration tests.

Test modules cannot import this file (``--import-mode=importlib``), so helpers
are exposed as fixtures.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]

PARSER_FORMAT = "nemotron_asr_streaming"
TOKENIZER_MODE = "nemotron_asr_streaming"
ARCH = "Nemotron3_5AsrStreamingForRNNT"
OFFLINE_ARCH = "Nemotron3_5AsrForRNNT"

#: Config-key names of the five control tokens, in ``blank + 1 .. blank + 5``
#: order. Only ``eos_token_id`` is named by the specs; the other four follow
#: the names the existing configuration class uses for the stream carrier and
#: EOU tokens, and extend them to the two new tokens (see the report).
CONTROL_FIELDS = (
    "eos_token_id",
    "audio_chunk_token_id",
    "eou_token_id",
    "file_carrier_token_id",
    "continuation_token_id",
)


@pytest.fixture
def repo_root() -> Path:
    return REPO


@pytest.fixture
def control_fields() -> tuple[str, ...]:
    return CONTROL_FIELDS


# --------------------------------------------------------------------------
# Fresh-process helper (REG-002, REG-007, REG-009, REG-010, REG-012, REG-013)
# --------------------------------------------------------------------------

_PRELUDE = """
import json, sys, warnings
warnings.filterwarnings("ignore")

def emit(**kw):
    print("RESULT:" + json.dumps(kw, default=str))
"""


@pytest.fixture
def run_fresh() -> Callable[..., dict[str, Any]]:
    """Run a snippet in a new interpreter and return the dict it ``emit``-ed."""

    def run(code: str, *, env: dict[str, str] | None = None, timeout: int = 600):
        full_env = dict(os.environ)
        full_env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO), *filter(None, [full_env.get("PYTHONPATH")])]
        )
        full_env["HF_HUB_OFFLINE"] = "1"
        full_env.update(env or {})
        proc = subprocess.run(
            [sys.executable, "-c", _PRELUDE + textwrap.dedent(code)],
            capture_output=True,
            text=True,
            env=full_env,
            timeout=timeout,
        )
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")]
        if proc.returncode != 0 or not lines:
            pytest.fail(
                f"fresh interpreter exited {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout[-2000:]}\n"
                f"--- stderr ---\n{proc.stderr[-4000:]}"
            )
        return json.loads(lines[-1][len("RESULT:") :])

    return run


# --------------------------------------------------------------------------
# Engine-tier fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def registered():
    """Run the plugin's ``register()`` once for the process (idempotent)."""
    from vllm_nemotron_asr_streaming import register

    register()
    return True


def require_registered() -> None:
    """Fail (not error, and not as an expected ``ValueError``) when the plugin has
    not registered its parser and tokenizer mode, so that tests expecting a
    refusal cannot pass because the format is simply unknown."""
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.transformers_utils.config import _CONFIG_FORMAT_TO_CONFIG_PARSER

    if PARSER_FORMAT not in _CONFIG_FORMAT_TO_CONFIG_PARSER:
        pytest.fail(f"config format {PARSER_FORMAT!r} is not registered")
    if TOKENIZER_MODE not in TokenizerRegistry.tokenizers:
        pytest.fail(f"tokenizer mode {TOKENIZER_MODE!r} is not registered")


@pytest.fixture
def require_plugin(registered) -> Callable[[], None]:
    return require_registered


@pytest.fixture
def parse_card(registered) -> Callable[..., tuple[dict, Any]]:
    """Call the registered card parser as core does (core calls with kwargs)."""

    def parse(path, **kwargs):
        from vllm.transformers_utils.config import get_config_parser

        require_registered()

        return get_config_parser(PARSER_FORMAT).parse(
            str(path), trust_remote_code=False, **kwargs
        )

    return parse


@pytest.fixture
def make_model_config(registered, monkeypatch) -> Callable[..., Any]:
    """Build a real ``ModelConfig`` on the card with the plugin's parser.

    The lazy model class is inspected by core in a child process, which needs
    the plugin importable there; the repository root goes on ``PYTHONPATH``.
    ``hf_overrides`` always carries ``num_lookahead_tokens`` (the Hub card has
    no default lookahead) unless ``lookahead=None``.
    """
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])]),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def make(path, *, overrides=None, lookahead: int | None = 3, **kwargs):
        from vllm.config import ModelConfig

        require_registered()
        hf_overrides: dict[str, Any] = {}
        if lookahead is not None:
            hf_overrides["num_lookahead_tokens"] = lookahead
        hf_overrides.update(copy.deepcopy(overrides or {}))
        kw: dict[str, Any] = {
            "model": str(path),
            "config_format": PARSER_FORMAT,
            "hf_overrides": hf_overrides,
            "enforce_eager": True,
            "dtype": "float32",
        }
        kw.update(kwargs)
        return ModelConfig(**kw)

    return make


@pytest.fixture
def make_vllm_config(registered) -> Callable[..., Any]:
    """A real ``VllmConfig`` on CPU around a ``ModelConfig``.

    ``VllmConfig`` runs the plugin's engine hook while constructing, so only
    acceptable engine settings construct; tests mutate the result afterwards to
    reach the model-state re-check.
    """

    def make(model_config, *, async_scheduling=False, prefix_caching=False):
        from vllm.config import (
            CacheConfig,
            DeviceConfig,
            SchedulerConfig,
            VllmConfig,
        )

        return VllmConfig(
            model_config=model_config,
            device_config=DeviceConfig(device="cpu"),
            cache_config=CacheConfig(enable_prefix_caching=prefix_caching),
            scheduler_config=SchedulerConfig.default_factory(
                async_scheduling=async_scheduling
            ),
        )

    return make


@pytest.fixture
def engine_config() -> Callable[..., SimpleNamespace]:
    """A ``VllmConfig``-shaped namespace for ``verify_and_update_config``.

    Attribute paths are the ones core's own persistent-state validation reads
    (``validate_state_config`` in ``vllm/v1/core/state_slot_manager.py``) plus
    ``model_config.enforce_eager``. Defaults are the accepted configuration.
    ``drop`` names ``parallel_config`` attributes to leave undefined.
    """

    def make(*, drop: tuple[str, ...] = (), **settings: Any) -> SimpleNamespace:
        values: dict[str, Any] = {
            "async_scheduling": False,
            "enable_prefix_caching": False,
            "speculative_config": None,
            "kv_transfer_config": None,
            "enable_dbo": False,
            "enable_sleep_mode": False,
            "enable_cumem_allocator": False,
            "hisparse_config": None,
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1,
        }
        unknown = set(settings) - set(values)
        assert not unknown, unknown
        values.update(settings)
        parallel = {
            name: values[name]
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
                "enable_dbo",
            )
            if name not in drop
        }
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(
                async_scheduling=values["async_scheduling"]
            ),
            cache_config=SimpleNamespace(
                enable_prefix_caching=values["enable_prefix_caching"]
            ),
            parallel_config=SimpleNamespace(**parallel),
            model_config=SimpleNamespace(
                enable_sleep_mode=values["enable_sleep_mode"],
                enable_cumem_allocator=values["enable_cumem_allocator"],
                enforce_eager=values["enforce_eager"],
            ),
            speculative_config=values["speculative_config"],
            kv_transfer_config=values["kv_transfer_config"],
            attention_config=SimpleNamespace(hisparse_config=values["hisparse_config"]),
        )

    return make


@pytest.fixture
def write_translated(registered, parse_card) -> Callable[..., Path]:
    """Write an already-translated checkpoint directory.

    Takes the parser's own translation of the synthetic card, applies
    ``mutate`` (a callable editing the dict), and writes it as ``config.json``
    beside the unchanged ``processor_config.json``, generation configuration
    and tokenizer.
    """

    def write(path: Path, mutate=None, **card_kwargs) -> Path:
        from support.card import write_card

        write_card(path, **card_kwargs)
        translated, _ = parse_card(path)
        translated = json.loads(json.dumps(translated, default=list))
        if mutate is not None:
            mutate(translated)
        (path / "config.json").write_text(json.dumps(translated))
        return path

    return write


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=1)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured_logs():
    """Capture log records from the plugin, vLLM and the root logger."""
    handler = _Capture()
    targets = [
        logging.getLogger(),
        logging.getLogger("vllm"),
        logging.getLogger("vllm_nemotron_asr_streaming"),
    ]
    saved = [(t, t.level) for t in targets]
    for target in targets:
        target.addHandler(handler)
        target.setLevel(min(target.level or logging.INFO, logging.INFO))
    try:
        yield handler
    finally:
        for target, level in saved:
            target.removeHandler(handler)
            target.setLevel(level)


@pytest.fixture
def build_model_state(registered) -> Callable[..., Any]:
    """Construct the model's real ``ModelState`` on CPU around ``vllm_config``.

    The state class is found as core finds it (``resolve_model_state_cls`` in
    ``vllm/v1/worker/gpu/model_states/__init__.py`` calls the model's
    ``get_model_state_cls``). The model instance is the real class without
    weights; ``ModelState.__init__`` (``vllm/v1/worker/gpu/model_states/
    interface.py``) only reads configuration.
    """

    def build(vllm_config):
        import torch

        from vllm_nemotron_asr_streaming.model import (
            Nemotron3_5AsrStreamingForRNNT as model_cls,
        )

        model = model_cls.__new__(model_cls)
        torch.nn.Module.__init__(model)
        state_cls = model_cls.get_model_state_cls()
        return state_cls(vllm_config, model, None, torch.device("cpu"))

    return build
