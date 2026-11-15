# SPDX-License-Identifier: Apache-2.0
"""Structure of the CUDA test harness, checked on CPU.

The ``gpu`` modules run only on a CUDA device, so a missing harness method
would surface only there. These checks read those modules' source and
require every name they use on ``support.txg_gpu`` (the module, ``GpuEnv``
and ``Driver``) to exist. They test the harness, not a spec.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from support import txg_gpu as g

GPU_MODULES = sorted(Path(__file__).parent.glob("test_*_gpu.py"))
ENV_NAMES = {"env", "same_cells"}
DRIVER_NAMES = {"d", "driver"}


def used_attributes(names: set[str]) -> set[tuple[str, str]]:
    found = set()
    for path in GPU_MODULES:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in names
            ):
                found.add((path.name, node.attr))
    return found


def dummy_env() -> g.GpuEnv:
    model = SimpleNamespace(config=SimpleNamespace())
    return g.GpuEnv(
        vllm_config=None,
        model=model,
        state=None,
        runner=None,
        lookahead=0,
        n=2,
        capacity=6,
    )


def test_gpu_modules_exist():
    assert GPU_MODULES


@pytest.mark.parametrize("module", GPU_MODULES, ids=lambda p: p.name)
def test_every_module_level_harness_name_exists(module):
    for node in ast.walk(ast.parse(module.read_text())):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "g"
        ):
            assert hasattr(g, node.attr), f"{module.name}: txg_gpu.{node.attr}"


def test_every_environment_attribute_exists():
    env = dummy_env()
    for module, attr in sorted(used_attributes(ENV_NAMES)):
        assert hasattr(type(env), attr) or attr in vars(env), f"{module}: GpuEnv.{attr}"


def test_every_driver_attribute_exists():
    driver = g.Driver(dummy_env())
    for module, attr in sorted(used_attributes(DRIVER_NAMES)):
        assert hasattr(type(driver), attr) or attr in vars(driver), (
            f"{module}: Driver.{attr}"
        )


def test_methods_the_tests_call_are_methods():
    for name in (
        "compare",
        "spy_replays",
        "silent",
        "option_off",
        "shutdown",
        "counters",
    ):
        assert callable(getattr(g.GpuEnv, name, None)), name
    for name in (
        "step",
        "run",
        "until_parked",
        "lockstep",
        "special",
        "add_stream",
        "add_file",
        "parked",
    ):
        assert callable(getattr(g.Driver, name, None)), name
