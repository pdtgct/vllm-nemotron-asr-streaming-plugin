# SPDX-License-Identifier: Apache-2.0
"""The synthetic card has the published card's shape, and the support
helpers that decide what a test asserts (condition matchers, traps, the
ownership walk) find what they claim to find."""

import json

from support.card import BLANK, LOCALES, NUM_PROMPTS


def test_synthetic_card_shape(card_dir):
    from transformers import AutoTokenizer

    config = json.loads((card_dir / "config.json").read_text())
    assert config["vocab_size"] == BLANK + 1
    assert len(set(LOCALES.values())) < len(LOCALES) < NUM_PROMPTS + len(LOCALES)
    tok = AutoTokenizer.from_pretrained(card_dir)
    assert tok.vocab_size == BLANK
    assert tok.convert_ids_to_tokens(BLANK) == "<pad>"
    assert tok.convert_ids_to_tokens(BLANK + 1) == "<blank>"
    # As on the published card: the language tag and <pad> are listed special,
    # <blank> is flagged special in the backend but not listed.
    assert {1, BLANK} <= set(tok.all_special_ids)
    assert BLANK + 1 not in tok.all_special_ids
    assert tok.decode([BLANK + 1], skip_special_tokens=True) == ""


# ---------------------------------------------------------------------------
# The harness's own matchers and walks


def test_a_current_setting_is_a_fact_not_a_remedy():
    """TXG-MEM-010's budget setting names an option with its current value;
    only a suggested change counts as a configuration remedy."""
    from support import txg_conditions as c

    settings = {"--kv-cache-memory-bytes": 1 << 30, "--max-num-seqs": 3}
    facts = (
        "pool access failed; budget setting: --kv-cache-memory-bytes 1073741824",
        "pool access failed; budget setting: --kv-cache-memory-bytes fixed at "
        "1073741824",
        "budget setting: a fixed budget (--kv-cache-memory-bytes 1073741824)",
        "max_num_seqs=3 rows",
        "the layout names nemotron.other_state; --max-num-seqs is 3",
    )
    for fact in facts:
        # A statement of the configuration, with or without the settings.
        assert c.configuration_remedies(fact, settings) == [], fact
        assert c.configuration_remedies(fact) == [], fact
    change = facts[1] + "; raise --num-gpu-blocks-override to at least 5"
    assert c.configuration_remedies(change, settings) == ["--num-gpu-blocks-override"]
    # A clause that states a setting and suggests a change: the stated
    # setting is the fact, the other option the remedy.
    mixed = (
        "--kv-cache-memory-bytes fixed at 1073741824 holds 3 pages, so raise "
        "--num-gpu-blocks-override"
    )
    assert c.configuration_remedies(mixed, settings) == ["--num-gpu-blocks-override"]
    assert "--max-num-seqs" in c.configuration_remedies(
        "or lower --max-num-seqs", settings
    )
    # The plugin's remedies are noun phrases (TXG-MEM-011 to TXG-MEM-013).
    assert c.configuration_remedies("a smaller --kv-cache-memory-bytes") == [
        "--kv-cache-memory-bytes"
    ]
    assert c.configuration_remedies(
        "remedies: --num-gpu-blocks-override or --kv-cache-memory-bytes"
    ) == ["--num-gpu-blocks-override", "--kv-cache-memory-bytes"]
    assert c.configuration_remedies(
        "a --num-gpu-blocks-override of at least 9", settings
    ) == ["--num-gpu-blocks-override"]


def test_state_layer_refusals_are_the_clauses_refusing_the_option_for_it():
    from support import txg_conditions as c

    accessor = (
        "transaction_graph_populations [4] needs ModelState.state_group_layouts, "
        "which the running core lacks; remove transaction_graph_populations"
    )
    proof_of_concept = (
        "transaction_graph_populations (transaction graphs) is not supported on "
        "the PR 6 state layer in this proof of concept; remove it"
    )
    assert c.state_layer_refusals(accessor) == [accessor.split(";")[0]]
    assert c.is_state_layer_condition(accessor)
    assert c.state_layer_refusals(proof_of_concept) == [proof_of_concept.split(";")[0]]
    assert not c.is_state_layer_condition(proof_of_concept)
    other = "experimental_native_burst is not supported on the state layer"
    assert c.state_layer_refusals(other) == []


def test_trapped_layouts_record_every_use_and_fail_while_armed():
    from types import SimpleNamespace

    import pytest
    from support import state_layer as sl

    layer = SimpleNamespace(component_pools={}, prefix="layer")
    model = SimpleNamespace(_persistent_state_layer=layer)
    state = SimpleNamespace(_state_group_layouts=("a layout",))
    trap = sl.trap_views(model, None, state)
    assert list(state._state_group_layouts) == ["a layout"]
    assert trap.reads and set(trap.reads) == {"state_group_layouts"}
    trap.armed = True
    for use in (len, list, bool, lambda layouts: layouts[0]):
        with pytest.raises(AssertionError, match="state_group_layouts"):
            use(state._state_group_layouts)


def _core_bound_layer(*shapes):
    """A state layer bound as core binds one: ``MambaBase.bind_kv_cache``
    unpacks one raw ``[B, 1, 1, C]`` int8 KV cache into per-state float32
    views, so every component tensor is a view of an intermediate tensor
    (its ``_base``), and a view derived from a component has that
    intermediate, not the component, as its ``_base``. Returns the layer and
    the raw tensor."""
    from math import prod
    from types import SimpleNamespace

    import torch
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    layer = SimpleNamespace(
        prefix="nemotron.state",
        state_alignment=8,
        get_state_shape=lambda: shapes,
        get_state_dtype=lambda: tuple(torch.float32 for _ in shapes),
    )
    page = sum(-(-prod(shape) * 4 // 8) * 8 for shape in shapes)
    raw = torch.zeros(4, 1, 1, page, dtype=torch.int8)
    MambaBase.bind_kv_cache(layer, raw)
    return layer, raw


def _holds_env(layer):
    """A model whose state layer is ``layer`` and whose plugin object
    ``binding`` keeps ``binding.kept``; its model state carries core's
    state-group layouts over the layer's bound views."""
    from types import SimpleNamespace

    from support import txgcap_harness as t

    binding = SimpleNamespace(kept=None)
    model = SimpleNamespace(_persistent_state_layer=layer, binding=binding)
    state = SimpleNamespace(
        _state_group_layouts=t.make_layouts({layer.prefix: layer.kv_cache}),
        model=model,
    )
    return SimpleNamespace(model=model, state=state), binding


def test_plugin_holds_finds_what_the_plugin_keeps_and_skips_cores(monkeypatch):
    """``txgcap_harness.plugin_holds`` reports a plugin object (or a plugin
    module's namespace) that keeps an accessor object, or any tensor sharing
    a component tensor's storage, and nothing that only core's attribute,
    the state layer or the harness keeps. Components are core-shaped (views
    of one raw tensor), so no view derived from one has a target as its
    ``_base``: only their shared storage connects them."""
    import pytest
    import torch

    pytest.importorskip("vllm.v1.worker.utils")
    pytest.importorskip("vllm.model_executor.layers.mamba.abstract")
    from support import txgcap_harness as t

    from vllm_nemotron_asr_streaming.nemotron_asr import advance

    layer, raw = _core_bound_layer((2,), (3,))
    env, binding = _holds_env(layer)
    state = env.state
    targets = t.accessor_objects(state)
    assert len(targets) == 1 + 1 + 2 * 2
    assert t.plugin_holds(env, targets) == []
    layout = state._state_group_layouts[0]
    (components,) = layout.layers.values()
    component = components[0].tensor
    derived = (
        components[1].tensor[1:],
        component.detach(),
        component.view(torch.int32),
    )
    target_ids = {id(obj) for _, obj in targets}
    for d in derived:
        # Neither the tensor nor its ``_base`` is a target, yet each views
        # the raw tensor's storage.
        assert id(d) not in target_ids and id(d._base) not in target_ids, d
        assert d.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    for kept in (
        state._state_group_layouts,
        layout,
        components[1],
        component,
        *derived,
        {"pools": [component[2:]]},
    ):
        binding.kept = kept
        held = t.plugin_holds(env, targets)
        assert len(held) == 1 and held[0].startswith("model -> "), (kept, held)
    binding.kept = component.clone()  # a copy holds none of the pool
    assert t.plugin_holds(env, targets) == []
    binding.kept = t.AccessorSpy  # a harness object is not the plugin's
    assert t.plugin_holds(env, targets) == []
    binding.kept = None
    # The plugin's module-level state is the plugin's too.
    monkeypatch.setattr(advance, "_batch_stats_slot", [("rows", component[1:])])
    held = t.plugin_holds(env, targets)
    assert len(held) == 1 and held[0].startswith(f"module {advance.__name__} -> "), held


def test_plugin_holds_finds_a_flattened_view_of_a_component(monkeypatch):
    """A whole-page component (one state per page) is contiguous, so its
    ``flatten()`` is a view, whose ``_base`` is not the component; the walk
    finds it, and finds nothing once the plugin keeps none."""
    import pytest

    pytest.importorskip("vllm.v1.worker.utils")
    pytest.importorskip("vllm.model_executor.layers.mamba.abstract")
    from support import txgcap_harness as t

    layer, raw = _core_bound_layer((2, 4))
    env, binding = _holds_env(layer)
    targets = t.accessor_objects(env.state)
    (component,) = layer.kv_cache
    flat = component.flatten()
    assert flat is not component and flat._base is not component
    assert flat.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    binding.kept = flat
    held = t.plugin_holds(env, targets)
    assert len(held) == 1 and "shares storage" in held[0], held
    binding.kept = None
    assert t.plugin_holds(env, targets) == []
