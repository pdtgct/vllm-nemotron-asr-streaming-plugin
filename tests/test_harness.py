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


def test_an_option_named_with_another_value_is_a_remedy():
    """An option named with a value other than its current one suggests that
    value, whatever the phrasing (a comparison, a wish, no imperative), so it
    is a configuration remedy; named with its current value, in any byte
    unit that renders it, it states the configuration. An option the
    settings leave unset differs from every value. Without the settings no
    value can be told from the current one."""
    from support import txg_conditions as c

    settings = {"--kv-cache-memory-bytes": 1 << 30, "--max-num-seqs": 3}
    changes = {
        "--num-gpu-blocks-override >= 5": ["--num-gpu-blocks-override"],
        "you may need --num-gpu-blocks-override 9": ["--num-gpu-blocks-override"],
        "num_gpu_blocks_override=9 would hold them": ["--num-gpu-blocks-override"],
        "--max-num-seqs 2 fits": ["--max-num-seqs"],
        "--max-num-active-seqs: 2": ["--max-num-active-seqs"],
        "kv_cache_memory_bytes = 2 GiB": ["--kv-cache-memory-bytes"],
        "--kv-cache-memory-bytes 1073741825": ["--kv-cache-memory-bytes"],
    }
    for message, remedies in changes.items():
        assert c.configuration_remedies(message, settings) == remedies, message
        assert c.changed_settings(message, settings) == remedies, message
        # Not a suggested change in words: only the settings tell it apart.
        assert c.configuration_remedies(message) == [], message
    facts = (
        "--kv-cache-memory-bytes 1073741824 bytes (1024.00 MiB)",
        "kv_cache_memory_bytes=1.00 GiB",
        "kv_cache_memory_bytes = 1024 MiB",
        "--max-num-seqs (3) rows",
        "max_num_seqs is 3",
        "the state pools hold 3 blocks",
    )
    for fact in facts:
        assert c.changed_settings(fact, settings) == [], fact
        assert c.configuration_remedies(fact, settings) == [], fact
    # A stated setting beside a changed one: only the changed one is a remedy.
    mixed = "budget --kv-cache-memory-bytes 1073741824; needs --max-num-seqs 2"
    assert c.configuration_remedies(mixed, settings) == ["--max-num-seqs"]


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


def test_plugin_holds_walks_the_state_layer_permitting_only_its_views(monkeypatch):
    """The walk covers the state layer: there only its bound ``kv_cache`` and
    the objects its ``pages`` and ``component_pools`` return (its view cache,
    REG-STATE-018) are the layer's. A pool slice, or a container of one,
    stored on the layer is found; a copy on the layer is not. The same view
    objects held by a plugin object off the layer are found."""
    from types import SimpleNamespace

    import pytest

    pytest.importorskip("vllm.v1.worker.utils")
    pytest.importorskip("vllm.model_executor.layers.mamba.abstract")
    from support import txgcap_harness as t

    layer, raw = _core_bound_layer((2,), (3,))
    # The layer's view cache: its own view objects over the bound storage.
    layer.component_pools = {
        name: state.view(state.shape)
        for name, state in zip(("first", "second"), layer.kv_cache, strict=True)
    }
    layer.pages = raw.view(raw.shape[0], -1)
    env, binding = _holds_env(layer)
    targets = t.accessor_objects(env.state)
    permitted = {id(obj) for obj in t.layer_permitted(layer)}
    own = (layer.kv_cache, layer.pages, layer.component_pools)
    assert {id(obj) for obj in (*own, *layer.component_pools.values())} == permitted
    assert t.plugin_holds(env, targets) == []
    view = layer.component_pools["first"]
    for kept in (view[1:3], view.detach(), {"pools": [view[2:]]}, [layer.pages[1]]):
        layer.txg_kept = kept
        held = t.plugin_holds(env, targets)
        assert len(held) == 1 and held[0].startswith("model -> "), (kept, held)
        assert "shares storage" in held[0], held
    layer.txg_kept = view.clone()  # a copy holds none of the pool
    assert t.plugin_holds(env, targets) == []
    del layer.txg_kept
    # The layer's own objects, held by a plugin object off the layer.
    for kept in (view, layer.component_pools, layer.pages, layer.kv_cache):
        binding.kept = kept
        held = t.plugin_holds(env, targets)
        assert held and all(line.startswith("model -> ") for line in held), (
            kept,
            held,
        )
    binding.kept = None
    assert t.plugin_holds(env, targets) == []
    # An unbound layer permits only its (empty) kv_cache, not a view it
    # might still hold.
    unbound = SimpleNamespace(kv_cache=[], component_pools={"first": view})
    assert [id(obj) for obj in t.layer_permitted(unbound)] == [id(unbound.kv_cache)]


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


# ---------------------------------------------------------------------------
# The block-access reading of TXG-CAP-018


def _scatter_op():
    """A CPU op named as the plugin's masked page scatter (its CUDA op is
    registered only on CUDA), with the same arguments and row predicate."""
    import torch
    from support import txgcap_harness as t
    from torch.library import custom_op

    name = f"txgharness::{t._MASKED_SCATTER}"
    namespace = getattr(torch.ops, "txgharness", None)
    if namespace is not None and hasattr(namespace, t._MASKED_SCATTER):
        return getattr(namespace, t._MASKED_SCATTER)

    @custom_op(name, mutates_args=["pool"])
    def scatter(
        pool: torch.Tensor,
        scratch: torch.Tensor,
        block_ids: torch.Tensor,
        row_status: torch.Tensor,
    ) -> None:
        for row in range(scratch.shape[0]):
            if int(row_status[row]) == 0:
                pool[int(block_ids[row])].copy_(scratch[row])

    return getattr(torch.ops.txgharness, t._MASKED_SCATTER)


def test_block_accesses_classify_the_masked_scatter_by_its_stored_rows():
    """The masked page scatter op addresses the block IDs of its rows whose
    status is 0, and nothing for rejected rows, which return before they
    read a block ID; an op the reading does not list addresses the whole
    view it takes."""
    import torch
    from support import txgcap_harness as t

    op = _scatter_op()
    pool = torch.zeros(6, 1, 1, 16, dtype=torch.uint8)
    view = pool.view(6, 16)
    scratch = torch.ones(3, 16, dtype=torch.uint8)
    cases = (
        ([0, 3, 4], [1, 0, 2], (3,)),
        ([0, 0, 0], [1, 2, 4], ()),
        ([5, 2, 1], [0, 0, 1], (2, 5)),
    )
    for ids, status, expected in cases:
        with t.BlockAccesses(pool) as rec:
            op(
                view,
                scratch,
                torch.tensor(ids, dtype=torch.int64),
                torch.tensor(status, dtype=torch.int32),
            )
        assert rec.accesses == [(t._MASKED_SCATTER, expected)], (ids, status, rec)
    with t.BlockAccesses(pool) as rec:
        view.sum()
    assert rec.blocks() == set(range(6)), rec.accesses


def test_block_accesses_charge_the_whole_view_for_values_they_cannot_read():
    """Index values the reading cannot read (a CUDA stream capturing) are
    taken from the same tensor's last reading, else the call is charged with
    every block of its view."""
    import torch
    from support import txgcap_harness as t

    pool = torch.zeros(4, 1, 1, 8, dtype=torch.uint8)
    rec = t.BlockAccesses(pool)
    index = torch.tensor([2])
    assert rec._rows(pool.view(4, 8), index) == {2}
    key = (index.device, index.data_ptr(), tuple(index.shape), index.dtype)
    assert rec._read[key] == [2]
    unread = t.BlockAccesses(pool)
    unread._values = lambda index: None
    assert unread._rows(pool.view(4, 8), index) == {0, 1, 2, 3}
    assert unread._scattered(pool.view(4, 8), 1, index, index) == {0, 1, 2, 3}
