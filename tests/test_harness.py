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

    fact = "pool access failed; budget setting: --kv-cache-memory-bytes 1073741824"
    settings = {"--kv-cache-memory-bytes": 1 << 30, "--max-num-seqs": 3}
    assert c.configuration_remedies(fact, settings) == []
    assert c.configuration_remedies(fact) == ["--kv-cache-memory-bytes"]
    change = fact + "; raise --num-gpu-blocks-override to at least 5"
    assert c.configuration_remedies(change, settings) == ["--num-gpu-blocks-override"]
    assert c.configuration_remedies("max_num_seqs=3 rows", settings) == []
    assert "--max-num-seqs" in c.configuration_remedies(
        "or lower --max-num-seqs", settings
    )


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


def test_plugin_holds_finds_what_the_plugin_keeps_and_skips_cores():
    """``txgcap_harness.plugin_holds`` reports a plugin object that keeps an
    accessor object, its tensor or a view of it, and nothing that only core's
    attribute, the state layer or the harness keeps."""
    from types import SimpleNamespace

    import pytest
    import torch

    pytest.importorskip("vllm.v1.worker.utils")
    from support import txgcap_harness as t

    views = (torch.zeros(4, 2), torch.zeros(4, 3))
    layer = SimpleNamespace(prefix="nemotron.state", kv_cache=views)
    binding = SimpleNamespace(kept=None)
    model = SimpleNamespace(_persistent_state_layer=layer, binding=binding)
    state = SimpleNamespace(
        _state_group_layouts=t.make_layouts({"nemotron.state": views}), model=model
    )
    env = SimpleNamespace(model=model, state=state)
    targets = t.accessor_objects(state)
    assert len(targets) == 1 + 1 + 2 * 2
    assert t.plugin_holds(env, targets) == []
    layout = state._state_group_layouts[0]
    (components,) = layout.layers.values()
    for kept in (
        state._state_group_layouts,
        layout,
        components[1],
        views[0],
        views[1][1:],
        {"pools": [views[0]]},
    ):
        binding.kept = kept
        held = t.plugin_holds(env, targets)
        assert len(held) == 1 and held[0].startswith("model -> "), (kept, held)
    binding.kept = t.AccessorSpy  # a harness object is not the plugin's
    assert t.plugin_holds(env, targets) == []
