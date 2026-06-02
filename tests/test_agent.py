"""Tests for the ``RandomAgent`` random-dispatch proxy in ``gr_synth.agent``."""

from unittest.mock import AsyncMock

import pytest

from gr_synth.agent import RandomAgent


def _fake_agent(name: str) -> AsyncMock:
    """An ``Agent`` stand-in whose async ``run`` echoes which agent ran."""
    agent = AsyncMock()
    agent.run.return_value = f"result-from-{name}"
    return agent


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_rejects_empty_pool():
    with pytest.raises(ValueError, match="at least one Agent"):
        RandomAgent([])


def test_init_stores_agents_and_builds_buckets():
    agents = [_fake_agent("a"), _fake_agent("b")]
    proxy = RandomAgent(agents)

    assert proxy._agents is agents
    assert len(proxy.buckets) == len(agents)


# --------------------------------------------------------------------------- #
# _make_buckets
# --------------------------------------------------------------------------- #


def test_buckets_single_agent_span_full_range():
    proxy = RandomAgent([_fake_agent("a")])
    assert proxy.buckets == [(0, 1.0)]


def test_buckets_two_agents_split_in_half():
    proxy = RandomAgent([_fake_agent("a"), _fake_agent("b")])
    assert proxy.buckets == [(0, 0.5), (0.5, 1.0)]


def test_buckets_four_agents_split_in_quarters():
    proxy = RandomAgent([_fake_agent(str(i)) for i in range(4)])
    assert proxy.buckets == [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


def test_buckets_are_contiguous_and_capped_at_one():
    proxy = RandomAgent([_fake_agent(str(i)) for i in range(4)])

    # each bucket starts where the previous one ended
    for (_, prev_high), (next_low, _) in zip(proxy.buckets, proxy.buckets[1:]):
        assert next_low == prev_high
    # the last bucket never exceeds 1
    assert proxy.buckets[-1][1] == 1


# --------------------------------------------------------------------------- #
# _is_in_bucket
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "number, expected_idx",
    [
        (0.25, 0),  # upper edge of bucket 0 -> bucket 0 (interval is (low, high])
        (0.30, 1),  # interior of bucket 1
        (0.50, 1),  # upper edge of bucket 1
        (0.75, 2),  # upper edge of bucket 2
        (0.90, 3),  # interior of bucket 3
        (1.0, 3),  # upper edge of last bucket
    ],
)
def test_is_in_bucket_maps_number_to_index(number, expected_idx):
    proxy = RandomAgent([_fake_agent(str(i)) for i in range(4)])
    assert proxy._is_in_bucket(number) == expected_idx


def test_is_in_bucket_lower_edge_zero_returns_zero():
    # Buckets are lower-inclusive, so exactly 0.0 maps to the first agent.
    # random.random() can return 0.0, so this edge is worth pinning down.
    proxy = RandomAgent([_fake_agent("a"), _fake_agent("b")])
    assert proxy._is_in_bucket(0.0) == 0


# --------------------------------------------------------------------------- #
# odd agent counts: buckets must tile [0, 1] with no gap (the 2nd edge case)
# --------------------------------------------------------------------------- #


def test_buckets_three_agents_tile_without_gap():
    proxy = RandomAgent([_fake_agent(str(i)) for i in range(3)])

    # buckets are contiguous: each starts exactly where the previous ended
    for (_, prev_high), (next_low, _) in zip(proxy.buckets, proxy.buckets[1:]):
        assert next_low == prev_high
    # and the top of the range is exactly 1 (no gap left near the top)
    assert proxy.buckets[0][0] == 0
    assert proxy.buckets[-1][1] == 1


@pytest.mark.parametrize(
    "number, expected_idx",
    [
        (0.0, 0),  # lower edge
        (0.2, 0),  # interior of bucket 0  (0 .. 1/3)
        (0.5, 1),  # interior of bucket 1  (1/3 .. 2/3)
        (0.9, 2),  # interior of bucket 2  (2/3 .. 1)
        (0.999, 2),  # near the top — previously fell into the rounding gap
        (1.0, 2),  # upper edge
    ],
)
def test_is_in_bucket_three_agents_covers_full_range(number, expected_idx):
    proxy = RandomAgent([_fake_agent(str(i)) for i in range(3)])
    assert proxy._is_in_bucket(number) == expected_idx


def test_is_in_bucket_above_range_clamps_to_last_agent():
    # Defensive fallback: a number outside [0, 1] never returns None.
    proxy = RandomAgent([_fake_agent(str(i)) for i in range(3)])
    assert proxy._is_in_bucket(1.5) == 2

# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


async def test_run_dispatches_to_bucket_for_random_value(monkeypatch):
    agents = [_fake_agent(str(i)) for i in range(4)]
    proxy = RandomAgent(agents)

    # 0.6 -> bucket index 2 (0.5 < 0.6 <= 0.75)
    monkeypatch.setattr("gr_synth.agent.random.random", lambda: 0.6)
    result = await proxy.run("prompt")

    assert result == "result-from-2"
    agents[2].run.assert_awaited_once_with("prompt")
    for i in (0, 1, 3):
        agents[i].run.assert_not_awaited()


async def test_run_forwards_args_and_kwargs(monkeypatch):
    agents = [_fake_agent("a"), _fake_agent("b")]
    proxy = RandomAgent(agents)

    monkeypatch.setattr("gr_synth.agent.random.random", lambda: 0.1)  # bucket 0
    await proxy.run("hello", message_history=[1, 2], deps="ctx")

    agents[0].run.assert_awaited_once_with("hello", message_history=[1, 2], deps="ctx")


async def test_run_returns_underlying_agent_result(monkeypatch):
    agents = [_fake_agent("a"), _fake_agent("b")]
    proxy = RandomAgent(agents)

    monkeypatch.setattr("gr_synth.agent.random.random", lambda: 0.9)  # bucket 1
    assert await proxy.run() == "result-from-b"


async def test_run_single_agent_always_dispatches_to_it(monkeypatch):
    agent = _fake_agent("only")
    proxy = RandomAgent([agent])

    monkeypatch.setattr("gr_synth.agent.random.random", lambda: 0.42)
    assert await proxy.run("x") == "result-from-only"
    agent.run.assert_awaited_once_with("x")
