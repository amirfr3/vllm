# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.gpu_aware_lru import (
    EvictionBand,
    GpuAwareLRUCachePolicy,
)
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


def key(name: bytes) -> OffloadKey:
    return make_offload_key(name, 0)


def ready_chunk(chunk_id: int) -> ChunkStatus:
    chunk = ChunkStatus(chunk_id)
    chunk.ref_cnt = 0
    return chunk


def keyed(chunks: list[tuple[OffloadKey, ChunkStatus]]) -> list[OffloadKey]:
    return [cache_key for cache_key, _ in chunks]


def insert(policy: GpuAwareLRUCachePolicy, name: OffloadKey, chunk_id: int) -> None:
    policy.insert(name, ready_chunk(chunk_id), None)


def test_gpu_backed_is_evicted_before_cpu_only():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"gpu"), 0)
    insert(policy, key(b"only"), 1)
    policy.set_gpu_resident([key(b"only")], False)

    evicted = policy.evict(1, set())

    assert evicted is not None
    assert keyed(evicted) == [key(b"gpu")]
    assert policy.get(key(b"only")) is not None


def test_move_keeps_the_chunk_in_one_band():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"key"), 0)

    policy.set_gpu_resident([key(b"key")], False)
    assert policy.get(key(b"key")) is not None
    evicted = policy.evict(1, set())
    assert evicted is not None and keyed(evicted) == [key(b"key")]

    # The old band no longer holds the chunk.
    assert policy.evict(1, set()) is None


def test_reinsert_takes_the_chunk_out_of_its_old_band():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"key"), 0)
    policy.set_gpu_resident([key(b"key")], False)

    insert(policy, key(b"key"), 1)

    chunk = policy.get(key(b"key"))
    assert chunk is not None and chunk.chunk_id == 1
    evicted = policy.evict(1, set())
    assert evicted is not None and keyed(evicted) == [key(b"key")]
    assert policy.evict(1, set()) is None


@pytest.mark.parametrize("method", ["remove", "mark_evictable", "mark_non_evictable"])
def test_unknown_key_raises_key_error(method: str):
    policy = GpuAwareLRUCachePolicy(4)

    with pytest.raises(KeyError):
        getattr(policy, method)(key(b"missing"))


def test_dropped_chunk_becomes_newest_in_cpu_only():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"first"), 0)
    insert(policy, key(b"second"), 1)
    insert(policy, key(b"gpu"), 2)

    policy.set_gpu_resident([key(b"first")], False)
    policy.set_gpu_resident([key(b"second")], False)
    # A repeated signal for a chunk already in the target band is inert.
    policy.set_gpu_resident([key(b"first")], False)

    evicted = policy.evict(1, set())
    assert evicted is not None and keyed(evicted) == [key(b"gpu")]
    evicted = policy.evict(1, set())
    assert evicted is not None and keyed(evicted) == [key(b"first")]


def test_load_returns_a_chunk_to_the_gpu_backed_band():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"loaded"), 0)
    insert(policy, key(b"only"), 1)
    policy.set_gpu_resident([key(b"loaded"), key(b"only")], False)

    policy.set_gpu_resident([key(b"loaded")], True)

    evicted = policy.evict(1, set())
    assert evicted is not None and keyed(evicted) == [key(b"loaded")]


def test_evict_is_atomic_when_the_bands_cannot_cover_it():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"gpu"), 0)
    insert(policy, key(b"only"), 1)
    policy.set_gpu_resident([key(b"only")], False)

    assert policy.evict(3, set()) is None

    assert policy.get(key(b"gpu")) is not None
    assert policy.get(key(b"only")) is not None


def test_evict_spans_bands_in_order():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"gpu1"), 0)
    insert(policy, key(b"gpu2"), 1)
    insert(policy, key(b"only"), 2)
    policy.set_gpu_resident([key(b"only")], False)

    evicted = policy.evict(3, set())

    assert evicted is not None
    assert keyed(evicted) == [key(b"gpu1"), key(b"gpu2"), key(b"only")]


def test_num_evictable_skips_protected_keys():
    pool = LRUCachePolicy(4)
    pool.insert(key(b"a"), ready_chunk(0), None)
    pool.insert(key(b"b"), ready_chunk(1), None)

    assert pool.num_evictable(set()) == 2
    assert pool.num_evictable({key(b"a")}) == 1
    assert pool.num_evictable({key(b"a"), key(b"b")}) == 0


def test_protected_key_survives_a_cross_band_eviction():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"gpu"), 0)
    insert(policy, key(b"only"), 1)
    policy.set_gpu_resident([key(b"only")], False)

    assert policy.evict(2, {key(b"gpu")}) is None
    assert policy.get(key(b"gpu")) is not None
    assert policy.get(key(b"only")) is not None

    evicted = policy.evict(1, {key(b"gpu")})
    assert evicted is not None and keyed(evicted) == [key(b"only")]


def test_pinned_chunk_moves_bands_without_becoming_evictable():
    # A load completion reports residency while the chunk is still pinned, so
    # ref_cnt is 1 at that moment.
    policy = GpuAwareLRUCachePolicy(4)
    chunk = ready_chunk(0)
    policy.insert(key(b"key"), chunk, None)
    policy.mark_non_evictable(key(b"key"))
    chunk.ref_cnt = 1

    policy.set_gpu_resident([key(b"key")], True)

    assert policy.get(key(b"key")) is chunk
    assert policy.evict(1, set()) is None

    chunk.ref_cnt = 0
    policy.mark_evictable(key(b"key"))
    evicted = policy.evict(1, set())
    assert evicted is not None and keyed(evicted) == [key(b"key")]


def test_clear_drops_both_bands():
    policy = GpuAwareLRUCachePolicy(4)
    insert(policy, key(b"gpu"), 0)
    insert(policy, key(b"only"), 1)
    policy.set_gpu_resident([key(b"only")], False)

    policy.clear()

    assert policy.get(key(b"gpu")) is None
    assert policy.get(key(b"only")) is None


def test_eviction_band_order():
    assert list(EvictionBand) == [EvictionBand.GPU_BACKED, EvictionBand.CPU_ONLY]
