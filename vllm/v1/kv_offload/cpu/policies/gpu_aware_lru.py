# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from enum import IntEnum

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
    GpuResidencyAware,
)
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class EvictionBand(IntEnum):
    """Eviction order of this policy. Lower is evicted first."""

    # A copy of the chunk also exists in GPU memory, so this copy is redundant.
    GPU_BACKED = 0
    # The CPU copy is the only one. Losing it needs a recompute or a flush.
    CPU_ONLY = 1


class GpuAwareLRUCachePolicy(CachePolicy, GpuResidencyAware):
    """LRU per eviction band, draining redundant chunks first.

    The band of a chunk is the pool it lives in: a chunk the GPU also holds is
    redundant, and one without a GPU copy is the only copy. Evicting a redundant
    chunk costs nothing, so those go first. Within a band the order is plain
    LRU.

    A band change resets the chunk's recency in the new band, which reads as
    "entered this band then". Only the GPU-backed pool records request recency,
    because a request stores chunks the GPU holds and a load moves chunks into
    that pool before its access is final.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self._pools: dict[EvictionBand, LRUCachePolicy] = {
            band: LRUCachePolicy(cache_capacity) for band in EvictionBand
        }

    def _holder(self, key: OffloadKey) -> EvictionBand | None:
        for band, pool in self._pools.items():
            if pool.get(key) is not None:
                return band
        return None

    def _cached_holder(self, key: OffloadKey) -> EvictionBand:
        band = self._holder(key)
        if band is None:
            raise KeyError(f"Chunk {key!r} is not cached")
        return band

    def _move(self, key: OffloadKey, band: EvictionBand) -> None:
        """Move a resident chunk to another band, dropping its old recency."""
        holder = self._holder(key)
        if holder is None or holder == band:
            return
        chunk = self._pools[holder].get(key)
        assert chunk is not None
        self._pools[holder].remove(key)
        self._pools[band].insert(key, chunk, None)

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        for pool in self._pools.values():
            chunk = pool.get(key)
            if chunk is not None:
                return chunk
        return None

    @override
    def insert(
        self, key: OffloadKey, chunk: ChunkStatus, req_context: ReqContext | None
    ) -> None:
        # Inserted as GPU-backed, the common case. A store without a request is
        # corrected by set_gpu_resident(resident=False) before it returns, and
        # a fresh chunk is not evictable yet, so no eviction sees the interim
        # band. A key that another band holds is overwritten, as in LRU.
        holder = self._holder(key)
        if holder is not None and holder != EvictionBand.GPU_BACKED:
            self._pools[holder].remove(key)
        self._pools[EvictionBand.GPU_BACKED].insert(key, chunk, req_context)

    @override
    def remove(self, key: OffloadKey) -> None:
        band = self._cached_holder(key)
        self._pools[band].remove(key)

    @override
    def on_store_miss(
        self, keys: Iterable[OffloadKey], req_context: ReqContext | None
    ) -> None:
        self._pools[EvictionBand.GPU_BACKED].on_store_miss(keys, req_context)

    @override
    def on_request_access(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        self._pools[EvictionBand.GPU_BACKED].on_request_access(keys, req_context)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        self._pools[EvictionBand.GPU_BACKED].on_request_finished(
            key_groups, insertion_only_keys, reused_keys, req_context
        )

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        band = self._cached_holder(key)
        self._pools[band].mark_evictable(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        band = self._cached_holder(key)
        self._pools[band].mark_non_evictable(key)

    @override
    def clear(self) -> None:
        for pool in self._pools.values():
            pool.clear()

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        # Both pools restore their own heap on a short eviction, so a partial
        # cross-band eviction must be ruled out before taking anything.
        available = sum(pool.num_evictable(protected) for pool in self._pools.values())
        if available < n:
            return None

        candidates: list[tuple[OffloadKey, ChunkStatus]] = []
        for pool in self._pools.values():
            remaining = n - len(candidates)
            take = min(remaining, pool.num_evictable(protected))
            if take == 0:
                continue
            evicted = pool.evict(take, protected)
            assert evicted is not None
            candidates.extend(evicted)
        assert len(candidates) == n
        return candidates

    @override
    def set_gpu_resident(self, keys: Iterable[OffloadKey], resident: bool) -> None:
        band = EvictionBand.GPU_BACKED if resident else EvictionBand.CPU_ONLY
        for key in keys:
            self._move(key, band)
