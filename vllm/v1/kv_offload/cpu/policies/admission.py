# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from typing import Any

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, StoreTrigger

logger = init_logger(__name__)


class StoreAdmissionPolicy(ABC):
    """Decides which store offers the CPU manager admits."""

    @property
    def triggers(self) -> frozenset[StoreTrigger]:
        """The store triggers that the manager asks the connector for."""
        return frozenset({StoreTrigger.ON_COMPUTE})

    @abstractmethod
    def admit(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext | None,
        trigger: StoreTrigger,
    ) -> Collection[OffloadKey]:
        """Choose the keys to store from a store offer.

        Args:
            keys: the offered keys that the CPU does not hold, in offer order.
            req_context: the request of the offer, or None for ON_OVERWRITE.
            trigger: why the connector offers the keys.

        Returns:
            The keys to store, a subset of keys. The other keys are refused.

        """


class EagerStoreAdmissionPolicy(StoreAdmissionPolicy):
    """Stores each chunk when it is computed."""

    @override
    def admit(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext | None,
        trigger: StoreTrigger,
    ) -> Collection[OffloadKey]:
        return keys


class LazyStoreAdmissionPolicy(StoreAdmissionPolicy):
    """Stores a chunk only when HBM is about to lose it."""

    @property
    @override
    def triggers(self) -> frozenset[StoreTrigger]:
        return frozenset({StoreTrigger.ON_COMPUTE, StoreTrigger.ON_OVERWRITE})

    @override
    def admit(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext | None,
        trigger: StoreTrigger,
    ) -> Collection[OffloadKey]:
        return () if trigger == StoreTrigger.ON_COMPUTE else keys


_BUILTIN_POLICIES: dict[str, type[StoreAdmissionPolicy]] = {
    "eager": EagerStoreAdmissionPolicy,
    "lazy": LazyStoreAdmissionPolicy,
}


def create_store_admission_policy(
    config: dict[str, Any] | None,
) -> StoreAdmissionPolicy:
    """Create a policy from the store_admission_policy config.

    Args:
        config: a dict with "type", an optional "module_path" for an
            out-of-tree policy, and other keys for the policy constructor.
            A built-in "type" ignores "module_path", as for the cache
            policy. None gives the eager policy.

    Returns:
        The policy, made with the other keys of config.

    Raises:
        ValueError: if "type" is unknown and "module_path" is not given.
        AttributeError: if "module_path" has no class named "type".
        TypeError: if the policy constructor does not accept the other keys.

    """
    kwargs = dict(config or {})
    name = kwargs.pop("type", "eager")
    module_path = kwargs.pop("module_path", None)
    if name in _BUILTIN_POLICIES:
        policy_cls = _BUILTIN_POLICIES[name]
    elif module_path is not None:
        logger.warning_once(
            "Loading out-of-tree store admission policy. This API is "
            "experimental and subject to change in the future "
            "as we iterate the design."
        )
        module = importlib.import_module(module_path)
        try:
            policy_cls = getattr(module, name)
        except AttributeError as e:
            raise AttributeError(f"Class {name} not found in {module_path}") from e
        assert issubclass(policy_cls, StoreAdmissionPolicy)
    else:
        raise ValueError(
            f"Unknown store admission policy: {name!r}. "
            f"Supported: {list(_BUILTIN_POLICIES)}. "
            "For an out-of-tree policy, also set module_path."
        )
    return policy_cls(**kwargs)
