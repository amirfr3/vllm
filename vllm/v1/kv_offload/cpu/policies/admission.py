# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, ClassVar

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, StoreTrigger

logger = init_logger(__name__)


class StoreAdmissionPolicy(ABC):
    """Decides which store offers the CPU manager admits."""

    # The store triggers that the manager asks the connector for.
    triggers: ClassVar[frozenset[StoreTrigger]]

    @abstractmethod
    def admit(self, keys: Collection[OffloadKey], trigger: StoreTrigger) -> bool:
        """Whether to admit a store offer.

        Args:
            keys: the offered keys.
            trigger: why the connector offers the keys.

        Returns:
            True to store the keys, False to refuse the whole offer.

        """


class EagerStoreAdmissionPolicy(StoreAdmissionPolicy):
    """Stores each chunk when it is computed."""

    triggers = frozenset({StoreTrigger.ON_COMPUTE})

    @override
    def admit(self, keys: Collection[OffloadKey], trigger: StoreTrigger) -> bool:
        return True


class LazyStoreAdmissionPolicy(StoreAdmissionPolicy):
    """Stores a chunk only when HBM is about to lose it."""

    triggers = frozenset(
        {StoreTrigger.ON_COMPUTE, StoreTrigger.ON_OVERWRITE, StoreTrigger.ON_PREEMPT}
    )

    @override
    def admit(self, keys: Collection[OffloadKey], trigger: StoreTrigger) -> bool:
        return trigger != StoreTrigger.ON_COMPUTE


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
            None gives the eager policy.

    Raises:
        ValueError: if "type" is unknown and "module_path" is not given.

    """
    kwargs = dict(config or {})
    name = kwargs.pop("type", "eager")
    module_path = kwargs.pop("module_path", None)
    if name in _BUILTIN_POLICIES:
        policy_cls = _BUILTIN_POLICIES[name]
    elif module_path is None:
        raise ValueError(
            f"Unknown store admission policy: {name!r}. "
            f"Supported: {list(_BUILTIN_POLICIES)}. "
            "For an out-of-tree policy, also set module_path."
        )
    else:
        logger.warning_once(
            "Loading out-of-tree store admission policy. This API is "
            "experimental and subject to change in the future "
            "as we iterate the design."
        )
        policy_cls = getattr(importlib.import_module(module_path), name)
        assert issubclass(policy_cls, StoreAdmissionPolicy)
    return policy_cls(**kwargs)
