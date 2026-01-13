# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch
from torch.cuda import nvtx

from ..memory_utils import MemorySnapshotSampler, enable_memory_visualize
from .config import ProfilerConfig, TorchMemoryToolConfig, TorchProfilerToolConfig
from .profile import DistProfiler

def mark_start_range(message: Optional[str] = None) -> None:
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
    """
    return nvtx.range_start(message=message)


def mark_end_range(range_id: str) -> None:
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return nvtx.range_end(range_id)


def mark_annotate(message: Optional[str] = None) -> Callable:
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
    """

    def decorator(func):
        return func

    return decorator


@contextmanager
def marked_timer(
    name: str, 
    timing_raw: dict[str, float], 
    *args: Any, 
    **kwargs: Any,
):
    """Context manager for timing with NVTX markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds MSTX markers for profiling.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    mark_range = mark_start_range(message=name)
    from verl.utils.profiler.performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


def get_torch_profiler(
    profile_save_path: str,
    role: Optional[str] = None,
    profile_step: Optional[str] = None,
):
    """Generate and return a Torch profiler object.

    Args:
        profile_save_path (str):
            The path to save the collected data.
        role (str, optional):
            The role of the current data collection. Defaults to None.
        profile_step(str, optional):
            The current training step. Defaults to None.
    """
    if profile_step:
        profile_save_path = os.path.join(profile_save_path, profile_step)
    if role:
        profile_save_path = os.path.join(profile_save_path, role)

    profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_save_path, use_gzip=True),
            record_shapes=True,
            with_stack=True,
        )

    return profile


class TorchProfiler(DistProfiler):
    """Torch profiler. Installed in a worker to control the Torch profiler."""

    _define_count = 0

    def __init__(self, rank: int, config: ProfilerConfig, tool_config: Optional[TorchProfilerToolConfig], **kwargs):
        """Initialize the TorchProfiler.

        Args:
            rank (int): The rank of the current process.
            config (Optional[ProfilerConfig]): Configuration for the profiler. If None, a default configuration is used.
        """
        # If no configuration is provided, create a default ProfilerConfig with an empty list of ranks
        if not config:
            config = ProfilerConfig(ranks=[], enable=False)
        if not tool_config:
            assert not config.enable, "tool_config must be set when profiler is enabled"
        self.enable: bool = config.enable
        if not config.enable:
            return
        self.this_step: bool = False
        self.discrete: bool = tool_config.discrete
        self.this_rank: bool = False
        self.profile = None
        self.profile_save_path = config.save_path
        if config.all_ranks:
            self.this_rank = True
        elif config.ranks:
            self.this_rank = rank in config.ranks

    def start(self, **kwargs):
        role, profile_step = kwargs.get("role", None), kwargs.get("profile_step", None)
        profile_step = str(profile_step) if profile_step is not None else None
        if self.enable and self.this_rank:
            self.this_step = True
            if not self.discrete and TorchProfiler._define_count == 0:
                self.profile = get_torch_profiler(
                    profile_save_path=self.profile_save_path,
                    role=role,
                    profile_step=profile_step,
                )
                self.profile.start()
                TorchProfiler._define_count += 1

    def stop(self):
        if self.enable and self.this_rank:
            self.this_step = False
            if not self.discrete and TorchProfiler._define_count == 1:
                self.profile.step()
                self.profile.stop()
                TorchProfiler._define_count -= 1

    def annotate(self, message: Optional[str] = None, role: Optional[str] = None, **kwargs_outer) -> Callable:
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker, which has a member field `profiler` with 
        TorchProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            role (str, optional):
                The role of the current data collection. Defaults to None.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                if not self.enable:
                    return func(*args, **kwargs_inner)

                profile_name = message or func.__name__

                if self.this_step:
                    if self.discrete:
                        profile = get_torch_profiler(
                            profile_save_path=self.profile_save_path,
                            role=role,
                        )
                        profile.start()
                    mark_range = mark_start_range(message=profile_name)

                result = func(*args, **kwargs_inner)

                if self.this_step:
                    mark_end_range(mark_range)
                    if self.discrete:
                        profile.step()
                        profile.stop()

                return result

            return wrapper

        return decorator


class DistProfiler:
    """A dispatcher that delegates to specific profilers based on config.tool.

    Supported tools:
    - nsys: NsightSystemsProfiler
    - npu: NPUProfiler (Ascend)
    - torch: PyTorch torch.profiler wrapper
    - torch_memory: Torch CUDA memory snapshot dump
    """

    def __init__(
        self, rank: int, config: Optional[ProfilerConfig] = None, tool_config: Optional[object] = None, **kwargs
    ):
        # Default config
        if not config:
            config = ProfilerConfig(ranks=[], enable=False, tool_config=None)

        if tool_config is None:
            tool_config = config.tool_config

        self._impl = None
        self._tool = getattr(config, "tool", None)

        # Normalize rank selection
        self._this_rank = False
        if config.all_ranks:
            self._this_rank = True
        elif config.ranks:
            self._this_rank = rank in config.ranks
        else:
            # default rank 0 if enabled but ranks unspecified
            self._this_rank = (rank == 0) if config.enable else False

        # Lazy import to avoid circular deps
        if self._tool == "nsys":
            from .nvtx_profile import NsightSystemsProfiler as _Nsight

            self._impl = _Nsight(rank=rank, config=config, tool_config=tool_config, **kwargs)
        elif self._tool == "npu":
            from .mstx_profile import NPUProfiler as _Npu

            self._impl = _Npu(rank=rank, config=config, tool_config=tool_config, **kwargs)
        elif self._tool == "torch":
            # Use the torch profiler wrapper defined above
            self._impl = TorchProfiler(rank=rank, config=config, tool_config=tool_config)
        elif self._tool == "torch_memory":
            self._impl = TorchMemoryProfiler(rank=rank, config=config, tool_config=tool_config)
        else:
            # Fallback to a no-op impl
            self._impl = _NoOpProfiler()

    def start(self, **kwargs):
        return getattr(self._impl, "start", lambda **_: None)(**kwargs)

    def stop(self):
        return getattr(self._impl, "stop", lambda: None)()

    @classmethod
    def annotate(
        cls,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ) -> Callable:
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_instance, *args, **kwargs_inner):
                profiler = getattr(self_instance, "profiler", None)
                if not profiler:
                    return func(self_instance, *args, **kwargs_inner)

                impl = profiler._impl
                if hasattr(impl, "annotate"):
                    try:
                        actual_decorator = impl.annotate(
                            message=message, color=color, domain=domain, category=category, **kwargs_outer
                        )

                        return actual_decorator(func)(self_instance, *args, **kwargs_inner)
                    except Exception:
                        return func(self_instance, *args, **kwargs_inner)
                return func(self_instance, *args, **kwargs_inner)

            return wrapper

        return decorator


class _NoOpProfiler:
    def start(self, **kwargs):
        return

    def stop(self):
        return


class TorchMemoryProfiler:
    """Profiler that dumps CUDA memory snapshots at step boundaries.

    Behavior:
    - On first construction (per process), enable memory history recording if CUDA is available
    - On start(step=X), remember sub_dir for this step
    - On stop(), dump a memory snapshot into config.save_path under the remembered sub_dir
    """

    _memory_history_enabled: bool = False

    def __init__(
        self, rank: int, config: Optional[ProfilerConfig], tool_config: Optional[TorchMemoryToolConfig] = None
    ):
        # Always respond to explicit start/stop calls for torch_memory tool,
        # regardless of per-role enable flag, to align with global step control.
        self.enable = True
        if not config:
            config = ProfilerConfig(ranks=[])
        self.config = config
        self.rank = rank
        self.this_step = False
        self.sub_dir = None
        self.sampler = MemorySnapshotSampler()

        # Get parameters from tool_config, with fallback to defaults
        if tool_config:
            trace_alloc_max_entries = tool_config.trace_alloc_max_entries
            stack_depth = tool_config.stack_depth
        else:
            trace_alloc_max_entries = 100_000
            stack_depth = 32

        # Best-effort enable memory history once
        if not TorchMemoryProfiler._memory_history_enabled:
            try:
                enable_memory_visualize(trace_alloc_max_entries=trace_alloc_max_entries, stack_depth=stack_depth)
            except Exception:
                # silently ignore if not supported
                pass
            TorchMemoryProfiler._memory_history_enabled = True

    def start(self, **kwargs):
        if not self.enable:
            return
        if not self._should_profile_this_rank():
            return
        profile_step = kwargs.get("profile_step", None)
        # Keep ranks aligned under same folder name
        self.sub_dir = f"step{profile_step}" if profile_step is not None else None
        self.this_step = True

    def stop(self):
        if not self.enable or not self.this_step:
            return
        self.this_step = False
        if not self._should_profile_this_rank():
            return
        out_dir = self.config.save_path or "outputs/profile"
        tag = "torch_memory"
        # Dump snapshot; all ranks write into same sub_dir
        try:
            self.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=self.sub_dir)
        except Exception:
            pass

    def _should_profile_this_rank(self) -> bool:
        if self.config.all_ranks:
            return True
        if self.config.ranks:
            return self.rank in self.config.ranks
        # default rank 0
        return self.rank == 0


class DistProfilerExtension:
    """An extension class for DistProfiler that provides distributed profiling capabilities.

    It is intended for workers in verl that single controller invokes.

    This class wraps a DistProfiler instance and provides methods to start/stop profiling
    that can be dispatched across multiple ranks in a distributed training environment.

    Args:
        profiler (DistProfiler): The base distributed profiler instance to extend
    """

    def __init__(self, profiler: DistProfiler):
        self.profiler = profiler

    from verl.single_controller.base.decorator import Dispatch, register

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()
