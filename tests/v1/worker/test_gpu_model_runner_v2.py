# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.lora.request import LoRARequest
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


def _make_capture_runner(captured: bool) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
    )
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda _cfg: contextlib.nullcontext()
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


def test_lora_state_reuses_supplied_active_requests_and_preserves_fallback(
    monkeypatch,
) -> None:
    lora_state = LoraState(max_num_reqs=4)
    shared_lora = LoRARequest("shared", 11, "/tmp/shared")
    other_lora = LoRARequest("other", 23, "/tmp/other")
    old_active_requests = {shared_lora, other_lora}

    lora_state.add_request("req-b", 0, shared_lora)
    lora_state.add_request("req-c", 1, shared_lora)
    lora_state.add_request("req-a", 2, other_lora)
    lora_state.add_request("base", 3, None)

    def fail_get_activate_loras(req_ids):
        pytest.fail(f"recomputed active LoRAs for {req_ids}")

    monkeypatch.setattr(lora_state, "get_activate_loras", fail_get_activate_loras)

    prompt_mapping, token_mapping, active_requests = lora_state.make_lora_inputs(
        ["base", "req-a", "req-b", "req-c"],
        np.array([3, 2, 0, 1], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.int32),
        active_lora_requests=old_active_requests,
    )

    assert active_requests is old_active_requests
    assert prompt_mapping == (0, 23, 11, 11)
    assert token_mapping == (0, 23, 23, 11, 11, 11, 11, 11, 11, 11)

    empty_active_requests: set[LoRARequest] = set()
    _, _, empty_requests = lora_state.make_lora_inputs(
        ["req-b"],
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        active_lora_requests=empty_active_requests,
    )
    assert empty_requests is empty_active_requests

    monkeypatch.undo()

    prompt_mapping, token_mapping, active_requests = lora_state.make_lora_inputs(
        ["base", "req-a", "req-b", "req-c"],
        np.array([3, 2, 0, 1], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.int32),
    )

    assert active_requests == old_active_requests
    assert prompt_mapping == (0, 23, 11, 11)
    assert token_mapping == (0, 23, 23, 11, 11, 11, 11, 11, 11, 11)


class _CountingLoraState(LoraState):
    def __init__(
        self,
        max_num_reqs: int,
        active_lora_requests: set[LoRARequest],
    ) -> None:
        super().__init__(max_num_reqs)
        self.active_lora_requests = active_lora_requests
        self.seen_req_ids: list[list[str]] = []

    def get_activate_loras(self, req_ids: list[str]) -> set[LoRARequest]:
        self.seen_req_ids.append(req_ids)
        return self.active_lora_requests


class _LoraActivationDone(Exception):
    pass


def test_mrv2_execute_model_reuses_one_active_lora_set_for_dispatch_and_activation(
    monkeypatch,
) -> None:
    shared_lora = LoRARequest("shared", 11, "/tmp/shared")
    other_lora = LoRARequest("other", 23, "/tmp/other")
    active_lora_requests = {shared_lora, other_lora}
    lora_state = _CountingLoraState(
        max_num_reqs=4,
        active_lora_requests=active_lora_requests,
    )
    lora_state.add_request("req-b", 0, shared_lora)
    lora_state.add_request("req-c", 1, shared_lora)
    lora_state.add_request("req-a", 2, other_lora)
    lora_state.add_request("base", 3, None)

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.update_pp_decode_requests = lambda: None
    runner.finish_requests = lambda _scheduler_output: None
    runner.free_states = lambda _scheduler_output: None
    runner.add_requests = lambda _scheduler_output: None
    runner.update_requests = lambda _scheduler_output: None
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None)
    runner.gather_batch_req_state = lambda _scheduler_output, _dummy_run: (
        SimpleNamespace(num_tokens=10),
        0,
    )
    runner.pcp_manager = None
    runner.lora_config = SimpleNamespace()
    runner.lora_state = lora_state
    runner.is_encoder_decoder = False
    runner.cudagraph_manager = object()
    runner.dp_size = 1
    runner.dp_rank = 0
    runner.parallel_config = SimpleNamespace()
    runner.ubatch_runner = None
    runner.decode_query_len = 1
    runner.observability_config = SimpleNamespace(cudagraph_metrics=False)
    runner.prepare_inputs = lambda *_args: SimpleNamespace(
        req_ids=["base", "req-a", "req-b", "req-c"],
        idx_mapping_np=np.array([3, 2, 0, 1], dtype=np.int32),
        num_scheduled_tokens=np.array([1, 2, 3, 4], dtype=np.int32),
    )
    runner.prepare_attn = lambda _input_batch: (object(), object())
    runner.kv_cache_config = SimpleNamespace()
    runner.req_states = SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=None))
    runner.model_state = SimpleNamespace(preprocess_state=lambda *_args: None)

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={
            "req-b": 3,
            "req-c": 4,
            "req-a": 2,
            "base": 1,
        },
        total_num_scheduled_tokens=10,
        scheduled_encoder_inputs={},
    )
    dispatch_kwargs = {}

    def fake_dispatch(*_args, **kwargs):
        dispatch_kwargs.update(kwargs)
        return SimpleNamespace(num_tokens=10, num_reqs=4, num_ubatches=1), None

    def fake_set_active_loras(prompt_mapping, token_mapping, active_requests):
        assert prompt_mapping == (0, 23, 11, 11)
        assert token_mapping == (0, 23, 23, 11, 11, 11, 11, 11, 11, 11)
        assert active_requests is active_lora_requests
        raise _LoraActivationDone

    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", fake_dispatch)
    runner._set_active_loras = fake_set_active_loras

    with pytest.raises(_LoraActivationDone):
        runner.execute_model(scheduler_output)

    assert dispatch_kwargs["num_active_loras"] == len(active_lora_requests)
    assert lora_state.seen_req_ids == [["req-b", "req-c", "req-a", "base"]]


def test_mrv2_dummy_lora_dispatch_uses_max_loras_plus_one(monkeypatch) -> None:
    active_lora_requests: set[LoRARequest] = set()
    lora_state = _CountingLoraState(
        max_num_reqs=1,
        active_lora_requests=active_lora_requests,
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.gather_batch_req_state = lambda _scheduler_output, _dummy_run: (None, 0)
    runner.pcp_manager = None
    runner.lora_config = SimpleNamespace(max_loras=4)
    runner.lora_state = lora_state
    runner.is_encoder_decoder = False
    runner.cudagraph_manager = object()
    runner.dp_size = 1
    runner.dp_rank = 0
    runner.parallel_config = SimpleNamespace()
    runner.ubatch_runner = None
    runner.decode_query_len = 1
    runner.kv_connector = SimpleNamespace(no_forward=lambda _scheduler_output: object())
    runner._merge_ec_connector_no_forward = lambda _scheduler_output, output: output

    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"_dummy_req_0": 3},
        total_num_scheduled_tokens=3,
        scheduled_encoder_inputs={},
    )
    dispatch_kwargs = {}

    def fake_dispatch(*_args, **kwargs):
        dispatch_kwargs.update(kwargs)
        return SimpleNamespace(num_tokens=0), None

    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", fake_dispatch)

    runner.execute_model(scheduler_output, dummy_run=True)

    assert dispatch_kwargs["num_active_loras"] == 5
    assert lora_state.seen_req_ids == []
