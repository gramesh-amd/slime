import dataclasses
import gc
import math
from contextlib import nullcontext
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model

import wandb
from slime.utils.memory_utils import clear_memory

from .checkpoint import load_checkpoint, save_checkpoint
from .data import get_batch
from .loss import get_log_probs_and_entropy, loss_function
from .model_provider import get_model_provider_func

if torch.version.hip:
    from vllm.device_allocator.cumem import CuMemAllocator


def get_optimizer_param_scheduler(args, optimizer):
    """Build the learning rate scheduler."""
    # Iteration-based training.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    args,
    no_wd_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    """Setup model and optimizer."""
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    with (
        CuMemAllocator.get_instance().use_memory_pool(tag="model")
        if args.offload and torch.version.hip
        else nullcontext()
    ):
        config = get_model_config(model[0])

        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs["grad_reduce_in_fp32"] = args.accumulate_allreduce_grads_in_fp32
        kwargs["check_for_nan_in_grad"] = args.check_for_nan_in_loss_and_grad
        kwargs["check_for_large_grads"] = args.check_for_large_grads
        kwargs["bucket_size"] = args.ddp_bucket_size
        kwargs["pad_buckets_for_high_nccl_busbw"] = args.ddp_pad_buckets_for_high_nccl_busbw
        kwargs["average_in_collective"] = args.ddp_average_in_collective
        ddp_config = DistributedDataParallelConfig(**kwargs)

        # In the custom FSDP and DDP use path, we need to initialize the bucket size.
        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
            )
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        model = [
            DDP(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Optimizer
        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        config.timers = None

        optimizer = get_megatron_optimizer(
            config,
            model,
            no_wd_decay_cond,
            scale_lr_cond,
            lr_mult,
            use_gloo_process_groups=args.enable_gloo_process_groups,
        )
        opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
        for optimizer in optimizer.chained_optimizers:
            if not getattr(optimizer, "init_state_fn", None):
                continue
            optimizer.init_state_fn(optimizer.optimizer, optimizer.config)

    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks, param_sync=True):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(args, model, data_iterator, num_microbatches, store_prefix=""):
    """Only do the forward pass and calculate the logprob."""

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])

    def forward_step(data_iterator, model: GPTModel):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """

        # Get the batch.
        batch = get_batch(data_iterator, ["tokens", "total_lengths", "response_lengths"])
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
        )

        return output_tensor, partial(
            get_log_probs_and_entropy,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
        )

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from slime.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    for step_id in range(num_steps_per_rollout):
        # collect_non_loss_data
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                origin_values = [None] * len(values)
                origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                for value, origin_index in zip(values, origin_indices):
                    origin_values[origin_index] = value
                values = origin_values
            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def train_one_step(args, rollout_id, step_id, data_iterator, model, optimizer, opt_param_scheduler, num_microbatches):
    """Single training step."""
    args = get_args()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(data_iterator, model: GPTModel):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "rollout_log_probs",
            ],
        )

        output_tensor = model(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=batch["packed_seq_params"],
        )

        return output_tensor, partial(loss_function, args, batch, num_microbatches)

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate.
        assert update_successful
        opt_param_scheduler.step(increment=args.global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        keys = losses_reduced[0]["keys"]
        values = None
        for x in losses_reduced:
            if values is None:
                values = x["values"]
            else:
                values += x["values"]
        assert len(keys) + 1 == values.numel()
        torch.distributed.all_reduce(values, group=mpu.get_data_parallel_group(with_context_parallel=True))

        loss_reduced = {}
        values = values.tolist()
        num_samples_or_tokens = values[0]
        for key, value in zip(keys, values[1:]):
            loss_reduced[key] = value * mpu.get_context_parallel_world_size() / num_samples_or_tokens
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args):
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches):
    """Training function: run train_step desired number of times."""
    args = get_args()

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            log_dict = {
                f"train/{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }
            log_dict["train/grad_norm"] = grad_norm
            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            if args.use_wandb:
                log_dict["train/step"] = accumulated_step_id
                wandb.log(log_dict)

            if args.ci_test:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    assert log_dict["train/ppo_kl"] == 0.0 and log_dict["train/pg_clipfrac"] == 0.0
                if accumulated_step_id == 0 and "train/kl_loss" in log_dict:
                    assert log_dict["train/kl_loss"] == 0.0

            print(f"step {accumulated_step_id}: {log_dict}")
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

def clear_gradients(model):
    """Completely clear all gradients from model."""
    for model_chunk in model:
        # Clear DDP gradient buffers
        if hasattr(model_chunk, 'zero_grad_buffer'):
            try:
                model_chunk.zero_grad_buffer()
            except Exception as e:
                print(f"[WARNING] Failed to zero grad buffer: {e}")
        
        # Clear parameter gradients
        for param in model_chunk.parameters():
            param.grad = None
            # Don't set main_grad to None if it exists - Megatron needs it
            if hasattr(param, 'main_grad') and param.main_grad is not None:
                param.main_grad.zero_()


def move_megatron_optimizer_to_cpu_complete(optimizer):
    """Move optimizer state keys (parameters) and values (tensors) to CPU"""
    print(f"[DEBUG] Moving optimizer to CPU...")
    
    # Create mapping from old params to new CPU params
    old_to_new_param_map = {}
    
    # Move param_groups parameters to CPU first and build mapping
    for group_idx, group in enumerate(optimizer.param_groups):
        print(f"[DEBUG] Processing param group {group_idx}")
        for i, param in enumerate(group['params']):
            cpu_param = param.cpu()
            old_to_new_param_map[param] = cpu_param
            group['params'][i] = cpu_param
            # Also move main_grad if it exists
            if hasattr(param, 'main_grad') and param.main_grad is not None:
                cpu_param.main_grad = param.main_grad.cpu()
            print(f"[DEBUG] Moved param {i} from {param.device} to {cpu_param.device}")
    
    # Rebuild state dict with CPU parameters as keys
    new_state = {}
    for param, state in optimizer.state.items():
        if param in old_to_new_param_map:
            cpu_param = old_to_new_param_map[param]
            new_state[cpu_param] = {}
            for k, v in state.items():
                if torch.is_tensor(v):
                    new_state[cpu_param][k] = v.cpu()
                    print(f"[DEBUG] Moved optimizer state '{k}' from {v.device} to cpu")
                else:
                    new_state[cpu_param][k] = v
    
    optimizer.state = new_state
    
    # Update optimizer's internal parameter mappings
    if hasattr(optimizer, '_param_groups'):
        optimizer._param_groups = optimizer.param_groups
    
    # Clear any cached parameter mappings
    if hasattr(optimizer, '_param_groups_backup'):
        delattr(optimizer, '_param_groups_backup')
    
    print(f"[DEBUG] Optimizer move to CPU complete")
    return optimizer

def move_model_to_gpu(model, device='cuda'):
    """Move model back to GPU and properly initialize gradients"""
    # Convert string device to device object if needed
    if isinstance(device, str):
        device = torch.device(device)
    
    print(f"[DEBUG] Moving model to {device}...")
    
    if isinstance(model, list):
        for i, model_chunk in enumerate(model):
            # Clear any existing gradients before moving
            for param in model_chunk.parameters():
                param.grad = None
                # Preserve main_grad if it exists
                if hasattr(param, 'main_grad') and param.main_grad is not None:
                    main_grad_cpu = param.main_grad
                    param.main_grad = None  # Temporarily set to None for the move
            
            model[i] = model_chunk.to(device)
            
            # Restore main_grad on GPU
            for param in model[i].parameters():
                if hasattr(param, 'main_grad'):
                    # Initialize main_grad as zeros on GPU if it was None or exists
                    param.main_grad = torch.zeros_like(param.data)
            
            print(f"[DEBUG] Moved model chunk {i} to {device}")
    else:
        # Clear any existing gradients before moving
        for param in model.parameters():
            param.grad = None
            if hasattr(param, 'main_grad') and param.main_grad is not None:
                param.main_grad = None
        
        model = model.to(device)
        
        # Initialize main_grad on GPU
        for param in model.parameters():
            if hasattr(param, 'main_grad'):
                param.main_grad = torch.zeros_like(param.data)
        
        print(f"[DEBUG] Moved model to {device}")
    
    return model




def move_megatron_optimizer_to_gpu(optimizer, model, device='cuda'):
    """Move optimizer parameters and state back to GPU, ensuring they match model device placement"""
    # Convert string device to device object if needed
    if isinstance(device, str):
        device = torch.device(device)
        
    print(f"[DEBUG] Moving optimizer to {device}...")
    
    # Get model parameters to ensure device alignment
    model_params = []
    if isinstance(model, list):
        for model_chunk in model:
            model_params.extend(list(model_chunk.parameters()))
    else:
        model_params = list(model.parameters())
    
    print(f"[DEBUG] Found {len(model_params)} model parameters")
    
    # Create mapping from CPU params to GPU model params by matching shapes
    cpu_to_gpu_param_map = {}
    used_gpu_params = set()
    
    # First pass: map optimizer params to model params
    for group_idx, group in enumerate(optimizer.param_groups):
        print(f"[DEBUG] Processing optimizer param group {group_idx}")
        new_params = []
        
        for i, cpu_param in enumerate(group['params']):
            # Find matching model param by shape
            gpu_param = None
            for model_param in model_params:
                if (model_param not in used_gpu_params and 
                    model_param.shape == cpu_param.shape and
                    model_param.dtype == cpu_param.dtype):
                    gpu_param = model_param
                    used_gpu_params.add(model_param)
                    break
            
            if gpu_param is not None:
                cpu_to_gpu_param_map[cpu_param] = gpu_param
                new_params.append(gpu_param)
                print(f"[DEBUG] Mapped optimizer param {i} to model param on {gpu_param.device}")
            else:
                # Fallback: manually move to GPU
                gpu_param = cpu_param.to(device)
                cpu_to_gpu_param_map[cpu_param] = gpu_param
                new_params.append(gpu_param)
                print(f"[DEBUG] Fallback: moved param {i} to {device}")
        
        group['params'] = new_params
    
    # Rebuild state dict with GPU parameters as keys
    new_state = {}
    for cpu_param, state in optimizer.state.items():
        if cpu_param in cpu_to_gpu_param_map:
            gpu_param = cpu_to_gpu_param_map[cpu_param]
            new_state[gpu_param] = {}
            for k, v in state.items():
                if torch.is_tensor(v):
                    new_state[gpu_param][k] = v.to(device)
                    print(f"[DEBUG] Moved optimizer state '{k}' to {device}")
                else:
                    new_state[gpu_param][k] = v
    
    optimizer.state = new_state
    print(f"[DEBUG] Optimizer move to GPU complete")
    return optimizer

def save(iteration, model, optimizer, opt_param_scheduler):
    """Save checkpoint with proper device handling."""
    args = get_args()
    print(f"[DEBUG] Starting save at iteration {iteration}...")
    
    if should_disable_forward_pre_hook(args):
        print("[DEBUG] Disabling forward pre-hook...")
        disable_forward_pre_hook(model)
    
    # Store original device
    original_device = next(model[0].parameters()).device
    print(f"[DEBUG] Original device: {original_device}")
    
    # Clear all gradients before moving to CPU
    print("[DEBUG] Clearing gradients before save...")
    clear_gradients(model)
    
    # Zero out optimizer gradients
    optimizer.zero_grad()
    
    # Move to CPU for saving
    print("[DEBUG] Moving model and optimizer to CPU for saving...")
    try:
        model[0] = model[0].cpu()
        print("[DEBUG] Model moved to CPU successfully")
    except Exception as e:
        print(f"[ERROR] Failed to move model to CPU: {e}")
        raise
    
    try:
        optimizer = move_megatron_optimizer_to_cpu_complete(optimizer)
        print("[DEBUG] Optimizer moved to CPU successfully")
    except Exception as e:
        print(f"[ERROR] Failed to move optimizer to CPU: {e}")
        raise
    
    # Save checkpoint
    print("[DEBUG] Saving checkpoint...")
    try:
        save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far=0,
            checkpointing_context=None,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=None,
        )
        print("[DEBUG] Checkpoint saved successfully")
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")
        raise
    
    # Move everything back to original device
    print("[DEBUG] Moving model and optimizer back to original device...")
    try:
        model = move_model_to_gpu(model, original_device)
        print("[DEBUG] Model moved back to GPU successfully")
    except Exception as e:
        print(f"[ERROR] Failed to move model back to GPU: {e}")
        raise
    
    try:
        optimizer = move_megatron_optimizer_to_gpu(optimizer, model, original_device)
        print("[DEBUG] Optimizer moved back to GPU successfully")
    except Exception as e:
        print(f"[ERROR] Failed to move optimizer back to GPU: {e}")
        raise
    
    # Clear gradients again and ensure clean state
    print("[DEBUG] Ensuring clean state after restore...")
    clear_gradients(model)
    optimizer.zero_grad()
    
    # For distributed optimizer, ensure internal buffers are reset
    if hasattr(optimizer, 'reset_grad_buffers'):
        print("[DEBUG] Resetting optimizer gradient buffers...")
        optimizer.reset_grad_buffers()
    
    # If using distributed optimizer, we need to ensure model groups are properly set
    if hasattr(optimizer, 'model_float16_groups') and hasattr(optimizer, 'float16_groups'):
        print("[DEBUG] Updating distributed optimizer model groups...")
        # Rebuild model groups with GPU parameters
        model_float16_groups = []
        for group in optimizer.float16_groups:
            model_group = []
            for param in group:
                # Find corresponding model parameter
                for model_chunk in model:
                    for p in model_chunk.parameters():
                        if p.shape == param.shape and p.dtype == param.dtype:
                            model_group.append(p)
                            break
            if model_group:
                model_float16_groups.append(model_group)
        
        if model_float16_groups:
            optimizer.model_float16_groups = model_float16_groups
            print(f"[DEBUG] Updated {len(model_float16_groups)} model float16 groups")
    
    # Final verification
    print("[DEBUG] Final verification of device placement...")
    model_device = next(model[0].parameters()).device
    print(f"[DEBUG] Model on device: {model_device}")
    
    # Check optimizer param devices
    for group_idx, group in enumerate(optimizer.param_groups):
        if len(group['params']) > 0:
            param_device = group['params'][0].device
            print(f"[DEBUG] Optimizer param group {group_idx} on device: {param_device}")
            break
    
    # Verify main_grad initialization
    main_grad_count = 0
    for model_chunk in model:
        for param in model_chunk.parameters():
            if hasattr(param, 'main_grad') and param.main_grad is not None:
                main_grad_count += 1
    print(f"[DEBUG] {main_grad_count} parameters have main_grad initialized")
    
    if should_disable_forward_pre_hook(args):
        print("[DEBUG] Re-enabling forward pre-hook...")
        enable_forward_pre_hook(model)
    
    print(f"[DEBUG] Save complete. Model ready for next training step.")

def initialize_model_and_optimizer(args):
    """Initialize model and optimizer, handling device placement correctly."""
    print("[DEBUG] Starting initialize_model_and_optimizer...")
    
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args)
    
    # Get the target device before loading
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f"[DEBUG] Target device: {device}")
    
    # Check current device of model
    current_device = next(model[0].parameters()).device
    print(f"[DEBUG] Model currently on device: {current_device}")
    
    clear_memory()
    
    # Load checkpoint
    print("[DEBUG] Loading checkpoint...")
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    print(f"[DEBUG] Loaded checkpoint at iteration {iteration}")
    
    # Check device after loading
    loaded_device = next(model[0].parameters()).device
    print(f"[DEBUG] Model on device after loading: {loaded_device}")
    
    # Ensure model is on the correct device after loading
    if loaded_device.type != device.type:
        print(f"[DEBUG] Need to move model from {loaded_device} to {device}")
        model = move_model_to_gpu(model, device)
    
    # Clear all gradients to ensure clean state
    print("[DEBUG] Clearing all gradients...")
    clear_gradients(model)
    
    # Initialize main_grad for all parameters (required by Megatron DDP)
    print("[DEBUG] Initializing main_grad for all parameters...")
    for i, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            if hasattr(param, 'main_grad') and param.main_grad is None:
                # Initialize main_grad with zeros of the same shape and dtype
                param.main_grad = torch.zeros_like(param.data)
                print(f"[DEBUG] Initialized main_grad for model[{i}].{name}")
    
    # Ensure optimizer state is also on the correct device
    if optimizer is not None:
        print("[DEBUG] Checking optimizer state devices...")
        needs_device_update = False
        
        # Check if any optimizer state is on wrong device
        for param, state in optimizer.state.items():
            param_device = param.device
            for k, v in state.items():
                if torch.is_tensor(v) and v.device != param_device:
                    print(f"[DEBUG] Found optimizer state '{k}' on {v.device}, param on {param_device}")
                    needs_device_update = True
                    break
            if needs_device_update:
                break
        
        if needs_device_update:
            print("[DEBUG] Updating optimizer state devices...")
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    if torch.is_tensor(v) and v.device != param.device:
                        optimizer.state[param][k] = v.to(param.device)
                        print(f"[DEBUG] Moved optimizer state '{k}' to {param.device}")
    
    # Force optimizer to update its internal state
    if hasattr(optimizer, 'reload_model_params'):
        print("[DEBUG] Reloading model params in optimizer...")
        optimizer.reload_model_params()
    
    clear_memory()
    
    print("[DEBUG] Model and optimizer initialization complete")
    return model, optimizer, opt_param_scheduler, iteration




