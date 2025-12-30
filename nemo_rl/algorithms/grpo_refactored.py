# no copy write yet will add later 
"""Refactored GRPO training using the new Trainer and Sampler interfaces.

This module provides a cleaner implementation of GRPO using the abstraction layer
defined in trainer_sampler.py. It uses the TrainerInterface and SamplerInterface
which provide a unified API matching the Tinker pattern:

- trainer.forward_backward(data, loss_fn, datastream_id=None)
- trainer.optim_step(optimizer_config=None)
- trainer.save_weights_and_get_sampling_client()
- sampler.sample(input_data, sampling_params, greedy=False)

All implementations (NeMoTrainer, MockTrainer, TinkerTrainer, etc.) implement
these interfaces, allowing the same training loop to work with different backends.
"""
import os
from typing import Optional

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    MasterConfig,
    dynamic_sampling,
    normalize_advantages_with_epsilon,
    scale_rewards,
    validate,
)
from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import ClippedPGLossDataDict
from nemo_rl.algorithms.trainer_sampler import (
    TrainerInterface,
    SamplerInterface,
    NeMoTrainer,
    NeMoSampler,
    Trainer,  # Type alias for backward compatibility
    Sampler,  # Type alias for backward compatibility
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import run_multi_turn_rollout

# Allow monkey-patching for mock testing
__all__ = ["grpo_train_refactored"]
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.timer import Timer, TimeoutChecker

# Re-export types for convenience
TokenizerType = PreTrainedTokenizerBase


def grpo_train_refactored(
    trainer: TrainerInterface,
    sampler: SamplerInterface,
    dataloader,
    val_dataloader: Optional,
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
    processor: Optional[AutoProcessor] = None,
) -> None:
    """Run GRPO training algorithm using the new Trainer and Sampler interfaces.

    This is a refactored version that uses the cleaner abstraction layer,
    making the code more modular and easier to understand.

    Args:
        trainer: TrainerInterface instance (e.g., NeMoTrainer, MockTrainer, TinkerTrainer)
        sampler: SamplerInterface instance (e.g., NeMoSampler, MockSampler, TinkerSampler)
        dataloader: Training data loader
        val_dataloader: Optional validation data loader
        tokenizer: Tokenizer
        loss_fn: Loss function
        task_to_env: Training environments
        val_task_to_env: Validation environments
        logger: Logger
        checkpointer: Checkpoint manager
        grpo_save_state: Training state
        master_config: Master configuration
        processor: Optional processor for VLM models
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    kv_scales_cache = None  # Cache reused for computed kv scales
    sync_kv_scales = getattr(sampler.generation, "requires_kv_scale_sync", False)

    # Extract config values
    grpo_config = master_config["grpo"]
    policy_config = master_config["policy"]
    loss_config = master_config["loss_fn"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Training state
    current_step = grpo_save_state["current_step"]
    total_steps = grpo_save_state["total_steps"]
    max_num_steps = grpo_config["max_num_steps"]
    current_epoch = grpo_save_state["current_epoch"]
    max_num_epochs = grpo_config["max_num_epochs"]
    consumed_samples = grpo_save_state["consumed_samples"]
    val_at_start = grpo_config["val_at_start"]
    val_period = grpo_config["val_period"]

    # Track if sampler needs weight sync
    sampler_stale = True

    # Run validation at the start if configured
    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if sampler_stale:
            # Sync weights using interface-compatible method
            if hasattr(sampler, "weight_sync"):
                # NeMoSampler has weight_sync() for backward compatibility
                sampler.weight_sync(kv_scales=kv_scales_cache, timer=timer)  # type: ignore
            elif hasattr(sampler, "_trainer") and sampler._trainer is not None:
                # For Mock/Tinker samplers, use trainer's save_weights_and_get_sampling_client
                sampler._trainer.save_weights_and_get_sampling_client(
                    kv_scales=kv_scales_cache, timer=timer
                )
            sampler_stale = False
        val_metrics, validation_timings = validate(
            sampler.generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
        )
        sampler.generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        batch_cache: BatchedDataDict[DatumSpec] = None
        dynamic_sampling_num_gen_batches = 0

        # Run grpo training loop
        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_num_steps)} {'=' * 25}",
                flush=True,
            )

            with timer.time("total_step_time"):
                # Prepare batch
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    # Repeat batch items
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(grpo_config["num_generations_per_prompt"])
                    )
                    # Convert to flat messages for generation
                    batched_flat, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    input_ids = batched_flat["token_ids"]

                # Generate responses using sampler
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    # Sync weights if needed before generation
                    if sampler_stale:
                        if sync_kv_scales and kv_scales_cache is None:
                            print("▶ Computing KV cache scales...", flush=True)
                            trainer.policy.prepare_for_lp_inference()
                            calib_flat, calib_input_lengths = (
                                batched_message_log_to_flat_message(
                                    repeated_batch["message_log"],
                                    pad_value_dict={
                                        "token_ids": tokenizer.pad_token_id
                                    },
                                    make_sequence_length_divisible_by=policy_config[
                                        "make_sequence_length_divisible_by"
                                    ],
                                )
                            )
                            calibration_data = BatchedDataDict(
                                {
                                    "input_ids": calib_flat["token_ids"],
                                    "input_lengths": calib_input_lengths,
                                }
                            )
                            calibration_data.update(
                                calib_flat.get_multimodal_dict(as_tensors=False)
                            )
                            calibration_data.to("cpu")
                            kv_scales_cache = trainer.policy.calibrate_qkv_fp8_scales(
                                calibration_data, include_q=True
                            )["layers"]

                        # Sync weights using interface-compatible method
                        if hasattr(sampler, "weight_sync"):
                            # NeMoSampler has weight_sync() for backward compatibility
                            sampler.weight_sync(kv_scales=kv_scales_cache, timer=timer)  # type: ignore
                        elif hasattr(sampler, "_trainer") and sampler._trainer is not None:
                            # For Mock/Tinker samplers, use trainer's save_weights_and_get_sampling_client
                            sampler._trainer.save_weights_and_get_sampling_client(
                                kv_scales=kv_scales_cache, timer=timer
                            )
                        sampler_stale = False

                dynamic_sampling_num_gen_batches += 1
                with timer.time("generation"):
                    # Generate responses using the sampler interface
                    generation_input = BatchedDataDict[GenerationDatumSpec](
                        {
                            "input_ids": input_ids,
                            "input_lengths": input_lengths,
                        }
                    )
                    generation_input.update(
                        batched_flat.get_multimodal_dict(as_tensors=False)
                    )
                    generation_input.to("cpu")

                    # Use sampler's generation interface for rollout
                    # Note: run_multi_turn_rollout uses sampler.generation directly
                    repeated_batch, rollout_metrics = run_multi_turn_rollout(
                        sampler.generation,  # Pass generation interface directly
                        input_batch=repeated_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=policy_config["max_total_sequence_length"],
                        max_rollout_turns=grpo_config["max_rollout_turns"],
                        greedy=False,
                    )

                # Scale rewards
                repeated_batch = scale_rewards(
                    repeated_batch, grpo_config["reward_scaling"]
                )

                # Calculate rewards & advantages
                print("▶ Processing rewards...", flush=True)
                with timer.time("reward_calculation"):
                    rewards = repeated_batch["total_reward"]
                    baseline, std = calculate_baseline_and_std_per_prompt(
                        input_ids,
                        rewards,
                        torch.ones_like(rewards),
                        leave_one_out_baseline=grpo_config[
                            "use_leave_one_out_baseline"
                        ],
                    )

                    # Apply dynamic sampling
                    (
                        repeated_batch,
                        is_batch_complete,
                        batch_cache,
                        ds_metrics,
                    ) = dynamic_sampling(
                        repeated_batch,
                        std,
                        baseline,
                        dynamic_sampling_num_gen_batches,
                        master_config,
                        timer,
                        batch_cache,
                    )

                    if not is_batch_complete:
                        continue

                    rewards = (
                        repeated_batch["total_reward"]
                        if not grpo_config["use_dynamic_sampling"]
                        else repeated_batch["filtered_reward"]
                    )
                    baseline = repeated_batch["baseline"]
                    std = repeated_batch["std"]
                    advantages = (rewards - baseline).unsqueeze(-1)

                    if grpo_config["normalize_rewards"]:
                        advantages = normalize_advantages_with_epsilon(
                            advantages=advantages, std=std
                        )

                # Prepare training data
                with timer.time("data_processing"):
                    # Add loss mask and advantages to messages
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        for j, message in enumerate(message_log):
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )
                            message["advantages"] = advantages[i].expand(
                                message["token_ids"].shape
                            )

                    # Convert to flat messages for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=policy_config[
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data
                    from nemo_rl.algorithms.loss_functions import (
                        ClippedPGLossDataDict,
                    )

                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "advantages": flat_messages["advantages"],
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    train_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
                    train_data.to("cpu")

                # Compute logprobs
                print("▶ Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    trainer.policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    fprop_logprobs = trainer.policy.get_logprobs(train_data)["logprobs"]
                    reference_logprobs = trainer.policy.get_reference_policy_logprobs(
                        train_data
                    )["reference_logprobs"]
                    train_data["prev_logprobs"] = fprop_logprobs
                    train_data["reference_policy_logprobs"] = reference_logprobs

                # Train using the trainer interface
                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    # Use new interface signature: forward_backward(data, loss_fn, datastream_id=None)
                    train_results = trainer.forward_backward(
                        data=train_data,
                        loss_fn=loss_fn,
                        datastream_id=f"step_{current_step}",
                    )
                    sampler_stale = True  # Mark sampler as needing weight sync

                # Recompute KV scales after training if needed
                if sync_kv_scales:
                    with timer.time("recompute_kv_scales"):
                        print(
                            "▶ Recomputing KV cache scales after policy update...",
                            flush=True,
                        )
                        kv_scales_cache = trainer.policy.calibrate_qkv_fp8_scales(
                            train_data, include_q=True
                        )["layers"]
                        sampler_stale = True

                # Validation
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    if sampler_stale:
                        # Sync weights using interface-compatible method
                        if hasattr(sampler, "weight_sync"):
                            # NeMoSampler has weight_sync() for backward compatibility
                            sampler.weight_sync(kv_scales=kv_scales_cache, timer=timer)  # type: ignore
                        elif hasattr(sampler, "_trainer") and sampler._trainer is not None:
                            # For Mock/Tinker samplers, use trainer's save_weights_and_get_sampling_client
                            sampler._trainer.save_weights_and_get_sampling_client(
                                kv_scales=kv_scales_cache, timer=timer
                            )
                        sampler_stale = False
                    val_metrics, validation_timings = validate(
                        sampler.generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    sampler.generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(val_metrics, total_steps + 1, prefix="validation")

                # Log metrics
                logger.log_metrics(train_results, total_steps + 1, prefix="train")
                if ds_metrics:
                    logger.log_metrics(ds_metrics, total_steps + 1)

                # Update state
                current_step += 1
                total_steps += 1
                consumed_samples += train_data.size
                timeout.mark_iteration()

                # Checkpointing
                is_last_step = (total_steps >= max_num_steps) or (
                    (current_epoch + 1 == max_num_epochs)
                    and (current_step == len(dataloader))
                )
                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"] == 0
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    trainer.policy.prepare_for_training()

                    # +1 because step is 0-indexed
                    grpo_save_state["current_step"] = current_step + 1
                    grpo_save_state["total_steps"] = total_steps + 1
                    grpo_save_state["current_epoch"] = current_epoch + 1
                    grpo_save_state["consumed_samples"] = consumed_samples

                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        total_steps + 1, grpo_save_state, master_config
                    )
                    trainer.policy.save_checkpoint(
                        weights_path=os.path.join(
                            checkpoint_path, "policy", "weights"
                        ),
                        optimizer_path=os.path.join(
                            checkpoint_path, "policy", "optimizer"
                        ),
                        tokenizer_path=os.path.join(
                            checkpoint_path, "policy", "tokenizer"
                        ),
                        checkpointing_cfg=master_config["checkpointing"],
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)

        current_epoch += 1
        current_step = 0

