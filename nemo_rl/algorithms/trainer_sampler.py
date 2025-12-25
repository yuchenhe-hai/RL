# no copy write yet will add later 
"""Abstraction layer for trainer and sampler interfaces.

This module provides a clean interface for creating and using trainers and samplers
in the GRPO algorithm, abstracting away the underlying implementation details.

Key Points for Compatibility with grpo.py:
    1. Policy must be initialized FIRST with the same configs used in grpo.py setup()
       (cluster, config=policy_config, tokenizer, processor, weights_path, optimizer_path, init_optimizer=True)
    
    2. When creating trainer, MUST provide:
       - policy: Already initialized policy instance
       - colocated_inference: Should match generation_config["colocated"]["enabled"]
       - refit_fn: Must be refit_policy_generation from nemo_rl.algorithms.grpo
                   (Required for weight_sync() to work)
    
    3. Weight updates work via trainer.weight_sync(sampler) which calls refit_policy_generation
       internally, using the same mechanism as grpo.py (IPC ZMQ for colocated, NCCL for non-colocated)
"""
from typing import Any, Callable, Optional, TypedDict

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface


class TrainerConfig(TypedDict):
    """Configuration for creating a trainer.
    
    This config should match the policy_config used in grpo.py setup.
    The policy itself is initialized separately and passed to create_trainer.
    """

    pass  # Policy configuration is handled by PolicyConfig in policy initialization


class SamplingConfig(TypedDict):
    """Configuration for creating a sampler."""

    pass  # Can be extended with sampler-specific config


class OptimizerConfig(TypedDict):
    """Configuration for optimizer step."""

    pass  # Can be extended with optimizer-specific config


class Trainer:
    """Wrapper around ColocatablePolicyInterface providing a clean training interface."""

    def __init__(
        self,
        policy: ColocatablePolicyInterface,
        colocated_inference: bool = False,
        refit_fn: Optional[
            Callable[
                [
                    ColocatablePolicyInterface,
                    GenerationInterface,
                    bool,
                    Optional[int],
                    Optional[Any],
                    Optional[dict[str, float]],
                ],
                None,
            ]
        ] = None,
    ):
        """Initialize trainer with a policy.

        Args:
            policy: The underlying policy that implements training operations
            colocated_inference: Whether inference is colocated with training
            refit_fn: Optional function for refitting/syncing weights to sampler.
                     If None, weight_sync will be a no-op.
        """
        self._policy = policy
        self._colocated_inference = colocated_inference
        self._refit_fn = refit_fn
        self._pending_gradients = False  # Track if we have pending gradients from forward_backward

    @property
    def policy(self) -> ColocatablePolicyInterface:
        """Access to the underlying policy for advanced operations."""
        return self._policy

    def forward_backward(
        self,
        datastream_id: str,
        loss_fn: LossFunction,
        data: Optional[BatchedDataDict] = None,
    ) -> dict[str, Any]:
        """Perform forward and backward pass on the given data.

        Wraps the underlying policy.train() method which performs:
        1. Forward pass through the model
        2. Loss computation via loss_fn
        3. Backward pass to compute gradients
        4. Optimizer step to update parameters

        Implementation Note:
            The current underlying policy.train() implementation performs both
            forward/backward AND optimization together in a single call. This method
            wraps policy.train() to provide a cleaner interface. The optimize() method
            is currently a no-op since optimization happens within this method.
            
            In the future, if the policy interface is extended to support true separation,
            this method would only do forward/backward, and optimize() would handle
            the optimizer step separately.

        Args:
            datastream_id: Identifier for the data stream (for future use with streaming).
                          Currently unused, but provided for API consistency.
            loss_fn: Loss function to use for training
            data: Training data batch. If None, data should be fetched using datastream_id

        Returns:
            Dictionary containing training metrics (loss, grad_norm, etc.) from policy.train()
        """
        if data is None:
            raise ValueError(
                "data must be provided. Future support for datastream_id-based data fetching not yet implemented."
            )
        
        # Prepare for training (set model to train mode, reload optimizer to GPU)
        self._policy.prepare_for_training()
        
        # Wrap the existing policy.train() method
        # This performs: forward pass -> loss computation -> backward pass -> optimizer step
        # TODO: In future, if policy interface supports separation, we would:
        #   1. Call policy.forward_backward(data, loss_fn) to only compute gradients
        #   2. Set self._pending_gradients = True
        #   3. Let optimize() handle the optimizer step
        train_results = self._policy.train(data, loss_fn)
        
        # In current implementation, optimizer step already happened in train()
        self._pending_gradients = False
        
        return train_results

    def optimize(self, optimizer_config: Optional[OptimizerConfig] = None) -> None:
        """Perform optimizer step.

        This method should be called after forward_backward() to update model parameters.

        Implementation Note:
            Currently, optimization is performed within forward_backward() via
            policy.train(). This method is provided for API consistency and future
            compatibility. When the policy interface supports true separation of
            forward_backward and optimization, this method will perform the actual
            optimizer step.

        Args:
            optimizer_config: Optional optimizer configuration (currently unused)
        """
        # In current implementation, optimizer step is done within policy.train()
        # which is called in forward_backward(). This is a placeholder for future
        # separation when policy interface supports it.
        
        # TODO: When policy interface supports separation:
        #   if self._pending_gradients:
        #       self._policy.optimizer_step()
        #       self._pending_gradients = False
        #   else:
        #       warnings.warn("optimize() called without pending gradients from forward_backward()")
        
        pass

    def weight_sync(
        self,
        sampler: Optional["Sampler"] = None,
        kv_scales: Optional[dict[str, float]] = None,
        timer: Optional[Any] = None,
    ) -> None:
        """Synchronize weights with a sampler (if provided).

        This triggers weight synchronization from trainer to sampler when needed.
        Uses the refit_fn (typically refit_policy_generation) provided during
        trainer creation.

        This method wraps the refit_policy_generation function from grpo.py,
        which handles weight synchronization between policy (trainer) and
        policy_generation (sampler) workers.

        Args:
            sampler: Optional sampler to sync weights to. If None, this is a no-op.
            kv_scales: Optional dictionary of KV cache scales for FP8 quantization.
                     Should be computed via policy.calibrate_qkv_fp8_scales() if needed.
            timer: Optional timer for timing the weight sync operation.
                  Should match the Timer used in grpo training loop.

        Raises:
            RuntimeError: If weight synchronization fails (from refit_policy_generation)
            ValueError: If refit_fn was not provided during trainer creation
        """
        if sampler is None:
            return  # No-op if no sampler provided
        
        if self._refit_fn is None:
            raise ValueError(
                "Cannot sync weights: refit_fn was not provided during trainer creation. "
                "Provide refit_policy_generation from nemo_rl.algorithms.grpo when creating trainer."
            )
        
        # Call refit_policy_generation with the same signature as in grpo.py
        # This handles weight synchronization via IPC ZMQ (colocated) or NCCL (non-colocated)
        self._refit_fn(
            self._policy,                    # policy: ColocatablePolicyInterface
            sampler.generation,              # policy_generation: GenerationInterface
            self._colocated_inference,       # colocated_inference: bool
            None,                            # _refit_buffer_size_gb: Optional[int] (use default)
            timer,                           # timer: Optional[Timer]
            kv_scales,                       # kv_scales: Optional[dict[str, float]]
        )


class Sampler:
    """Wrapper around GenerationInterface providing a clean sampling interface."""

    def __init__(
        self,
        generation: GenerationInterface,
        trainer: Optional[Trainer] = None,
    ):
        """Initialize sampler with a generation interface.

        Args:
            generation: The underlying generation interface that implements sampling
            trainer: Optional trainer reference for weight synchronization
        """
        self._generation = generation
        self._trainer = trainer
        self._stale = True  # Track if weights need sync

    @property
    def generation(self) -> GenerationInterface:
        """Access to the underlying generation interface for advanced operations."""
        return self._generation

    def stream(
        self,
        input_data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
        require_sync: bool = True,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Stream/generate responses for the given input data.

        Args:
            input_data: Input data containing prompts for generation
            greedy: Whether to use greedy decoding (True) or sampling (False)
            require_sync: If True and weights are stale, sync weights before generation

        Returns:
            Generated responses with output_ids, logprobs, etc.
        """
        # Sync weights if needed
        if require_sync and self._stale and self._trainer is not None:
            self.weight_sync()
            self._stale = False

        # Prepare for generation
        self._generation.prepare_for_generation()
        # Generate responses
        output = self._generation.generate(input_data, greedy=greedy)
        # Finish generation
        self._generation.finish_generation()
        return output

    def weight_sync(
        self,
        kv_scales: Optional[dict[str, float]] = None,
        timer: Optional[Any] = None,
    ) -> None:
        """Synchronize weights from trainer to sampler.

        This should be called when trainer weights have been updated.

        Args:
            kv_scales: Optional dictionary of KV cache scales for FP8 quantization
            timer: Optional timer for timing the weight sync operation
        """
        if self._trainer is not None:
            self._trainer.weight_sync(
                sampler=self, kv_scales=kv_scales, timer=timer
            )
            self._stale = False

    def mark_stale(self) -> None:
        """Mark sampler weights as stale (needing sync)."""
        self._stale = True


def create_trainer(
    checkpoint: Optional[str],
    trainer_config: TrainerConfig,
    policy: ColocatablePolicyInterface,
    colocated_inference: bool = False,
    refit_fn: Optional[
        Callable[
            [
                ColocatablePolicyInterface,
                GenerationInterface,
                bool,
                Optional[int],
                Optional[Any],
                Optional[dict[str, float]],
            ],
            None,
        ]
    ] = None,
) -> Trainer:
    """Factory function to create a trainer instance.

    This function wraps an already-initialized policy (created with Policy config
    from grpo.py setup) into the Trainer interface for clean API usage.

    Usage example matching grpo.py setup:
        ```python
        # After initializing policy in grpo.py setup (lines 464-476):
        # policy = Policy(cluster=train_cluster, config=policy_config, ...)
        
        from nemo_rl.algorithms.grpo import refit_policy_generation
        
        trainer = create_trainer(
            checkpoint=last_checkpoint_path,  # Same as used in policy init
            trainer_config=TrainerConfig(),   # Empty, policy config handled separately
            policy=policy,                    # Already initialized policy
            colocated_inference=colocated_inference,  # From generation_config
            refit_fn=refit_policy_generation,  # REQUIRED for weight sync to work
        )
        ```

    Args:
        checkpoint: Optional path to checkpoint for loading weights.
                   Should match the checkpoint path used when initializing policy.
                   Note: Checkpoint loading is done during policy initialization,
                   this is just for reference/tracking.
        trainer_config: Configuration for the trainer (currently unused, but kept
                       for API consistency. Policy configuration is handled separately
                       when creating the Policy instance).
        policy: The policy instance to wrap (must be already initialized with
               the same configs used in grpo.py setup, including cluster, config,
               tokenizer, processor, weights_path, optimizer_path, init_optimizer=True).
        colocated_inference: Whether inference is colocated with training.
                            Should match generation_config["colocated"]["enabled"]
                            from grpo.py setup. Required for correct weight sync behavior.
        refit_fn: Function for refitting/syncing weights to sampler.
                 MUST be provided (typically refit_policy_generation from grpo module)
                 for weight_sync() to work correctly. If None, weight_sync() will be a no-op.

    Returns:
        Trainer instance that can be used for training and weight synchronization

    Raises:
        ValueError: If policy is not properly initialized (will fail during usage)
    """
    # Verify that policy is initialized (basic check)
    if policy is None:
        raise ValueError("policy must be provided and already initialized")
    
    # Checkpoint loading is done during policy initialization, not here
    # This factory just wraps the policy in the Trainer interface
    # The colocated_inference flag and refit_fn are critical for weight sync to work
    if refit_fn is None:
        import warnings
        warnings.warn(
            "refit_fn is None. weight_sync() will not work. "
            "Provide refit_policy_generation from nemo_rl.algorithms.grpo for weight synchronization.",
            UserWarning,
        )
    
    return Trainer(policy, colocated_inference=colocated_inference, refit_fn=refit_fn)


def create_sampler(
    checkpoint: Optional[str],
    sampling_config: SamplingConfig,
    generation: Optional[GenerationInterface] = None,
    trainer: Optional[Trainer] = None,
) -> Sampler:
    """Factory function to create a sampler instance.

    Can create a standalone sampler or one linked to a trainer.

    Args:
        checkpoint: Optional path to checkpoint for loading weights
        sampling_config: Configuration for the sampler
        generation: The generation interface instance to wrap (already initialized).
                    If None and trainer is provided, uses trainer's policy if it implements GenerationInterface.
        trainer: Optional trainer reference for combined trainer-sampler setup.
                If provided, enables automatic weight synchronization.

    Returns:
        Sampler instance
    """
    if generation is None:
        if trainer is not None:
            # Try to use trainer's policy as generation interface if it implements it
            policy = trainer.policy
            if isinstance(policy, GenerationInterface):
                generation = policy  # type: ignore
            else:
                raise ValueError(
                    "generation must be provided if trainer.policy does not implement GenerationInterface"
                )
        else:
            raise ValueError(
                "Either generation or trainer must be provided to create_sampler"
            )

    # Checkpoint loading is assumed to be done during generation initialization
    # This factory just wraps the generation in the Sampler interface
    return Sampler(generation, trainer=trainer)


# ===============================================================================
# Usage Example
# ===============================================================================
#
# Example integration into setup function (around lines 464-600 in grpo.py):
#
# ```python
# from nemo_rl.algorithms.trainer_sampler import (
#     create_trainer,
#     create_sampler,
#     TrainerConfig,
#     SamplingConfig,
# )
# from nemo_rl.algorithms.grpo import refit_policy_generation
#
# # In grpo.py setup(), after initializing policy and policy_generation:
# # (around lines 494-563, after policy and policy_generation are created)
#
# # Extract colocated_inference from config (same as used in grpo.py setup)
# colocated_inference = generation_config["colocated"]["enabled"]
#
# # Create trainer with same configs as grpo.py
# # The policy must be initialized first with the same configs:
# #   policy = Policy(
# #       cluster=train_cluster,
# #       config=policy_config,  # From master_config["policy"]
# #       tokenizer=tokenizer,
# #       processor=processor,
# #       weights_path=weights_path,  # From checkpoint
# #       optimizer_path=optimizer_path,  # From checkpoint
# #       init_optimizer=True,
# #   )
#
# trainer = create_trainer(
#     checkpoint=last_checkpoint_path,  # Same checkpoint path used for policy init
#     trainer_config=TrainerConfig(),   # Empty config (policy config handled separately)
#     policy=policy,                    # Already initialized policy from grpo setup
#     colocated_inference=colocated_inference,  # CRITICAL: Must match generation_config
#     refit_fn=refit_policy_generation,  # CRITICAL: Required for weight_sync() to work
# )
#
# # Create sampler, optionally linked to trainer for automatic weight sync
# if policy_generation is not None:
#     sampler = create_sampler(
#         checkpoint=last_checkpoint_path,
#         sampling_config=SamplingConfig(),
#         generation=policy_generation,  # Already initialized VllmGeneration or Policy
#         trainer=trainer,  # Link to trainer enables weight sync via trainer.weight_sync()
#     )
# else:
#     # Standalone sampler (uses policy as generation interface, e.g., Megatron backend)
#     sampler = create_sampler(
#         checkpoint=last_checkpoint_path,
#         sampling_config=SamplingConfig(),
#         trainer=trainer,  # Will use trainer.policy as generation (must implement GenerationInterface)
#     )
#
# # Usage in training loop:
# # 1. Generate responses
# response = sampler.stream(input_data, greedy=False)
#
# # 2. Train on data (forward_backward wraps policy.train() which does forward,
# #    backward, AND optimizer step in one call)
# train_results = trainer.forward_backward(
#     datastream_id="batch_0",  # For future use with datastreams
#     loss_fn=loss_fn,
#     data=train_data,
# )
# # Note: train_results contains metrics like 'loss', 'grad_norm', etc.
#
# # 3. Optimize (currently a no-op since optimization happens in forward_backward,
# #    but included for API consistency and future compatibility)
# trainer.optimize(optimizer_config=None)
#
# # 4. Sync weights when needed (e.g., after training step)
# sampler.mark_stale()  # Mark sampler as needing weight sync
# trainer.weight_sync(sampler=sampler, kv_scales=kv_scales_cache, timer=timer)
# # Or use sampler's weight_sync:
# sampler.weight_sync(kv_scales=kv_scales_cache, timer=timer)
#
# # Alternative: If sampler is linked to trainer, it can auto-sync when streaming:
# # sampler.stream(input_data, require_sync=True)  # Will sync if stale
# ```

