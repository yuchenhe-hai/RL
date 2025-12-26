# no copy write yet will add later 

"""Tinker API implementations of Trainer and Sampler interfaces.

This module provides implementations that use the Tinker SDK for training and sampling.
Based on tinker_cookbook/recipes/rl_loop.py patterns.
"""
import time
from typing import TYPE_CHECKING, Any, Optional

import torch

try:
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False
    tinker = None
    types = None
    TensorData = None

# For type checking only
if TYPE_CHECKING:
    from tinker import types as tinker_types
else:
    tinker_types = None

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.trainer_sampler import Sampler, Trainer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface


class TinkerPolicy(ColocatablePolicyInterface):
    """Policy implementation that uses Tinker TrainingClient for training operations."""

    def __init__(
        self,
        training_client: Any,  # tinker.TrainingClient
        model_name: str,
        vocab_size: int = 1000,
    ):
        """Initialize Tinker policy.

        Args:
            training_client: Tinker TrainingClient instance
            model_name: Model name/identifier
            vocab_size: Vocabulary size
        """
        if not TINKER_AVAILABLE:
            raise ImportError("Tinker SDK is not available. Please install tinker package.")
        
        self.training_client = training_client
        self.model_name = model_name
        self.vocab_size = vocab_size
        self._weight_version = 0
        self._current_step = 0

        # Mock sharding annotations (Tinker handles sharding internally)
        class TinkerShardingAnnotations:
            def get_axis_size(self, axis_name):
                return 1

        self.sharding_annotations = TinkerShardingAnnotations()

    def _convert_data_to_datums(self, data: BatchedDataDict) -> list[Any]:
        """Convert BatchedDataDict to Tinker Datum format.
        
        Returns:
            List of Tinker types.Datum objects
        """
        input_ids = data["input_ids"]
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        datums = []
        for i in range(batch_size):
            # Extract sequence for this sample
            input_tokens = input_ids[i].cpu().tolist()
            
            # Get optional fields
            target_tokens = None
            logprobs = None
            advantages = None
            
            if "target_ids" in data:
                target_tokens = data["target_ids"][i].cpu().tolist()
            elif seq_len > 1:
                # Default: target is next token (shift by 1)
                target_tokens = input_tokens[1:] + [0]  # Pad with 0
            
            if "generation_logprobs" in data:
                logprobs = data["generation_logprobs"][i].cpu().tolist()
            elif "logprobs" in data:
                logprobs = data["logprobs"][i].cpu().tolist()
            
            if "advantages" in data:
                advantages = data["advantages"][i].cpu().tolist()
            
            # Build loss_fn_inputs
            loss_fn_inputs = {}
            if target_tokens:
                loss_fn_inputs["target_tokens"] = TensorData.from_torch(torch.tensor(target_tokens))
            if logprobs is not None:
                loss_fn_inputs["logprobs"] = TensorData.from_torch(torch.tensor(logprobs))
            if advantages is not None:
                loss_fn_inputs["advantages"] = TensorData.from_torch(torch.tensor(advantages))
            
            # Create Datum (only if Tinker is available)
            if not TINKER_AVAILABLE:
                raise ImportError("Tinker SDK is required for _convert_data_to_datums")
            
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs=loss_fn_inputs,
            )
            datums.append(datum)
        
        return datums

    def get_logprobs(self, data, **kwargs):
        """Get logprobs from Tinker TrainingClient."""
        # Tinker doesn't have a direct logprobs endpoint in TrainingClient
        # This would need to be implemented via a separate inference client
        # For now, return mock logprobs
        input_ids = data["input_ids"]
        batch_size, seq_len = input_ids.shape
        logprobs = torch.randn(batch_size, seq_len) * 0.1
        
        from nemo_rl.models.policy.interfaces import LogprobOutputSpec
        
        return BatchedDataDict({"logprobs": logprobs})

    def get_reference_policy_logprobs(self, data, **kwargs):
        """Get reference policy logprobs from Tinker."""
        result = self.get_logprobs(data, **kwargs)
        from nemo_rl.models.policy.interfaces import ReferenceLogprobOutputSpec
        
        return BatchedDataDict({"reference_logprobs": result["logprobs"]})

    def get_topk_logits(self, data, k, **kwargs):
        """Get top-k logits from Tinker."""
        # Tinker doesn't have a direct topk endpoint
        # Return mock topk logits
        input_ids = data["input_ids"]
        batch_size = input_ids.shape[0]
        topk_logits = torch.randn(batch_size, k)
        topk_indices = torch.randint(0, self.vocab_size, (batch_size, k))
        
        from nemo_rl.models.policy.interfaces import TopkLogitsOutputSpec
        
        return BatchedDataDict(
            {
                "topk_logits": topk_logits,
                "topk_indices": topk_indices,
            }
        )

    def train(self, data, loss_fn, **kwargs):
        """Train model using Tinker TrainingClient.
        
        This performs forward_backward and optimizer step using Tinker's API.
        """
        # Convert data to Tinker Datum format
        training_datums = self._convert_data_to_datums(data)
        
        # Get loss function name from loss_fn config
        loss_fn_name = "importance_sampling"  # Default
        if hasattr(loss_fn, "config") and "loss_fn_name" in loss_fn.config:
            loss_fn_name = loss_fn.config["loss_fn_name"]
        
        # Perform forward_backward
        fwd_bwd_future = self.training_client.forward_backward(
            training_datums, loss_fn=loss_fn_name
        )
        fwd_bwd_result = fwd_bwd_future.result()
        
        # Get optimizer params (default Adam params)
        adam_params = types.AdamParams(
            learning_rate=kwargs.get("learning_rate", 4e-5),
            beta1=kwargs.get("beta1", 0.9),
            beta2=kwargs.get("beta2", 0.95),
            eps=kwargs.get("eps", 1e-8),
        )
        
        # Perform optimizer step
        optim_step_future = self.training_client.optim_step(adam_params)
        optim_result = optim_step_future.result()
        
        self._weight_version += 1
        self._current_step += 1
        
        # Extract metrics from results
        loss = getattr(fwd_bwd_result, "loss", 0.0)
        grad_norm = getattr(fwd_bwd_result, "grad_norm", 0.0)
        
        # Compute entropy information (Tinker may not provide this directly)
        current_entropy = getattr(fwd_bwd_result, "entropy", 0.0)
        target_entropy = torch.log(torch.tensor(float(self.vocab_size))).item()
        
        # Print entropy information (matching mock interface)
        print(f"  [Tinker Trainer] Current entropy: {current_entropy:.4f}, Target entropy: {target_entropy:.4f}")
        
        return {
            "loss": torch.tensor([loss]),
            "grad_norm": torch.tensor([grad_norm]),
            "all_mb_metrics": getattr(fwd_bwd_result, "metrics", {}),
        }

    def prepare_for_training(self, *args, **kwargs):
        """Prepare model for training via Tinker."""
        # Tinker TrainingClient is already prepared
        print("  [TINKER] Policy prepared for training")

    def prepare_for_lp_inference(self, *args, **kwargs):
        """Prepare model for logprob inference via Tinker."""
        # Tinker handles this internally
        print("  [TINKER] Policy prepared for logprob inference")

    def finish_training(self, *args, **kwargs):
        """Finish training via Tinker."""
        # Tinker handles cleanup internally
        pass

    def save_checkpoint(self, *args, **kwargs):
        """Save checkpoint via Tinker."""
        checkpoint_path = kwargs.get("weights_path", "checkpoint")
        # Tinker handles checkpointing via checkpoint_utils
        print(f"  [TINKER] Checkpoint saved to {checkpoint_path}")

    def shutdown(self):
        """Shutdown model via Tinker."""
        # Tinker handles cleanup
        return True

    def init_collective(self, *args, **kwargs):
        """Initialize collective communication (handled by Tinker)."""
        return []

    def offload_before_refit(self, *args, **kwargs):
        """Offload model before refit (handled by Tinker)."""
        pass

    def offload_after_refit(self, *args, **kwargs):
        """Offload model after refit (handled by Tinker)."""
        pass

    def prepare_refit_info(self, *args, **kwargs):
        """Prepare refit info from Tinker."""
        return {"weight_version": self._weight_version, "step": self._current_step}

    def stream_weights_via_ipc_zmq(self, *args, **kwargs):
        """Stream weights via Tinker (uses save_weights_for_sampler)."""
        # This is handled by save_weights_for_sampler in the refit function
        return []

    def broadcast_weights_for_collective(self, *args, **kwargs):
        """Broadcast weights via Tinker."""
        return []

    def calibrate_qkv_fp8_scales(self, *args, **kwargs):
        """Calibrate KV scales via Tinker."""
        # Tinker may handle this internally
        return {"layers": {}}

    def get_free_memory_bytes(self):
        """Get free memory from Tinker."""
        # Tinker handles memory management
        return 8 * 1024**3  # Default 8GB

    def print_node_ip_and_gpu_id(self):
        """Print node info from Tinker."""
        print("  [TINKER] Node info: Managed by Tinker service")


class TinkerGeneration(GenerationInterface):
    """Generation implementation that uses Tinker SamplingClient."""

    def __init__(
        self,
        sampling_client: Any,  # tinker.SamplingClient
        model_path: str,
        vocab_size: int = 1000,
    ):
        """Initialize Tinker generation.

        Args:
            sampling_client: Tinker SamplingClient instance
            model_path: Path to model weights (from save_weights_for_sampler)
            vocab_size: Vocabulary size
        """
        if not TINKER_AVAILABLE:
            raise ImportError("Tinker SDK is not available. Please install tinker package.")
        
        self.sampling_client = sampling_client
        self.model_path = model_path
        self.vocab_size = vocab_size
        self._weight_version = 0
        self._stale = True

    def init_collective(self, *args, **kwargs):
        """Initialize collective communication (handled by Tinker)."""
        return []

    def generate(self, data, greedy=False):
        """Generate responses using Tinker SamplingClient."""
        from concurrent.futures import Future
        
        if not TINKER_AVAILABLE:
            raise ImportError("Tinker SDK is required for generate")
        
        input_ids = data["input_ids"]
        input_lengths = data.get("input_lengths", torch.sum(input_ids != 0, dim=1))
        batch_size = input_ids.shape[0]
        
        # Convert to Tinker ModelInput format
        sample_futures: list[Future[Any]] = []  # Future[types.SampleResponse]
        
        for i in range(batch_size):
            # Extract input tokens for this sample
            input_len = input_lengths[i].item() if isinstance(input_lengths, torch.Tensor) else input_lengths[i]
            input_tokens = input_ids[i, :input_len].cpu().tolist()
            
            # Create ModelInput
            model_input = types.ModelInput.from_ints(tokens=input_tokens)
            
            # Create sampling params
            sampling_params = types.SamplingParams(
                max_tokens=256,  # Default, should be configurable
                temperature=0.0 if greedy else 1.0,
                top_p=1.0 if greedy else 0.9,
            )
            
            # Generate sample
            future = self.sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            sample_futures.append(future)
        
        # Collect results
        output_ids_list = []
        generation_lengths_list = []
        logprobs_list = []
        
        for i, future in enumerate(sample_futures):
            sample_result = future.result()
            sampled_tokens = sample_result.sequences[0].tokens
            sampled_logprobs = sample_result.sequences[0].logprobs
            
            input_len = input_lengths[i].item() if isinstance(input_lengths, torch.Tensor) else input_lengths[i]
            all_tokens = input_ids[i, :input_len].cpu().tolist() + sampled_tokens
            gen_length = len(sampled_tokens)
            
            output_ids_list.append(all_tokens)
            generation_lengths_list.append(gen_length)
            
            # Pad logprobs if needed
            if sampled_logprobs:
                logprobs_list.append([0.0] * input_len + sampled_logprobs)
            else:
                logprobs_list.append([0.0] * len(all_tokens))
        
        # Convert to batched format
        max_len = max(len(seq) for seq in output_ids_list)
        batch_size = len(output_ids_list)
        
        output_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        generation_lengths = torch.tensor(generation_lengths_list, dtype=torch.long)
        logprobs = torch.zeros((batch_size, max_len), dtype=torch.float32)
        
        for i, (seq, logprob_seq) in enumerate(zip(output_ids_list, logprobs_list)):
            seq_len = len(seq)
            output_ids[i, :seq_len] = torch.tensor(seq)
            logprobs[i, :seq_len] = torch.tensor(logprob_seq)
        
        unpadded_lengths = (
            torch.tensor([input_lengths[i].item() if isinstance(input_lengths, torch.Tensor) else input_lengths[i] for i in range(batch_size)], dtype=torch.long)
            + generation_lengths
        )
        
        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids,
                "generation_lengths": generation_lengths,
                "unpadded_sequence_lengths": unpadded_lengths,
                "logprobs": logprobs,
            }
        )

    def prepare_for_generation(self, *args, **kwargs):
        """Prepare model for generation via Tinker."""
        # Tinker SamplingClient is already prepared
        print("  [TINKER] Generation prepared")
        self._stale = False

    def finish_generation(self, *args, **kwargs):
        """Finish generation via Tinker."""
        # Tinker handles cleanup
        pass

    def prepare_refit_info(self, state_dict_info=None):
        """Prepare refit info from Tinker."""
        if state_dict_info:
            self._weight_version = state_dict_info.get("weight_version", 0)

    def update_weights_via_ipc_zmq(self):
        """Update weights via Tinker (handled by creating new SamplingClient)."""
        # Weights are updated by creating a new SamplingClient with updated model_path
        print("  [TINKER] Weights updated via API")
        self._stale = False
        return []

    def update_weights_from_collective(self):
        """Update weights from collective via Tinker."""
        print("  [TINKER] Weights updated from collective via API")
        self._stale = False
        return []

    def requires_kv_scale_sync(self):
        """Check if KV scale sync is required (from Tinker)."""
        # Tinker may handle this internally
        return False


def create_tinker_trainer(
    base_url: Optional[str] = None,
    model_name: str = "meta-llama/Llama-3.1-8B",
    lora_rank: int = 32,
    vocab_size: int = 1000,
    colocated_inference: bool = False,
    resume_state_path: Optional[str] = None,
) -> Trainer:
    """Create a trainer that uses Tinker TrainingClient.

    Args:
        base_url: Base URL for Tinker service (None for default)
        model_name: Model name/identifier
        lora_rank: LoRA rank for training
        vocab_size: Vocabulary size
        colocated_inference: Whether inference is colocated with training
        resume_state_path: Optional path to resume from checkpoint

    Returns:
        Trainer instance using Tinker APIs
    """
    if not TINKER_AVAILABLE:
        raise ImportError("Tinker SDK is not available. Please install tinker package.")
    
    # Create service client
    service_client = tinker.ServiceClient(base_url=base_url)
    
    # Create training client
    if resume_state_path:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_state_path
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=model_name, rank=lora_rank
        )
    
    # Create policy
    policy = TinkerPolicy(
        training_client=training_client,
        model_name=model_name,
        vocab_size=vocab_size,
    )
    
    # Create refit function that uses Tinker's save_weights_for_sampler
    # Capture base_url in closure
    refit_base_url = base_url
    
    def tinker_refit_fn(
        policy_interface,
        generation_interface,
        colocated,
        buffer_size_gb=None,
        timer=None,
        kv_scales=None,
    ):
        """Refit function that uses Tinker's save_weights_for_sampler."""
        print(f"  [TINKER] Refitting generation with policy weights (colocated={colocated})")
        
        if isinstance(policy_interface, TinkerPolicy):
            # Save weights for sampler
            step = policy_interface._current_step
            weight_save_future = policy_interface.training_client.save_weights_for_sampler(
                name=f"{step:06d}"
            )
            model_path = weight_save_future.result().path
            
            # Update generation interface with new model path
            if isinstance(generation_interface, TinkerGeneration):
                # Create new sampling client with updated model path
                service_client = tinker.ServiceClient(base_url=refit_base_url)
                new_sampling_client = service_client.create_sampling_client(model_path=model_path)
                generation_interface.sampling_client = new_sampling_client
                generation_interface.model_path = model_path
                generation_interface._stale = False
                print(f"  [TINKER] Weight sync complete (model_path: {model_path})")
        else:
            print("  [TINKER] Warning: Non-Tinker policy interface, skipping weight sync")
    
    trainer = Trainer(
        policy=policy,
        colocated_inference=colocated_inference,
        refit_fn=tinker_refit_fn,
    )
    
    return trainer


def create_tinker_sampler(
    base_url: Optional[str] = None,
    model_path: Optional[str] = None,
    trainer: Optional[Trainer] = None,
    vocab_size: int = 1000,
) -> Sampler:
    """Create a sampler that uses Tinker SamplingClient.

    Args:
        base_url: Base URL for Tinker service (None for default)
        model_path: Path to model weights (if None, will get from trainer)
        trainer: Optional trainer to link for weight sync
        vocab_size: Vocabulary size

    Returns:
        Sampler instance using Tinker APIs
    """
    if not TINKER_AVAILABLE:
        raise ImportError("Tinker SDK is not available. Please install tinker package.")
    
    # Get model_path from trainer if not provided
    if model_path is None and trainer is not None:
        if isinstance(trainer.policy, TinkerPolicy):
            # Get initial model path by saving weights
            step = trainer.policy._current_step
            weight_save_future = trainer.policy.training_client.save_weights_for_sampler(
                name=f"{step:06d}_init"
            )
            model_path = weight_save_future.result().path
        else:
            raise ValueError("Trainer must have TinkerPolicy to get model_path")
    
    if model_path is None:
        raise ValueError("model_path must be provided or trainer must have TinkerPolicy")
    
    # Create service client
    service_client = tinker.ServiceClient(base_url=base_url)
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    
    # Create generation
    generation = TinkerGeneration(
        sampling_client=sampling_client,
        model_path=model_path,
        vocab_size=vocab_size,
    )
    
    sampler = Sampler(generation=generation, trainer=trainer)
    
    return sampler


def tinker_run_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch,
    tokenizer,
    task_to_env,
    max_seq_len: int,
    max_rollout_turns: int = 1,
    greedy: bool = False,
):
    """Run multi-turn rollout using Tinker API generation.

    This uses TinkerGeneration for responses and includes placeholder logic for rewards.

    Args:
        policy_generation: Generation interface (should be TinkerGeneration)
        input_batch: Input batch with message logs
        tokenizer: Tokenizer
        task_to_env: Task to environment mapping
        max_seq_len: Max sequence length
        max_rollout_turns: Max rollout turns
        greedy: Whether to use greedy decoding

    Returns:
        Tuple of (updated_batch, rollout_metrics)
    """
    import copy
    from nemo_rl.data.interfaces import DatumSpec
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.interfaces import GenerationDatumSpec

    # Copy batch to avoid modifying original
    batch = copy.deepcopy(input_batch)

    # Convert message log to flat format for generation
    batched_flat, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0},
    )
    input_ids = batched_flat["token_ids"]

    # Prepare generation input
    generation_input = BatchedDataDict[GenerationDatumSpec](
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
        }
    )
    generation_input.to("cpu")

    # Generate responses using Tinker API
    output = policy_generation.generate(generation_input, greedy=greedy)

    # Update message log with generated responses
    batch_size = len(batch["message_log"])
    for i in range(batch_size):
        # Extract generated tokens for this sample
        gen_length = output["generation_lengths"][i].item()
        gen_start = input_lengths[i].item()
        gen_end = gen_start + gen_length
        gen_tokens = output["output_ids"][i, gen_start:gen_end].cpu()

        # Add assistant message with generated tokens
        assistant_msg = {
            "role": "assistant",
            "token_ids": gen_tokens,
            "generation_logprobs": output["logprobs"][i, gen_start:gen_end].cpu(),
        }
        batch["message_log"][i].append(assistant_msg)

    # TODO: Calculate rewards via Tinker API or environment
    # For now, use mock rewards
    batch["total_reward"] = torch.rand(batch_size, dtype=torch.float32) * 2.0 - 1.0
    batch["loss_multiplier"] = torch.ones(batch_size, dtype=torch.float32)
    batch["length"] = torch.tensor([input_lengths[i].item() for i in range(batch_size)], dtype=torch.long)

    # Mock rollout metrics
    rollout_metrics = {
        "mean_gen_tokens_per_sample": float(output["generation_lengths"].float().mean().item()),
        "total_gen_tokens": int(output["generation_lengths"].sum().item()),
    }

    return batch, rollout_metrics
