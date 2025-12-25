# no copy write yet will add later 

"""Tinker API implementations of Trainer and Sampler interfaces.

This module provides implementations that call Tinker APIs for training and sampling.
Tinker APIs are external services that handle model training and inference operations.

To use this module:
1. Configure Tinker API endpoints and credentials
2. Implement the Tinker API client (see TinkerAPIClient class)
3. Use create_tinker_trainer() and create_tinker_sampler() to create instances
"""
import time
from typing import Any, Optional

import torch

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.trainer_sampler import Sampler, Trainer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface


class TinkerAPIClient:
    """Client for interacting with Tinker APIs.

    This class handles communication with Tinker API endpoints for:
    - Model training operations
    - Model inference/generation
    - Weight synchronization
    - Checkpoint management

    TODO: Implement actual API calls based on Tinker API documentation.
    """

    def __init__(
        self,
        api_endpoint: str,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        """Initialize Tinker API client.

        Args:
            api_endpoint: Base URL for Tinker API (e.g., "https://api.tinker.com/v1")
            api_key: Optional API key for authentication
            model_id: Optional model identifier
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_id = model_id
        self._session = None  # Placeholder for HTTP session

    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make HTTP request to Tinker API.

        TODO: Implement actual HTTP request logic using requests library or similar.

        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint path
            **kwargs: Additional request parameters

        Returns:
            Response data as dictionary
        """
        # Placeholder implementation
        # TODO: Replace with actual API call
        # Example:
        # import requests
        # url = f"{self.api_endpoint}/{endpoint}"
        # headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # response = requests.request(method, url, headers=headers, **kwargs)
        # return response.json()
        return {"status": "success", "data": {}}

    def train_step(self, data: dict[str, Any], loss_config: dict[str, Any]) -> dict[str, Any]:
        """Submit training step to Tinker API.

        Args:
            data: Training data dictionary
            loss_config: Loss function configuration

        Returns:
            Training results dictionary with loss, grad_norm, current_entropy, target_entropy, etc.
        """
        response = self._make_request(
            "POST",
            f"models/{self.model_id}/train",
            json={"data": data, "loss_config": loss_config},
        )
        
        # Ensure entropy information is included in response (for interface consistency)
        if "current_entropy" not in response:
            # If Tinker API doesn't provide entropy, compute from logits if available
            # Otherwise use placeholder values
            response["current_entropy"] = 0.0
        if "target_entropy" not in response:
            # Target entropy is log(vocab_size) for uniform distribution
            # This should ideally come from the model config, but use a default
            response["target_entropy"] = 6.9078  # log(1000) as default
        
        return response

    def generate(self, input_data: dict[str, Any], greedy: bool = False) -> dict[str, Any]:
        """Generate responses using Tinker API.

        Args:
            input_data: Input data for generation
            greedy: Whether to use greedy decoding

        Returns:
            Generation results with output_ids, logprobs, etc.
        """
        return self._make_request(
            "POST",
            f"models/{self.model_id}/generate",
            json={"input_data": input_data, "greedy": greedy},
        )

    def sync_weights(self, source_model_id: str, target_model_id: str) -> dict[str, Any]:
        """Synchronize weights between models via Tinker API.

        Args:
            source_model_id: Source model identifier
            target_model_id: Target model identifier

        Returns:
            Sync operation result
        """
        return self._make_request(
            "POST",
            "models/sync_weights",
            json={
                "source_model_id": source_model_id,
                "target_model_id": target_model_id,
            },
        )

    def get_logprobs(self, data: dict[str, Any]) -> dict[str, Any]:
        """Get log probabilities from Tinker API.

        Args:
            data: Input data for logprob computation

        Returns:
            Logprobs results
        """
        return self._make_request(
            "POST",
            f"models/{self.model_id}/logprobs",
            json={"data": data},
        )


class TinkerPolicy(ColocatablePolicyInterface):
    """Policy implementation that uses Tinker APIs for training operations."""

    def __init__(
        self,
        api_client: TinkerAPIClient,
        model_id: str,
        vocab_size: int = 1000,
    ):
        """Initialize Tinker policy.

        Args:
            api_client: Tinker API client instance
            model_id: Model identifier
            vocab_size: Vocabulary size
        """
        self.api_client = api_client
        self.model_id = model_id
        self.vocab_size = vocab_size
        self._weight_version = 0

        # Mock sharding annotations (Tinker API may provide this)
        class TinkerShardingAnnotations:
            def get_axis_size(self, axis_name):
                # TODO: Get actual sharding info from Tinker API
                return 1

        self.sharding_annotations = TinkerShardingAnnotations()

    def get_logprobs(self, data, **kwargs):
        """Get logprobs from Tinker API."""
        # Convert data to API format
        api_data = {
            "input_ids": data["input_ids"].cpu().numpy().tolist(),
            "input_lengths": data["input_lengths"].cpu().numpy().tolist(),
        }

        # Call Tinker API
        response = self.api_client.get_logprobs(api_data)

        # Convert response back to BatchedDataDict format
        logprobs = torch.tensor(response.get("logprobs", []))
        from nemo_rl.models.policy.interfaces import LogprobOutputSpec

        return BatchedDataDict({"logprobs": logprobs})

    def get_reference_policy_logprobs(self, data, **kwargs):
        """Get reference policy logprobs from Tinker API."""
        # For reference policy, use same API but with reference flag
        api_data = {
            "input_ids": data["input_ids"].cpu().numpy().tolist(),
            "input_lengths": data["input_lengths"].cpu().numpy().tolist(),
            "use_reference": True,
        }

        response = self.api_client.get_logprobs(api_data)
        reference_logprobs = torch.tensor(response.get("logprobs", []))

        from nemo_rl.models.policy.interfaces import ReferenceLogprobOutputSpec

        return BatchedDataDict({"reference_logprobs": reference_logprobs})

    def get_topk_logits(self, data, k, **kwargs):
        """Get top-k logits from Tinker API."""
        api_data = {
            "input_ids": data["input_ids"].cpu().numpy().tolist(),
            "input_lengths": data["input_lengths"].cpu().numpy().tolist(),
            "k": k,
        }

        # TODO: Add topk endpoint to Tinker API client
        response = self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/topk",
            json=api_data,
        )

        topk_logits = torch.tensor(response.get("topk_logits", []))
        topk_indices = torch.tensor(response.get("topk_indices", []))

        from nemo_rl.models.policy.interfaces import TopkLogitsOutputSpec

        return BatchedDataDict(
            {
                "topk_logits": topk_logits,
                "topk_indices": topk_indices,
            }
        )

    def train(self, data, loss_fn, **kwargs):
        """Train model using Tinker API.
        
        This method calls the Tinker API for training and prints entropy information
        instead of computing cross-entropy loss directly.
        """
        # Convert data to API format
        api_data = {
            "input_ids": data["input_ids"].cpu().numpy().tolist(),
            "input_lengths": data["input_lengths"].cpu().numpy().tolist(),
        }

        # Add optional fields if present
        if "advantages" in data:
            api_data["advantages"] = data["advantages"].cpu().numpy().tolist()
        if "generation_logprobs" in data:
            api_data["generation_logprobs"] = (
                data["generation_logprobs"].cpu().numpy().tolist()
            )
        if "prev_logprobs" in data:
            api_data["prev_logprobs"] = data["prev_logprobs"].cpu().numpy().tolist()
        if "reference_policy_logprobs" in data:
            api_data["reference_policy_logprobs"] = (
                data["reference_policy_logprobs"].cpu().numpy().tolist()
            )

        # Get loss config from loss function
        loss_config = {}
        if hasattr(loss_fn, "config"):
            loss_config = loss_fn.config

        # Call Tinker API for training
        response = self.api_client.train_step(api_data, loss_config)

        # Get entropy information from Tinker API response
        current_entropy = response.get("current_entropy", 0.0)
        target_entropy = response.get("target_entropy", torch.log(torch.tensor(float(self.vocab_size))).item())
        
        # Print entropy information (matching mock interface)
        print(f"  [Tinker Trainer] Current entropy: {current_entropy:.4f}, Target entropy: {target_entropy:.4f}")

        self._weight_version += 1

        # Convert response to expected format
        return {
            "loss": torch.tensor([response.get("loss", 0.0)]),
            "grad_norm": torch.tensor([response.get("grad_norm", 0.0)]),
            "all_mb_metrics": response.get("metrics", {}),
        }

    def prepare_for_training(self, *args, **kwargs):
        """Prepare model for training via Tinker API."""
        # TODO: Call Tinker API to prepare model
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/prepare_training",
        )
        print("  [TINKER] Policy prepared for training")

    def prepare_for_lp_inference(self, *args, **kwargs):
        """Prepare model for logprob inference via Tinker API."""
        # TODO: Call Tinker API to prepare for inference
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/prepare_inference",
        )
        print("  [TINKER] Policy prepared for logprob inference")

    def finish_training(self, *args, **kwargs):
        """Finish training via Tinker API."""
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/finish_training",
        )

    def save_checkpoint(self, *args, **kwargs):
        """Save checkpoint via Tinker API."""
        checkpoint_path = kwargs.get("weights_path", "checkpoint")
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/checkpoint",
            json={"checkpoint_path": str(checkpoint_path)},
        )
        print("  [TINKER] Checkpoint saved")

    def shutdown(self):
        """Shutdown model via Tinker API."""
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/shutdown",
        )
        return True

    def init_collective(self, *args, **kwargs):
        """Initialize collective communication (may not be needed for Tinker API)."""
        # Tinker API may handle collective communication internally
        return []

    def offload_before_refit(self, *args, **kwargs):
        """Offload model before refit via Tinker API."""
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/offload",
        )

    def offload_after_refit(self, *args, **kwargs):
        """Offload model after refit via Tinker API."""
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/offload",
        )

    def prepare_refit_info(self, *args, **kwargs):
        """Prepare refit info from Tinker API."""
        response = self.api_client._make_request(
            "GET",
            f"models/{self.model_id}/refit_info",
        )
        return response.get("refit_info", {"weight_version": self._weight_version})

    def stream_weights_via_ipc_zmq(self, *args, **kwargs):
        """Stream weights via Tinker API (may use different mechanism)."""
        # Tinker API may handle weight streaming differently
        print("  [TINKER] Weight streaming via API")
        return []

    def broadcast_weights_for_collective(self, *args, **kwargs):
        """Broadcast weights via Tinker API."""
        print("  [TINKER] Weight broadcast via API")
        return []

    def calibrate_qkv_fp8_scales(self, *args, **kwargs):
        """Calibrate KV scales via Tinker API."""
        # TODO: Implement KV scale calibration API call
        return {"layers": {}}

    def get_free_memory_bytes(self):
        """Get free memory from Tinker API."""
        response = self.api_client._make_request(
            "GET",
            f"models/{self.model_id}/memory",
        )
        return response.get("free_memory_bytes", 8 * 1024**3)

    def print_node_ip_and_gpu_id(self):
        """Print node info from Tinker API."""
        response = self.api_client._make_request(
            "GET",
            f"models/{self.model_id}/node_info",
        )
        node_info = response.get("node_info", {})
        print(
            f"  [TINKER] Node: {node_info.get('ip', 'N/A')}, "
            f"GPU: {node_info.get('gpu_id', 'N/A')}"
        )


class TinkerGeneration(GenerationInterface):
    """Generation implementation that uses Tinker APIs."""

    def __init__(
        self,
        api_client: TinkerAPIClient,
        model_id: str,
        vocab_size: int = 1000,
    ):
        """Initialize Tinker generation.

        Args:
            api_client: Tinker API client instance
            model_id: Model identifier
            vocab_size: Vocabulary size
        """
        self.api_client = api_client
        self.model_id = model_id
        self.vocab_size = vocab_size
        self._weight_version = 0
        self._stale = True

    def init_collective(self, *args, **kwargs):
        """Initialize collective communication (may not be needed for Tinker API)."""
        return []

    def generate(self, data, greedy=False):
        """Generate responses using Tinker API."""
        # Convert data to API format
        api_data = {
            "input_ids": data["input_ids"].cpu().numpy().tolist(),
            "input_lengths": data["input_lengths"].cpu().numpy().tolist(),
            "greedy": greedy,
        }

        # Call Tinker API
        response = self.api_client.generate(api_data, greedy=greedy)

        # Convert response to BatchedDataDict format
        batch_size = len(response.get("output_ids", []))
        max_len = max(
            len(seq) for seq in response.get("output_ids", [[]])
        ) if response.get("output_ids") else 0

        # Pad output_ids to same length
        output_ids_list = response.get("output_ids", [])
        output_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        generation_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, seq in enumerate(output_ids_list):
            seq_len = len(seq)
            output_ids[i, :seq_len] = torch.tensor(seq)
            generation_lengths[i] = seq_len

        unpadded_lengths = (
            torch.tensor(response.get("input_lengths", [])) + generation_lengths
        )
        logprobs = torch.tensor(response.get("logprobs", []))

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids,
                "generation_lengths": generation_lengths,
                "unpadded_sequence_lengths": unpadded_lengths,
                "logprobs": logprobs,
            }
        )

    def prepare_for_generation(self, *args, **kwargs):
        """Prepare model for generation via Tinker API."""
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/prepare_generation",
        )
        print("  [TINKER] Generation prepared")
        self._stale = False

    def finish_generation(self, *args, **kwargs):
        """Finish generation via Tinker API."""
        self.api_client._make_request(
            "POST",
            f"models/{self.model_id}/finish_generation",
        )

    def prepare_refit_info(self, state_dict_info=None):
        """Prepare refit info from Tinker API."""
        if state_dict_info:
            self._weight_version = state_dict_info.get("weight_version", 0)

    def update_weights_via_ipc_zmq(self):
        """Update weights via Tinker API."""
        # Tinker API may handle weight updates differently
        print("  [TINKER] Weights updated via API")
        self._stale = False
        return []

    def update_weights_from_collective(self):
        """Update weights from collective via Tinker API."""
        print("  [TINKER] Weights updated from collective via API")
        self._stale = False
        return []

    def requires_kv_scale_sync(self):
        """Check if KV scale sync is required (from Tinker API)."""
        response = self.api_client._make_request(
            "GET",
            f"models/{self.model_id}/kv_scale_sync",
        )
        return response.get("requires_sync", False)


def create_tinker_trainer(
    api_endpoint: str,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    vocab_size: int = 1000,
    colocated_inference: bool = False,
) -> Trainer:
    """Create a trainer that uses Tinker APIs.

    Args:
        api_endpoint: Base URL for Tinker API
        api_key: Optional API key for authentication
        model_id: Model identifier (if None, will be created via API)
        vocab_size: Vocabulary size
        colocated_inference: Whether inference is colocated with training

    Returns:
        Trainer instance using Tinker APIs
    """
    from nemo_rl.algorithms.trainer_sampler import Trainer, TrainerConfig

    # Create API client
    api_client = TinkerAPIClient(
        api_endpoint=api_endpoint,
        api_key=api_key,
        model_id=model_id,
    )

    # Create or get model ID from API if not provided
    if model_id is None:
        # TODO: Call Tinker API to create/register model
        response = api_client._make_request("POST", "models/create")
        model_id = response.get("model_id")
        if model_id is None:
            raise ValueError("Failed to create model via Tinker API")

    # Create policy
    policy = TinkerPolicy(api_client=api_client, model_id=model_id, vocab_size=vocab_size)

    # Create refit function that uses Tinker API
    def tinker_refit_fn(
        policy_interface,
        generation_interface,
        colocated,
        buffer_size_gb=None,
        timer=None,
        kv_scales=None,
    ):
        """Refit function that uses Tinker API for weight sync."""
        print(f"  [TINKER] Refitting generation with policy weights (colocated={colocated})")
        if timer:
            with timer.time("tinker_weight_sync"):
                # Call Tinker API to sync weights
                if isinstance(policy_interface, TinkerPolicy) and isinstance(
                    generation_interface, TinkerGeneration
                ):
                    api_client.sync_weights(
                        source_model_id=policy_interface.model_id,
                        target_model_id=generation_interface.model_id,
                    )
                else:
                    # Fallback for non-Tinker interfaces
                    time.sleep(0.01)
        else:
            if isinstance(policy_interface, TinkerPolicy) and isinstance(
                generation_interface, TinkerGeneration
            ):
                api_client.sync_weights(
                    source_model_id=policy_interface.model_id,
                    target_model_id=generation_interface.model_id,
                )
            else:
                time.sleep(0.01)

        print("  [TINKER] Weight sync complete")

    trainer = Trainer(
        policy=policy,
        colocated_inference=colocated_inference,
        refit_fn=tinker_refit_fn,
    )

    return trainer


def create_tinker_sampler(
    api_endpoint: str,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    trainer: Optional[Trainer] = None,
    vocab_size: int = 1000,
) -> Sampler:
    """Create a sampler that uses Tinker APIs.

    Args:
        api_endpoint: Base URL for Tinker API
        api_key: Optional API key for authentication
        model_id: Model identifier (if None, will use trainer's model_id if available)
        trainer: Optional trainer to link for weight sync
        vocab_size: Vocabulary size

    Returns:
        Sampler instance using Tinker APIs
    """
    from nemo_rl.algorithms.trainer_sampler import Sampler, SamplingConfig

    # Create API client
    api_client = TinkerAPIClient(
        api_endpoint=api_endpoint,
        api_key=api_key,
        model_id=model_id,
    )

    # Get model ID from trainer if not provided
    if model_id is None and trainer is not None:
        if isinstance(trainer.policy, TinkerPolicy):
            model_id = trainer.policy.model_id
        else:
            # Create new model for generation
            response = api_client._make_request("POST", "models/create")
            model_id = response.get("model_id")
            if model_id is None:
                raise ValueError("Failed to create model via Tinker API")

    if model_id is None:
        raise ValueError("model_id must be provided or trainer must have TinkerPolicy")

    # Create generation
    generation = TinkerGeneration(
        api_client=api_client, model_id=model_id, vocab_size=vocab_size
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

    This is a simplified version that uses Tinker API for generation.
    For full multi-turn support, you may need to implement environment
    interactions via Tinker API as well.

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

