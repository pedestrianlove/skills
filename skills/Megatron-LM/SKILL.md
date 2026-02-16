---
name: Megatron-LM
description: Skills for agents to consume for Megatron-LM
---
third_party/Megatron-LM/tools/checkpoint/convert.py
def load_plugin(plugin_type, name):
def main():

third_party/Megatron-LM/pretrain_mamba.py
def get_batch(data_iterator, vp_stage=None):
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[MambaModel] = None):
def forward_step(data_iterator, model: MambaModel):
def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
def core_gpt_dataset_config_from_args(args):
def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):

third_party/Megatron-LM/pretrain_vision_classify.py
def model_provider(pre_process=True, post_process=True):
def get_batch(data_iterator):
def loss_func(labels, output_tensor):
def forward_step(data_iterator, model):
def train_valid_test_datasets_provider(train_val_test_num_samples):

third_party/Megatron-LM/pretrain_bert.py
def model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
def get_batch(data_iterator):
def loss_func(loss_mask, sentence_order, output_tensor):
def forward_step(data_iterator, model):
def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):

third_party/Megatron-LM/pretrain_vision_inpaint.py
def model_provider(pre_process=True, post_process=True):
def get_batch(data_iterator):
def loss_func(images, masks, masked_images, outputs, non_loss_data=False):
def forward_step(data_iterator, model):
def process_non_loss_data(data, iteration, writer):
def train_valid_test_datasets_provider(train_val_test_num_samples):

third_party/Megatron-LM/pretrain_vision_dino.py
def model_provider(pre_process=True, post_process=True):
def get_batch(data_iterator):
def loss_func(model, labels, output_tensor, collect_data=False):
def forward_step(data_iterator, model):
def train_valid_test_datasets_provider(train_val_test_num_samples):

third_party/Megatron-LM/tools/checkpoint/loader_llama_mistral.py
def add_arguments(parser):
def verify_transformers_version():
def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
def read_json(path):
def write_json(text, path):
def convert_to_hf(model_path, input_base_path, model_size, tokenizer_path):
def load_args_from_checkpoint(args, model_size):
def set_preprocess_state(args, model, hf_model):
def set_postprocess_state(args, model, hf_model):
def set_attn_state(args, layer, hf_layer):
def set_mlp_state(args, layer, hf_layer):
def set_layer_state(args, model, hf_model, layer_idx):
def load_checkpoint_to_model(args):
def _load_checkpoint(queue, args):
def load_checkpoint(queue, args):

third_party/Megatron-LM/pretrain_ict.py
def pretrain_ict_model_provider(pre_process=True, post_process=True):
def get_group_world_size_rank():
def loss_func(output_tensor):
def forward_step(data_iterator, model):
def train_valid_test_datasets_provider(train_val_test_num_samples):

third_party/Megatron-LM/examples/mimo/avlm_inference.py
def init_distributed(tp_size: int = 1, pp_size: int = 1):
def get_input_data(
    processor: AutoProcessor,
    image_processor: AutoProcessor,
    audio_processor: AutoProcessor,
    audio_path: str,
    image_path: str,
    prompt: str,
    device: Union[int, str] = 0):
def main():
def load_distributed_checkpoint(model: torch.nn.Module, ckpt_dir: str):

third_party/Megatron-LM/tasks/finetune_utils.py
def process_batch(batch):
def cross_entropy_loss_func(labels, output_tensor):
def _cross_entropy_forward_step(batch, model):
def build_data_loader(dataset, micro_batch_size, num_workers, drop_last,
        task_collate_fn=None):
def _build_infinite_size_dataloader(dataloader):
def _build_train_valid_dataloaders(train_dataset, valid_dataset, 
    task_collate_fn=None):
def _train(model, optimizer, opt_param_scheduler, forward_step,
           train_dataloader, valid_dataloader, end_of_epoch_callback):
def finetune(train_valid_datasets_provider, model_provider,
             model_type=ModelType.encoder_or_decoder,
             forward_step=_cross_entropy_forward_step,
             end_of_epoch_callback_provider=None,
             task_collate_fn=None):

third_party/Megatron-LM/tools/checkpoint/saver_llava.py
def add_arguments(parser):
def save_checkpoint(queue, args):

third_party/Megatron-LM/tools/checkpoint/saver_legacy.py
def add_arguments(parser):
def save_checkpoint(queue, args):

third_party/Megatron-LM/tasks/eval_utils.py
def accuracy_func_provider(single_dataset_provider):
def calculate_correct_answers(name, model, dataloader,
                              epoch, output_predictions):

third_party/Megatron-LM/tools/checkpoint/utils.py
def chunk_bias(bias, parallel_mode, tp_size=1, ep_size=1):
def chunk_weight(weight, parallel_mode, tp_size=1, ep_size=1):
def print_memory_usage(key, rank, num_ranks):

third_party/Megatron-LM/examples/rl/environments/countdown/countdown.py
def extract_solution(solution_str: str, remove_prompt: bool = False):
def validate_equation(equation_str, available_numbers):
def evaluate_equation(equation_str):
def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.0):

third_party/Megatron-LM/tasks/data_utils.py
def clean_text(text):
def build_sample(ids, types, paddings, label, unique_id):
def build_tokens_types_paddings_from_text(text_a, text_b,
                                          tokenizer, max_seq_length):
def build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids, max_seq_length,
                                         cls_id, sep_id, pad_id):

third_party/Megatron-LM/examples/mimo/configs/llava_vlm.py
def get_vicuna_language_model_config(  
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig tuned for **Vicuna-7B**.
    The hyper-parameters follow the published Vicuna-7B weights (same sizes as
    Llama-7B).
    """
    cfg = TransformerConfig(num_layers=32, hidden_size=4096, num_attention_heads=32)
    # Feed-forward / MLP hidden size (11008 in original Vicuna).
    cfg.ffn_hidden_size = 11008
    # SwiGLU (SiLU-gate) activation.
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True
    # Normalisation – RMSNorm
    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = 1e-5
    # Positional embeddings – RoPE.
    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 10000
    cfg.rotary_percent = 1.0
    # Sequence length.
    cfg.seq_length = 4096
    cfg.max_position_embeddings = 4096
    # Attention / dropout.
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0
    # GQA disabled (queries == heads).
    cfg.num_query_groups = 32
    # Bias usage.
    cfg.add_bias_linear = False
    # Weight sharing.
    cfg.untie_embeddings_and_output_weights = False
    # Kernel / TE fusions.
    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True
    # Apply user overrides last.
    if config is not None:
        for field, value in vars(config).items():
def get_llava_projection_config( 
    hidden_size: int = 4096,
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the vision projection MLP."""
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = 4096
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu
    # Allow caller overrides.
    if config is not None:
        for field, value in vars(config).items():

third_party/Megatron-LM/examples/inference/t5/simple_t5_batch_inference.py
def add_text_generate_args(parser):
def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Utility to get the relevant backend for running inference
    This function will automatically chose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. TRT LLM Backend is not implmented yet.
    Args:
        args (Namespace):
def main():

third_party/Megatron-LM/tools/checkpoint/loader_llava.py
def add_arguments(parser):
def load_checkpoint(queue, args):

third_party/Megatron-LM/megatron/legacy/fused_kernels/__init__.py
def load(args):
def _get_cuda_bare_metal_version(cuda_dir):
def _create_build_dir(buildpath):

third_party/Megatron-LM/examples/post_training/modelopt/finetune.py
def add_finetune_args(parser):
def get_eos_id():
def train_valid_test_sft_datasets_provider(train_val_test_num_samples):
def get_batch(data_iterator):
def non_loss_data_func(model: GPTModel):
def forward_step(data_iterator, model: GPTModel):

third_party/Megatron-LM/examples/mimo/train.py
def add_mimo_args(parser):
def get_batch(data_iterator: Iterator[Dict[str, Any]]):
def loss_func(loss_mask, output_tensor):
def forward_step(data_iterator, model):
def train_valid_test_datasets_provider(*provider_args, **provider_kwargs):
def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = 32000,
    audio_special_token_id: int = 32002,
):

third_party/Megatron-LM/examples/mimo/configs/mock.py
def get_mock_language_model_config(config: Optional[TransformerConfig] = None) -> TransformerConfig:
    """
    Create a mock language model configuration.
    Args:
        config: Optional base configuration to modify
    Returns:
        TransformerConfig: Mock configuration for a language model
    """
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
    if config is not None:
        for field_name, field_value in vars(config).items():
def get_mock_vision_model_config(config: Optional[TransformerConfig] = None) -> TransformerConfig:
    """
    Create a mock vision model configuration.
    Args:
        config: Optional base configuration to modify
    Returns:
        TransformerConfig: Mock configuration for a vision model
    """
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = config.hidden_size * 4
    config.gated_linear_unit = False
    config.kv_channels = 64
    config.layernorm_zero_centered_gamma = False
    config.apply_query_key_layer_scaling = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = 'LayerNorm'
    config.apply_rope_fusion = False
    return config
def get_mock_projection_config(hidden_size: int = 128) -> TransformerConfig:
    """
    Create a mock projection layer configuration.
    Args:
        hidden_size: Hidden dimension size (used as the vision projection output size)
    Returns:
        TransformerConfig: Mock configuration for a projection layer
    """
    config = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    config.ffn_hidden_size = hidden_size * 4
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = False
    config.normalization = 'LayerNorm'
    return config
def get_mock_language_layer_spec():
def get_mock_vision_layer_spec():
def get_mock_projection_layer_spec():

third_party/Megatron-LM/tools/checkpoint/saver_core.py
def add_arguments(parser):
def save_checkpoint(queue, args):

third_party/Megatron-LM/examples/inference/gpt/gpt_static_inference.py
def add_static_inference_args(parser):
def get_inference_engine(args: Namespace, model: MegatronModule) -> StaticInferenceEngine:
    """Utility to get the relevant backend for running inference
    This function will automatically choose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. TRT LLM Backend is not implmented yet.
    Args:
        args (Namespace):
def main():

third_party/Megatron-LM/examples/post_training/modelopt/convert_model.py
def add_convert_args(parser):
def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
def check_arguments():

third_party/Megatron-LM/tools/checkpoint/loader_core.py
def add_arguments(parser):
def load_checkpoint(queue, args):

third_party/Megatron-LM/megatron/legacy/fused_kernels/tests/test_fused_kernels.py
def test_load_fused_kernels():
def test_fused_softmax():
def test_fused_upper_triangle_mask_softmax():
def test_layer_norm():
def attention_mask_func(attention_scores, attention_mask):
def forward_torch_softmax(input, mask, scale):
def test_masked_softmax_forward():
def test_masked_softmax_backward():
def test_allmasked_softmax_forward():
def test_allmasked_softmax_backward():

third_party/Megatron-LM/tools/run_vlm_text_generation.py
def add_text_generation_args(parser):
def preprocess_image(target_h, target_w, img):
def generate_samples(model):
def generate_and_write_samples(model):
def main():

third_party/Megatron-LM/examples/mimo/configs/llava_avlm.py
def get_llava_projection_config( 
    hidden_size: int = 4096,
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the vision projection MLP."""
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = 4096
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu
    # Allow caller overrides.
    if config is not None:
        for field, value in vars(config).items():

third_party/Megatron-LM/examples/post_training/modelopt/offline_feature_extract.py
def add_extract_args(parser):
def extract_feature(dataset, model, output_dir, idx_start, idx_end):

third_party/Megatron-LM/examples/mimo/model_providers/llava_vlm.py
def model_provider_llava_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    image_special_token_id: int = 32000,
    is_video_input: bool = False
):

third_party/Megatron-LM/examples/post_training/modelopt/generate.py
def add_generate_args(parser):
def check_arguments():
def mtbench_to_oai_chat(example):
def get_conversations(example):

third_party/Megatron-LM/tools/checkpoint/saver_hf_llava.py
def add_arguments(parser):
def recover_qkv(new_tensor, num_head, head_dim):
def save_checkpoint(queue, args):

third_party/Megatron-LM/megatron/legacy/model/t5_model.py
def t5_extended_attention_mask(attention_mask_list):
def t5_position_ids(token_ids):

third_party/Megatron-LM/tools/preprocess_data.py
def get_args():
def get_file_name(args, file_id):
def check_files_exist(in_ss_out_names, key, num_partitions):
def main():

third_party/Megatron-LM/examples/mimo/model_providers/mock.py
def model_provider_mock_vlm_single_encoder(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    special_token_id: int = 32000,
):

third_party/Megatron-LM/megatron/legacy/model/realm_model.py
def general_ict_model_provider(only_query_model=False, only_block_model=False):

third_party/Megatron-LM/examples/mimo/model_providers/llava_avlm.py
def model_provider_llava_avlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    image_special_token_id: int = 32000,
    audio_special_token_id: int = 32002,
):

third_party/Megatron-LM/pretrain_gpt.py
def get_batch(data_iterator, vp_stage: Optional[int] = None):
def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
def is_dataset_built_on_rank(vp_stage=None):
def core_gpt_dataset_config_from_args(args):
def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
def get_embedding_ranks(pp_ranks: List[int]):

third_party/Megatron-LM/tools/checkpoint/hybrid_conversion.py
def get_split_dim(tensor_name):
def combine_tp_tensors(params, key, dim, tensors):
def split_tensor_for_tp(params, key, dim, tensor):
def finalize_checkpoint(sample_model, model, params, verbose=False):
def main(args):

third_party/Megatron-LM/tools/preprocess_mmdata.py
def get_args():
def main():

third_party/Megatron-LM/megatron/legacy/model/fused_bias_gelu.py
def bias_gelu(bias, y):
def bias_gelu_back(g, bias, y):

third_party/Megatron-LM/train_rl.py
def _gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
def loss_func(
    loss_mask: torch.Tensor,
    kl_term: torch.Tensor,
    ratios: torch.Tensor,
    entropy_term: torch.Tensor,
    truncated_from_above: torch.Tensor,
    truncated_from_below: torch.Tensor,
    output_tensor: torch.Tensor,
):
def forward_step(data_iterator, model: GPTModel, loss_only: bool = False):
def train_valid_test_datasets_provider(train_val_test_num_samples):

third_party/Megatron-LM/examples/inference/gpt/gpt_dynamic_inference.py
def add_dynamic_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """Dynamic inference arguments."""
    add_common_inference_args(parser)
    group = parser.add_argument_group(title='Dynamic inference')
    group.add_argument(
        "--inference-ckpt-non-strict",
        action="store_true",
        help="Load checkpoint with `strict=False`.",
    )
    group.add_argument(
        "--termination-id", type=int, default=None,
        help="Termination ID that overrides `tokenizer.eod`.",
    )
    group.add_argument(
        "--suspend-resume-interval", type=int, default=None,
        help="Suspend and resume the dynamic engine every "
        "`suspend_resume_interval` steps. This is used to tet the suspend/resume "
        "system.",
    )
    group.add_argument(
        "--inference-repeat-n", type=int, default=1,
        help="Repeat inference iterations N times for benchmarking."
    )
    group.add_argument(
        "--throughput-check-only",
        action='store_true',
        default=False,
        help="If true, only run throughput check without verifying outputs."
    )
    return parser
def get_model() -> MegatronModule:
    """Initialize model and load checkpoint."""
    args = get_args()
    if args.model_provider == "gpt":
        model_builder = gpt_builder
    elif args.model_provider == "mamba":
        model_builder = mamba_builder
    else:
        raise ValueError(f"Invalid model provider {args.model_provider}")
    # Build model.
    model = _get_model(
        partial(model_provider, model_builder),
        wrap_with_ddp=False
    )
    # Load checkpoint.
    assert args.load is not None
    args.exit_on_missing_checkpoint = True
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=not args.inference_ckpt_non_strict,
    )
    # No virtual PP.
    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    # Eval mode.
    model.eval()
    return model
def get_inference_context(
    requests: List[Request],
    sampling_params: Optional[SamplingParams] = None,
    calculate_max_sequence_length_from_requests: bool = True,
    mamba_inference_state_config: Optional[MambaInferenceStateConfig] = None,
):
def get_inference_controller(
    model: MegatronModule, context: DynamicInferenceContext
) -> TextGenerationController:
    """Buid text generation controller, which manages the model inference context.
    Args:
        model (MegatronModule):
def run_inference(
    requests: List[Request],
    engine: DynamicInferenceEngine,
    sampling_params: Optional[SamplingParams] = None,
) -> List[Dict[str, float]]:
    """Add requests to engine and generate tokens.
    Args:
        requests (List[Request]):
def main():

third_party/Megatron-LM/examples/post_training/modelopt/quantize.py
def add_text_generate_ptq_args(parser):
def check_arguments():
def _is_first_layers(name: str, num_layers: int = 1, num_layers_to_disable: int = 1) -> bool:
    if "layers." not in name:
        return False
    try:
        layer_idx = int(name.split("layers.")[-1].split(".")[0])
    except ValueError:
        return False
    return layer_idx < num_layers_to_disable
def _is_last_layers(name: str, num_layers: int = 1, num_layers_to_disable: int = 1) -> bool:
    if "layers." not in name:
        return False
    try:
        layer_idx = int(name.split("layers.")[-1].split(".")[0])
    except ValueError:
        return False
    return layer_idx >= num_layers - num_layers_to_disable
def get_first_layers_disabled_config(config, num_layers: int = 1, num_layers_to_disable: int = 1):
def get_last_layers_disabled_config(config, num_layers: int = 1, num_layers_to_disable: int = 1):
def get_modelopt_torch_quantization_config():
def get_calib_dataloader(calib_size=512, max_sequence_length=512):

third_party/Megatron-LM/examples/mimo/utils/model_helpers.py
def load_submodule_ckpt(module: torch.nn.Module, ckpt_dir: str):

third_party/Megatron-LM/examples/multimodal/dataset_helpers.py
def search_for_fit(numbers: List[int], capacity: int) -> int:
    """Finds the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)
# Based on https://github.com/hiyouga/LLaMA-Factory/blob/641d0dab08d96a93c34657742213d8994d9ed476/src/llamafactory/data/processors/processor_utils.py#L27
# Copyright (c) 2024 LLaMA-Factory. Apache license 2.0.
def greedy_knapsack(item_sizes: List[int], samples: List, max_capacity: int) -> List:
    """Greedy algorithm with binary search for the knapsack problem.
    Pack as many samples as possible given a maximum capacity and capacities of individual samples.
    Used if sequence packing is enabled.
    """
    assert len(item_sizes) == len(samples), "sample lengths and samples must have the same length."
    knapsacks = []
    if len(item_sizes) == 0:
        return knapsacks
    # Sort sample lengths and samples together.
    sorted_item_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples), key=lambda x: x[0]))
    sorted_item_sizes = list(sorted_item_sizes)
    sorted_samples = list(sorted_samples)
    # Check if all samples fit in the knapsack capacity.
    if sorted_item_sizes[-1] > max_capacity:
        raise ValueError(f"knapsack: A sample is larger {sorted_item_sizes[-1]} than the max_sequence_length {max_capacity}.")
    while sorted_item_sizes:
        current_knapsack = []
        remaining_capacity = max_capacity
        while True:
            idx = search_for_fit(sorted_item_sizes, remaining_capacity)
            if idx == -1:
                break   # Can't fit more samples.
            remaining_capacity -= sorted_item_sizes[idx]
            sorted_item_sizes.pop(idx)
            sample = sorted_samples.pop(idx)
            current_knapsack.append(sample)
        knapsacks.append(current_knapsack)
    return knapsacks
class TaskEncoder(DefaultTaskEncoder[OCRSample, OCRSample, ImageTaskBatchPacked, dict]):
def print_error_handler(exc: Exception, key: Optional[str]):
def format_multichoice_question(question, multichoice_options):
def format_multichoice_answer(idx):

third_party/Megatron-LM/tools/retro/text_generation/evaluate.py
def normalize_answer(s):
def compute_f1_score(predicted_answers, groundtruth_answer, exp_name="default"):
def load_groundtruth_file(data_file):
def read_prediction(prediction_file):
def exact_match_score(prediction, ground_truth):
def ems(prediction, ground_truths):
def evaluate_ems(prediction_file, ground_truth_file, dev_num=3000):
def load_prediction(data_file):
def evaluate_f1(ground_truth_file, prediction_file, reduced_test_only=False):

third_party/Megatron-LM/model_provider.py
def model_provider(
    model_builder: Callable, pre_process=True, post_process=True, vp_stage: Optional[int] = None, config=None, pg_collection=None,
) -> Union[GPTModel, megatron.legacy.model.GPTModel, MambaModel]:
    """Builds the model.
    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.
    Args:
        model_builder: A callable that builds the actual model, its signature is the same as model_provider's with an exception of the first argument which is a builder itself. In addition might take a config passed from outside to skip its own config loading. See gpt_builder or mamba_builder for an example, see _gpt_model_builder in train_rl.py to see how to augment a default gpt builder and pass the config from outside
        pre_process (bool, optional):
def count_parameters_in_layer(model, layer_name):

third_party/Megatron-LM/tools/run_text_generation_server.py
def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference
    This function will automatically choose the TRTLLMBackend when possible, and default to Mcore
    backend if the user does not specify any backends. TRTLLMBackend is not implmented yet.
    Args:
        args (Namespace):
def add_text_generate_args(parser):
def main(model_type: str = "gpt"):

third_party/Megatron-LM/examples/multimodal/layer_scaling.py
def _bias_dropout_add_func_layer_scaling(ls, x_with_bias, residual, prob, training):
def bias_dropout_add_unfused_layer_scaling(ls, training):
def get_bias_dropout_add_layer_scaling(ls, training, fused):

third_party/Megatron-LM/examples/mimo/utils/logging.py
def print_mimo_structure(model):

third_party/Megatron-LM/examples/post_training/modelopt/mmlu.py
def add_mmlu_args(parser):
def get_all_subjects():
def format_example(example, include_answer: bool = True):
def generate_prompt(test_example, dev_examples, few_shots=0, no_subject_prompt=False):

third_party/Megatron-LM/tools/retro/text_generation/metrics.py
def normalize_answer(s):

third_party/Megatron-LM/tools/retro/sft/sft_retro.py
def get_tasks_args(parser):
def get_batch(data_iterator):
def loss_func(loss_mask, output_tensor):
def forward_step(data_iterator, model):
def train_valid_test_datasets_provider(train_val_test_num_samples):

third_party/Megatron-LM/megatron/legacy/model/vision/knn_monitor.py
def build_data_loader(dataset, drop_last=True, shuffle=False):
def compute_feature_bank(model):
def get_feature_bank():
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):

third_party/Megatron-LM/tools/retro/text_generation/retro_generation.py
def retro_generate_tokens_probs_and_return_on_first_stage(
        model, tokens, lengths, neighbours_array=None,
        return_output_log_probs=False,
        top_k=0, top_p=0.0,
        temperature=1.0,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False,
        logits_mask=None):

third_party/Megatron-LM/pretrain_vlm.py
def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
    config=None,
    pg_collection=None,
) -> LLaVAModel:
    """Builds the model.
    Note: currently, only LLaVA model is supported. Follow-up changes will make this configurable.
    Args:
        pre_process (bool):
def train_valid_test_datasets_provider(train_val_test_num_samples):
def _preprocess_data_for_llava(data):
def get_batch(data_iterator):
def forward_step(data_iterator, model: LLaVAModel):
def add_vlm_extra_args(parser):
def llava_embedding_ranks(pp_ranks):
def llava_position_embedding_ranks(pp_ranks):

third_party/Megatron-LM/tools/linter.py
def recursively_lint_files():

third_party/Megatron-LM/tools/bert_embedding/embed.py
def collate_batch(samples):
def get_data_loader(dataset, batch_size):
def embed_data_loader(models, data_loader, tag):

third_party/Megatron-LM/tools/checkpoint/checkpoint_inspector.py
def rank0_echo(message):
def print_header(text, color="white"):
def cli():
def inspect(checkpoint_dir, enable_msc, not_ignore_param_to_group_meta):
def print_tensor(checkpoint_dir, key):
def check_gpu_memory(threshold=0.9):
def flatten(obj, parent_key="", sep="."):
def save_checkpoint_with_pickle_protocol(state_dict, output_dir, pickle_protocol=4):
def convert_checkpoint(
    input_dir,
    output_dir,
    swiglu,
    process_group,
    optimizer_param_to_group_prefix="optimizer.param_to_group_meta.module.module.module",
    optimizer_state_prefix="optimizer.state.module.module.module",
    model_weight_prefix="model.module",
    param_to_param_group_map={},
):
def convert_torch_dist_to_fsdp_dtensor(
    input_dir,
    output_dir,
    swiglu,
    oom_traceback,
    enable_msc,
    output_optimizer_state_prefix,
    output_model_weight_prefix,
    param_to_param_group_map_json,
):
def _modify_state_dict(input_dir, output_dir, ops, process_group, enable_msc=False):
def modify_state_dict(input_dir, output_dir, op, enable_msc):
def _compare_two_checkpoint(checkpoint_1, checkpoint_2):
def compare_two_checkpoint(checkpoint_1, checkpoint_2, enable_msc):
def print_torch_dcp_in_json(torch_dcp_dir, model_weight_prefix="model.module"):
def init_process_group(message):

third_party/Megatron-LM/tools/build_sequences_per_dataset.py
def get_paths_from_blend(
    blend: Optional[Tuple[List[str], Optional[List[float]]]],
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]],
) -> List[str]:
    """Extract all dataset paths from blend and blend_per_split.
    Args:
        blend (Optional[Tuple[List[str], Optional[List[float]]]]):
def build_sequences_per_dataset(args):

third_party/Megatron-LM/examples/mimo/utils/data_helpers.py
def flatten(
    nested: Dict[str, Any], prefix: Tuple[str, ...] = ()
) -> List[Tuple[Tuple[str, ...], torch.Tensor]]:
    """Recursively flatten nested dict into [(key_path, tensor), …]."""
    flat = []
    for k, v in nested.items():
def regroup(flat: List[Tuple[Tuple[str, ...], torch.Tensor]]) -> Dict[str, Any]:
    """Rebuild the nested dict from [(key_path, tensor), …]."""
    root = {}
    for path, tensor in flat:
        cur = root
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = tensor
    return root
def broadcast_nested_data_batch(nested_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively broadcast nested dictionaries of tensors using each tensor's own dtype."""
    
    tp_group = mpu.get_tensor_model_parallel_group()
    src      = mpu.get_tensor_model_parallel_src_rank()
    # ---------- rank-0 prepares metadata ----------
    if mpu.get_tensor_model_parallel_rank() == 0:
        flat = flatten(nested_dict)                # [(path,tensor), …]
        paths, tensors = zip(*flat) if flat else ([], [])
        dtypes = [t.dtype for t in tensors]
    else:
        paths, dtypes = [], []
        tensors = []
    # ---------- 1. broadcast schema (paths + dtypes) ----------
    meta = [paths, dtypes]                         # small, picklable
    obj_list = [meta]
    torch.distributed.broadcast_object_list(obj_list, src=src, group=tp_group)
    paths, dtypes = obj_list[0]                    # now identical on all ranks
    # ---------- 2. group tensors by dtype and broadcast ----------
    # build maps keyed by dtype for convenience
    dtype_to_keys = {}
    for p, dt in zip(paths, dtypes):

third_party/Megatron-LM/tools/retro/text_generation/retro_text_generation.py
def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.
    Args:
        pre_process (bool, optional):
def pad_neighbours_for_query_only(args, nb_tokens, pad_id, ft_neighbours):
def add_text_generate_args(parser):
def generate_samples_conditional(model):
def generate_and_write_samples_conditional(model):
def main():

third_party/Megatron-LM/tools/checkpoint/schema_core.py
def get_core_transformer_block_key(model_key):

third_party/Megatron-LM/tools/preprocess_data_nmt.py
def get_args():
def main():

third_party/Megatron-LM/pretrain_t5.py
def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    config=None,
    pg_collection=None,
) -> Union[megatron.legacy.model.T5Model, T5Model]:
    """Builds the model.
    Args:
        pre_process (bool, optional):
def get_batch(data_iterator, use_local):
def forward_step(data_iterator, model: T5Model):
def train_valid_test_datasets_provider(train_val_test_num_samples: int):
def t5_embedding_ranks(pp_ranks):
def t5_position_embedding_ranks(pp_ranks):

third_party/Megatron-LM/tools/retro/sft/dataset_conv.py
def format_multichoice(multichoice_options):
def format_multichoice_question(question, multichoice_options):
def format_answer(answer):
def preprocess(dataset_path: str, config: JsonQADatasetConfig):
def count_stat(dataset, tokenizer, k):
def reformat_prompt_retro(query, neighbours, dataset_name, ft_neighbours, \
                          max_output_len, tokenizer, max_seq_length):
def flan_format(system, context, dialogue_turn, template_id=0):
def reformat_prompt(query, neighbours, dataset_name, ft_neighbours, \
                    max_output_len, tokenizer, max_seq_length, template_id=0):
def reformat_prompt_short(query, neighbours, dataset_name, ft_neighbours, \
                          max_output_len, tokenizer, max_seq_length):
def pad_and_convert_to_numpy(input_ids, output_ids,
                             pad_id, max_seq_length,
                             eos_id):

third_party/Megatron-LM/tools/check_copyright.py
def has_correct_header(file_path):
def main():

third_party/Megatron-LM/examples/post_training/modelopt/prune.py
def add_prune_args(parser):
def check_arguments(args):
def get_calib_dataloader(calib_size=1024, max_sequence_length=512):
def get_params(model):

third_party/Megatron-LM/tools/retro/preprocess_data.py
def add_retro_args(parser):
def initialize_megatron_retro():
def get_bert_embedders(config):
def get_gpt_chunk_datasets(config):
def get_gpt_tokenizer(config):
def get_bert_tokenizer(config):
def get_tokenizers(config):
def get_retro_preprocessing_config():
def save_config(config):

third_party/Megatron-LM/mamba_builders.py
def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):

third_party/Megatron-LM/megatron/legacy/model/vision/esvit_swin_backbone.py
def window_partition(x, window_size):
def window_reverse(windows, window_size, H, W):
def get_swin(is_teacher=False):

third_party/Megatron-LM/tools/run_dynamic_text_generation_server.py
def add_text_generation_server_args(parser: argparse.ArgumentParser):

third_party/Megatron-LM/examples/post_training/modelopt/export.py
def add_modelopt_export_args(parser):

third_party/Megatron-LM/tools/checkpoint/loader_legacy.py
def add_arguments(parser):
def _load_checkpoint(queue, args):
def load_checkpoint(queue, args):

third_party/Megatron-LM/megatron/legacy/model/vision/utils.py
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):

third_party/Megatron-LM/tools/merge_datasets.py
def get_args():
def main():

third_party/Megatron-LM/examples/mimo/data/energon_vlm_task_encoder.py
def llava_vlm_dataloader_provider(train_val_test_num_samples, is_video_input=False):

third_party/Megatron-LM/tools/retro/text_generation/retro_api.py
def tokenize_prompts(prompts=None, tokens_to_generate=None,
                     add_BOS=None, rank=0):
def _tokenize_prompts_and_batch(prompts, tokens_to_generate, add_BOS):
def retro_generate_and_post_process(model,
                              prompts=None,
                              neighbours_array=None,
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              random_seed=-1,
                              logits_mask=None):
def retro_generate(model,
             prompts=None,
             neighbours_array=None,
             tokens_to_generate=0,
             return_output_log_probs=False,
             top_k_sampling=0,
             top_p_sampling=0.0,
             temperature=1.0,
             add_BOS=False,
             use_eod_token_for_early_termination=True,
             stop_on_double_eol=False,
             stop_on_eol=False,
             random_seed=-1,
             logits_mask=None):

third_party/Megatron-LM/tools/retro/config_utils.py
def verify_and_get_config_attr_descs(config_cls, strict_docstring_match=True):
def add_config_args(parser, config_cls):
def get_config_leaf_field_names(config_cls):
def config_from_args(args, config_cls, add_custom_args=False):
def flatten_config(config, base_config_cls=None):

third_party/Megatron-LM/examples/post_training/modelopt/validate.py
def add_ar_validation_args(parser):
def check_arguments():
def get_current_memory_info():
def report_current_memory_info():

third_party/Megatron-LM/examples/multimodal/run_text_generation.py
def is_first_rank():
def add_text_generation_args(parser):
def get_evaluation_dataloader(
    task,
    input_image_path,
    gt_path,
    img_h,
    img_w,
    use_tiling,
    max_num_tiles,
    use_thumbnail,
    num_samples_per_partition,
    num_partitions,
    partition_id,
    num_frames,
    num_workers,
    vision_model_type,
    split="validation"
):
def generate_samples(model, config: EvaluationConfig, print_output):
def get_evaluation_configs(config_path=None) -> Dict[str, EvaluationConfig]:
    """Get evaluation config(s) from a config file or command-line arguments.
    Args:
        config_path: Optional path to config file. If not provided, will check args.config_path
                    or fall back to command-line arguments.
    Returns:
        Dict[str, EvaluationConfig]: dict of configs.
    """
    args = get_args()
    configs = {}
    # Use provided config_path or fall back to args.config_path
    config_file = config_path or args.config_path
    # We check if we're trying to run a single config evals by checking for the task and output_path
    # args.
    if hasattr(args, "task") and args.task and hasattr(args, "output_path") and args.output_path:
        # Single config from args
        config = EvaluationConfig(
            task=args.task,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            out_seq_length=args.out_seq_length,
            output_path=args.output_path,
            input_image_path=args.input_image_path,
            gt_path=args.gt_path,
            num_partitions=args.num_partitions,
            partition_id=args.partition_id,
            num_samples_per_partition=args.num_samples_per_partition,
        )
        if not config.output_path:
            default_output_dir = args.output_path if args.output_path else "generated"
            os.makedirs(default_output_dir, exist_ok=True)
            config.output_path = os.path.join(default_output_dir, args.language_model_type)
        return {args.task: config}
    elif config_file:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        if 'datasets' not in config_data:
            print("Error: 'datasets' key not found in config file for batch mode.")
            sys.exit(1)
        config_dict = config_data['datasets']
        for key, value in config_dict.items():
def get_output_path(config, dp_rank):
def generate_and_write_samples(model, config, print_output=True):
def get_conversation(task, question, metadata=None):
def get_prompt_and_generated(prompt_and_generation, prompt_format):
def run_eval(config, iteration=None):
def run_evaluation_loop(model, configs, output_dir_override=None, iteration=None, print_output=True):
def eval_tasks():

third_party/Megatron-LM/megatron/legacy/model/bert_model.py
def bert_extended_attention_mask(attention_mask):
def bert_position_ids(token_ids):
def post_language_model_processing(lm_output, pooled_output,
                                   lm_head, binary_head,
                                   lm_labels,
                                   logit_weights,
                                   fp16_lm_cross_entropy):

third_party/Megatron-LM/tools/checkpoint/loader_mixtral_hf.py
def add_arguments(parser):
def load_args_from_checkpoint(args):
def verify_transformers_version():
def set_preprocess_state(args, model, hf_model):
def set_postprocess_state(args, model, hf_model):
def set_attn_state(args, layer, hf_layer):
def set_mlp_state(args, layer, hf_layer):
def set_layer_state(args, model, hf_model, layer_idx):
def load_checkpoint_to_model(args):
def _load_checkpoint(queue, args):
def load_checkpoint(queue, args):

third_party/Megatron-LM/tools/run_inference_performance_test.py
def add_inference_benchmarking_args(parser):
def get_inference_engine(args: argparse.Namespace, model: MegatronModule) -> AbstractEngine:
    """Utility to get the relevant backend for running inference
    Args:
        args (Namespace):
def get_random_prompt_tokens(tokenizer, num_input_tokens) -> List[int]:
    # Get the set of special token IDs to exclude
    special_token_ids = set()
    try:
        if hasattr(tokenizer, 'bos') and tokenizer.bos is not None:
            special_token_ids.add(tokenizer.bos)
        if hasattr(tokenizer, 'eos') and tokenizer.eos is not None:
            special_token_ids.add(tokenizer.eos)
        if hasattr(tokenizer, 'eod') and tokenizer.eod is not None:
            special_token_ids.add(tokenizer.eos)
        if (
            hasattr(tokenizer, 'additional_special_tokens_ids')
            and tokenizer.additional_special_tokens_ids
        ):
def generate_dynamic(
    args: argparse.Namespace,
    inference_requests: List[InferenceRequest],
    inference_engine: DynamicInferenceEngine,
):
def main():

third_party/Megatron-LM/megatron/post_training/non_loss_data_func.py
def report_draft_acceptance_length(model, osl: int = 64, draft_steps: int = 7):

third_party/Megatron-LM/pretrain_retro.py
def get_retro_config():
def core_model_provider(pre_process=True, post_process=True):
def model_provider(pre_process=True, post_process=True):
def get_batch(data_iterator):
def forward_step(data_iterator, model):
def train_valid_test_datasets_provider(train_valid_test_num_samples):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_video_phys_game_bench.py
def merge_input_files(input_path):
def check_ans(pred, gt):
def compute_all_acc(result_list):
def phys_game_bench_eval(input_path):

third_party/Megatron-LM/examples/inference/gpt/utils.py
def add_common_inference_args(parser: ArgumentParser) -> ArgumentParser:
    """Common inference arguments."""
    group = parser.add_argument_group(title='Common inference')
    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1, help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument(
        "--return-log-probs",
        action='store_true',
        default=False,
        help='Return the log probabilities of the final output tokens',
    )
    group.add_argument(
        "--prompts",
        metavar='N',
        type=str,
        nargs='+',
        help='Input prompts with each prompt within quotes and seperated by space',
    )
    group.add_argument(
        "--num-tokens-to-prompt",
        type=int,
        nargs="+",
        default=[64, 1024],
        help='Number of tokens to use for simulated prompts. This should be a '
        'space-separated pair of integers, and the generated prompt lengths will '
        'be uniformly sampled within this range.',
    )
    group.add_argument(
        "--num-tokens-to-generate",
        type=int,
        default=30,
        help='Number of tokens to generate for each prompt',
    )
    group.add_argument(
        "--num-tokens-from-file",
        action='store_true',
        default=False,
        help='Use per-prompt num_tokens_to_generate from prompt file',
    )
    group.add_argument(
        "--top-n-logprobs",
        type=int,
        default=0,
        help='Return the top n logprobs for the generated tokens and their corresponding token as a dictionary',
    )
    group.add_argument(
        "--incoming-requests-per-step",
        type=int, default=None,
        help="Add a deterministic number of requests per step. This arg is "
        "prioritized over `--incoming-requests-per-sec` below (which is non-"
        "deterministic). Note that the number of requests added per step is "
        "additionally limited by the inference context's `max_requests`, "
        "`max_tokens`, and KV buffer size.",
    )
    group.add_argument(
        "--incoming-requests-per-sec",
        type=float,
        default=100.0,
        help="Simulated number of requests per second. Set to -1 to add all requests together.",
    )
    group.add_argument(
        "--incoming-requests-duration",
        type=float,
        default=10.0,
        help="Total amount of time to simulate that requests are "
        "arriving. Multiply this value with "
        "`--incoming-requests-per-sec` to get the approximate "
        "total number of requests. Set to -1 to add all requests together.",
    )
    group.add_argument(
        "--model-provider",
        choices=["mamba", "gpt"],
        default="gpt",
        help="Model provider",
    )
    group.add_argument(
        "--skip-prompt-log-probs",
        action='store_true',
        default=False,
        help='Skip prompt log probs.',
    )
    group.add_argument(
        "--stop-words",
        metavar='WORD',
        type=str,
        nargs='+',
        default=None,
        help='Stop words to terminate generation. Each word should be quoted and '
        'separated by space. Example: --stop-words "\\n\\n" "END" "###"',
    )
    group.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save generations as JSON",
    )
    group.add_argument(
        "--output-every-n-results",
        type=int,
        default=1,
        help="To minimize the output file size of larger runs, only write the "
        "results of every `n` requests.",
    )
    group.add_argument(
        "--prompt-file",
        help='Jsonl file containing input prompts, where each item (i.e., line) '
        'contains the field \'text\' where the value is the prompt. All other '
        'fields within each item are ignored, and may be customized for each '
        'application.',
    )
    group.add_argument(
        "--prompt-file-num-truncate",
        type=int,
        help='Number of samples to use from the loaded prompt file (see '
        '`--prompt-file` above). The first `--prompt-file-num-truncate` samples '
        'will be used, in order.',
    )
    group.add_argument(
        "--use-flashinfer-fused-rope",
        action='store_true',
        default=False,
        help='Use flashinfer fused rope implementation.',
    )
    group.add_argument(
        "--no-record-throughput",
        action='store_false',
        dest="record_throughput",
        help="Disable throughput recording in --output-file"
        
    )
    return parser
def get_default_sampling_params(termination_id: int = None):
def get_curr_time() -> float:
    """Get synchronized time across ranks."""
    curr_time = torch.cuda.LongTensor([time.time_ns()])
    if torch.distributed.is_initialized():
def get_time_offsets(
    seed: int | None,
    incoming_requests_per_step: int,
    incoming_requests_per_sec: float,
    num_requests: int,
) -> list[float]:
    """Get example time offsets."""
    # Time offsets to add all requests at once.
    if incoming_requests_per_step is not None or incoming_requests_per_sec <= 0:
        return [-1] * num_requests
    # if num_requests is not None:
    incoming_requests_duration = num_requests / incoming_requests_per_sec
    incoming_requests_duration *= 2 # extra margin, to accomodate time sampling
    random.seed(seed)
    
    import simpy  # Guard against this import in test case
    # Generate random time offsets.
    def arrival(r):
def get_cli_requests(
        args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    # Get time offsets.
    t_offsets = get_time_offsets(
        args.seed,
        args.incoming_requests_per_step,
        args.incoming_requests_per_sec,
        len(args.prompts),
    )
    # Init requests.
    requests = [Request(p, t, tokenizer, sampling_params) for p,t in zip(args.prompts, t_offsets)]
    return requests
def get_synthetic_requests(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    """Get example requests."""
    # Get time offsets.
    time_offsets = get_time_offsets(
        args.seed,
        args.incoming_requests_per_step,
        args.incoming_requests_per_sec,
        int(args.incoming_requests_per_sec * args.incoming_requests_duration),
    )
    # Build prompts with expected lengths.
    assert (
        len(args.num_tokens_to_prompt) == 2
        and
        args.num_tokens_to_prompt[1] >= args.num_tokens_to_prompt[0]
    )
    max_prompt_length = args.num_tokens_to_prompt[1]
    max_prompt_text = "hi " * max_prompt_length
    max_prompt_tokens = tokenizer.tokenize(max_prompt_text)
    prompt_lengths = [
        random.randint(*args.num_tokens_to_prompt)
        for _ in time_offsets
    ]
    prompt_tokens_list = [ max_prompt_tokens[:l] for l in prompt_lengths ]
    prompt_texts = [ tokenizer.detokenize(tt) for tt in prompt_tokens_list ]
    # Init requests.
    assert len(prompt_texts) == len(time_offsets)
    requests = [
        Request(t, o, tokenizer, sampling_params=sampling_params)
        for t, o in zip(prompt_texts, time_offsets)
    ]
    return requests
def get_requests_from_file(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    """Get requests from a file."""
    if not args.prompt_file:
        raise ValueError("Prompt file is required to read requests from a file.")
    # Load prompts.
    n_prompts = sum(1 for _ in open(args.prompt_file))
    prompts = []
    if sampling_params is None:
        sampling_params = get_default_sampling_params(tokenizer.eod)
    sampling_params_list = []
    with open(args.prompt_file) as f:
        for line in tqdm(f.readlines(), "read prompt file", total=n_prompts):
def build_requests(
    args: Namespace, tokenizer: Any, sampling_params: Optional[SamplingParams] = None
) -> list[Request]:
    # Check if we have any prompts (from command line or JSONL)
    if args.prompts:
        if args.prompt_file:
            raise ValueError("Cannot use both --prompts and --prompt-file")
        return get_cli_requests(args, tokenizer, sampling_params)
    elif args.prompt_file:
        return get_requests_from_file(args, tokenizer, sampling_params)
    else:
        return get_synthetic_requests(args, tokenizer, sampling_params)
def get_model_size_str(model):
def build_dynamic_engine_setup_prefix(
    args: Namespace,
    model: MegatronModule,
    context: DynamicInferenceContext,
    requests: list[DynamicInferenceRequest],
):
def get_global_peak_memory_stats_bytes() -> dict:
    """Peak allocated CUDA memory aggregated across ranks (MAX), in bytes.
    Uses `torch.cuda.max_memory_allocated()` and assumes peak stats were reset
    before the benchmark run.
    """
    peak_alloc = int(torch.cuda.max_memory_allocated())
    if torch.distributed.is_available() and torch.distributed.is_initialized():

third_party/Megatron-LM/megatron/legacy/model/vision/dino.py
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
def get_student_backbone_and_num_features(config, pre_process=True, post_process=True):
def get_teacher_backbone_and_num_features(config, pre_process=True, post_process=True):

third_party/Megatron-LM/megatron/legacy/mpu/tests/commons.py
def set_random_seed(seed):
def initialize_distributed(backend='nccl'):
def print_separator(message):

third_party/Megatron-LM/examples/export/trtllm_export/single_device_export/gpt_single_device_cpu_export.py
def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
def model_provider():
def load_distributed_checkpoint(checkpoint_path, gpt_model):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_chartqa.py
def merge_input_files(input_path):
def chartqa_eval(input_path):

third_party/Megatron-LM/megatron/legacy/model/utils.py
def init_method_normal(sigma):
def scaled_init_method_normal(sigma, num_layers):
def attention_mask_func(attention_scores, attention_mask):
def get_linear_layer(rows, columns, init_method):
def gelu_impl(x):
def openai_gelu(x):
def erf_gelu(x):
def get_norm(config):

third_party/Megatron-LM/megatron/legacy/model/vision/vit_backbone.py
def isPerfectSquare(x):
def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

third_party/Megatron-LM/megatron/post_training/utils.py
def modelopt_version_higher_than(target_version: str):
def modelopt_version_at_least(target_version: str):
def function_has_parameter(function, argument_name: str) -> bool:
    """Check if a function has a specific argument."""
    sig = inspect.signature(function)
    return argument_name in sig.parameters
def get_current_memory_info():
def report_current_memory_info():
def get_mtbench_chat_data():
def to_empty_if_meta(module: torch.nn.Module, *, device: torch.device, recurse=True):
def print_distributed_quant_summary(model, msg=""):

third_party/Megatron-LM/examples/mimo/data/mock.py
def create_mock_image(image_size: int = 336) -> torch.Tensor:
    """
    Create a simple mock image (all zeros).
    Args:
        image_size: Size of the square image
    Returns:
        Tensor of shape [3, H, W] with all zeros
    """
    return torch.zeros(3, image_size, image_size)
def create_mock_caption() -> str:
    """
    Create a simple mock caption.
    Returns:
        A simple caption string
    """
    return "This is an image."
class MockVLMDataset(Dataset):
def get_mock_vlm_dataloader(
    batch_size: int = 8,
    dataset_size: int = 100,
    image_size: int = 224,
    seq_len: int = 77,
    image_seq_length: int = 32,
    num_workers: int = 0,
    pad_token_id: int = 0,
    image_token_id: int = 50000,
) -> DataLoader:
    """
    Create a DataLoader for mock VLM data.
    Args:
        batch_size: Batch size
        dataset_size: Size of the dataset
        image_size: Size of the square images
        seq_len: Total length of the token sequence (image + text)
        image_seq_length: Number of image tokens to pad
        num_workers: Number of worker processes for data loading
        pad_token_id: ID for padding token
        image_token_id: ID for image placeholder token
    Returns:
        DataLoader for the mock VLM dataset
    """
    dataset = MockVLMDataset(
        size=dataset_size,
        image_size=image_size,
        seq_len=seq_len,
        image_seq_length=image_seq_length,
        pad_token_id=pad_token_id,
        image_token_id=image_token_id,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_fn(batch),
    )
    return dataloader
def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for the DataLoader.
    Args:
        batch: List of dictionaries from the dataset
    Returns:
        Dictionary of batched tensors
    """
    images = torch.stack([item["images"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_inputs": {
            "clip_encoder": {
                "images": images,
            }
        },
    }
def train_valid_test_datasets_provider(train_val_test_num_samples):

third_party/Megatron-LM/megatron/legacy/mpu/tests/test_initialize.py
def test_initialize_model_parallel(tensor_model_parallel_size):
def test_get_tensor_model_parallel_src_rank(tensor_model_parallel_size_):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_mathvista.py
def merge_input_files(input_path):
def extra_processing(text):
def extract_answer(text):
def compute_mathvista_accuracy(result_file):
def mathvista_eval(input_path):

third_party/Megatron-LM/examples/export/trtllm_export/distributed_export/gpt_distributed_gpu_export.py
def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
def model_provider():
def load_distributed_checkpoint(checkpoint_path, gpt_model):

third_party/Megatron-LM/megatron/legacy/model/gpt_model.py
def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):

third_party/Megatron-LM/megatron/post_training/generate.py
def simple_generate(
    model,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    osl: int = 32,
    eos_token_id: List[int] = [],
    disable_tqdm: bool = False,
):
def simple_speculative_generate(
    model,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    osl: int = 32,
    steps: int = 0,
    eos_token_id: List[int] = [],
    disable_tqdm: bool = False,
):

third_party/Megatron-LM/examples/mimo/data/utils/calculate_audio_tokens.py
def calculate_num_mel_frames(audio_length, sample_rate, window_stride, window_length=None):
def calculate_num_audio_tokens(audio_tensor, model_name):

third_party/Megatron-LM/megatron/legacy/model/language_model.py
def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
def get_language_model(
    config,
    num_tokentypes,
    add_pooler,
    encoder_attn_mask_type,
    add_encoder=True,
    add_decoder=False,
    decoder_attn_mask_type=AttnMaskType.causal,
    pre_process=True,
    post_process=True,
):

third_party/Megatron-LM/megatron/post_training/arguments.py
def add_modelopt_args(parser):

third_party/Megatron-LM/examples/mimo/data/prepare_video_llava_data.py
def _extract_archives(root: str):
def convert_llava_video_to_wds(dataset_root: str, shard_size: int = 8000):

third_party/Megatron-LM/tests/unit_tests/test_sequence_packing_utils.py
def test_get_actual_sequence_lengths():
def test_get_actual_sequence_lengths_with_interior_padding():
def test_get_actual_sequence_lengths_invalid_shape():
def test_sequence_packing_basic():
def test_sequence_packing_with_generation_masks():
def test_sequence_packing_empty_bins():
def test_sequence_packing_integration():
def test_update_inference_logprobs_group_stats():
def test_update_inference_logprobs_group_stats_empty_mask():
def test_update_inference_logprobs_group_stats_with_mismatch():
def test_compute_packed_inference_logprobs_stats():
def test_compute_packed_inference_logprobs_stats_with_mismatch():
def test_compute_packed_inference_logprobs_stats_shape_mismatch():

third_party/Megatron-LM/megatron/legacy/mpu/tests/test_layers.py
def test_parallel_embedding(tensor_model_parallel_size):
def test_initialize_affine_weight(tensor_model_parallel_size):
def test_column_parallel_linear(tensor_model_parallel_size):
def test_row_parallel_linear(tensor_model_parallel_size):
def parallel_self_attention(tensor_model_parallel_size, num_att_heads_per_partition,
                            hidden_size_per_att_head, dropout_prob, batch_size,
                            sequence_length):
def test_parallel_self_attention(tensor_model_parallel_size):
def parallel_transformer(tensor_model_parallel_size, num_att_heads_per_partition,
                         hidden_size_per_att_head, batch_size, sequence_length):
def test_parallel_transformer_layer(tensor_model_parallel_size):

third_party/Megatron-LM/megatron/legacy/model/biencoder_model.py
def get_model_provider(only_query_model=False, only_context_model=False,
        biencoder_shared_query_context_model=False):
def biencoder_model_provider(only_query_model=False,
                             only_context_model=False,
                             biencoder_shared_query_context_model=False,
                             pre_process=True,
                             post_process=True):

third_party/Megatron-LM/megatron/legacy/model/vision/swin_backbone.py
def window_partition(x, window_size):
def window_reverse(windows, window_size, H, W):
def get_swin(drop_path_rate=0.3, output_avg=False):

third_party/Megatron-LM/megatron/post_training/model_builder.py
def count_parameters_in_layer(model, layer_name):
def _add_load_convert_hooks(model: MCoreGPTModel):
def _load_teacher_model_config(checkpoint_path: str) -> Namespace:
    """Reads teacher config from a file.
    The config provided via --teacher-model-config should specify
    (in NEMO format) any model architecture settings which differ from the main student model's.
    This function will translate NEMO field names to MCore as needed.
    """
    required_teacher_fields = (
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
    )
    args = get_args()
    config_path = os.path.join(checkpoint_path, "model_config.yaml") if args.teacher_model_config is None else args.teacher_model_config
    if not os.path.exists(config_path):
def _load_teacher_model(config, config_raw: Namespace, model_kwargs: Dict[str, Any]) -> MCoreGPTModel:
    """Teacher model creator."""
    args = get_args()
    if config.is_hybrid_model:
        # These parameters are not part of the TransformerConfig and need to be passed separately.
        if "hybrid_override_pattern" in config_raw:
            model_kwargs["hybrid_override_pattern"] = config_raw.hybrid_override_pattern
        if "hybrid_attention_ratio" in config_raw:
            model_kwargs["hybrid_attention_ratio"] = config_raw.hybrid_attention_ratio
        if "hybrid_mlp_ratio" in config_raw:
            model_kwargs["hybrid_mlp_ratio"] = config_raw.hybrid_mlp_ratio
        teacher = MCoreMambaModel(config=config, **model_kwargs)
    else:
        # GPT layer spec needs re-creation since it depends on number of model layers.
        if config.heterogeneous_block_specs:
            model_kwargs["transformer_layer_spec"] = get_gpt_heterogeneous_layer_spec(
                config=config,
                use_te=(args.transformer_impl == "transformer_engine"),
            )
        else:
            model_kwargs["transformer_layer_spec"] = get_gpt_modelopt_spec(
                config=config,
                local_core_attention=False if config.context_parallel_size > 1 else args.export_force_local_attention,
                remap_te_layernorm=args.export_te_mcore_model,
                real_quant_cfg=args.export_real_quant_cfg,
                use_arbitrary_attention_mask=False if config.context_parallel_size > 1 else True,
            )
        teacher = MCoreGPTModel(config=config, **model_kwargs)
    _add_load_convert_hooks(teacher)
    print_rank_0(f"Loading teacher as {type(teacher).__name__} from {args.export_kd_teacher_load} ...")
    # [WAR]: load checkpoint will check checkpoint's saved args and rng state if not finetune.
    # To avoid error out on loading teacher's checkpoint, we temporarily set args.finetune to
    # True while loading the teacher checkpoint.
    original_args_finetune, original_ckpt_format = args.finetune, args.ckpt_format
    args.finetune = True
    if args.export_kd_teacher_ckpt_format is not None:
        args.ckpt_format = args.export_kd_teacher_ckpt_format
    load_modelopt_checkpoint([teacher], load_arg='export_kd_teacher_load')
    args.finetune, args.ckpt_format = original_args_finetune, original_ckpt_format
    print_rank_0("...teacher loaded successfully.")
    return teacher
def modelopt_gpt_mamba_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
) -> MCoreGPTModel | MCoreMambaModel:
    """Builds the model.
    Args:
        args (Namespace):

third_party/Megatron-LM/megatron/legacy/mpu/tests/test_random.py
def test_set_cuda_rng_state(tensor_model_parallel_size):
def test_cuda_rng_tracker(tensor_model_parallel_size):
def test_model_parallel_cuda_manual_seed(tensor_model_parallel_size):

third_party/Megatron-LM/examples/mimo/data/energon_avlm_task_encoder.py
def llava_avlm_dataloader_provider(train_val_test_num_samples):

third_party/Megatron-LM/megatron/legacy/model/module.py
def param_is_not_shared(param):
def conversion_helper(val, conversion):
def fp32_to_float16(val, float16_convertor):
def float16_to_fp32(val):

third_party/Megatron-LM/megatron/post_training/loss_func.py
def _mask_loss(output_tensor, loss_mask):
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: GPTModel):

third_party/Megatron-LM/examples/academic_paper_scripts/detxoify_lm/generate_samples_gpt.py
def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.
    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the core GPT model.
    Args:
        pre_process (bool, optional):
def add_text_generate_args(parser):
def generate_samples_unconditional(model):
def generate_samples_conditional(model):
def generate_and_write_samples_unconditional(model):
def generate_and_write_samples_conditional(model):
def main():

third_party/Megatron-LM/megatron/legacy/mpu/tests/test_data.py
def test_broadcast_data(tensor_model_parallel_size):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluation_datasets.py
def _get_partition_bounds(
    total_num_samples, num_samples_per_partition, num_partitions, partition_id
):
def get_evaluation_dataset(
    task,
    input_image_path,
    gt_path,
    img_h,
    img_w,
    use_tiling,
    max_num_tiles,
    use_thumbnail,
    num_samples_per_partition,
    num_partitions,
    partition_id,
    num_frames,
    vision_model_type,
    split="validation",
):

third_party/Megatron-LM/tools/retro/cli/cli.py
def shorten_str(s: str, n: int) -> str:
    s = "\\n".join(s.splitlines())
    return s if len(s) <= n else "%s ... %s" % (s[: n // 2], s[-n // 2 :])
class retro:
    config = None
    ##############################################
    # initialize.
    ##############################################
    @classmethod
    def init(cls, project_dir: str) -> None:
        '''Initialize Megatron, tokenizers, and datasets.'''
        # Megatron args.
        args = parse_args(extra_args_provider=None, ignore_unknown_args=False)
        args.retro_project_dir = project_dir
        args.micro_batch_size = 1
        args.num_layers = 1
        args.hidden_size = 1
        args.num_attention_heads = 1
        args.async_tensor_model_parallel_allreduce = False
        args.retro_add_retriever = True # for building RetroDataset
        validate_args(args)
        set_global_variables(args)
        update_train_iters(args)
        # Retro config.
        cls.config = load_retro_config(project_dir)
        cls.config.retro_project_dir = project_dir
        cls.config.retro_tokenizers = get_tokenizers(cls.config)
        # Chunk database dataset.
        cls.db_indexed_dataset_infos = get_db_indexed_dataset_infos(project_dir)
        cls.db_dataset = get_db_dataset(project_dir,
                                        cls.config.retro_gpt_chunk_length,
                                        cls.config.retro_tokenizers.gpt.eod)
        # Pretraining datasets.
        pt_train_ds, pt_valid_ds, pt_test_ds = build_train_valid_test_datasets(
            train_valid_test_datasets_provider)
        cls.pt_datasets = SimpleNamespace(
            train=pt_train_ds,
            valid=pt_valid_ds,
            test=pt_test_ds,
        )
        # Print usage.
        cls.print_usage()
    ##############################################
    # utils.
    ##############################################
    @classmethod
    def gpt_to_text(cls, token_ids: np.ndarray) -> str:
        '''GPT tokens to text.'''
        return cls.config.retro_tokenizers.gpt.detokenize(
            token_ids.tolist() if isinstance(token_ids, np.ndarray) else token_ids
        )
    @classmethod
    def text_to_bert(cls, text: str) -> np.ndarray:
        '''Text to Bert tokens.'''
        return cls.config.retro_tokenizers.bert.tokenize(text)
    ##############################################
    # chunk db.
    ##############################################
    @classmethod
    def get_db_num_indexed_datasets(cls) -> int:
        '''Number of indexed datasets within blended dataset.'''
        return len(cls.db_indexed_dataset_infos)
    @classmethod
    def get_db_indexed_dataset_infos(cls) -> T.List[T.Tuple[float, str]]:
        '''Dataset infos, including number of training & sampled sets.'''
        return [(info["ratio"], info["prefix"]) for info in cls.db_indexed_dataset_infos]
    @classmethod
    def get_db_dataset(cls) -> DBDataset:
        return cls.db_dataset
    @classmethod
    def get_db_num_chunks(cls) -> int:
        '''Number of DB chunks.'''
        return len(cls.get_db_dataset())
    @classmethod
    def get_db_chunk_gpt(cls, idx: int) -> T.List[int]:
        '''Get DB chunk as GPT token ids.'''
        return cls.get_db_dataset()[idx]["text"].tolist()
    @classmethod
    def get_db_chunk_bert(cls, idx: int) -> T.List[int]:
        '''Get DB chunk as Bert token ids.'''
        return cls.text_to_bert(cls.get_db_chunk_text(idx))
    @classmethod
    def get_db_chunk_text(cls, idx: int) -> str:
        '''Get DB chunk as text.'''
        return cls.gpt_to_text(cls.get_db_chunk_gpt(idx))
    @classmethod
    def get_db_chunk_and_continuation_text(cls, idx: int) -> T.List[str]:
        '''Get DB chunk along with continuation, as text.'''
        # Modulus used here to match original implementation (i.e., last
        # chunks continuation wraps around to first chunk).
        return [
            cls.get_db_chunk_text(idx),
            cls.get_db_chunk_text((idx + 1) % len(cls.get_db_dataset())),
        ]
    ##############################################
    # pretraining corpus.
    ##############################################
    @classmethod
    def get_pt_num_samples_and_chunks(cls, data_key: str) -> T.Tuple[int, int]:
        '''Number of samples & chunks (e.g., 32*n_samples) in corpus.'''
        assert hasattr(cls.pt_datasets, data_key), (
            "pretraining set '%s' not found (choices: %s)."
            % (data_key, ", ".join(vars(cls.pt_datasets).keys()))
        )
        chunk_dataset = getattr(cls.pt_datasets, data_key).chunk_dataset
        return (
            len(chunk_dataset.sample_dataset),
            len(chunk_dataset),
        )
    @classmethod
    def get_pt_num_samples(cls, data_key: str) -> int:
        '''Number of pretraining samples.'''
        return cls.get_pt_num_samples_and_chunks(data_key)[0]
    @classmethod
    def get_pt_num_chunks(cls, data_key: str) -> int:
        '''Number of pretraining chunks (e.g., 32*n_samples).'''
        return cls.get_pt_num_samples_and_chunks(data_key)[1]
    @classmethod
    def get_pt_dataset(cls, data_key: str) -> RetroDataset:
        return getattr(cls.pt_datasets, data_key)
    @classmethod
    def get_pt_sample(cls, data_key: str, idx: int) -> dict:
        return getattr(cls.pt_datasets, data_key)[idx]
    @classmethod
    def get_neighbor_tokens(cls, sample_id: int, chunk_id: int, data_key: str="train") -> T.Optional[dict]:
        try:
            sample = cls.get_pt_sample(data_key, sample_id)
            sample_token_ids = sample["text"]
            chunk_length = cls.args.retro_gpt_chunk_length
            chunk_start_idx = chunk_id * chunk_length
            chunk_end_idx = min(sample_token_ids.shape[0], chunk_start_idx + chunk_length)
            chunk_token_ids = sample_token_ids[chunk_start_idx:chunk_end_idx]
            neighbor_token_ids = sample["neighbor_tokens"][chunk_id]
            return {
                "chunk_tokens": chunk_token_ids,
                "neighbor_tokens": neighbor_token_ids,
            }
        except Exception:
            return None
    @classmethod
    def print_neighbor_texts(cls, sample_id: int, chunk_id: int, data_key: str="train") -> None:
        tokens: dict = cls.get_neighbor_tokens(sample_id, chunk_id, data_key)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        try:
            print("PRETRAINING CHUNK:")
            print("  - %s" % shorten_str(cls.gpt_to_text(tokens["chunk_tokens"]), 150))
            print("NEIGHBOR_CHUNKS:")
            for token_ids in tokens["neighbor_tokens"]:
                print("  - %s" % shorten_str(cls.gpt_to_text(token_ids), 150))
        except Exception:
            print("<no neighbors for sample %d>" % sample_id)
    ##############################################
    # usage.
    ##############################################
    @classmethod
    def print_usage(cls) -> None:
        '''Print usage.'''
        print()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("examples ... [ *note*: 'db' = chunk db; 'pt' = pretraining corpus. ]")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()
        print("~~~~ indexed datasets ~~~~")
        print("retro.get_db_num_indexed_datasets() :

third_party/Megatron-LM/tests/unit_tests/inference/contexts/test_dynamic_context.py
def set_rounder(value):

third_party/Megatron-LM/tests/unit_tests/a2a_overlap/test_cuda_graphed_schedule_chunk_1f1b.py
def is_deep_ep_available():
def is_hybrid_ep_available():
def save(fn, message):

third_party/Megatron-LM/megatron/legacy/mpu/tests/test_cross_entropy.py
def torch_cross_entropy(batch_size, seq_length, vocab_size,
                        logits_scale, seed):
def mpu_cross_entropy(batch_size, seq_length, vocab_size,
                      logits_scale, seed):
def test_cross_entropy(tensor_model_parallel_size):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_ocrbench_v2.py
def convert_to_ocrbench_v2_format(input_path, groundtruth_path):
def ocrbench_v2_eval(input_path, groundtruth_path, output_path):
def main():

third_party/Megatron-LM/scripts/check_api_backwards_compatibility.py
def has_exempt_decorator(obj: Object) -> bool:
    """Check if a Griffe object has any exempt decorator.
    
    Args:
        obj: A Griffe Object to check for exempt decorators
        
    Returns:
        bool: True if the object has any decorator matching EXEMPT_DECORATORS list
    """
    if not hasattr(obj, 'decorators'):
def get_filtered_paths(package: Object, package_name: str) -> set:
    """Recursively collect all object paths with exempt decorators from a package.
    
    This function traverses the entire package tree and identifies objects that are
    decorated with any of the EXEMPT_DECORATORS, building a set of their full paths.
    
    Args:
        package: The Griffe package object to traverse
        package_name: The full package name (e.g., "megatron.core") for path construction
        
    Returns:
        set: A set of full object paths (e.g., "megatron.core.ModelParallelConfig") 
             that should be filtered from compatibility checks
    """
    filtered = set()
    visited = set()
    
    def visit(obj, path, depth=0, is_root=False):
def strip_ansi_codes(text):
def get_object_path(change) -> str:
    """Extract the full object path from a Griffe breaking change.
    
    Tries multiple sources to get the object path:
    1. Direct path attributes (new_path, old_path, path)
    2. Path from new_value or old_value objects
    3. Parse from the explanation string as last resort
    
    Args:
        change: A Griffe breaking change object
        
    Returns:
        str: The full object path (e.g., "megatron.core.ModelParallelConfig.__init__")
             or None if unable to extract
    """
    # Try different attributes
    path = (getattr(change, 'new_path', None) or 
            getattr(change, 'old_path', None) or
            getattr(change, 'path', None))
    
    if path:
        return strip_ansi_codes(path)
    
    # Try from values
    if hasattr(change, 'new_value') and change.new_value:
        path = getattr(change.new_value, 'path', None)
        if path:
            return strip_ansi_codes(path)
    
    if hasattr(change, 'old_value') and change.old_value:
        path = getattr(change.old_value, 'path', None)
        if path:
            return strip_ansi_codes(path)
    
    # Last resort: parse from explanation
    # Format: "filepath:line: object_path: description"
    # Example: "megatron/core/model_parallel_config.py:338: ModelParallelConfig.cpu_offloading_weights: Attribute value was changed"
    try:
        explanation = change.explain()
        # Split by ": " and get the second part (object path)
        parts = explanation.split(': ')
        if len(parts) >= 2:
            # Get the part after "filepath:line" but before the description
            # It's usually the second part
            object_path = parts[1]
            
            # Extract the module path from file path (first part)
            file_part = parts[0].split(':')[0]  # Get just the file path, remove line number
            
            # Convert file path to module path
            # e.g., "megatron/core/model_parallel_config.py" -> "megatron.core.model_parallel_config"
            module_path = file_part.replace('/', '.').replace('\\', '.').replace('.py', '')
            
            # If object_path doesn't start with module, prepend it
            if not object_path.startswith(module_path):
def should_skip_change(change, filtered_paths: set) -> bool:
    """Determine if a breaking change should be skipped.
    
    A change is skipped if:
    - The change kind is in IGNORED_BREAKAGE_KINDS (not a signature change)
    - The change kind is in IGNORED_FOR_INIT_METHODS and affects an __init__ method
    - The changed object itself is in filtered_paths (exact match)
    - The changed object is a child of an exempt object (prefix match)
    
    Args:
        change: A Griffe breaking change object
        filtered_paths: Set of paths with exempt decorators
        
    Returns:
        bool: True if the change should be skipped (filtered out)
    """
    # Check if this breakage kind should be ignored globally (not a signature change)
    change_kind = type(change).__name__
    if change_kind in IGNORED_BREAKAGE_KINDS:
        return True
    
    path = get_object_path(change)
    if not path:
        return False
    
    # Strip parameter names from path for matching
    # e.g., "Class.__init__(param)" -> "Class.__init__"
    clean_path = path.split('(')[0] if '(' in path else path
    
    # Check if this is a breakage kind we ignore for __init__ methods
    # Config dataclasses should use keyword args, so parameter reordering is safe
    if change_kind in IGNORED_FOR_INIT_METHODS:
        if '.__init__' in clean_path:
            return True
    
    # Check exact match
    if clean_path in filtered_paths or path in filtered_paths:
        return True
    
    # Check if it's a child of a filtered object
    # e.g., MyClass.__init__ is child of MyClass, MyClass.attr is child of MyClass
    for filtered_path in filtered_paths:
        if clean_path.startswith(filtered_path + '.'):
def main():

third_party/Megatron-LM/examples/academic_paper_scripts/detxoify_lm/annotations/perspective_api_annotate.py
def test():
def split_lines(lines, split):
def get_score(line):
def get_scores(lines):
def get_annotated_datasets(lines, threads=10):
def main():

third_party/Megatron-LM/examples/run_simple_mcore_train_loop.py
def initialize_distributed(
    tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1
) -> None:
    """
    Initialize torch.distributed and Megatron-Core model parallel groups.
    Args:
        tensor_model_parallel_size: Number of GPUs for tensor model parallelism.
        pipeline_model_parallel_size: Number of GPUs for pipeline model parallelism.
    """
    parallel_state.destroy_model_parallel()
    # Torch setup for distributed training
    rank: int = int(os.environ["RANK"])
    world_size: int = int(os.environ["WORLD_SIZE"])
    local_rank: int = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )
    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )
def model_provider() -> GPTModel:
    """
    Build and return a simple GPT model for demonstration.
    Returns:
        GPTModel: A small GPT model with 2 layers for testing.
    """
    transformer_config: TransformerConfig = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    gpt_model: GPTModel = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=_SEQUENCE_LENGTH,
    )
    return gpt_model
def get_train_data_iterator() -> Iterator:
    """
    Create a mock dataset and return a data iterator.
    Returns:
        Iterator: Data iterator for training batches.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
def forward_step_func(
    data_iterator: Iterator, model: torch.nn.Module
) -> Tuple[torch.Tensor, Callable]:
    """
    Forward step function that computes model output and returns loss function.
    Args:
        data_iterator: Iterator providing training batches.
        model: The GPT model to train.
    Returns:
        Tuple of (output_tensor, loss_function) where loss_function is a partial
        function that will compute the final loss when called.
    """
    def loss_func(
        loss_mask: torch.Tensor, output_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: torch.Tensor = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss: torch.Tensor = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.
        return loss, {"lm loss": loss}
    data: Dict[str, torch.Tensor] = next(data_iterator)
    tokens: torch.Tensor = data["tokens"].to(device)
    attention_mask: torch.Tensor = data["attention_mask"].to(device)
    position_ids: torch.Tensor = data["position_ids"].to(device)
    labels: torch.Tensor = data["labels"].to(device)
    loss_mask: torch.Tensor = data["loss_mask"].to(device)
    output_tensor: torch.Tensor = model(
        tokens, position_ids, attention_mask, labels=labels
    )
    return output_tensor, partial(loss_func, loss_mask)
def save_distributed_checkpoint(
    checkpoint_path: str, gpt_model: torch.nn.Module
) -> None:
    """
    Save model checkpoint using Megatron-Core distributed checkpointing.
    Args:
        checkpoint_path: Directory path to save checkpoint.
        gpt_model: The model to checkpoint (may be wrapped with DDP).
    """
    # Access underlying model if wrapped with DDP
    model: torch.nn.Module = (
        gpt_model.module if hasattr(gpt_model, "module") else gpt_model
    )
    sharded_state_dict: Dict = model.sharded_state_dict(prefix="")
    dist_checkpointing.save(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
def load_distributed_checkpoint(
    checkpoint_path: str, gpt_model: torch.nn.Module
) -> torch.nn.Module:
    """
    Load model checkpoint using Megatron-Core distributed checkpointing.
    Args:
        checkpoint_path: Directory path to load checkpoint from.
        gpt_model: The model to load into (may be wrapped with DDP).
    Returns:
        The model with loaded checkpoint weights.
    """
    # Access underlying model if wrapped with DDP
    model: torch.nn.Module = (
        gpt_model.module if hasattr(gpt_model, "module") else gpt_model
    )
    sharded_state_dict: Dict = model.sharded_state_dict(prefix="")
    checkpoint: Dict = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    model.load_state_dict(checkpoint)
    return gpt_model
if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    gpt_model: GPTModel = model_provider()
    device: torch.device = torch.device("cuda")
    gpt_model.to(device)
    # Wrap model with DistributedDataParallel for proper gradient synchronization.
    # This provides the finish_grad_sync() method required by finalize_model_grads().
    config: TransformerConfig = gpt_model.config
    ddp_config: DistributedDataParallelConfig = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    gpt_model = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=gpt_model,
    )
    optim: Adam = Adam(gpt_model.parameters())
    train_iterator: Iterator = get_train_data_iterator()
    forward_backward_func: Callable[..., Dict[str, Any]] = get_forward_backward_func()
    # Running the model for 5 iterations
    for iteration in range(5):

third_party/Megatron-LM/tests/unit_tests/test_api_backwards_compat_setup.py
def test_griffe_installed():
def test_decorator_module():
def test_checker_script():
def test_workflow():
def test_decorators_work():
def test_basic_comparison():
def example_func(x, y):
def example_func(x, y, z=None):
def main():

third_party/Megatron-LM/megatron/legacy/data/ict_dataset.py
def make_attention_mask(source_block, target_block):
def get_ict_dataset(use_titles=True, query_in_block_prob=1):

third_party/Megatron-LM/tests/unit_tests/a2a_overlap/utils.py
def build_data(seq_len=1024):
def deterministic_mode():
def reset_model(model, params=None):
def compare_captures(capture_ref, capture_a2a_overlap, verbose=False, skip_embedding=False):
def get_test_config(num_layers=1, num_moe_experts=8, extra_kwargs={}, moe_grouped_gemm=True):
def get_valid_token_dispatcher_types():
def get_valid_fp8_flags():

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_mmmu.py
def get_input_output_paths(input_path, task):
def extract_answer(text):
def convert_to_mmmu_format(input_path):
def mmmu_eval(input_path, groundtruth_path):
def main():

third_party/Megatron-LM/gpt_builders.py
def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
def _get_transformer_layer_spec(use_te, config):

third_party/Megatron-LM/megatron/post_training/checkpointing.py
def has_modelopt_state(checkpoint_path: str) -> bool:
    """Check if modelopt_state folder exists inside the checkpoint.
    Args:
        checkpoint_path: Path to the checkpoint directory
    Returns:
        True if modelopt_state exists, False otherwise
    """
    args = get_args()
    try:
        if args.ckpt_format == "torch":
            # Non-sharded
            state_dict, _, _ = _load_base_checkpoint(checkpoint_path, rank0=False)
            if state_dict is None:
                return False
            if "modelopt_state" not in state_dict:
                return False
            return True
        else:
            # Sharded
            load_dir, _ = get_sharded_load_dir(checkpoint_path)
            if load_dir is None:
                return False
            if not (load_dir / "modelopt_state").is_dir():
def get_sharded_load_dir(load_dir: str) -> Tuple[Union[Path, None], str]:
    """Helper to retrieve the sharded load directory and its prefix, if any."""
    load_dir = Path(load_dir)
    # Skip if load_dir is nonexistent or empty
    if not load_dir.is_dir() or not any(load_dir.iterdir()):
def load_modelopt_state(model: nn.Module, load_dir: Optional[str] = None) -> None:
    """Loading modelopt_state without loading the model.
    If distributed checkpointing in use, we try to load from the sharded modelopt_state. This will not
    load the model state_dict. Otherwise, if the checkpoint is not sharded, we load the base checkpoint
    (which contains the model state as well) and extract the modelopt_state.
    Args:
        model: the model to load the modelopt_state into
        load_dir: optionally provide a different loading path
    """
    args = get_args()
    load_dir = load_dir or args.load
    if args.ckpt_format == "torch":
        # Non-sharded
        print_rank_0(f"Loading ModelOpt state from base checkpoint ({load_dir})")
        try:
            state_dict, _, _ = _load_base_checkpoint(args.load, rank0=False)
        except Exception:
            print_rank_0("Failed to load base checkpoint via megatron _load_base_checkpoint!")
            return
        if state_dict is None:
            print_rank_0("No checkpoint state_dict found. Skipping loading ModelOpt state.")
            return
        modelopt_state = state_dict.get("modelopt_state", None)
        if modelopt_state is not None:
            mto.restore_from_modelopt_state(model, modelopt_state)
    else:
        # Sharded
        sharded_load_dir, _ = get_sharded_load_dir(load_dir)
        if sharded_load_dir is None:
            print_rank_0("No sharded checkpoint found. Skipping loading modelopt_state.")
            return
        restore_sharded_modelopt_state([model], sharded_load_dir)
def load_modelopt_checkpoint(
    model,
    optimizer=None,
    opt_param_scheduler=None,
    strict: bool = True,
    additional_sharded_prefix: str = "",
    load_arg: str = "load",
) -> None:
    """Load a sharded (untar .nemo or megatron --use-dist-ckpt) or unsharded checkpoint.
    Essentially, the function is detecting whether the checkpoint is a .nemo sharded checkpoint.
    If so, we load the sharded state_dict with additional_sharded_prefix `model.`.
    This additional prefix is tha artifact of the lightning module wrapper. Once the sharded
    state_dict is loaded, we use a state_dict pre_hook to pop this additional prefix (`model.`)
    from all state_dict keys.
    If this is not a .nemo sharded checkpoint, then this function will simply call
    load_checkpoint. See megatron.checkpointing.load_checkpoint for explanation.
    Args:
        additional_sharded_prefix: append additional prefix to align the sharded checkpoint keys.
            When loading an .nemo sharded checkpoint, this is usually `model.`. Otherwise, this is
            typically an empty string.
    """
    def _remove_prefix_state_dict_pre_hook(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):

third_party/Megatron-LM/tests/unit_tests/a2a_overlap/test_schedule_chunk_1f1b.py
def build_model(config):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_spdocvqa.py
def merge_input_files(input_path):
def spdocvqa_eval(input_path):

third_party/Megatron-LM/examples/academic_paper_scripts/detxoify_lm/annotations/filter-selfgeneration.py
def get_corpus_scores(lines):
def main():

third_party/Megatron-LM/examples/academic_paper_scripts/detxoify_lm/finetune_gpt.py
def model_provider(pre_process=True, post_process=True):
def get_batch(data_iterator):
def loss_func(loss_mask, output_tensor):
def forward_step(data_iterator, model):
def train_valid_test_datasets_provider(train_val_test_num_samples):
def add_validation_args(parser):

third_party/Megatron-LM/tests/unit_tests/inference/contexts/attention_metadata/test_tensor_ops.py
def tensor_get_slice_after_pytorch(
    input_tensor: torch.Tensor, output_tensor: torch.Tensor, pos_on_device: torch.Tensor
) -> None:
    """Reference PyTorch implementation of tensor_get_slice_after."""
    assert input_tensor.ndim == output_tensor.ndim, "Rank mismatch"
    for i in range(1, input_tensor.ndim):
def tensor_merge_pytorch(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    output_tensor: torch.Tensor,
    pos_on_device: torch.Tensor,
) -> None:
    """Reference PyTorch implementation of tensor_merge."""
    assert tensor_a.ndim == tensor_b.ndim == output_tensor.ndim, "Rank mismatch across tensors"
    for i in range(1, tensor_a.ndim):
def device():
def slice_params():
def test_get_slice_after_basic(device, slice_params):
def test_get_slice_after_pos_zero(device, slice_params):
def test_get_slice_after_pos_full(device, slice_params):
def test_get_slice_after_exact_fit(device):
def test_get_slice_after_nd(device):
def test_get_slice_after_bounds(device, slice_params):
def test_get_slice_after_consistency(device):
def merge_params():
def test_tensor_merge_basic(device, merge_params, in_place):
def test_tensor_merge_pos_zero(device, merge_params):
def test_tensor_merge_pos_full(device, merge_params):
def test_tensor_merge_small(device):
def test_tensor_masked_update(device, ndim):

third_party/Megatron-LM/examples/academic_paper_scripts/detxoify_lm/perspective_api.py
def test():
def get_score(x):
def main():

third_party/Megatron-LM/tests/unit_tests/tokenizers/test_tokenizer.py
def get_conversation():
def get_chat_template():
def test_sp_tokenizer():
def test_hf_tokenizer():
def test_megatron_tokenizer():
def test_tiktoken_tokenizer():
def test_null_tokenizer():
def test_bytelevel_tokenizer():
def test_write_metadata():

third_party/Megatron-LM/megatron/legacy/data/orqa_wiki_dataset.py
def get_open_retrieval_wiki_dataset():
def get_open_retrieval_batch(data_iterator):
def build_tokens_types_paddings_from_text(row, tokenizer, max_seq_length):
def build_tokens_types_paddings_from_ids(text_ids, max_seq_length,
                                         cls_id, sep_id, pad_id):
def build_sample(row_id, context_ids, context_types, context_pad_mask):

third_party/Megatron-LM/tests/unit_tests/utils/test_experimental_log_once.py
def _get_test_logger():
def test_experimental_fn_logs_once(caplog):
def test_experimental_cls_logs_once(caplog):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_vqav2.py
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    dist = levenshtein_distance(s1, s2)
    length = max(len(s1.upper()), len(s2.upper()))
    return 0.0 if length == 0 else dist / length
def similarity_function(prediction: str, gold_label: str, threshold: float) -> float:
    nl_score = normalized_levenshtein_distance(prediction, gold_label)
    return 1 - nl_score if nl_score < threshold else 0.0
def anls_score(
    prediction: str, gold_labels: List[str], threshold: float = 0.5
) -> float:
    # not case sensitive, but space sensitive
    y_pred = " ".join(prediction.strip().lower().split())
    anls_scores: List[float] = []
    for gold_label in gold_labels:
        # not case sensitive, but space sensitive
        y_true = " ".join(gold_label.strip().lower().split())
        anls_score = similarity_function(y_pred, y_true, threshold)
        anls_scores.append(anls_score)
    score = max(anls_scores)
    return score
def merge_input_files(input_path):
def is_number(n: str):
def compute_vqa_accuracy(result_file, task):
def vqav2_eval(input_path):

third_party/Megatron-LM/tests/unit_tests/test_parallel_state.py
def test_initialize_and_destroy_model_parallel(order):
def test_pipeline_parallel_initializations(order):
def test_data_parallel_initializations(order):
def test_tensor_model_parellel_world_size(order):
def test_expert_tensor_parellel_world_size(order):
def test_pipeline_model_parallel_world_size(order):
def test_tensor_model_parallel_rank(order):
def test_moe_tensor_model_parellel_rank(order):
def test_pipeline_model_parallel_rank(order):
def test_context_parallel_rank():
def test_expert_model_parallel_rank():
def test_is_pipeline_first_stage(order):
def test_is_pipeline_last_stage(order):
def test_virtual_pipeline_model_parallel_rank(order):
def test_get_tensor_model_parallel_src_rank(order):
def test_different_initialize_order_consistency(src_tp_pp, ep_size):
def test_different_initialize_order_unconsistency(src_tp_pp, ep_size):
def test_rank_generator_for_tp_dp_pp(nodes, num_gpu, tp, pp, cp, ep):
def test_hybrid_dp_cp_groups(world_size, tp_size, cp_size, dp_size):

third_party/Megatron-LM/megatron/legacy/data/image_folder.py
def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string):
def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string):
def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    data_per_class_fraction: float,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    Args:
        directory (str):
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
class ImageFolder(DatasetFolder):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_video_motionbench.py
def merge_input_files(input_path):
def motionbench_eval(input_path):

third_party/Megatron-LM/examples/multimodal/model_converter/clip_converter.py
def convert(download_root, output_path, tensor_parallel_size, use_te):

third_party/Megatron-LM/tests/unit_tests/optimizer/test_optimizer_config.py
def test_paramkey_matches():

third_party/Megatron-LM/tests/unit_tests/a2a_overlap/test_schedule_layer_1f1b.py
def run_transformer_layer_ref_with_capture(model, input_tensors, iterations):
def run_transformer_layer_a2a_overlap_with_capture(model, input_tensors, microbatches):
def run_mtp_layer_ref_with_capture(
    model,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    attention_mask,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    microbatches,
):
def run_mtp_layer_a2a_overlap_with_capture(
    model,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    attention_mask,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    microbatches,
):

third_party/Megatron-LM/megatron/legacy/data/biencoder_dataset_utils.py
def make_attention_mask(source_block, target_block):
def get_one_epoch_dataloader(dataset, micro_batch_size=None):
def get_ict_batch(data_iterator):
def join_str_list(str_list):
def get_block_samples_mapping(block_dataset, title_dataset, data_prefix, num_epochs,
                              max_num_samples, max_seq_length, seed, name, use_one_sent_docs=False):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_realworldqa.py
def merge_input_files(input_path):
def realworldqa_eval(input_path):

third_party/Megatron-LM/megatron/core/inference/contexts/fused_kv_append_kernel.py
def _append_kv_cache_kernel(
    # --- Pointers to Tensors ---
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_idx_ptr,
    local_kv_seq_idx_ptr,
    # --- Strides for Tensor Memory Layout ---
    stride_key_token,
    stride_key_head,
    stride_key_hdim,
    stride_value_token,
    stride_value_head,
    stride_value_hdim,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_cache_hdim,
    # --- Other Parameters ---
    n_tokens: tl.int32,
    num_heads: tl.int32,
    H_DIM: tl.int32,
    # --- Compile-Time Constants ---
    BLOCK_SIZE_H: tl.constexpr,
):
def triton_append_key_value_cache(
    layer_number: int,
    key: Tensor,
    value: Tensor,
    memory_buffer: Tensor,
    padded_active_token_count: int,
    token_to_block_idx: Tensor,
    token_to_local_position_within_kv_block: Tensor,
) -> None:
    """
    Append to KV cache using a high-performance, standalone Triton kernel.
    Args:
        layer_number (int):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_infovqa.py
def merge_input_files(input_path):
def infovqa_eval(input_path):

third_party/Megatron-LM/tests/unit_tests/conftest.py
def pytest_addoption(parser):
def experimental(request):
def pytest_sessionfinish(session, exitstatus):
def set_env():
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.
    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """
    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'
    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
def ensure_test_data():
def reset_env_vars():

third_party/Megatron-LM/megatron/legacy/model/transformer.py
def sinkhorn(cost, tol=0.0001):
def get_router_linear_layer(config):
def bias_dropout_add(x, bias, residual, prob, training):
def get_bias_dropout_add(training):
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)
@jit_fuser
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: Optional[torch.Tensor],
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)
class ParallelTransformerLayer(MegatronModule):
def _get_num_layers(args, model_type, is_decoder=False):
def _get_layer_type(model_type, default_layer_type, retro_layer_numbers,
                    layer_number):

third_party/Megatron-LM/megatron/legacy/data/realm_dataset_utils.py
def get_one_epoch_dataloader(dataset, micro_batch_size=None):
def get_ict_batch(data_iterator):
def join_str_list(str_list):
def get_block_samples_mapping(block_dataset, title_dataset, data_prefix, num_epochs,
                              max_num_samples, max_seq_length, seed, name, use_one_sent_docs=False):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_ocrbench.py
def merge_input_files(input_path):
def compute_ocrbench_score(result_file):
def ocrbench_eval(input_path):

third_party/Megatron-LM/examples/multimodal/model_converter/radio_converter.py
def convert_radio_h(output_path, tensor_parallel_size, use_te, version):
def convert_radio_g(output_path, tensor_parallel_size, use_te, version):
def convert(output_path, tensor_parallel_size, use_te, model_type, version):

third_party/Megatron-LM/megatron/legacy/data/vit_dataset.py
def build_train_valid_datasets(data_path, image_size=224):

third_party/Megatron-LM/tests/unit_tests/test_basic.py
def test_import():

third_party/Megatron-LM/megatron/legacy/data/realm_index.py
def detach(tensor):

third_party/Megatron-LM/tests/unit_tests/test_tokenizer.py
def offsets_to_substrs(offsets, string):
def local_test_specs():
def gpt2_tiktok_vocab(tmp_path_factory):
def test_tokenizer(args):
def test_gpt2_tiktok_tokenizer(gpt2_tiktok_vocab):
def run_tokenizer_tests(tok):
def test_null_tokenizer():
def test_multimodal_tokenizer():

third_party/Megatron-LM/examples/multimodal/model_converter/siglip_converter.py
def convert(output_path, tensor_parallel_size, use_te):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_rd_tablebench.py
def convert_to_rdtablebench_format(input_path):
def rdtablebench_eval(input_path):
def main():

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_video_mvbench.py
def merge_input_files(input_path):
def check_ans(pred, gt):
def create_result_dict(result_list):
def combine_all_res(acc_dict):
def mvbench_eval(input_path):

third_party/Megatron-LM/megatron/legacy/data/multimodal_dataset.py
def _convert_image_to_rgb(image):
def _transform(img_h, img_w):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_textvqa.py
def merge_input_files(input_path):
def textvqa_eval(input_path):

third_party/Megatron-LM/examples/multimodal/model_converter/vision_model_tester.py
def run_mcore_vision(model_path):
def run_hf_vision(model_name):
def main(mcore_model, hf_model):

third_party/Megatron-LM/tests/unit_tests/rl/test_rl_batch_invariant.py
def test_selective_log_softmax_batch_invariant():

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_ai2d.py
def merge_input_files(input_path):
def ai2d_eval(input_path):

third_party/Megatron-LM/examples/multimodal/model_converter/internvit_converter.py
def convert(model_name, output_path, tensor_parallel_size, use_te):

third_party/Megatron-LM/examples/multimodal/evaluation/mmmu_utils.py
def load_yaml(file_path):
def parse_img_path(text):
def process_single_sample(data):
def construct_prompt(sample, config):
def parse_multi_choice_response(response, all_choices, index2ans):
def check_is_number(string):
def normalize_str(string):
def extract_numbers(string):
def parse_open_response(response):
def eval_multi_choice(gold_i, pred_i):
def eval_open(gold_i, pred_i):
def evaluate(samples):
def calculate_ins_level_acc(results: Dict):
def mmmu_main_eval(output_dict, task_cfg):

third_party/Megatron-LM/tests/unit_tests/data/test_preprocess_data.py
def dummy_jsonl(odir):
def build_datasets(idir, odir, extra_args=[]):
def merge_datasets(idir):
def do_test_preprocess_data(temp_dir, extra_args=[]):
def gpt2_vocab(odir):
def gpt2_merge(odir):
def test_preprocess_data_gpt():
def bert_vocab(odir):
def test_preprocess_data_bert():

third_party/Megatron-LM/tests/unit_tests/inference/engines/test_dynamic_engine.py
def skip_if_mamba_sequence_packing_not_available(model_provider: str):
def set_rounder(value):

third_party/Megatron-LM/tests/unit_tests/test_local_multi_tensor_fns.py
def test_local_multi_tensor_l2_norm_and_scale():
def test_local_multi_tensor_apply():

third_party/Megatron-LM/examples/multimodal/model.py
def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> LLaVAModel:
    """Builds the model.
    Args:
        pre_process (bool):
def _get_tile_tags(args, tokenizer):

third_party/Megatron-LM/tests/unit_tests/data/test_fim_dataset.py
def test_fim_gpt_dataset(spm_rate, split_sample):

third_party/Megatron-LM/examples/multimodal/evaluation/evaluate_coco.py
def convert_to_coco_format(input_path):
def coco_captioning_eval(input_path, groundtruth_file):

third_party/Megatron-LM/tests/unit_tests/resharding/test_model_swap.py
def _build_pg_collection(
    tp_size: int, pp_size: int = None, ep_size: int = 1
) -> ProcessGroupCollection:
    cp_size = mpu.get_context_parallel_world_size()
    if pp_size is None:
        pp_size = mpu.get_pipeline_model_parallel_world_size()
    world_size = dist.get_world_size()
    dp_size = world_size // (tp_size * cp_size * ep_size * pp_size)
    assert dp_size >= 1 and (tp_size * cp_size * ep_size * pp_size * dp_size) == world_size
    grid = HyperCommGrid(
        [tp_size, cp_size, ep_size, pp_size, dp_size], ["tp", "cp", "ep", "pp", "dp"]
    )
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pp_group = grid.create_pg("pp")
    ep_group = grid.create_pg("ep")
    dp_group = grid.create_pg("dp")
    # Composite groups required by MoE/router and some utilities
    tp_cp_group = grid.create_pg(["tp", "cp"])
    mp_group = grid.create_pg(["tp", "cp", "ep", "pp"])
    tp_ep_group = grid.create_pg(["tp", "ep"])
    tp_ep_pp_group = grid.create_pg(["tp", "ep", "pp"])
    dp_cp_group = grid.create_pg(["cp", "dp"])
    tp_dp_cp_group = grid.create_pg(["tp", "cp", "dp"])
    embd_group_ranks = mpu.default_embedding_ranks(dist.get_process_group_ranks(pp_group))
    embd_group = dist.new_group(ranks=embd_group_ranks)
    pos_embd_group_ranks = mpu.default_position_embedding_ranks(
        dist.get_process_group_ranks(pp_group)
    )
    pos_embd_group = dist.new_group(ranks=pos_embd_group_ranks)
    return ProcessGroupCollection(
        tp=tp_group,
        cp=cp_group,
        pp=pp_group,
        ep=ep_group,
        embd=embd_group,
        pos_embd=pos_embd_group,
        dp=dp_group,
        tp_cp=tp_cp_group,
        mp=mp_group,
        expt_tp=tp_group,
        expt_dp=dp_group,
        tp_ep=tp_ep_group,
        tp_ep_pp=tp_ep_pp_group,
        dp_cp=dp_cp_group,
        tp_dp_cp=tp_dp_cp_group,
    )
def _build_gpt(
    config: TransformerConfig,
    vocab_size: int,
    seq_len: int,
    pg_collection,
    parallel_output: bool = True,
    num_moe_experts: Optional[int] = None,
) -> GPTModel:
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_moe_experts, moe_grouped_gemm=(num_moe_experts is not None)
        ),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=True,
        post_process=True,
        fp16_lm_cross_entropy=False,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        pg_collection=pg_collection,
    )
    return model
def _mp_config() -> ModelParallelConfig:
    return ModelParallelConfig(
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
    )
def _set_pg_collection(module, tp_group, dp_group):
def test_swap_gpt_parametrized(
    refit_backend: str,
    src_tp: int,
    src_pp: int,
    src_ep: int,
    dst_tp: int,
    dst_pp: int,
    dst_ep: int,
    num_experts: Optional[int],
):

third_party/Megatron-LM/megatron/core/inference/contexts/dynamic_context.py
def get_mem_size_str(n_bytes: int) -> str:
    """Convert number of bytes to human-readable string."""
    for exp, suffix in ((4, "TB"), (3, "GB"), (2, "MB"), (3, "KB"), (0, "bytes")):

third_party/Megatron-LM/tests/unit_tests/test_model_configs.py
def get_yaml_files(directory):
def load_yaml(file_path):
def test_model_config_tracks_memory(yaml_file, metric):

third_party/Megatron-LM/tests/unit_tests/models/test_mamba_moe_model.py
def serialize_config(cfg: Any) -> Dict[str, Any]:
    """Normalize a config object into a JSON-serializable dict."""
    data = {k: v for k, v in vars(cfg).items() if k not in SKIP_FIELDS}
    return _ser(data)
def assert_config_matches_golden(cfg: Any) -> None:
    """Compare live config to golden snapshot with readable diffs."""
    current = serialize_config(cfg)
    golden = GOLDEN_CONFIG
    added, removed, changed = _diff_configs(golden, current)
    # Ignore added fields that are explicitly allowed.
    added = [k for k in added if k not in ALLOW_ADDED_FIELDS]
    if added or removed or changed:
        # Build actionable guidance for each type of drift
        guidance_parts = []
        if added:
            guidance_parts.append(
                f"\n\n[ADDED ARGS]: {sorted(added)}\n"
                "  → Update GOLDEN_CONFIG in this test file to include the new arg(s) with "
                "their default value(s).\n"
                "  ⚠️  CAUTION: Review any logic associated with new args to ensure it doesn't "
                "silently affect downstream model configs or behavior.\n"
            )
        if changed:
            guidance_parts.append(
                f"\n\n[CHANGED DEFAULTS]: {sorted(changed)}\n"
                "  → Please don't change the default values of existing args unless "
                "it is absolutely necessary for a bug fix.\n"
                "  → If you must change the default value, please update the GOLDEN_CONFIG "
                "in this test file to reflect the new default value.\n"
            )
        if removed:
            guidance_parts.append(
                f"\n\n[REMOVED ARGS]: {sorted(removed)}\n"
                "  → Do NOT remove args directly. Instead, deprecate them with a warning message "
                "to maintain backwards compatibility.\n"
            )
        guidance_parts.append(
            "Please contact NV-username @jbarker if you are unsure how to proceed.\n"
        )
        header = "Mamba MoE config drift detected!\n" "═" * 60 + "".join(guidance_parts)
        parts = [header]
        if changed:
            formatted = {k: {"expected": golden[k], "actual": current[k]} for k in sorted(changed)}
            parts.append(
                f"Changed field details:\n{json.dumps(formatted, indent=2, sort_keys=True)}"
            )
        pytest.fail("\n".join(parts))
def regenerate_mamba_moe_golden(cfg: Any) -> Dict[str, Any]:
    """Helper to regenerate the golden config; copy/paste into GOLDEN_CONFIG."""
    serialized = serialize_config(cfg)
    return serialized
def _ser(obj: Any) -> Any:
    """Recursively convert objects to JSON-friendly structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
def _diff_configs(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Tuple[set, set, set]:
    """Return added, removed, and changed top-level keys between dicts."""
    expected_keys = set(expected)
    actual_keys = set(actual)
    added = actual_keys - expected_keys
    removed = expected_keys - actual_keys
    changed = {k for k in expected_keys & actual_keys if expected[k] != actual[k]}
    return added, removed, changed
class TestMambaMoEModel:
    """Test the initialization and use of an MoE Mamba model."""
    def create_test_args(self):

third_party/Megatron-LM/tests/unit_tests/data/test_bin_reader.py
def _msc_download_file(remote_path, local_path):
def _msc_resolve_storage_client(path):
def test_bin_reader():

third_party/Megatron-LM/tests/unit_tests/test_rl_utils.py
def mock_pipeline_stuff():
def test_get_logprobs():
def test_get_logprobs_with_sequence_packing():
def test_prepare_trajectories(mock_rank):
def test_prepare_trajectories_with_packing(mock_rank):
def test_grpo_loss_calculation_all_pi_eq():
def test_grpo_loss_calculation_2x_ratios():
def test_entropy_calculation():
def test_grpo_loss_truncation():
def test_prepare_data_for_update():
def test_prepare_trajectories_with_sequence_packing(mock_rank):

third_party/Megatron-LM/tests/unit_tests/test_num_microbatches_calculator.py
def test_init_num_microbatches_calculator():
def test_reconfigure_num_microbatches_calculator():
def test_get_num_microbatches():
def test_get_current_global_batch_size():
def test_get_micro_batch_size():
def test_update_num_microbatches():
def test_build_num_microbatches_calculator():
def test_ramp_up():

third_party/Megatron-LM/megatron/rl/inference/inference_interface.py
def grouper(iterable, n, fillvalue=None):
def ensure_template(value: Any) -> ConversationTemplate:
    if isinstance(value, ConversationTemplate):

third_party/Megatron-LM/examples/multimodal/config.py
def get_language_model_config(config):
def get_vision_model_config(config, apply_query_key_layer_scaling):
def get_vision_projection_config(config, hidden_size):

third_party/Megatron-LM/tests/unit_tests/data/test_multimodal_dataset.py
def test_mock_multimodal_dataset():

third_party/Megatron-LM/examples/multimodal/combine_state_dicts.py
def combine(input_files, module_prefixes, output_files):

third_party/Megatron-LM/tests/unit_tests/models/test_mimo_audio_submodules.py
def calculate_num_mel_frames(audio_length, sample_rate, window_stride, window_length=None):

third_party/Megatron-LM/tests/unit_tests/test_optimizer_param_scheduler.py
def mock_optimizer():
def test_initialization(mock_optimizer):
def test_get_wd_constant(mock_optimizer):
def test_get_wd_linear(mock_optimizer):
def test_get_wd_cosine(mock_optimizer):
def test_get_lr_linear(mock_optimizer):
def test_get_lr_cosine(mock_optimizer):
def test_step_function(mock_optimizer):
def test_state_dict(mock_optimizer):
def test_load_state_dict(mock_optimizer):

third_party/Megatron-LM/megatron/legacy/data/dataset_utils.py
def get_datasets_weights_and_num_samples(data_prefix,
                                         train_valid_test_num_samples):
def get_a_and_b_segments(sample, np_rng):
def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, np_rng):
def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
def is_start_piece(piece):
def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False,
                                 geometric_dist=False,
                                 masking_style="bert"):
def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
def build_train_valid_test_datasets_with_prefixes(train_valid_test_num_samples,
                                                  max_seq_length,
                                                  seed,
                                                  train_data_prefix=None,
                                                  valid_data_prefix=None,
                                                  test_data_prefix=None,
                                                  binary_head=False,
                                                  max_seq_length_dec=None,
                                                  dataset_type='standard_bert'):
def build_train_valid_test_datasets(data_prefix, splits_string,
                                    train_valid_test_num_samples,
                                    max_seq_length, seed,
                                    binary_head=False,
                                    max_seq_length_dec=None,
                                    dataset_type='standard_bert'):
def _build_train_valid_test_datasets(data_prefix, splits_string,
                                     train_valid_test_num_samples,
                                     max_seq_length, seed,
                                     binary_head,
                                     max_seq_length_dec,
                                     dataset_type='standard_bert'):
def build_dataset(name, data_prefix, max_num_samples,
                  max_seq_length, seed, binary_head,
                  max_seq_length_dec, dataset_type='standard_bert',
                  indexed_dataset=None):
def get_indexed_dataset_(data_prefix, dataset_type):
def get_train_valid_test_split_(splits_string, size):
def get_samples_mapping(indexed_dataset,
                        data_prefix,
                        num_epochs,
                        max_num_samples,
                        max_seq_length,
                        short_seq_prob,
                        seed,
                        name,
                        binary_head):

third_party/Megatron-LM/tests/unit_tests/ssm/test_gated_delta_net.py
def test_parallel_gated_delta_net_correctness(tmp_path_dist_ckpt, tp, sp, cp):

third_party/Megatron-LM/tests/unit_tests/models/test_clip_vit_model.py
def test_get_num_image_embeddings(vision_model, pixel_shuffle, tile_tags, expected):

third_party/Megatron-LM/tests/unit_tests/data/test_preprocess_mmdata.py
def dummy_img(odir_txt, odir_img):
def build_datasets(idir_txt, idir_img, odir, extra_args=[]):
def merge_datasets(idir):
def do_test_preprocess_mmdata(temp_dir, extra_args=[]):
def test_preprocess_mmdata():

third_party/Megatron-LM/tests/unit_tests/test_utils.py
def test_divide_properly():
def test_divide_improperly():
def test_experimental_cls_init():
def test_experimental_cls_static():
def test_experimental_cls_exception_init():
def test_experimental_cls_exception_static():
def test_global_memory_buffer():
def test_make_viewless_tensor():
def test_safely_set_viewless_tensor_data():
def test_assert_viewless_tensor():
def _init_distributed(world, rank):
def _deinit_distributed():
def test_nvtx_range(msg, suffix):
def test_nvtx_decorator():
def test_check_param_hashes_across_dp_replicas():
def test_cross_check_param_hashes_across_dp_replicas():
def test_param_norm_linear(use_distributed_optimizer: bool):
def test_param_norm_moe(use_distributed_optimizer: bool):
def test_straggler_detector():

third_party/Megatron-LM/tests/unit_tests/tensor_parallel/test_layers.py
def test_LinearWithFrozenWeight(tensor_parallel, allreduce_dgrad):

third_party/Megatron-LM/megatron/rl/inference/megatron.py
def get_static_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Get the relevant backend for running inference.
    This function will automatically choose the TRTLLMBackend when possible,
    and default to Mcore backend if the user does not specify any backends.
    TRTLLMBackend is not implmented yet.
    Args:
        args (Namespace):
def get_dynamic_inference_engine(
    args: Namespace,
    model: MegatronModule,
    inference_logging_step_interval: int = 0,
    metrics_writer = None
) -> AbstractEngine:
    """Get the relevant backend for running inference.
    This function will automatically choose the TRTLLMBackend when possible,
    and default to Mcore backend if the user does not specify any backends.
    TRTLLMBackend is not implmented yet.
    Args:
        args (Namespace):

third_party/Megatron-LM/examples/multimodal/train.py
def get_batch(data_iterator, image_token_index, img_seq_len):
def get_ltor_masks_and_position_ids(input_ids, target, pad_token):
def get_mask_start_and_end_idx(arr):
def scaled_loss_func(loss_mask, output_tensor):
def loss_func(loss_mask, output_tensor):
def forward_step(data_iterator, model: LLaVAModel):
def llava_embedding_ranks(pp_ranks):
def llava_position_embedding_ranks(pp_ranks):
def run_online_eval(model):
def write_eval_to_tensorboard(data, iteration, writer, walltime=None):
def write_online_eval_to_tensorboard(data, iteration, writer, walltime=None):

third_party/Megatron-LM/tests/unit_tests/inference/test_wandb_logging.py
def set_rounder(value):

third_party/Megatron-LM/tests/unit_tests/find_test_cases.py
def get_test_cases(yaml_file):
def get_base_path(pattern):
def is_child_of_bucket(test_case, bucket):
def expand_pattern(pattern):
def main():

third_party/Megatron-LM/examples/multimodal/dataloader_provider.py
def datasets_provider(task_encoder,worker_config=None):
def is_first_or_last_stage(pp_size):
def is_dataloader_rank():
def train_valid_test_dataloaders_provider(train_val_test_num_samples, task_encoder=None):
def cyclic_iter(iter):

third_party/Megatron-LM/tests/unit_tests/models/test_gpt_model.py
def test_get_mlp_module_spec_interface():
def test_gpt_with_te_activation_func(num_experts, gated_linear_unit):

third_party/Megatron-LM/examples/multimodal/multimodal_args.py
def add_multimodal_extra_args(parser):

third_party/Megatron-LM/tests/unit_tests/tensor_parallel/test_random.py
def test_cuda_rng_states_tracker():
def test_double_fork_cuda_rng_states_tracker(use_cudagraphable_rng):
def test_convert_cuda_rng_state():
def test_model_parallel_cuda_manual_seed():
def test_checkpoint():
def test_checkpoint_without_output():

third_party/Megatron-LM/tests/unit_tests/data/test_builder.py
def create_file_prefixes(tokenizer, number_of_files, maximum_number_of_documents, dataset_dir):
def do_setup(odir):
def test_builder():
def test_fast_builder(
    use_split,
    add_weights,
    fast_cache_load,
    sequences_per_dataset,
    defer_npy_index_mmap,
    vocab_size,
    mid_level_dataset_surplus,
    tmp_path_dist_ckpt,
    sequence_length: int = 5,
    number_of_files: int = 10,
    number_of_documents: int = 10,
):

third_party/Megatron-LM/examples/multimodal/image_processing.py
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
def find_closest_area_weighted_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False,
    find_closest_aspect_ratio_fn=find_closest_aspect_ratio):
def _build_transform(input_size, vision_model_type):

third_party/Megatron-LM/tests/unit_tests/pipeline_parallel/test_bridge_communicator.py
def _create_transformer_block(
    dtype=torch.bfloat16, hidden_size=4096, pg_collection=None
) -> TransformerBlock:
    torch.manual_seed(12345)
    model_parallel_cuda_manual_seed(
        123,
        tp_rank=(
            pg_collection.tp.rank()
            if pg_collection is not None
            else get_tensor_model_parallel_rank()
        ),
        ep_rank=torch.distributed.get_rank(),
        etp_rank=torch.distributed.get_rank(),
    )
    if pg_collection is not None:
        cp_size = pg_collection.cp.size()
    else:
        cp_size = get_context_parallel_group().size()
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=8,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=dtype == torch.bfloat16,
        context_parallel_size=cp_size,
    )
    block = (
        TransformerBlock(
            transformer_config,
            get_gpt_layer_with_transformer_engine_spec(),
            pg_collection=pg_collection,
        )
        .cuda()
        .to(dtype)
    )
    with torch.no_grad():
def _shard_and_copy_(
    ref_block: TransformerBlock, tgt_block: TransformerBlock, tp_size: int, tp_rank: int
) -> None:
    """Copy weights from *ref_block* into a tensor-parallel *tgt_block*."""
    ref_sd = ref_block.state_dict()
    tgt_sd = tgt_block.state_dict()
    for name, tgt_param in tgt_sd.items():
def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
def _get_pg_collection_from_grid(grid):
def _avg_params(module: torch.nn.Module, group: dist.ProcessGroup = None) -> None:
    world = dist.get_world_size(group=group or dist.group.WORLD)
    for p in module.parameters():
def get_transformer_block_and_grid(
    ref_block,
    tp_size=1,
    cp_size=1,
    pp_size=1,
    dp_size=1,
    grid_offset: int = 0,
    use_global_parallel_state: bool = False,
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
):

third_party/Megatron-LM/megatron/core/inference/contexts/attention_context/triton/tensor_ops.py
def _tensor_get_slice_after_kernel(
    INPUT_TENSOR,
    OUTPUT_TENSOR,
    POS_ON_DEVICE,
    INPUT_BATCH_SIZE: tl.constexpr,
    OUTPUT_BATCH_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
def _tensor_merge_kernel(
    TENSOR_A,
    TENSOR_B,
    OUTPUT_TENSOR,
    POS_ON_DEVICE,
    TENSOR_B_BATCH_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_BATCH_SIZE: tl.constexpr,
    IS_INPLACE: tl.constexpr,
):
def _tensor_masked_update_kernel_2d(
    STATES_PTR,
    IDX_PTR,
    NEW_STATES_PTR,
    stride_state_b,
    stride_state_d0,
    stride_new_b,
    stride_new_d0,
    ROW_SIZE,
    BLOCK_SIZE: tl.constexpr,
):
def _tensor_masked_update_kernel_3d(
    STATES_PTR,
    IDX_PTR,
    NEW_STATES_PTR,
    stride_state_b,
    stride_state_d0,
    stride_state_d1,
    stride_new_b,
    stride_new_d0,
    stride_new_d1,
    SIZE_D0,
    SIZE_D1,  # Dimensions of the non-batch axes
    ROW_SIZE,  # Total elements per batch item (D0 * D1)
    BLOCK_SIZE: tl.constexpr,
):
def _tensor_masked_update_kernel_4d(
    STATES_PTR,
    IDX_PTR,
    NEW_STATES_PTR,
    stride_state_b,
    stride_state_d0,
    stride_state_d1,
    stride_state_d2,
    stride_new_b,
    stride_new_d0,
    stride_new_d1,
    stride_new_d2,
    SIZE_D0,
    SIZE_D1,
    SIZE_D2,  # Dimensions (C, H, W)
    ROW_SIZE,  # Total elements (C * H * W)
    BLOCK_SIZE: tl.constexpr,
):
def _compute_row_size(tensor):
def tensor_get_slice_after(input_tensor, output_tensor, pos_on_device, check_bounds: bool = False):
def tensor_merge(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    pos_on_device: torch.Tensor,
    output_tensor: Optional[torch.Tensor] = None,
    check_bounds: bool = False,
):
def tensor_masked_update(states: torch.Tensor, idx: torch.Tensor, new_states: torch.Tensor):

third_party/Megatron-LM/tests/unit_tests/tensor_parallel/test_tensor_parallel_utils.py
def test_split_tensor_along_last_dim():
def test_split_tensor_into_1d_equal_chunks():
def test_gather_split_1d_tensor():
def test_vocab():

third_party/Megatron-LM/tests/unit_tests/data/test_gpt_dataset.py
def sample_N(dataset, N, randomize):
def test_mock_gpt_dataset():

third_party/Megatron-LM/tests/unit_tests/pipeline_parallel/test_fine_grained_activation_offloading.py
def _reset_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
def _build_gpt_model(
    *,
    seed: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    vocab_size: int,
    seq_length: int,
    num_experts: Optional[int],
    fine_grained_activation_offloading: bool,
    offload_modules: Optional[List[str]],
    min_offloaded_tensor_size: int,
    is_mla: bool,
) -> GPTModel:
    """Build a GPTModel that uses TE-based transformer layer spec."""
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    ConfigClass = MLATransformerConfig if is_mla else TransformerConfig
    transformer_config = ConfigClass(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.unfused,
        bf16=True,
        # Recompute
        recompute_modules=["layernorm", "moe_act"] if num_experts is not None else ["layernorm"],
        recompute_granularity="selective",
        # MoE
        num_moe_experts=num_experts,
        moe_grouped_gemm=(num_experts is not None),
        # Fine-grained activation offloading
        fine_grained_activation_offloading=fine_grained_activation_offloading,
        offload_modules=offload_modules,
        min_offloaded_tensor_size=min_offloaded_tensor_size,
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=num_experts is not None,
            moe_use_legacy_grouped_gemm=False,
            multi_latent_attention=is_mla,
        ),
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
    ).bfloat16()
    return gpt_model
def _make_gpt_inputs(
    *, seq_length: int, micro_batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = list(range(seq_length))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
    attention_mask = torch.ones((micro_batch_size, 1, seq_length, seq_length), dtype=bool).to(
        device
    )
    return input_ids, position_ids, attention_mask
def _run_one_iter_and_capture(
    model: GPTModel,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    enable_offload_reset: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
    """
    Run a single forward+backward iteration.
    Returns:
      - logits (CPU float32)
      - selected grads (CPU float32)
      - peak_memory_allocated (bytes) during the iteration
    """
    if enable_offload_reset:
        off_interface.reset()
    # for p in model.parameters():
def test_gpt_fine_grained_activation_offloading_correctness_and_memory(
    is_moe: bool, is_mla: bool, offload_modules: List[str]
):
def test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility(
    dispatcher_backend: str, is_mla: bool, offload_modules: List[str]
):

third_party/Megatron-LM/tests/functional_tests/python_test_utils/test_grpo_training_loop.py
def test_grpo_training_loop(golden_values_path: str, test_values_path: str) -> None:
    with open(golden_values_path, 'r') as f1, open(test_values_path, 'r') as f2:
        golden_values_content = f1.read()
        tensorboard_content = f2.read()
    output_groundtruth = json.loads(golden_values_content)
    if isinstance(output_groundtruth, str):

third_party/Megatron-LM/tests/unit_tests/test_checkpointing.py
def create_checkpoint(load_path, ckpt_format):
def create_args():
def create_ckpt_load_args(create_args):
def init_model_parallel():
def test_load_base_checkpoint(
    init_model_parallel, create_ckpt_load_args, ckpt_format, tmp_path_dist_ckpt
):
def test_save_checkpoint(init_model_parallel, create_args, tmp_path_dist_ckpt, ckpt_format):
def test_load_checkpoint(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, ckpt_format
):
def test_dist_checkpoint_versioning(init_model_parallel, tmp_path_dist_ckpt, create_ckpt_load_args):
def test_read_metadata_non_distributed(tmp_path, metadata_content, expected_iter, expected_release):

third_party/Megatron-LM/tests/unit_tests/test_muon_optimizer.py
def test_muon_optimizer_smoke():
def test_muon_optimizer_different_modes_single_rank(mode):
def test_muon_optimizer_coefficient_types(coefficient_type_and_steps):
def test_muon_optimizer_scale_modes(scale_mode):
def test_muon_optimizer_nesterov(use_nesterov):
def test_muon_optimizer_multiple_steps():
def test_muon_optimizer_qkv_split():
def test_muon_optimizer_extra_scale_factor():
def test_muon_optimizer_num_ns_steps(num_ns_steps):

third_party/Megatron-LM/tests/unit_tests/test_fp8_param.py
def enable_forward_pre_hook(model_chunks):
def disable_forward_pre_hook(model_chunks, param_sync=True):
def should_disable_forward_pre_hook(args):

third_party/Megatron-LM/examples/multimodal/nvlm/pp_checkpoint_converter.py
def split(input_dir, base_output_dir, input_pp, output_pp, num_tp, num_layers_per_pp_rank):
def combine(input_dir, base_output_dir, input_pp, output_pp, num_tp, num_layers_per_pp_rank):

third_party/Megatron-LM/tests/unit_tests/tensor_parallel/test_mappings.py
def test_CopyToModelParallelRegion():
def test_ReduceFromModelParallelRegion():
def test_ScatterToModelParallelRegion():
def test_GatherFromModelParallelRegion():
def test_ScatterToSequenceParallelRegion():
def test_GatherFromSequenceParallelRegion():
def test_ReduceScatterToSequenceParallelRegion():

third_party/Megatron-LM/tests/unit_tests/tensor_parallel/test_data.py
def test_broadcast_data():

third_party/Megatron-LM/tests/unit_tests/transformer/test_full_cuda_graph.py
def test_forward_backward_func_with_full_cuda_graph(mocker):

third_party/Megatron-LM/tests/unit_tests/pipeline_parallel/test_pipeline_layout.py
def initialize_gpt_model(
    seed,
    layer_spec_fn=gpt_te_spec,
    vocab_size=128,
    virtual_pipeline_model_parallel_size=None,
    is_moe=False,
    with_mtp=False,
    **config_kwargs,
):
def create_args():
def test_forward_vpp(create_args, tmp_path_dist_ckpt, tp_pp_vpp, pp_layout, is_moe, with_mtp):
def get_batch_iterator(seq_length, micro_batch_size, num_batches=None):

third_party/Megatron-LM/examples/multimodal/nvlm/internvit.py
def get_mlp_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )
# Override a few things that are special in InternViT and not supported by the SelfAttention class.
class InternViTSelfAttention(SelfAttention):

third_party/Megatron-LM/tests/unit_tests/tensor_parallel/test_cross_entropy.py
def test_vocab_parallel_cross_entropy():

third_party/Megatron-LM/tests/unit_tests/pipeline_parallel/test_helpers.py
def compare_helpers(pipeline_parallel_size, num_microbatches, num_model_chunks):
def test_helpers():

third_party/Megatron-LM/tests/unit_tests/pipeline_parallel/test_schedules.py
def _populate_embedding_and_position_groups(pp_group):
def test_get_forward_backward_func():
def test_deallocate_output_tensor():
def test_get_pipeline_parallel_order(
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    num_microbatches,
    microbatch_group_size_per_vp_stage,
):
def test_forward_backward_func_without_pipeline_parallel(mocker):
def test_forward_backward_func_with_pipeline_parallel(mocker):
def test_forward_backward_func_with_interleaving(mocker):
def test_forward_backward_func_with_uneven_interleaving(mocker):
def test_forward_backward_pipelining_without_interleaving_with_custom_pgs(mocker):
def test_forward_backward_pipelining_with_interleaving_with_custom_pgs(mocker):
def test_forward_backward_no_pipelining_with_custom_pgs(mocker):

third_party/Megatron-LM/tests/unit_tests/models/test_llava_model.py
def setup_and_teardown_llava_model(request):
def create_test_args(cp_size, sequence_parallel):
def count_parameters(model):
def test_get_padding(cp_size, tp_size, has_sp, seq_len, fp8_enabled, expected_padding):
def test_get_packed_seq_params(tokens, img_seq_len, padding_needed, cp_size, expected_seq_len):

third_party/Megatron-LM/tests/unit_tests/test_imports.py
def import_class_by_path(path: str):
def _build_import_path(subdomains: list, imp):
def _get_class_from_path(subdomains, imp):
def _test_domain_module_imports(module, subdomains: list):
def test_domain_mcore():

third_party/Megatron-LM/tests/functional_tests/python_test_utils/common.py
def read_tb_logs_as_list(
    path, index: int = 0, train_iters: int = 50, start_idx: int = 1, step_size: int = 5
) -> Optional[Dict[str, GoldenValueMetric]]:
    """Reads a TensorBoard Events file from the input path, and returns the
    summary specified as input as a list.
    Args:
        path: str, path to the dir where the events file is located.
        summary_name: str, name of the summary to read from the TB logs.
    Returns:
        summary_list: list, the values in the read summary list, formatted as a list.
    """
    files = glob.glob(f"{path}/events*tfevents*")
    files += glob.glob(f"{path}/results/events*tfevents*")
    if not files:
        logger.error(f"File not found matching: {path}/events* || {path}/results/events*")
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, pathlib.Path(x).name)))
    accumulators = []
    if index == -1:
        for event_file in files:
            ea = event_accumulator.EventAccumulator(event_file, size_guidance=SIZE_GUIDANCE)
            ea.Reload()
            accumulators.append(ea)
    else:
        event_file = files[index]
        ea = event_accumulator.EventAccumulator(event_file, size_guidance=SIZE_GUIDANCE)
        ea.Reload()
        accumulators.append(ea)
    summaries = {}
    for ea in accumulators:
        for scalar_name in ea.Tags()["scalars"]:
            if scalar_name in summaries:
                for x in ea.Scalars(scalar_name):
def read_golden_values_from_json(
    golden_values_path: Union[str, pathlib.Path]
) -> Dict[str, GoldenValueMetric]:
    with open(golden_values_path) as f:
        if os.path.exists(golden_values_path):
def _filter_checks(
    checks: List[Union[ApproximateTest, DeterministicTest]], filter_for_type_of_check
):
def pipeline(
    compare_approximate_results: bool,
    golden_values: Dict[str, GoldenValueMetric],
    actual_values: Dict[str, GoldenValueMetric],
    checks: Dict[str, List[Union[ApproximateTest, DeterministicTest]]],
):

third_party/Megatron-LM/tests/unit_tests/transformer/moe/test_upcycling.py
def _find_submodule(model, submodule_name):
def model_provider(
    pre_process=True,
    post_process=True,
    layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
    **config_kwargs,
):
def create_test_args(tp, grouped_gemm, swiglu, squared_relu, use_te):
def set_upcycling_args(ep, granularity, num_experts=8):
def set_bias_value(dense_model):
def get_batch(data_iterator):

third_party/Megatron-LM/tests/unit_tests/test_optimizer_cpu_offloading.py
def setup_seed(seed):
def test_multi_device_hybrid_optimizer(
    with_param_groups, optimizer, offload_fraction, overlap_cpu_optimizer_d2h_h2d, n_steps
):

third_party/Megatron-LM/tests/unit_tests/models/test_gpt_model_batch_invariant.py
def _configure_flash_attention_env():
def _build_flash_attn_bik_model(seq_len: int, vocab_size: int, hidden_size: int = 128) -> GPTModel:
    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        normalization="RMSNorm",
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
    )
    cfg.fp16 = False
    cfg.bf16 = True
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
    )
    return model.cuda().eval()
def _train_forward_logprobs(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = tokens.shape
    position_ids = (
        torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
    )
    attention_mask = torch.ones(
        batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=tokens.device
    )
    with torch.no_grad():

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/test_replication.py
def equal_(a, b):
def test_all_gather_batch(tp, pp):

third_party/Megatron-LM/megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/common.py
def send_do_generate():

third_party/Megatron-LM/tests/unit_tests/post_training/test_modelopt_module_spec.py
def model_forward(model: torch.nn.Module, config: TransformerConfig, micro_batch_size: int = 2):
def test_get_gpt_modelopt_spec_interface():
def test_get_mamba_stack_modelopt_spec_interface():

third_party/Megatron-LM/tests/unit_tests/test_optimizer.py
def test_get_param_groups_no_overrides(mock_get_world_size):
def test_get_param_groups_default_overrides(mock_get_world_size):
def test_get_param_groups_with_overrides(mock_get_world_size):
def test_get_param_groups_multiple_matches(mock_get_world_size):
def test_get_param_groups_overlapping_matches(mock_get_world_size):
def test_get_param_groups_with_standard_config_overrides(apply_wd_to_qk_layernorm: bool):
def test_get_param_groups_appling_wd_to_qk_layernorm(apply_wd_to_qk_layernorm: bool):
def test_chained_optimizer():
def test_precision_aware_fused_adam():
def test_precision_aware_optimizer(
    precision: str,
    main_params_dtype: torch.dtype,
    main_grads_dtype: torch.dtype,
    moment_dtype: torch.dtype,
):
def test_optim_sharded_state_dict(use_distributed_optimizer: bool, precision: str):
def test_optimizer_reload_model_params():
def test_get_megatron_optimizer_with_custom_process_groups(world_size, tp_size, cp_size, dp_size):
def test_get_megatron_optimizer_custom_process_groups_validation():

third_party/Megatron-LM/tests/unit_tests/models/test_heterogeneous_gpt_model.py
def heterogeneous_gpt_model(request, tmp_path):

third_party/Megatron-LM/tests/unit_tests/transformer/moe/test_token_dispatcher.py
def token_permutation(token_dispatcher, hidden_states, probs, indices):
def token_unpermutation(token_dispatcher, hidden_states):
def is_deep_ep_available():
def is_hybrid_ep_available():

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/test_layer_wise_optimizer.py
def check_equal(input_1, input_2):
def initialize_real_model(
    seed,
    pre_process,
    post_process,
    vp_stage=None,
    is_moe=False,
    is_mla=False,
    virtual_pipeline_model_parallel_size=None,
    **config_kwargs,
):
def load_checkpoint_no_arg_checks(*args, **kwargs):

third_party/Megatron-LM/megatron/core/inference/text_generation_server/dynamic_text_gen_server/flask_server.py
def temp_log_level(level, logger=None):

third_party/Megatron-LM/tests/unit_tests/post_training/test_modelopt_model_builder.py
def _sentinel_builder(return_value, calls):
def test_model_provider_switches_to_modelopt_builder(monkeypatch):

third_party/Megatron-LM/tests/unit_tests/transformer/test_cuda_graphs.py
def test_cuda_graph_determine_first_last_layer_logic(
    total_num_layers,
    pp,
    vpp,
    account_for_embedding_in_pipeline_split,
    account_for_loss_in_pipeline_split,
    num_layers_in_first_pipeline_stage,
    num_layers_in_last_pipeline_stage,
    pp_layout,
    first_layer_numbers_golden,
    last_layer_numbers_golden,
):
def is_deep_ep_available():
def is_hybrid_ep_available():

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/conftest.py
def pytest_sessionfinish(session, exitstatus):
def tmp_dir_per_class(tmp_path_factory):
def set_default_dist_ckpt_strategy():

third_party/Megatron-LM/tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py
def test_placeholder():

third_party/Megatron-LM/tests/functional_tests/test_cases/common/moe_perf/__main__.py
def _build_transformer_config(case: MoEPerformanceCase) -> TransformerConfig:
    model = case.model
    config_kwargs = dict(
        num_layers=1,
        hidden_size=model.hidden_size,
        moe_ffn_hidden_size=model.moe_ffn_hidden_size,
        num_attention_heads=model.num_attention_heads,
        # MoE Arguments
        num_moe_experts=model.num_experts,
        moe_router_topk=model.router_topk,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1.0,
        moe_token_dispatcher_type=case.token_dispatcher,
        moe_flex_dispatcher_backend=case.moe_flex_dispatcher_backend,
        use_cpu_initialization=True,
        add_bias_linear=False,
        # Router Arguments
        moe_router_num_groups=model.moe_router_num_groups,
        moe_router_group_topk=model.moe_router_group_topk,
        moe_router_score_function=model.moe_router_score_function,
        moe_router_dtype=model.moe_router_dtype,
        moe_router_enable_expert_bias=model.moe_router_enable_expert_bias,
        # Parallelism Arguments
        sequence_parallel=case.tensor_model_parallel_size > 1,
        tensor_model_parallel_size=case.tensor_model_parallel_size,
        pipeline_model_parallel_size=case.pipeline_model_parallel_size,
        expert_model_parallel_size=case.expert_model_parallel_size,
        expert_tensor_parallel_size=case.expert_tensor_parallel_size,
        context_parallel_size=case.context_parallel_size,
        params_dtype=case.input_dtype,
        bf16=True,
        fp8=case.fp8,
        moe_permute_fusion=case.moe_permute_fusion,
        moe_router_fusion=case.moe_router_fusion,
        moe_router_force_load_balancing=case.moe_router_force_load_balancing,
    )
    if case.fp8:
        config_kwargs.update(
            dict(fp8="hybrid", fp8_margin=0, fp8_interval=1, fp8_recipe="blockwise")
        )
    return TransformerConfig(**config_kwargs)
# NOTE: Only TE backend is covered in this test.
def _resolve_moe_submodules(case: MoEPerformanceCase):
def _load_baselines() -> Dict[str, Dict[str, float]]:
    if not BASELINES_PATH.exists():
def _persist_baselines(data: Dict[str, Dict[str, float]]) -> None:
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BASELINES_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
def _serialize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    forward_ms = metrics["forward_ms"]
    backward_ms = metrics["backward_ms"]
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "max_allocated_bytes": metrics["max_allocated_bytes"],
        "max_regression_ratio": DEFAULT_MAX_REGRESSION_RATIO,
    }
def _assert_within_baseline(
    case_name: str, metrics: Mapping[str, Any], baselines: Dict[str, Dict[str, float]]
):
def _benchmark_moe_layer(layer: MoELayer, case: MoEPerformanceCase):
def _maybe_update_baseline(
    case: MoEPerformanceCase, metrics: Dict[str, float], baselines: Dict[str, Dict[str, float]]
):
def _prepare_moe_layer(case: MoEPerformanceCase) -> MoELayer:
    config = _build_transformer_config(case)
    submodules = _resolve_moe_submodules(case)
    layer = MoELayer(config=config, submodules=submodules).cuda().to(dtype=torch.bfloat16)
    layer.train()
    return layer
def _check_env():
def _check_dependencies(case: MoEPerformanceCase):
def test_moe_layer_performance(perf_case: MoEPerformanceCase, debug_mode: bool = False):

third_party/Megatron-LM/megatron/core/inference/text_generation_server/dynamic_text_gen_server/tokenization.py
def tokenize_prompts(
    tokenizer, prompts=None, tokens_to_generate=None, add_BOS=None, rank=0, data_parallel=False
):
def _tokenize_prompts_and_batch(tokenizer, prompts, tokens_to_generate, add_BOS):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/models/test_gpt_model.py
def initialize_gpt_model(seed, layer_spec_fn=gpt_te_spec, vocab_size=128, **config_kwargs):

third_party/Megatron-LM/tests/unit_tests/distributed/test_torch_fully_sharded_parallel.py
def init_model_parallel():
def test_fsdp2_constructor(init_model_parallel):
def test_fsdp2_constructor_with_process_group(init_model_parallel):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/models/test_mamba.py
def initialize_mamba(seed, glu=True, **config_kwargs):

third_party/Megatron-LM/tests/unit_tests/export/trtllm/test_distributed_fp8.py
def _model_provider():
def _get_train_data_iterator():
def _forward_step_func(data_iterator, model):

third_party/Megatron-LM/megatron/core/inference/text_generation_server/run_mcore_engine.py
def run_mcore_engine(
    engine,
    prompts=None,
    temperature=1.0,
    top_k=0,
    top_p=0.0,
    logprobs=True,
    tokens_to_generate=0,
    top_n_logprobs=0,
    random_seed=-1,
):

third_party/Megatron-LM/tests/unit_tests/models/test_mimo_model.py
def get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim):
def get_audio_submodules_spec(hidden_size):
def get_language_model_spec(hidden_size, vocab_size, seq_len):
def get_avlm_mimo_model(
    hidden_size, vocab_size, seq_len, img_h, img_w, patch_dim, special_token_ids
):
def get_vlm_mimo_model(
    hidden_size, vocab_size, seq_len, img_h, img_w, patch_dim, special_token_ids
):

third_party/Megatron-LM/tests/functional_tests/python_test_utils/test_optimizer_grads_match.py
def _as_iter(x: TensorLike):
def _fro_norm(x: TensorLike) -> torch.Tensor:
    """Frobenius norm; supports sharded tensors (sum of shard ||·||_F^2)."""
    it = _as_iter(x)
    s = torch.tensor(0.0, device=next(iter(it)).device if it else "cpu")
    for t in it:
        s = s + t.float().pow(2).sum()
    return torch.sqrt(s)
def machine_epsilon_for_dtype(dtype: torch.dtype) -> float:
    """Return machine epsilon for dtype. For FP8, use BF16 epsilon per paper."""
    # Standard types
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
def relative_grad_diff(g_hat: TensorLike, g_ref: TensorLike, eps_den: float = 1e-30) -> float:
    """
    Relative difference ||g_hat - g_ref||_F / ||g_ref||_F.
    Accepts a single tensor or an iterable of shards for each argument.
    """
    # If sharded, assume shards align 1:1; otherwise pass the merged tensors.
    gh_iter, gr_iter = _as_iter(g_hat), _as_iter(g_ref)
    if len(list(gh_iter)) != len(list(gr_iter)):
def expected_rel_bound(
    l: int,
    *,
    L: int = 32,
    C: float = 1.03,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    k: float = 4.0,
) -> float:
    """
    Bound ~ k * (C ** (L + 1 - l)) * eps_mch, with 1-based layer index l.
    - L is hard-coded default to 32 per your request.
    - C is 'close to 1'; 1.01–1.05 are reasonable defaults.
    - k absorbs the hidden constant in big-O; 2–8 are common choices.
    - dtype controls eps_mch; for FP8 use BF16 epsilon (see https://www.arxiv.org/pdf/2506.09280 theorem 5.3).
    """
    eps_mch = machine_epsilon_for_dtype(dtype or torch.bfloat16)
    depth = L + 1 - l  # 1-based depth from the top (as in the theorem)
    depth = max(depth, 0)
    return float(k * (C**depth) * eps_mch)
def check_gradient(
    g_hat: TensorLike,
    g_ref: TensorLike,
    l: int,
    *,
    L: int = 32,
    C: float = 1.03,
    dtype: Optional[torch.dtype] = None,
    k: float = 4.0,
) -> Tuple[float, float, bool]:
    """
    Compute (rel_error, bound, ok) for layer l.
    - If dtype is None, infer from g_ref (or g_hat if needed).
    # See https://www.arxiv.org/pdf/2506.09280 theorem 5.3
    """
    # Infer dtype if not provided
    if dtype is None:
        t0 = next(iter(_as_iter(g_ref)))
        dtype = t0.dtype
    rel = relative_grad_diff(g_hat, g_ref)
    bnd = expected_rel_bound(l, L=L, C=C, dtype=dtype, k=k)
    return rel, bnd, (rel <= bnd)
def _filter_optimizer_tensors(plain_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return only optimizer-related tensors from a flat checkpoint tensor dict."""
    return {
        k: v for k, v in plain_tensors.items() if k.startswith("optimizer.") and ".exp_avg." in k
    }
def assert_grads_close(left: torch.Tensor, right: torch.Tensor):
def unshard_row_parallel_state(saved_state, out_features, in_features, tp):
def _assert_optimizer_tensors_equal(
    left: Dict[str, torch.Tensor],
    right: Dict[str, torch.Tensor],
    left_empty: Dict[str, torch.Tensor],
    right_empty: Dict[str, torch.Tensor],
    eps=1e-4,
):
def load_dist_checkpoint_pt(
    ckpt_dir,
    metadata_ckpt_dir=None,
    pattern=r"optimizer",
    device="cpu",
    return_full_empty: bool = False,
):
def test_optimizer_states_match(checkpoint_dirs):
def main():

third_party/Megatron-LM/tests/unit_tests/distributed/test_mcore_fully_sharded_data_parallel.py
def setup_seed(seed):

third_party/Megatron-LM/tests/unit_tests/transformer/moe/test_routers.py
def test_router_gating_linear(router_dtype):
def test_router_gating_linear_bias(router_dtype):

third_party/Megatron-LM/tests/unit_tests/transformer/test_attention_variant_dsa.py
def mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mock implementation of hadamard_transform for testing without the library installed.
    This is a simple identity-like transformation that preserves shape and applies scaling.
    """
    return x * scale
@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():

third_party/Megatron-LM/tests/functional_tests/python_test_utils/get_test_results_from_tensorboard_logs.py
def collect_train_test_metrics(
    logs_dir: str,
    train_iters: str,
    output_path: str,
    is_convergence_test: bool,
    is_second_run: bool,
    step_size: int,
):

third_party/Megatron-LM/megatron/core/inference/text_generation_server/endpoints/common.py
def send_do_generate():

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/test_optimizer.py
def get_param_state_dp_zero(optimizer):
def initialize_pp_agnostic_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
def initialize_pp_agnostic_gpt_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
def initialize_small_model(pre_process=True, post_process=True, seed=0, **config_kwargs):
def initialize_1d_flatten_tensor_model(
    pre_process=True, post_process=True, seed=0, **config_kwargs
):
def initialize_real_model(
    seed,
    pre_process,
    post_process,
    vp_stage=None,
    is_moe=False,
    is_mla=False,
    virtual_pipeline_model_parallel_size=None,
    **config_kwargs,
):
def load_checkpoint_no_arg_checks(*args, **kwargs):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/models/common.py
def common_test_simple_sharded_state_dict_save_load(
    initialize_model_fn, tmp_path_dist_ckpt, src_layer_spec_fn, dst_layer_spec_fn
):
def common_test_parallel_reconfiguration_e2e(
    initialize_model_fn,
    tmp_path_dist_ckpt,
    src_tp_pp,
    dest_tp_pp,
    src_layer_spec_fn,
    dst_layer_spec_fn,
    use_fpsl,
    load_order="tp-dp-pp",
    store_order="tp-dp-pp",
    src_tp_pp_kwargs=None,
    dst_tp_pp_kwargs=None,
    src_model_init_kwargs=None,
    dst_model_init_kwargs=None,
    metadata=None,
):
def common_test_state_dict_comparison(initialize_model_fn, tmp_path_dist_ckpt):
def common_test_vocab_size_padding_change(
    initialize_model_fn, tmp_path_dist_ckpt, vocab_size_base, src_tp_pp, dest_tp_pp
):

third_party/Megatron-LM/tests/unit_tests/transformer/test_attention_packed_seq.py
def make_test_packed_seq_params(sequence_length):
def make_test_packed_padded_seq_params(sequence_length):

third_party/Megatron-LM/tests/functional_tests/test_cases/common/ckpt_converter/__main__.py
def is_model_parallel_rank_0():
def broadcast(item):
def get_gpt_pipelines():
def get_moe_pipelines():
def get_llava_pipelines():
def test_all_pipelines():

third_party/Megatron-LM/tests/functional_tests/python_test_utils/test_pretraining_regular_pipeline.py
def test_regular_pipeline(
    compare_approximate_results: bool,
    golden_values: Dict[str, common.GoldenValueMetric],
    actual_values: Dict[str, common.GoldenValueMetric],
    model_config_path: str,
    checks: Optional[Dict[str, List[common.Test]]] = None,
):

third_party/Megatron-LM/tests/unit_tests/export/trtllm/test_single_device_fp8.py
def _model_provider():
def _get_train_data_iterator():
def _forward_step_func(data_iterator, model):

third_party/Megatron-LM/megatron/core/inference/text_generation_server/endpoints/completions.py
def detokenize(prompt, tok) -> list[str]:
    """Detokenizes the given prompt."""
    if isinstance(prompt, str):

third_party/Megatron-LM/tests/unit_tests/fusions/test_bias_dropout_fusion.py
def test_bias_dropout_add(dtype, training):

third_party/Megatron-LM/tests/unit_tests/distributed/test_param_and_grad_buffer.py
def get_model_and_buffers(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    bias: bool,
    shared_embedding: bool,
    bucket_size: int,
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    num_distributed_optimizer_instances: int = 1,
):
def test_bucket_sizes(
    bucket_size: Optional[int], use_distributed_optimizer: bool, bias: bool, shared_embedding: bool
):
def test_grad_sync(
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    num_distributed_optimizer_instances: int,
):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/models/test_bert_model.py
def initialize_bert_model(
    seed, layer_spec=bert_layer_with_transformer_engine_spec, vocab_size=128, **config_kwargs
):

third_party/Megatron-LM/tests/functional_tests/python_test_utils/test_inference_regular_pipeline.py
def _median_as_float(value):
def _bytes_to_gib(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0**3)
def test_inference_pipeline(golden_values_path: str, test_values_path: str) -> None:
    with open(golden_values_path, 'r') as f1, open(test_values_path, 'r') as f2:
        golden_values_content = f1.read()
        tensorboard_content = f2.read()
    output_groundtruth = json.loads(golden_values_content)
    if isinstance(output_groundtruth, str):

third_party/Megatron-LM/tests/unit_tests/distributed/test_reduce_scatter_with_fp32_accumulation.py
def get_non_matching_values(tensor1_shard, tensor2_shard):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/test_pipeline_parallel_layout.py
def initialize_gpt_model(
    seed,
    layer_spec_fn=gpt_te_spec,
    vocab_size=128,
    virtual_pipeline_model_parallel_size=None,
    is_moe=False,
    **config_kwargs,
):
def test_save_and_load_checkpoint_pp(tmp_path_dist_ckpt):
def create_args():
def test_save_and_load_checkpoint_vpp(
    create_args,
    tmp_path_dist_ckpt,
    src_tp_pp_vpp,
    src_pp_layout,
    dst_tp_pp_vpp,
    dst_pp_layout,
    is_moe,
):

third_party/Megatron-LM/tests/unit_tests/fusions/test_weighted_squared_relu_fusion.py
def test_weighted_squared_relu_fusion(input_dtype):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/models/test_t5_model.py
def initialize_t5_model(seed, encoder_decoder_spec_fn, num_layers=8, **config_kwargs):

third_party/Megatron-LM/megatron/core/inference/text_generation_server/tokenization.py
def tokenize_prompts(
    tokenizer, prompts=None, tokens_to_generate=None, add_BOS=None, rank=0, data_parallel=False
):
def _tokenize_prompts_and_batch(tokenizer, prompts, tokens_to_generate, add_BOS):

third_party/Megatron-LM/tests/unit_tests/distributed/test_grad_reduce_for_replicated_embedder.py
def test_allreduce_conditional_embedding_grads():

third_party/Megatron-LM/tests/unit_tests/transformer/moe/conftest.py
def pytest_sessionfinish(session, exitstatus):
def cleanup():
def set_env():
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.
    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """
    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'
    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):

third_party/Megatron-LM/tests/functional_tests/python_test_utils/test_pretraining_resume_checkpoint_pipeline.py
def test_resume_checkpoint_pipeline(
    compare_approximate_results: bool,
    actual_values_first_run: Dict[str, common.GoldenValueMetric],
    actual_values_second_run: Dict[str, common.GoldenValueMetric],
    train_iters: int,
    model_config_path: str,
):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/models/test_moe_experts.py
def initialize_expert_layer(seed, glu=True, expert_type='sequential', fp8=False, **config_kwargs):

third_party/Megatron-LM/tests/unit_tests/distributed/fsdp/test_mfsdp_fully_shard.py
def destroy_device_mesh(device_mesh):
def build_toy_model(model_type: str, init_model_with_meta_device: bool, seed=None):
def build_distributed_environment(mesh_dim_config: tuple):

third_party/Megatron-LM/tests/unit_tests/fusions/test_mla_yarn_rope_apply.py
def dtype_tols(dtype):
def _test_fused_apply_mla_rope_for_q(input_format):
def _test_fused_apply_mla_rope_for_kv(input_format):

third_party/Megatron-LM/tests/unit_tests/transformer/test_transformer_block_custom_pgs.py
def create_reference_mlp(hidden_size, ffn_hidden_size, seed=12345):
def copy_weights_to_tp_mlp(ref_mlp, tp_mlp, tp_group):
def _gpt_te_layer_spec_with_hetro_pgs(attn_pg_collection, mlp_pg_collection):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py
def initialize_mlp(glu=True):

third_party/Megatron-LM/tests/unit_tests/fusions/test_swiglu_fusion.py
def test_weighted_bias_swiglu(input_dtype):

third_party/Megatron-LM/tests/unit_tests/test_inference.py
def gpt2_tiktoken_tokenizer(gpt2_tiktok_vocab):
def static_inference_engine(gpt2_tiktoken_tokenizer):
def app(static_inference_engine):
def client(app):
def test_generations_endpoint(mock_send_do_generate, client, gpt2_tiktoken_tokenizer):
def test_completions_endpoint(mock_send_do_generate, client, gpt2_tiktoken_tokenizer):

third_party/Megatron-LM/tests/functional_tests/python_test_utils/conftest.py
def pytest_addoption(parser):
def compare_approximate_results(request) -> bool:
    """Simple fixture returning whether to check against results approximately."""
    return request.config.getoption("--allow-nondeterministic-algo") is True
@pytest.fixture
def golden_values_path(request):
def golden_values(request):
def actual_values(request):
def actual_values_first_run(request):
def actual_values_second_run(request):
def scope(request):
def train_iters(request):
def tensorboard_logs(request, train_iters):
def test_values_path(request):
def tensorboard_path(request):
def model_config_path(request):

third_party/Megatron-LM/megatron/core/inference/engines/dynamic_engine.py
def format_mem_bytes(mem_bytes):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/__init__.py
def empty_dir(path: Path):

third_party/Megatron-LM/tests/unit_tests/transformer/test_transformer_layer.py
def get_tensor_shapes_for_tp(transformer_config, tp_size):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/test_async_save.py
def write_data_os_err_mock_fn(
    transform_list, local_proc_idx, write_bucket, results_queue, count_queue, use_fsync, **kwargs
):

third_party/Megatron-LM/tests/unit_tests/distributed/test_grad_sync_with_expert_parallel.py
def get_moe_model_and_buffers(
    num_layers: int,
    hidden_size: int,
    num_moe_experts: int,
    moe_grouped_gemm: bool,
    ep_size: int,
    bucket_size: Optional[int],
    etp_size: int,
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    num_distributed_optimizer_instances: int,
):
def test_grad_sync(
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
    average_in_collective: bool,
    ep_size: int,
    etp_size: int,
    num_distributed_optimizer_instances: int,
):

third_party/Megatron-LM/tests/unit_tests/transformer/test_submodule_callables.py
def run_model_ref_with_capture(model, input_tensors, iterations):
def run_model_submodules_with_capture(model, input_tensors, microbatches):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/test_mapping.py
def test_is_main_replica():

third_party/Megatron-LM/tests/unit_tests/test_training.py
def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
def create_test_args():

third_party/Megatron-LM/tests/unit_tests/transformer/test_core_attention.py
def core_attention(transformer_config):

third_party/Megatron-LM/tests/unit_tests/extension/test_kitchen_sdpa.py
def get_attention_implementation(
    impl: Literal["megatron", "te-fa", "te-unfused", "kitchen", "kitchen-fa"],
    config: TransformerConfig,
    layer_number: int,
    attn_mask_type: AttnMaskType,
    attention_type: str,
    attention_dropout: float,
    softmax_scale: float,
    cp_comm_type: str = "a2a",
) -> MegatronModule:
    if impl == "megatron":
        return DotProductAttention(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type,
            pg_collection,
        )
    elif impl == "te-fa" or impl == "te-unfused":
        if attention_type == "self_attention":
            attention_type = "self"
        return TEDotProductAttention(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
    elif impl == "kitchen":
        attn = KitchenDotProductAttention(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type,
            pg_collection,
        )
        attn.finish_init(
            get_quant_config_or_none("self_attention.core_attention", config.quant_recipe)
        )
        return attn
    elif impl == "kitchen-fa":
        if attention_type == "self_attention":
            attention_type = "self"
        attn = KitchenFlashAttention(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type,
            pg_collection,
        )
        attn.finish_init(
            get_quant_config_or_none("self_attention.core_attention", config.quant_recipe)
        )
        return attn
    else:
        raise ValueError(f"Invalid implementation: {impl}")
class DotProductAttentionModel(torch.nn.Module):

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/utils.py
def initialize_gpt_model(
    pre_process=True, post_process=True, seed=0, use_glu=True, **config_kwargs
):
def initialize_moe_model(
    pre_process=True,
    post_process=True,
    seed=0,
    use_glu=True,
    use_sp=False,
    use_te=False,
    use_grouped_mlp=False,
    **config_kwargs,
):
def init_basic_mock_args(args, tp, pp, bf16=True):
def init_checkpointing_mock_args(args, ckpt_dir, fully_parallel=False):
def setup_model_and_optimizer(
    seed, tp, pp, initialize_fn=initialize_gpt_model, bf16=True, dist_opt=True, optimizer='adam'
):
def find_matching_values(
    x: Union[dict, list], predicate: Callable[[Any], bool]
) -> Tuple[Union[dict, list], Union[dict, list]]:
    """Return matching values in a single list
    Args:
        x (Union[dict, list]) :
def setup_moe_model_and_optimizer(
    seed,
    tp,
    pp,
    ep,
    initialize_fn=initialize_moe_model,
    bf16=True,
    dist_opt=True,
    use_te=False,
    use_grouped_mlp=False,
    use_glu=False,
    optimizer='adam',
):

third_party/Megatron-LM/megatron/core/jit.py
def noop_decorator(func):
def enable_jit_fuser():
def disable_jit_fuser():

third_party/Megatron-LM/tests/unit_tests/dist_checkpointing/test_fp8.py
def to_float8(tensor: torch.Tensor) -> Float8Tensor:
    """Convert a tensor to FP8 format."""
    try:
        return Float8Tensor.to_float8(tensor)
    except Exception as e:
        # Handle the case where the method fails (due to API changes in TransformerEngine)
        # https://github.com/NVIDIA/TransformerEngine/commit/544dd14b4301beb47136f273deff3f532cdde181
        import transformer_engine_torch as tex
        from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
        fp8_dtype = tex.DType.kFloat8E4M3
        scale = 1.0
        # Create a quantizer for FP8 conversion
        quantizer = Float8Quantizer(
            scale=torch.full([1], scale, dtype=torch.float32, device="cuda"),
            amax=torch.empty([1], dtype=torch.float32, device="cuda"),
            fp8_dtype=fp8_dtype,
        )
        # Return the quantized tensor
        return quantizer(tensor.cuda())
class TestFP8:
    @pytest.mark.parametrize('dtype', ['bf16', 'fp16', 'fp8'])
    @pytest.mark.parametrize('src_rank', [0, 6])
    def test_simple_broadcast(self, dtype, src_rank):

third_party/Megatron-LM/megatron/core/inference/utils.py
def get_attention_mask(seq_length: int) -> torch.Tensor:
    """Constructs an attention mask given the input sequence length."""
    attention_mask = torch.tril(
        torch.ones((1, seq_length, seq_length), device=torch.cuda.current_device())
    ).view(1, 1, seq_length, seq_length)
    # Convert to boolean
    attention_mask = attention_mask < 0.5
    return attention_mask
# Initialize cache for sequence parallel modules
moe_layer_cache = None
def _init_moe_expert_cache(model):
def set_decode_expert_padding(model, set_to: bool = False, capacity_factor: int = None):
def tensor_swap(x, src_idxs, dst_idxs):

third_party/Megatron-LM/megatron/core/tensor_parallel/inference_layers.py
def _te_rms_norm_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float):

third_party/Megatron-LM/tests/unit_tests/transformer/test_te_layers_batch_invariant.py
def _split_concat_equal(layer, x_full, dim=0, forward_kwargs=None, out_dim_concat=0):
def _split_many_concat_equal(layer, x_full, splits, dim=0, forward_kwargs=None, out_dim_concat=0):
def _random_splits(total, num_parts):
def test_te_column_parallel_linear_batch_invariant_randomized():
def test_te_row_parallel_linear_batch_invariant_randomized():
def test_te_layernorm_column_parallel_linear_batch_invariant_randomized():
def test_te_norm_batch_invariant_randomized():
def test_column_parallel_linear_batch_invariant_randomized():
def test_te_attention_layer_batch_invariant_randomized():
def test_te_column_parallel_linear_parity():
def test_te_rmsnorm_parity():
def test_te_layernorm_linear_parity():
def _tols(dtype: torch.dtype):
def _device(dtype=torch.float16):
def _te_general_gemm(*args, **kwargs):
def test_bik_te_general_gemm_chunking_deterministic(dtype):
def test_bik_te_general_gemm_numerical_parity(dtype):

third_party/Megatron-LM/tests/unit_tests/transformer/test_quantization_config.py
def test_recipe_config_matching() -> None:
    recipe_config = RecipeConfig(
        [
            GlobMatcher("*fc2", "fc2_cfg"),
            GlobMatcher("*fc*", "fc_cfg"),
            GlobMatcher("*", "default"),
        ],
        {"fc2_cfg": {"fc2": "foo"}, "fc_cfg": {"fc1": "bar"}, "default": {"default": "baz"}},
    )
    assert (
        recipe_config.match_to_config_key(MatchContext("decoder.1.linear_fc2", layer_number=1))
        == "fc2_cfg"
    )
    assert (
        recipe_config.match_to_config_key(MatchContext("decoder.1.linear_fc1", layer_number=1))
        == "fc_cfg"
    )
    assert (
        recipe_config.match_to_config_key(MatchContext("decoder.1.linear_qkv", layer_number=1))
        == "default"
    )
@pytest.mark.skipif(not HAVE_KITCHEN, reason="Kitchen required for using kitchen backend.")
def test_parse_qlinear_params_example() -> None:
    qat_params = 2
    config = {"kitchen_config_type": "QLinearParams", "recipe_idx": qat_params}
    qlinear_params_actual = QLinearParamsConfigSchema.parse_config_dict(config).to_kitchen_qlinear()
    qlinear_params_expected = get_qlinear_params_from_predefined(QuantizeRecipe.FP8_CS)
    assert qlinear_params_actual.x_params == qlinear_params_expected.x_params
    assert qlinear_params_actual.w_params == qlinear_params_expected.w_params
    assert qlinear_params_actual.g_params == qlinear_params_expected.g_params
    assert qlinear_params_actual.mm_fprop == qlinear_params_expected.mm_fprop
    assert qlinear_params_actual.mm_dgrad == qlinear_params_expected.mm_dgrad
    assert qlinear_params_actual.mm_wgrad == qlinear_params_expected.mm_wgrad
    assert (
        qlinear_params_actual.autograd_function_implementation
        == AutogradFunctionImplementation.QUANTIZED
    )
    qat_params = 6001
    config = {"kitchen_config_type": "QAttentionParams", "recipe_idx": qat_params}
    qattention_params_actual = QAttentionParamsConfigSchema.parse_config_dict(
        config
    ).to_kitchen_qattention()
    qattention_params_expected = get_qattention_params_from_predefined(
        QuantizeRecipeAttnBMM.MXFP8_EMULATION
    )
    assert type(qattention_params_actual.quantizer_bmm1) == type(
        qattention_params_expected.quantizer_bmm1
    )
    assert type(qattention_params_actual.quantizer_bmm2) == type(
        qattention_params_expected.quantizer_bmm2
    )
    assert type(qattention_params_actual.get_quantizer(True)) == type(
        qattention_params_expected.get_quantizer(True)
    )
    assert type(qattention_params_actual.get_quantizer(False)) == type(
        qattention_params_expected.get_quantizer(False)
    )
@pytest.mark.skipif(not HAVE_KITCHEN, reason="Kitchen required for using kitchen backend.")
def test_error_from_malformed() -> None:
    qat_params = 2
    config: Dict[Any, Any] = {"recipe_idx": qat_params}
    with pytest.raises(KeyError, match="Missing required keys"):
def test_parse_qflash_attention_params_example() -> None:
    recipe_name = "triton_fa_bf16_for_all_base_2"
    config = {"kitchen_config_type": "QFlashAttentionParams", "recipe_name": recipe_name}
    qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config).to_kitchen_qfa()
    qfa_params_expected = get_qfa_params_from_recipe_name(recipe_name)
    # Verify they are the same object (since recipes are cached)
    assert qfa_params_actual is qfa_params_expected
    assert qfa_params_actual.backend == "triton"
    assert qfa_params_actual.qk_dot_precisions == "bf16@bf16"
    assert qfa_params_actual.pv_dot_precisions == "bf16@bf16"
    assert qfa_params_actual.use_natural_transcendental_func is False
    # Test with natural recipe
    recipe_name = "triton_fa_bf16_for_all_natural"
    config = {"kitchen_config_type": "QFlashAttentionParams", "recipe_name": recipe_name}
    qfa_params_actual = QFlashAttentionParamsConfigSchema.parse_config_dict(config).to_kitchen_qfa()
    qfa_params_expected = get_qfa_params_from_recipe_name(recipe_name)
    assert qfa_params_actual is qfa_params_expected
    assert qfa_params_actual.backend == "triton"
    assert qfa_params_actual.use_natural_transcendental_func is True
@pytest.mark.skipif(not HAVE_KITCHEN, reason="Kitchen required for using kitchen backend.")
def test_error_from_malformed_qflash_attention_params() -> None:
    # Missing recipe_name
    config: Dict[Any, Any] = {"kitchen_config_type": "QFlashAttentionParams"}
    with pytest.raises(KeyError, match="Missing required keys"):

third_party/Megatron-LM/megatron/rl/rl_utils.py
def _maybe_prefetch_separate_inference_model_weights(model_core, *, to_cpu: bool) -> None:
    """Prefetch RL *separate inference model* weights to CPU/GPU (UVM-only path).
    Gated only by user args; this assumes the separate inference model was allocated with UVM when enabled.
    """
    args = get_args()
    if not args.rl_offload_inference_model_weights_when_idle:
        return
    if args.rl_inference_model_unified_memory_level != 1:
        return
    device = -1 if to_cpu else int(torch.cuda.current_device())
    advise_managed_module_parameters_preferred_location(model_core, device=device, include_buffers=True)
    nbytes = prefetch_managed_module_parameters(model_core, device=device, include_buffers=True)
    # Ensure pages are resident before we enter CUDA-graph capture / inference, or before training continues.
    torch.cuda.synchronize()
    if to_cpu:
        print_rank_0(f"[Rank 0] offloaded {nbytes / 1024**2:.2f} MB of separate RL inference model weights to CPU (other ranks may vary)")
    else:
        print_rank_0(f"[Rank 0] prefetched {nbytes / 1024**2:.2f} MB of separate RL inference model weights to GPU (other ranks may vary)")
def verify_model_weights_swap(
    train_model: LanguageModule,
    inference_model: LanguageModule,
    seq_len: int = 8,
    batch_size: int = 2,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Verify that the inference model produces the same forward pass outputs
    as the training model after the weights have been swapped.
    This function should be called after swap_model_weights to ensure the weight
    transfer was successful. It runs a forward pass on both models and asserts
    the outputs match.  This is meant for debugging purposes only.
    Args:
        train_model: The training model (source of weights).
        inference_model: The inference model (target of weights).
        seq_len: Sequence length for test input.
        batch_size: Batch size for test input.
        atol: Absolute tolerance for comparing outputs.
        rtol: Relative tolerance for comparing outputs.
    Raises:
        AssertionError: If forward pass outputs do not match within tolerance.
    """
    args = get_args()
    # Unwrap models to get the core module
    train_lm = train_model[0] if isinstance(train_model, (list, tuple)) else train_model
    inf_lm = inference_model[0] if isinstance(inference_model, (list, tuple)) else inference_model
    train_core = unwrap_model(train_lm)
    inf_core = unwrap_model(inf_lm)
    actual_vocab_size = getattr(args, 'padded_vocab_size', 128256)
    actual_seq_len = min(seq_len, getattr(args, 'seq_length', seq_len))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    # Generate deterministic test input - same across ALL ranks
    torch.manual_seed(1234)
    test_tokens = torch.randint(
        low=0, high=actual_vocab_size, size=(batch_size, actual_seq_len),
        device=device, dtype=torch.long
    )
    test_position_ids = (
        torch.arange(actual_seq_len, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    test_attention_mask = torch.ones(
        (batch_size, 1, actual_seq_len, actual_seq_len), device=device, dtype=torch.bool
    )
    # Save and restore training state
    train_was_training = train_core.training
    inf_was_training = inf_core.training
    train_core.eval()
    inf_core.eval()
    try:
        with torch.no_grad():
def get_rl_runtime_state():
def update_inference_logprobs_group_stats(
    old_logprobs: torch.Tensor,
    inference_logprobs: torch.Tensor,
    mask: torch.Tensor,
    group_stats: Any,
) -> None:
    """Update group statistics with inference/train logprobs comparison metrics.
    This is the common statistics computation used by both packed and unpacked cases.
    Args:
        old_logprobs: Old logprobs tensor (train side)
        inference_logprobs: Inference logprobs tensor (aligned to match old_logprobs shape)
        mask: Boolean mask indicating valid positions for statistics
        group_stats: Statistics object to update with computed metrics
    """
    n_elems = mask.sum()
    if n_elems > 0:
        ratios = (old_logprobs - inference_logprobs).exp()[mask]
        abs_diffs = (old_logprobs.exp() - inference_logprobs.exp()).abs()[mask]
        group_stats.min_piold_to_inf_prob = ratios.min().item()
        group_stats.max_piold_to_inf_prob = ratios.max().item()
        group_stats.mean_piold_to_inf_prob = (ratios.sum() / n_elems).item()
        group_stats.min_inf_train_prob_abs_diff = abs_diffs.min().item()
        group_stats.max_inf_train_prob_abs_diff = abs_diffs.max().item()
        group_stats.mean_inf_train_prob_abs_diff = (abs_diffs.sum() / n_elems).item()
        inf_probs = inference_logprobs.exp()[mask]
        group_stats.min_inf_prob = inf_probs.min().item()
        group_stats.max_inf_prob = inf_probs.max().item()
        group_stats.mean_inf_prob = inf_probs.mean().item()
def align_unpacked_inference_logprobs(
    inference_logprobs: List[torch.Tensor],
    old_logprobs_for_data: torch.Tensor,
    generation_masks: torch.Tensor,
    group_stats: Any,
) -> torch.Tensor:
    """Align inference logprobs with old_logprobs for unpacked sequences and compute statistics.
    Args:
        inference_logprobs: List of inference logprobs tensors for each sequence
        old_logprobs_for_data: Template tensor with correct shape for alignment
        generation_masks: Tensor indicating which tokens were generated
        group_stats: Statistics object to update with computed metrics
    Returns:
        Aligned inference logprobs tensor
    """
    # Get first occurrence of a generation token
    # In get_logprobs() we chop off the first token -> the generation mask is shifted by one
    gen_masks_for_alignment = generation_masks
    first_gen_tok = gen_masks_for_alignment.int().argmax(dim=1) - 1
    # Align inference logprobs with old_logprobs
    # Note: We use old_logprobs_for_data as template since it has correct shape
    padded_inference_logprobs = old_logprobs_for_data.clone()
    # We need to align old_logprobs and inference logprobs as the latter are only for generations
    for i, inf_logprobs in enumerate(inference_logprobs):
def get_agent(args, parallel_generation_tasks: int | None = None):
def get_inference_interface(args, loop, model):
def get_rollout_generator(args, inference_interface, n_prompts, samples_per_group):
def get_environment_rollouts(
    model: LanguageModule, inference_model: LanguageModule, optimizer: MegatronOptimizer, n_prompts: int, samples_per_group: int
):
def selective_log_softmax(logits, index):
def get_logprobs(model, tokens, position_ids, no_grad=False, sequence_packing=False, packed_seq_params=None):
def compute_group_stats(
    rollouts: GroupedRollouts, tokenizer: MegatronLegacyTokenizer
) -> RolloutStats:
    """Add group-based rollout stats for logging.
    Args:
        rollouts: Rollouts to generate the stats for. Each inner list is a group (as in GRPO group), i.e. all rollouts are for the same prompt.
        tokenizer: Tokenizer to tokenize the rollouts in case they are raw strings.
    Returns:
       RolloutStats object containing all the stats.
    """
    args = get_args()
    # TODO (rkirby) Maybe do some of this after the tensor building
    group_reward_means = []
    group_reward_stds = []
    group_length_means = []
    group_length_stds = []
    group_length_maxs = []
    group_length_mins = []
    group_rollout_similarities = []
    for group in rollouts:
        group_rewards = []
        group_lengths = []
        for rollout in group:
            if isinstance(rollout, TokenRollout):
def maybe_log_training_metrics(
    group_stats: RolloutStats,
    current_iteration: int,
    tokenizer: MegatronLegacyTokenizer,
    example_group: list[TokenRollout | Rollout],
    wandb_writer: wandb_run.Run | None = None,
    tb_writer: SummaryWriter | None = None,
):
def prepare_trajectories(
    rollouts: GroupedRollouts, tokenizer: MegatronLegacyTokenizer, seq_length: int
):
def prepare_data_for_update(
    model: list[LanguageModule],
    ref_state_dict: Dict[str, Any],
    rollouts: GroupedRollouts,
    tokenizer: MegatronLegacyTokenizer,
) -> RerunDataIterator:
    """Extract data for the update from raw rollouts.
    Args:
        model: Current policy as the zero-eth element.
        ref_state_dict: Reference policy state dict.
        rollouts: Rollouts to extract the data from.
        tokenizer: Tokenizer to pad/tokenize data.
    Returns:
        Cycled iterator over dataset batches. In GRPO we might want to go over the same data multiple times.
    """
    args = get_args()
    wandb_writer = get_wandb_writer()
    tb_writer = get_tensorboard_writer()
    nvtx_range = get_nvtx_range()
    runtime_state = get_rl_runtime_state()
    if args.cuda_graph_impl != "none" and not args.rl_training_cuda_graphs:
        lang_module = (
            model[0].module.module if hasattr(model[0].module, "module") else model[0].module
        )
        toggle_cuda_graphs(lang_module, "none", reset_cuda_graphs=False)
    model = model[0]
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    with nvtx_range("prepare-data-for-update"):
def get_rollout_data_iterator(
    model: LanguageModule,
    inference_model: LanguageModule | None,
    optimizer: MegatronOptimizer,
    iteration: int,
    ref_state_dict: Dict[str, torch.Tensor],
) -> RerunDataIterator:
    args = get_args()
    tokenizer = get_tokenizer()
    buffered_rollouts = get_environment_rollouts(
        model, inference_model, optimizer, args.grpo_prompts_per_step, args.grpo_group_size
    )
    buffered_rollouts = prepare_data_for_update(model, ref_state_dict, buffered_rollouts, tokenizer)
    return buffered_rollouts
def setup_grpo_data_iterator(
    model: LanguageModule,
    inference_model: LanguageModule | None,
    optimizer: MegatronOptimizer,
    iteration: int,
    ref_state_dict: Dict[str, torch.Tensor],
    buffered_rollouts: RerunDataIterator | None = None,
) -> RerunDataIterator:
    """
    Set up the data iterator for GRPO training.
    Args:
        model: The language model
        optimizer: The Megatron optimizer
        iteration: Current training iteration
        ref_state_dict: Reference model state dict for GRPO
        buffered_rollouts: Previously collected rollouts (if any)
    Returns:
        RerunDataIterator for the current training step
    """
    args = get_args()
    runtime_state = get_rl_runtime_state()
    if inference_model is not None:
        inference_pg_collection = unwrap_model(inference_model[0]).pg_collection
    else:
        inference_pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    # We collect new rollouts when we've gone over the collected data 'grpo_iterations' times.
    if (
        buffered_rollouts is None or
        iteration == runtime_state.last_collection_iteration + 
        (args.grpo_iterations * runtime_state.global_batches_per_collection)
    ):
def evaluate_and_print_results_rl(
    data_iterator: Iterator[TensorDataset],
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    iteration: int,
    write_to_tensorboard: bool = True,
):
def calculate_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clamp_eps_lower: float,
    clamp_eps_upper: float,
    kl_beta: float,
    entropy_weight: float,
    inference_logprobs: torch.Tensor | None = None,
    is_truncation_coef: float | None = None,
    seq_starts: list | None = None,
    seq_lengths: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get GRPO loss, the kl term of the loss and the pi/pi_{old} ratios.
    Args:
        current_logprobs: pi logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        old_logprobs: pi_{old} logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        ref_logprobs: pi_{ref} logprobs, [batch, seq] for unpacked or [1, bin_size] for packed.
        advantages: advantages tensor, [batch,] for unpacked or [num_sequences_in_bin,] for packed.
        clamp_eps_lower: eps to clamp ratios from below.
        clamp_eps_upper: eps to clamp ratios from above, if vanilla GRPO, this should be equal to clamp_eps_lower.
        kl_beta: weight for the KL penalty term measuring the distance between pi and pi_{ref}.
        entropy_weight: weight for the entropy term.
        inference_logprobs: pi_{old} logprobs calculated by the inference engine.
            If not None, importance sampling correction will be applied.
        is_truncation_coef: importance sampling truncation coefficient. Will be applied if it is not None and inference_logprobs are present.
        seq_starts: (optional) For packed sequences: start positions of each sequence in the bin.
        seq_lengths: (optional) For packed sequences: original lengths of each sequence.
    Returns:
        total per-token GRPO loss [batch, seq] or [1, bin_size],
        kl_term of the loss [batch, seq] or [1, bin_size],
        pi/pi_{old} ratios [batch, seq] or [1, bin_size],
        entropy_term of the loss [batch, seq] or [1, bin_size],
        truncated_from_above [batch, seq] or [1, bin_size] (whether we clamped the ratios or not),
        truncated_from_below [batch, seq] or [1, bin_size] (whether we clamped the ratios or not).
    """
    # Ensure shapes match before computation
    if current_logprobs.shape != old_logprobs.shape:
        log_single_rank(
            logger,
            logging.WARNING,
            f"WARNING: Shape mismatch - current_logprobs: {current_logprobs.shape}, old_logprobs: {old_logprobs.shape}",
        )
    ratios = (current_logprobs - old_logprobs).exp()
    clamped_ratios = ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)
    truncated_from_above = torch.gt(ratios, 1 + clamp_eps_upper)
    truncated_from_below = torch.lt(ratios, 1 - clamp_eps_lower)
    # Handle advantages based on whether this is packed or unpacked
    if seq_starts is not None and seq_lengths is not None:
        # Packed sequences: map each sequence's advantage to its tokens
        bin_size = current_logprobs.shape[1]
        packed_advantages = torch.zeros(
            (1, bin_size), device=current_logprobs.device, dtype=current_logprobs.dtype
        )
        for seq_idx, (start, seq_len) in enumerate(zip(seq_starts, seq_lengths)):
def megatron_rl_inference_mode(
    model: list[LanguageModule],
    optimizer: MegatronOptimizer,
    cuda_graph_impl: str,
    reset_cuda_graphs: bool,
    offload_optimizer_during_inference: bool,
    offload_kv_cache_during_training: bool,
    remove_kv_cache_during_training: bool,
):
def rl_inference_interface_shutdown():
def get_iteration_sequence_count(args):

third_party/Megatron-LM/megatron/core/safe_globals.py
def register_safe_globals():

third_party/Megatron-LM/megatron/core/full_cuda_graph.py
def copy_tensors_in_struct(src):
def clone_tensors_in_struct(tgt, src):

third_party/Megatron-LM/megatron/core/tensor_parallel/utils.py
def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.
    Args:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list
def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False, tp_group=None):
def gather_split_1d_tensor(tensor, tp_group=None):

third_party/Megatron-LM/megatron/core/msc_utils.py
def open_file(*args, **kwargs):

third_party/Megatron-LM/tests/unit_tests/transformer/test_multi_latent_attention.py
def make_test_packed_seq_params(sequence_length=None, cu_seqlens=None):
def make_test_packed_seq_params_with_padding(
    sequence_length=None, cu_seqlens=None, cu_seqlens_padded=None
):
def get_mla_self_attn_submodules(linear_qkv_down_proj=None):
def test_parallel_multi_latent_attention_correctness(
    tmp_path_dist_ckpt, rope_type, apply_rope_fusion, tp, sp, cp
):

third_party/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py
def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):

third_party/Megatron-LM/megatron/core/nccl_allocator.py
def _build_nccl_allocator():
def get_func_args(func):
def create_nccl_mem_pool(symmetric=None):
def init() -> None:
    """
    Initialize the NCCL allocator.
    PyTorch tracks memory registration at the pool level, not per allocation.
    If a pool already contains allocations from a previous context, attempting
    to register it again will re-register all existing allocations and may
    trigger NCCL errors. To avoid this, the pool is explicitly deregistered
    on entry and re-registered on exit for each context use.
    """
    # Enables NCCL NVLS algorithm
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    # Disables the use of the tensor register allocator hook
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "0"
    _build_nccl_allocator()
    logging.info(f"[MCORE][NCCL_ALLOCATOR] Initialized NCCL Allocator")
# register_mem_pool/deregister_mem_pool are used for manual (de)registration of the memory pool.
# They are used in the case of FSDP manual registration.
def register_mem_pool(pool, group, symmetric=True):
def deregister_mem_pool(pool, group):

third_party/Megatron-LM/megatron/training/datasets/data_samplers.py
def build_pretraining_data_loader(dataset, consumed_samples):

third_party/Megatron-LM/megatron/core/typed_torch.py
def apply_module(m: _Module[P, R_co], *, check_subclass: bool = True) -> Callable[P, R_co]:
    """Returns the provided module unchanged, but with correct type hints.
    Args:
      m: An instance of a subclass of `torch.nn.Module`.
      check_subclass: If `True`, checks that `m` is a subclass of
            `torch.nn.Module` and raises a `TypeError` if not.
    Returns:
      That module unchanged, but with correct type hints.
    """
    if check_subclass and not issubclass(type(m), torch.nn.Module):

third_party/Megatron-LM/megatron/core/rerun_state_machine.py
def initialize_rerun_state_machine(**kwargs) -> None:
    """Helper function to initialize the rerun machine instance.
    Check the RerunStateMachine class for the details.
    """
    rerun_state_machine: RerunStateMachine = RerunStateMachine(**kwargs)
    _set_rerun_state_machine(rerun_state_machine)
def destroy_rerun_state_machine() -> None:
    """Helper function to shut down the rerun machine instance."""
    global _GLOBAL_RERUN_STATE_MACHINE
    _GLOBAL_RERUN_STATE_MACHINE = None
def get_rerun_state_machine() -> RerunStateMachine:
    """Helper function to return the singleton instance of the rerun machine."""
    if _GLOBAL_RERUN_STATE_MACHINE is None:
        logger.warning("Implicit initialization of Rerun State Machine!")
        initialize_rerun_state_machine()
        assert _GLOBAL_RERUN_STATE_MACHINE is not None
    return _GLOBAL_RERUN_STATE_MACHINE
def _set_rerun_state_machine(rerun_state_machine) -> None:
    """Internal function to set the singleton instance of the rerun machine."""
    global _GLOBAL_RERUN_STATE_MACHINE
    assert _GLOBAL_RERUN_STATE_MACHINE is None, "Rerun state machine is already initialized"
    _GLOBAL_RERUN_STATE_MACHINE = rerun_state_machine
def _safe_get_rank() -> int:
    """Internal function that safely checks and returns the rank of the caller."""
    if torch.distributed.is_initialized():
def _compare_floats(a: torch.Tensor, b: torch.Tensor) -> float:
    """Internal function that implements the default compare_func.
    Check the validate_result() method of the RerunStateMachine class for details.
    """
    af: float = a.item()
    bf: float = b.item()
    if (af == bf) or (math.isnan(af) and math.isnan(bf)):

third_party/Megatron-LM/megatron/core/post_training/modelopt/gpt/state_dict_hooks.py
def mcore_gpt_load_te_state_dict_pre_hook(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):

third_party/Megatron-LM/megatron/core/post_training/modelopt/gpt/model_specs.py
def get_gpt_modelopt_spec(
    config: TransformerConfig,
    local_core_attention: bool = False,
    remap_te_layernorm: bool = False,
    real_quant_cfg: str = "None",
    qk_l2_norm: bool = False,
    use_arbitrary_attention_mask: bool = False,
):

third_party/Megatron-LM/megatron/core/inference/unified_memory.py
def _compile_timeout(timeout_s: int):
def compile_allocator():
def create_unified_mempool() -> "MemPool":
    """Create a unified memory mempool using CUDA managed memory.
    Returns:
        (MemPool) Unified memory mempool.
    """
    # Attempt to compile allocator.
    compile_allocator()
    # Return mempool.
    if _compilation_state != CompilationState.SUCCESS:
        details = _compilation_error
        if details is None:
            details = "Unknown reason (allocator compilation did not succeed)."
        raise UnifiedMemoryUnsupportedError(
            "Unified virtual memory (UVM) mempool is unsupported or failed to initialize: "
            + details
        )
    else:
        return MemPool(allocator=_alloc)
def _get_ctypes_lib() -> "ctypes.CDLL":
    """Return a ctypes handle to the compiled UVM extension (.so)."""
    global _ctypes_lib
    compile_allocator()
    if _compilation_state != CompilationState.SUCCESS or _so_path is None:
        raise UnifiedMemoryUnsupportedError()
    if _ctypes_lib is not None:
        return _ctypes_lib
    with _ctypes_lock:
        if _ctypes_lib is None:
            _ctypes_lib = ctypes.CDLL(_so_path)
            # Configure argtypes/restype for exported helpers.
            _ctypes_lib.managed_prefetch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            _ctypes_lib.managed_prefetch.restype = ctypes.c_int
            _ctypes_lib.managed_advise_preferred_location.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _ctypes_lib.managed_advise_preferred_location.restype = ctypes.c_int
            _ctypes_lib.managed_advise_accessed_by.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            _ctypes_lib.managed_advise_accessed_by.restype = ctypes.c_int
    return _ctypes_lib
def prefetch_managed_tensor(tensor, *, device: int, stream=None) -> None:
    """Prefetch a CUDA tensor allocated from the UVM mempool to a specific device.
    This uses `cudaMemPrefetchAsync` to physically migrate the pages backing the tensor.
    The virtual address (pointer) remains unchanged, making this safe for use with
    recorded CUDA graphs.
    Args:
        tensor (torch.Tensor):
def advise_managed_tensor_preferred_location(tensor, *, device: int) -> None:
    """Set the preferred physical location hint for a managed tensor.
    This uses `cudaMemAdviseSetPreferredLocation`. It tells the CUDA driver where the
    pages should ideally reside. Unlike prefetch, this is a hint and does not
    immediately trigger migration unless the driver decides it is necessary.
    Args:
        tensor (torch.Tensor):
def advise_managed_tensor_accessed_by(tensor, *, device: int) -> None:
    """Hint that a specific device will access the managed tensor.
    This uses `cudaMemAdviseSetAccessedBy`. It ensures that the mapping for this
    memory region is established in the page tables of the specified device,
    reducing page fault latency when the device first touches the data.
    Args:
        tensor (torch.Tensor):
def prefetch_managed_module_parameters(
    module, *, device: int, include_buffers: bool = False
) -> int:
    """Prefetch all UVM-allocated parameters (and optionally buffers) of a module.
    Iterates through all parameters of the module and initiates an asynchronous
    migration to the target device. This is typically used to offload weights to
    CPU during training or prefetch them to GPU before inference.
    Args:
        module (torch.nn.Module):
def advise_managed_module_parameters_preferred_location(
    module, *, device: int, include_buffers: bool = False
) -> None:
    """Set the preferred physical location hint for all UVM parameters in a module.
    Args:
        module (torch.nn.Module):

third_party/Megatron-LM/megatron/core/fp8_utils.py
def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor.
    Note that in TE2.x, in order to support more recipes, the design of the fp8 tensor class has
    changed. Now Float8Tensor is only used for current scaling and delayed scaling. And mxfp8
    and blockwise scaling have their own fp8 tensor classes. These different fp8 tensor classes
    are both inherited from QuantizedTensor. So, for TE1.x, FP8_TENSOR_CLASS is Float8Tensor,
    and for TE2.x, FP8_TENSOR_CLASS is QuantizedTensor.
    """
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)
def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)
def dequantize_fp8_tensor(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp8 tensor to a higher precision tensor."""
    if is_te_min_version("2.0"):
def _resolve_callable_from_python_import_path(dotted_path: str):
def _get_custom_recipe(quantizer_factory_python_path: str) -> Union[Fp8Recipe, Fp4Recipe]:
    quantizer_factory = _resolve_callable_from_python_import_path(quantizer_factory_python_path)
    try:
        custom_recipe = transformer_engine.common.recipe.CustomRecipe(qfactory=quantizer_factory)
    except AttributeError:
        raise ValueError(
            """CustomRecipe recipe is not available in this version of 
            Transformer Engine. Please make sure you are using TE version 
            >= 2.9.0.dev0."""
        )
    return custom_recipe
def get_fp8_align_size(fp8_recipe: Fp8Recipe) -> int:
    """Get the alignment size required for fp8 GEMM."""
    if fp8_recipe == Fp8Recipe.mxfp8:
        return 32
    else:
        return 16
def is_column_parallel_linear(module):
def is_row_parallel_linear(module):
def modify_underlying_storage(tensor: torch.Tensor, new_raw_data: torch.Tensor):
def quantize_param_shard(
    model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params=None
):
def correct_amax_history_if_needed(model: List[torch.nn.Module]):
def post_all_gather_processing(model_params):
def is_first_last_bf16_layer(config: TransformerConfig, layer_no: int):

third_party/Megatron-LM/megatron/core/inference/inference_request.py
def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes.
    Args:
        tensor (Tensor):
def deserialize_tensor(tensor_bytes: bytes) -> torch.Tensor:
    """Deserialize tensor from bytes.
    Args:
        tensor_bytes (bytes):

third_party/Megatron-LM/megatron/core/tensor_parallel/layers.py
def param_is_not_tensor_parallel_duplicate(param, tp_group=None):
def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1, is_expert=False):
def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
    rank=None,
    world_size=None,
    skip_set_tensor_parallel_attributes=False,
):
def linear_with_frozen_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    tp_group: Optional[torch.distributed.ProcessGroup],
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: None = None,
    async_grad_allreduce: Optional[bool] = None,
) -> torch.Tensor:
    """Linear layer execution with weight.requires_grad == False.
    This function handles linear layers with weight frozen (untrainable).
    In the forward, it only saves weight and does not save input activations.
    In the backward, it does not perform weight gradient calculation, or
    weight gradient allreduce.
    Args:
    input (torch.Tensor required):
def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: Optional[int] = 0,
    async_grad_allreduce: Optional[bool] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.
    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.
    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.
    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calculation of the
    weight gradients.
    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.
    Args:
        input (torch.Tensor required):

third_party/Megatron-LM/megatron/training/tokenizer/tokenizer.py
def build_tokenizer(args, **kwargs):
def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
def reload_mergeable_ranks(path: str, max_vocab: Optional[int] = None) -> Dict[bytes, int]:
    """
    Reload our tokenizer JSON file and convert it to Tiktoken format.
    """
    from ..utils import print_rank_0  # To prevent circular import.
    assert path.endswith(".json")
    # reload vocab
    with open(path, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    print_rank_0(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        print_rank_0(f"Cutting vocab to first {len(vocab)} tokens.")
    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):

third_party/Megatron-LM/megatron/core/fp4_utils.py
def is_nvfp4tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine NVFP4Tensor."""
    return HAVE_TE_FP4_TENSOR_CLASS and isinstance(tensor, FP4_TENSOR_CLASS)
def get_fp4_align_size(fp4_recipe: Fp4Recipe) -> int:
    """
    Get the alignment size required for FP4 GEMM.
    FP4 GEMM requires Blackwell and later architectures.
    The value 32 is a hardware requirement: TMA (Tensor Memory Accelerator) requires
    a 16-byte aligned address for efficient memory access. Since FP4 uses 4 bits per value,
    16 bytes (128 bits) corresponds to 32 FP4 values. Therefore, the alignment size for FP4
    is 32. With this alignment, NVFP4 GEMM can be performed efficiently.
    Note that since we are also random hadamard transform for NVFP4 training, we want
    fused group nvfp4 quantize plus hadamard transform. Hadamard transform will leverage
    tensor core instructions for better performance, while group quantize kernels also
    prefer a more aligned size in token dimension M. The efficiently leverage grouped
    kernels, padding needs to be 64 multiple, but 128 multiple will bring even faster.
    When it comes to MOE cuda graph support, the number of tokens for each expert should
    be a buffer on device memory, which means that we don't know the token dimension for
    each expertin host, therefore we cannot calculate the zero padded scaling factors shape
    on host to comply with the NVFP4 GEMM scaling factor layout. However, if we have already
    zero padded the tokens to 128 multiple, then there is no need for such padding, so that
    host doesn't need to copy the token distribution from device to host (which will break
    the CUDA graph).
    Paper link: https://arxiv.org/pdf/2509.25149
    Scaling factor layout: https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
    TE NVFP4 Grouped Quantization: https://github.com/NVIDIA/TransformerEngine/pull/2411
    """
    # pylint: disable=unused-argument
    return 128
def dequantize_fp4_tensor(fp4_tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a fp4 tensor to a higher precision tensor."""
    if is_te_min_version("2.7.0.dev0"):

third_party/Megatron-LM/megatron/core/extensions/transformer_engine.py
def _get_fp8_autocast_for_quant_recipe(qrecipe: TEQuantizationRecipe):
def _get_fp8_autocast_for_quant_params(qparams: TEQuantizationParams | None, training: bool):
def _get_should_context_be_quantized_recipe(
    qrecipe: TEQuantizationRecipe, is_original_context_quantized: bool
):
def _get_should_context_be_quantized_params(
    qparams: TEQuantizationParams | None, training: bool, is_context_quantized: bool
):
def _get_extra_te_kwargs(config: TransformerConfig):
def condition_init_method(config, init_method):
def split_te_layernorm_column_parallel_linear(
    fused_layer,
    config,
    init_method: Optional[callable] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
def te_checkpoint(
    forward_func, distribute_saved_activations, get_rng_state_tracker, tp_group, *args, **kwargs
):
def set_save_original_input(module):

third_party/Megatron-LM/megatron/core/inference/communication_utils.py
def is_pipeline_first_stage(pp_group: ProcessGroup):
def is_pipeline_last_stage(pp_group: ProcessGroup):
def _is_cuda(tensor):
def _is_cuda_contiguous(tensor):
def broadcast_from_last_pipeline_stage(
    size: List[int],
    dtype: torch.dtype,
    tensor: Optional[torch.Tensor] = None,
    pp_group: Optional[ProcessGroup] = None,
):
def recv_from_prev_pipeline_rank_(
    recv_buffer: torch.Tensor = None, pp_group: Optional[ProcessGroup] = None
):
def send_to_next_pipeline_rank(
    tensor: torch.Tensor = None, pp_group: Optional[ProcessGroup] = None
):
def broadcast_tensor(size, dtype, tensor=None, rank=0, data_parallel=False):
def broadcast_list(size, dtype, list_values=None, rank=0, data_parallel=False):
def broadcast_int_list(size, int_list=None, rank=0, data_parallel=False):
def broadcast_float_list(size, float_list=None, rank=0, data_parallel=False):

third_party/Megatron-LM/megatron/core/optimizer_param_scheduler.py
def param_group_override_to_tuple(
    param_group_override: ParamGroupOverride | None,
) -> tuple[tuple[str, Any], ...] | None:
    """Convert a param group override to a tuple for use as a key in a dictionary.
    The tuple is sorted by the keys of the param group override to handle different orderings of
     the keys in different override dictionaries which still mean the same thing.
    """
    if param_group_override is None:
        return None
    return tuple(sorted(param_group_override.items()))
def combine_param_group_overrides(
    param_group_overrides: list[ParamGroupOverride | None],
) -> ParamGroupOverride:
    """Combine a list of param group overrides into a single param group override.
    This function ensures that the overrides are not conflicting as well.
    Args:
        param_group_overrides (list[ParamGroupOverride]):

third_party/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py
def _get_write_results_queue():
def _split_by_size_and_type(bins: int, items: List[WriteItem]) -> List[List[WriteItem]]:
    """
    Splits write items according to item size into close to uniform bins.
    Same as torch.distributed.checkpoint.filesystem._split_by_size_and_type,
    but with a fixed _item_size function.
    Args:
        bins (int):
def _split_by_separation_hint(
    buckets: List[List[WriteItem]], separation_hint: Optional[str] = None
) -> Dict[str, List[List[WriteItem]]]:
    """
    Splits buckets into those whose keys begin with the separation_hint and those whose keys do not
    Args:
        buckets (List[List[WriteItem]]):
def _item_size(item: WriteItem) -> int:
    """
    Calculates size (in bytes) of a single write item.
    Same as torch.distributed.checkpoint.filesystem._item_size,
    but fixes computing chunk size (with item.tensor_data.chunk.sizes)
    Args:
        item (WriteItem):
def _process_memory() -> int:
    """
    Get memory used by current process.
    Returns (int):

third_party/Megatron-LM/megatron/core/transformer/multi_token_prediction.py
def tie_word_embeddings_state_dict(
    sharded_state_dict: ShardedStateDict,
    word_emb_weight: Tensor,
    word_emb_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """tie the embedding of the mtp processing stage in a given sharded state dict.
    Args:
        sharded_state_dict (ShardedStateDict):
def tie_output_layer_state_dict(
    sharded_state_dict: ShardedStateDict,
    output_layer_weight: Tensor,
    output_layer_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """tie the output layer of the mtp processing stage in a given sharded state dict.
    Args:
        sharded_state_dict (ShardedStateDict):
def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
def _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group=None):
def get_mtp_layer_spec(
    transformer_layer_spec: ModuleSpec, use_transformer_engine: bool
) -> ModuleSpec:
    """Get the MTP layer spec.
    Returns:
        ModuleSpec: Module specification with TE modules
    """
    return get_mtp_layer_spec_for_backend(
        transformer_layer_spec,
        backend=TESpecProvider() if use_transformer_engine else LocalSpecProvider(),
    )
def get_mtp_layer_spec_for_backend(
    transformer_layer_spec: ModuleSpec, backend: BackendSpecProvider
) -> ModuleSpec:
    """Get the MTP layer spec.
    Returns:
        ModuleSpec: Module specification with modules from the backend.
    """
    column_parallel_linear_impl: type = backend.column_parallel_linear()
    layer_norm_impl: type = backend.layer_norm()
    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=layer_norm_impl,
            hnorm=layer_norm_impl,
            eh_proj=column_parallel_linear_impl,
            transformer_layer=transformer_layer_spec,
            layer_norm=layer_norm_impl,
        ),
    )
    return mtp_layer_spec
def mtp_on_this_rank(
    config: TransformerConfig, ignore_virtual: Optional[bool] = True, vp_stage: Optional[int] = None
) -> bool:
    """
    Check if there is MTP on the current rank.
    Behavior:
        - If a custom pipeline model parallel layout is provided in the config:
            - If virtual pipeline parallelism is enabled (and `ignore_virtual` is False), checks
              whether any MTP layers are present on this (pp_rank, vp_stage) pair.
            - Otherwise, checks all virtual pipeline ranks of the current pipeline rank. Returns
              True if any virtual sub-rank includes at least one MTP layer.
        - If no custom layout is provided, assumes all MTP layers (if any) are placed on the last
          pipeline stage. The function returns True only on the last pipeline stage.
    """
    mtp_on_this_rank = False
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if config.pipeline_model_parallel_layout is not None:
        # with custom PP layout, we support put MTP layers on any pipeline stage
        layout = config.pipeline_model_parallel_layout.layout
        if (
            not ignore_virtual
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
def get_mtp_ranks(pp_ranks: List[int], config: TransformerConfig) -> List[int]:
    """Get the ranks of the MTP layers."""
    mtp_ranks = set()
    if config.mtp_num_layers is None:
        return []
    if config.pipeline_model_parallel_layout is None:
        return [pp_ranks[-1]]
    layout = config.pipeline_model_parallel_layout.layout
    for pp_rank in range(len(layout)):
def get_mtp_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """Get the offset of the MTP layer."""
    if config.pipeline_model_parallel_size > 1:
        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.mtp, vp_stage=vp_stage
            )
        else:
            offset = 0
    else:
        offset = 0
    return offset
def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Get the number of MTP layers to build."""
    if config.pipeline_model_parallel_layout is not None:
        # If we have a custom PP layout, get the number of mtp layers in the layout array.
        num_layers_to_build = config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.mtp, vp_stage=vp_stage
        )
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, we only support put all of MTP layers on the last pipeline stage, "
            f"so the number of MTP layers to build ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
    else:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
def _get_mtp_block_submodules(
    config: TransformerConfig, spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]
) -> MultiTokenPredictionBlockSubmodules:
    """
    Retrieve or construct MultiTokenPredictionBlockSubmodules based on the provided specification.
    Args:
        config (TransformerConfig):

third_party/Megatron-LM/megatron/training/tokenizer/bert_tokenization.py
def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
def convert_to_unicode(text):
def printable_text(text):
def load_vocab(vocab_file):
def convert_by_vocab(vocab, items):
def convert_tokens_to_ids(vocab, tokens):
def convert_ids_to_tokens(inv_vocab, ids):
def whitespace_tokenize(text):
def _is_whitespace(char):
def _is_control(char):
def _is_punctuation(char):

third_party/Megatron-LM/megatron/core/extensions/kitchen.py
def _get_extra_kitchen_kwargs(config: TransformerConfig):

third_party/Megatron-LM/megatron/core/distributed/finalize_model_grads.py
def _get_main_grad_attr(param: torch.nn.Parameter):
def _unshard_if_dtensor(tensor: Union[torch.Tensor, "DTensor"]) -> torch.Tensor:
    """
    Unshards the input tensor if it is a DTensor and otherwise returns the
    tensor unmodified.
    Args:
        tensor (Union[torch.Tensor, DTensor]):
def _reshard_if_dtensor(
    tensor_to_shard: torch.Tensor, reference_tensor: Union[torch.Tensor, "DTensor"]
) -> Union[torch.Tensor, "DTensor"]:
    """
    Reshards the input tensor to match the sharding configuration of the
    reference tensor if the reference tensor is a DTensor. Otherwise, returns
    the reference tensor unmodified.
    Args:
        tensor_to_shard (torch.Tensor):
def _allreduce_conditional_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
):
def _get_shared_word_embedding_weight(
    model_module: torch.nn.Module, config: TransformerConfig
) -> Optional[torch.nn.Parameter]:
    """Return the shared word-embedding weight if it is duplicated across stages.
    Args:
        model_module: The model module from which to extract the
            word-embedding weight.
        config: Transformer config.
    Returns:
        The shared embedding or output weight if available; otherwise ``None``.
    """
    # Only reduce if weights are duplicated across stages.
    if model_module.share_embeddings_and_output_weights or getattr(config, 'mtp_num_layers', 0):
def _get_position_embedding_weight(model_module: torch.nn.Module) -> torch.nn.Parameter:
    """Return the position-embedding weight tensor from the given model module.
    Args:
        model_module: The model module that owns the
            position-embedding parameter.
    Returns:
        The position-embedding weight tensor.
    """
    return getattr(model_module, 'position_embeddings').weight  # type: ignore[attr-defined]
def _allreduce_word_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    embd_group: Optional[torch.distributed.ProcessGroup] = None,
    pp_group: Optional[torch.distributed.ProcessGroup] = None,
):
def _allreduce_embedding_grad(
    model: List[torch.nn.Module],
    embd_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
    weight_getter: Callable[[torch.nn.Module], Optional[torch.nn.Parameter]],
    skip_if_none: bool = True,
    config: TransformerConfig = None,
):
def _allreduce_position_embedding_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    pos_emb_group: torch.distributed.ProcessGroup,
    pp_group: torch.distributed.ProcessGroup,
):
def reset_model_temporary_tensors(config: TransformerConfig, model: List[torch.nn.Module]):
def _update_router_expert_bias(model: List[torch.nn.Module], config: TransformerConfig):
def _allreduce_non_tensor_model_parallel_grads(
    model: List[torch.nn.Module],
    config: TransformerConfig,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
def finalize_model_grads(
    model: List[torch.nn.Module],
    num_tokens: Optional[torch.Tensor] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):

third_party/Megatron-LM/megatron/core/inference/communication/torch_symm_triton/fused_collectives.py
def unpack_bf16x2(x, mask):
def sum_sq(x, y, z, w, mask):
def apply_norm(x, y, z, w, wx, wy, wz, ww, rrms, mask):
def _multimem_reduce_scatter_residual_add_kernel(
    residual_output_ptr,
    residual_input_ptr,
    rms_norm_weights_ptr,
    multicast_ptr,  # points to symmetric memory buffer
    signal_pad_ptrs,
    num_tokens,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):

third_party/Megatron-LM/megatron/training/tokenizer/gpt2_tokenization.py
def bytes_to_unicode():
def get_pairs(word):

third_party/Megatron-LM/megatron/core/tensor_parallel/random.py
def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda", clone: bool = False, graph_safe: bool = False
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU.
    Arguments:
        device (int):
def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
def convert_cuda_rng_state(
    state: Union[torch.Tensor, torch.Generator], to_graphable: bool = False
) -> Union[torch.Tensor, torch.Generator]:
    """
    Convert the cuda rng state tensor to the graphable version,
    or from the graphable version to the non-graphable tensor version.
    """
    if to_graphable:
        if isinstance(state, torch.Tensor):
def get_expert_parallel_rng_tracker_name():
def get_data_parallel_rng_tracker_name():
def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    force_reset: bool = False,
):
def get_cuda_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
def get_all_rng_states():
def model_parallel_cuda_manual_seed(
    seed: int,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    tp_rank: Optional[int] = None,
    ep_rank: Optional[int] = None,
    etp_rank: Optional[int] = None,
    force_reset_rng: bool = False,
):
def is_graph_safe_cuda_rng_tracker(cuda_rng_tracker):
def _get_all_rng_states():
def _set_all_rng_states(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
def _fork_rng():
def checkpoint(function, distribute_saved_activations, *args):

third_party/Megatron-LM/megatron/core/inference/communication/torch_symm_triton/barrier.py
def _send_signal(addrs, sem: tl.constexpr):
def _wait_signal(addrs, sem: tl.constexpr):
def symm_mem_sync(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    hasPreviousMemAccess: tl.constexpr = False,
    hasSubsequentMemAccess: tl.constexpr = False,
):

third_party/Megatron-LM/megatron/core/tensor_parallel/data.py
def _check_data_types(keys, data, target_dtype):
def _build_key_size_numel_dictionaries(keys, data, tp_group=None):
def broadcast_data(keys, data, datatype, tp_group=None):

third_party/Megatron-LM/megatron/core/transformer/transformer_block.py
def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig):
def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.
    Args:
        config (TransformerConfig):

third_party/Megatron-LM/megatron/core/pipeline_parallel/utils.py
def is_pp_first_stage(pp_group: torch.distributed.ProcessGroup):
def is_pp_last_stage(pp_group: torch.distributed.ProcessGroup):
def is_vp_first_stage(vp_stage: int, vp_size: int | None):
def is_vp_last_stage(vp_stage: int, vp_size: int | None):
def get_pp_first_rank(pp_group: torch.distributed.ProcessGroup):
def get_pp_last_rank(pp_group: torch.distributed.ProcessGroup):
def get_pp_next_rank(pp_group: torch.distributed.ProcessGroup):
def get_pp_prev_rank(pp_group: torch.distributed.ProcessGroup):
def make_viewless(e):
def set_ideal_affinity_for_current_gpu():
def stream_acquire_context(stream, event):
def set_streams(comp_stream=None, comm_stream=None):
def get_comp_stream():
def get_comm_stream():

third_party/Megatron-LM/megatron/training/global_vars.py
def get_args():
def get_tokenizer():
def get_tensorboard_writer():
def get_wandb_writer():
def get_one_logger():
def get_adlr_autoresume():
def get_timers():
def get_energy_monitor():
def get_signal_handler():
def _set_signal_handler(exit_signal):
def set_global_variables(args, build_tokenizer=True):
def unset_global_variables():
def set_args(args):
def _build_tokenizer(args):
def rebuild_tokenizer(args):
def _set_tensorboard_writer(args):
def _set_wandb_writer(args):
def _set_one_logger(args):
def _set_adlr_autoresume(args):
def _set_timers(args):
def _set_energy_monitor(args):
def _ensure_var_is_initialized(var, name):
def _ensure_var_is_not_initialized(var, name):
def destroy_global_vars():

third_party/Megatron-LM/megatron/core/export/trtllm/trtllm_weights_converter/utils.py
def is_gated_activation(helper):

third_party/Megatron-LM/megatron/core/export/trtllm/trtllm_weights_converter/distributed_trtllm_model_weights_converter.py
def str_dtype_to_torch(dtype: DataType):

third_party/Megatron-LM/megatron/core/inference/communication/torch_symm_triton/collectives.py
def _multimem_all_gather_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
def multimem_all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    **kwargs,
) -> torch.Tensor:
    """
    Calls a multicast all-gather triton kernel on the given tensor.
    Output tensor must be a symmetric memory buffer.
    Input tensor can be a regular torch tensor
    Arguments:
        output_tensor: torch.Tensor - output tensor to be all-gathered into
        input_tensor: torch.Tensor - input tensor to be all-gathered from
        symm_mem_hdl: _SymmetricMemory - handle to the symmetric memory buffer for output_tensor
    Returns:
        torch.Tensor - all-gathered tensor, which is output_tensor
    """
    assert HAVE_TRITON, "Triton is required for multimem all-gather."
    config = {
        "max_num_blocks": kwargs.get("max_num_blocks", 24),
        "num_warps": kwargs.get("num_warps", 32),
        "BLOCK_SIZE": kwargs.get("BLOCK_SIZE", 1024),
    }
    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert output_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    numel_per_thread = 128 // (input_tensor.element_size() * 8)
    assert (
        output_tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."
    num_threads = triton.cdiv(output_tensor.numel() // numel_per_thread, symm_mem_hdl.world_size)
    num_blocks = min(triton.cdiv(num_threads, config["BLOCK_SIZE"]), config["max_num_blocks"])
    _multimem_all_gather_kernel[(num_blocks, 1, 1)](
        input_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel=output_tensor.numel(),
        BLOCK_SIZE=config["BLOCK_SIZE"],
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=config["num_warps"],
    )
    return output_tensor
@triton.jit
def _multimem_reduce_scatter_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
def multimem_reduce_scatter(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    symm_mem_hdl: _SymmetricMemory,
    **kwargs,
) -> torch.Tensor:
    """
    Calls a multicast reduce-scatter triton kernel on the given tensor.
    Input tensor must be a symmetric memory buffer.
    Output tensor can be a regular torch tensor
    Arguments:
        output_tensor: torch.Tensor - output tensor to be reduce-scattered into
        input_tensor: torch.Tensor - input tensor to be reduce-scattered from
        symm_mem_hdl: _SymmetricMemory - handle to the symmetric memory buffer for input_tensor
        **kwargs: Additional keyword arguments for kernel configuration:
            max_num_blocks (int, optional):

third_party/Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py
def combined_1f1b_schedule_for_no_pipelining(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    output_tensor_grad,
    forward_data_store,
    config,
    collect_non_loss_data,
    first_val_step,
    forward_only,
    no_sync_func,
    total_num_tokens,
    check_first_val_step,
):
def combined_1f1b_schedule_for_interleaved_pipelining(
    config,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    forward_data_store,
    forward_step_helper_preprocess,
    forward_step_helper_postprocess,
    backward_step_helper_preprocess,
    backward_step_helper_postprocess,
    get_microbatch_id_in_model_chunk,
    get_model_chunk_id,
    check_first_val_step,
    is_first_microbatch_for_model_chunk,
    collect_non_loss_data,
    f_virtual_microbatch_id=None,
    b_virtual_microbatch_id=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
):
def combined_forward_backward_step(
    forward_step_func,
    data_iterator,
    f_model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    b_model,
    b_input_tensor,
    b_output_tensor,
    b_output_tensor_grad,
    config,
    f_model_chunk_id=None,
    pre_forward=None,
    pre_backward=None,
    post_forward=None,
    post_backward=None,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
):

third_party/Megatron-LM/megatron/core/export/trtllm/trtllm_weights_converter/single_device_trtllm_model_weights_converter.py
def pad_vocab_size(vocab_size: int, tp_size: int):
def str_dtype_to_torch(dtype: DataType):

third_party/Megatron-LM/megatron/core/tensor_parallel/mappings.py
def _reduce(input_, group):
def _split_along_last_dim(input_, group):
def _split_along_first_dim(input_, group):
def _gather_along_last_dim(input_, group):
def _reduce_scatter_along_last_dim(input_, group):
def _gather_along_first_dim(input_, group, output_split_sizes=None, use_global_buffer=False):
def _reduce_scatter_along_first_dim(input_, group, input_split_sizes=None, use_global_buffer=False):
def copy_to_tensor_model_parallel_region(input_, group=None):
def reduce_from_tensor_model_parallel_region(input_, group=None):
def scatter_to_tensor_model_parallel_region(input_, group=None):
def gather_from_tensor_model_parallel_region(input_, group=None):
def scatter_to_sequence_parallel_region(input_, group=None):
def gather_from_sequence_parallel_region(
    input_,
    tensor_parallel_output_grad=True,
    group=None,
    output_split_sizes=None,
    use_global_buffer=False,
):
def reduce_scatter_to_sequence_parallel_region(
    input_, group=None, input_split_sizes=None, use_global_buffer=False
):
def all_gather_last_dim_from_tensor_parallel_region(input_, group=None):
def reduce_scatter_last_dim_to_tensor_parallel_region(input_, group=None):
def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
def all_to_all_sp2hp(input_, group=None):
def all_to_all_hp2sp(input_, group=None):

third_party/Megatron-LM/megatron/core/utils.py
def null_decorator(*args, **kwargs):
def experimental_fn(introduced_with_version: str):
def experimental_cls(introduced_with_version: str):
def get_torch_version():
def get_te_version():
def is_te_min_version(version, check_equality=True):
def get_torch_version():
def is_torch_min_version(version, check_equality=True):
def get_fa_version():
def is_fa_min_version(version, check_equality=True):
def get_mamba_version():
def is_mamba_min_version(version, check_equality=True):
def get_causal_conv1d_version():
def is_causal_conv1d_min_version(version, check_equality=True):
def ensure_divisibility(numerator, denominator):
def divide(numerator, denominator):
def deprecate_inference_params(inference_context, inference_params):
def get_tensor_model_parallel_group_if_none(tp_group, is_expert=False, check_initialized=True):
def get_pg_size(group=None):
def get_pg_rank(group=None):
def get_pg_src_rank(group=None):
def get_attr_wrapped_model(model, attr, allow_none=True, return_model_obj=False):
def get_model_type(model):
def get_model_xattn(model):
def get_model_config(model):
def _kernel_make_viewless_tensor(inp, requires_grad):
def make_viewless_tensor(inp, requires_grad, keep_graph):
def assert_viewless_tensor(tensor, extra_msg=None):
def safely_set_viewless_tensor_data(tensor, new_data_tensor):
def init_method_normal(sigma):
def scaled_init_method_normal(sigma, num_layers, multiplier=2.0):
def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
def log_on_each_pipeline_stage(
    logger: logging.Logger,
    *args: Any,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
    **kwargs: Any,
):
def check_param_hashes_across_dp_replicas(
    model: List[torch.nn.Module], cross_check: bool = False
) -> bool:
    """Computes hashes of all parameters in model, all-gathers hashes across DP replicas,
    and then checks for equality between the locally-computed hashes and those of other ranks.
    NOTE: This function computes SHA-1 hashes on the CPU and thus needs to move all param
    tensors from GPU to CPU first; as a result, this function is not intended to be called
    very frequently in the main training loop.
    Args:
        model (List[torch.nn.Module]):
def make_tp_sharded_tensor_for_checkpoint(
    tensor, key, tp_axis=0, replica_id=None, prepend_offsets=(), **kwargs
):
def make_sharded_tensor_for_checkpoint(tensor, key, prepend_offsets=(), replica_id=None, **kwargs):
def get_full_tensor_if_necessary(tensor):
def to_local_if_dtensor(tensor: Union[torch.Tensor, "DTensor"]) -> torch.Tensor:
    """Returns the local shard of the given tensor if it is a DTensor."""
    with torch.no_grad():
def get_data_parallel_group_if_dtensor(
    tensor: Union[torch.Tensor, "DTensor"], data_parallel_group: "ProcessGroup" = None
) -> Optional["ProcessGroup"]:
    """Gets the data parallel group of the given tensor if it is a DTensor."""
    if HAVE_DTENSOR and isinstance(tensor, DTensor):
def prepare_input_tensors_for_wgrad_compute(grad_output, all_gathered_input):
def drain_embedding_wgrad_compute(
    config, embedding_activation_buffer, grad_output_buffer, weight, tp_group
):
def local_multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
def local_multi_tensor_l2_norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
def local_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
def is_submodule(module, parent_module, strict=True):
def get_batch_on_this_cp_rank(
    batch: Dict[str, Any], cp_group: Optional[torch.distributed.ProcessGroup] = None
):
def get_thd_batch_on_this_cp_rank(
    batch: Dict[str, Any],
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    max_seqlen: torch.Tensor,
    cp_size: Optional[int] = None,
    cp_rank: Optional[int] = None,
):
def get_batch_on_this_hybrid_cp_rank(
    batch: Dict[str, Any],
    local_cp_size: int,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
):
def configure_nvtx_profiling(enabled: bool) -> None:
    """Configure NVTX range profiling to be enabled or disabled.
    Args:
        enabled (bool):
def _nvtx_range_get_func_path():
def nvtx_range_push(msg=None, suffix=None) -> None:
    """Push NVTX range onto stack. If msg is not provided, use the calling function's path.
    Args:
        msg (str, optional):
def nvtx_range_pop(msg=None, suffix=None) -> None:
    """Pop NVTX range from stack. If msg is not provided, use the calling function's path.
    Args:
        msg (str, optional):
def _nvtx_decorator_get_func_path(func):
def nvtx_decorator(message: Optional[str] = None, color: Optional[str] = None):
def unwrap_model(model, module_instances=None):
def get_asyncio_loop(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.AbstractEventLoop:
    """Creates an asyncio loop if necessary and then returns the current asyncio loop."""
    global _ASYNC_IO_LOOP
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            if _ASYNC_IO_LOOP is not None:
                return _ASYNC_IO_LOOP
            else:
                _ASYNC_IO_LOOP = loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
    return loop
def is_using_quantization_scales(config):
def trace_async_exceptions(func: Optional[Callable] = None, *, verbose: bool = False):
def get_mamba_inference_state_config_from_model(model) -> Optional["MambaInferenceStateConfig"]:
    """Returns Mamba inference state config from the model if it is a hybrid model."""
    from megatron.core.inference.contexts.attention_context.mamba_metadata import (
        MambaInferenceStateConfig,
    )
    from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
    decoder = get_attr_wrapped_model(model, "decoder")
    layer_type_list = getattr(decoder, "layer_type_list", None)
    if layer_type_list is not None and Symbols.MAMBA in layer_type_list:
        (mamba_conv_states_shape, mamba_ssm_states_shape) = decoder.mamba_state_shapes_per_request()
        return MambaInferenceStateConfig(
            layer_type_list=layer_type_list,
            mamba_conv_states_shape=mamba_conv_states_shape,
            mamba_ssm_states_shape=mamba_ssm_states_shape,
        )
    return None
# ============================================================================
# Backward Compatibility Decorators
# ============================================================================
def deprecated(
    version: str,
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
    reason: Optional[str] = None,
) -> Callable:
    """
    Mark a function as deprecated.
    This decorator:
    1. Adds deprecation metadata to the function
    2. Issues a DeprecationWarning when the function is called
    3. Allows the compatibility checker to track deprecation lifecycle
    Args:
        version: Version where deprecation starts (e.g., "1.0.0")
        removal_version: Version where function will be removed (e.g., "2.0.0")
        alternative: Name of the recommended replacement function
        reason: Optional explanation for the deprecation
    Returns:
        Decorator function
    Example:
        @deprecated(
            version="1.0.0",
            removal_version="2.0.0",
            alternative="new_train_model",
            reason="Improved performance and cleaner API"
        )
        def old_train_model(config):
def internal_api(func: Callable) -> Callable:
    """
    Mark a function or class as internal API (not for external use).
    Use this decorator for:
    - Internal APIs not intended for public consumption
    - Experimental features that may change without notice
    - Implementation details that are not part of the stable API
    Objects marked with this decorator will be exempt from backward
    compatibility checks.
    Args:
        func: The function or class to mark as internal
    Returns:
        The original function/class with an internal API marker
    Example:
        @internal_api
        def _internal_helper():
def experimental_api(func: Callable) -> Callable:
    """
    Mark a function or class as experimental API.
    Use this decorator for:
    - Experimental features that may change without notice
    - New APIs under active development
    - Features that are not yet stable
    Objects marked with this decorator will be exempt from backward
    compatibility checks, allowing rapid iteration during development.
    Args:
        func: The function or class to mark as experimental
    Returns:
        The original function/class with an experimental API marker
    Example:
        @experimental_api
        def new_experimental_feature():

third_party/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py
def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
def partition_buckets(
    buffers: List[_ParamAndGradBuffer], force_single_bucket_group: bool = False
) -> List[_ParamAndGradBucketGroup]:
    """
    Automatically regroup the buckets of input buffers and return a list of bucket groups.
    In some scenarios, we need to put buckets from different buffers into a group so that their
    communication can be aggregated.
    For example, when there are both fp8 weights and bf16 biases in the model and virtual
    pipeline parallelism is enabled, each model chunk will have an fp8 bucket and a bf16 bucket,
    which doubles the number of communication kernels, and because of the use of
    CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back communications will prevent the
    overlap of communication kernels with computation kernels.
    The grouping strategy is:
    1. If force_single_bucket_group is True, put all buckets across all buffers into a single
       bucket group.
    2. If force_single_bucket_group is False, when there is no fp8 buffer in the input buffers,
       let each bucket group have only one bucket.
    3. If force_single_bucket_group is False, when using fp8 params, merge all non-fp8 buckets
       into the last fp8 bucket group.
       - Since the non-fp8 parameters (typically the biases of various layers) are relatively
         small, they are likely to be grouped into a single non-fp8 bucket.
       - The fp8 buckets start from the end of the model, i.e., the first bucket corresponds to
         the end of the model, while the last bucket corresponds to the beginning.
       - If we combine the non-fp8 bucket with the first fp8 bucket, we cannot initiate the
         reduce-scatter to synchronize gradients after the backward pass at the end of the model
         has completed. This is because we need to wait for the non-fp8 params from the beginning
         layers to obtain their gradients.
       - Combining the non-fp8 bucket with the last fp8 bucket can help avoid this issue.
    Args:
        buffers (list):

third_party/Megatron-LM/megatron/core/inference/communication/torch_symm_triton/utils.py
def get_tid():
def get_ntid():
def get_flat_tid():
def get_flat_bid():
def sync_threads():

third_party/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py
def register_default_torch_strategies():
def flatten_state_dict(
    state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, Dict[str, OBJ_PATH]]:
    """Flattens state dict into a single level dict.
    It's a copy of torch.distributed.checkpoint._nested_dict.flatten_state_dict
    which also accepts ShardedBase tensors as terminal objects
    Args:
        state_dict (ShardedStateDict):
def sharded_tensor_to_torch_sharded_tensor(
    sh_tens: List[ShardedTensor],
    rank: Optional[int] = None,
    load_legacy_1d_flatten_tensors: bool = False,
) -> TorchShardedTensor:
    """Convert MCore ShardedTensor to PyT ShardedTensor. PyT requires information about all chunks.
    On high-level, this function follows the logic of
    torch.distributed.fsdp._shard_utils._create_chunk_sharded_tensor.
    Additionally, it saves `prepend_axis_num` and `has_flattened_range` (specific to MCore)
    as attributes for further restoration in `_unwrap_pyt_sharded_tensor`.
    NOTE: this function assumes regular (grid) sharding of the MCore ShardedTensor.
    The only local irregularities could be introduced with a `flattened_range` attribute.
    This function handles 2 different type of ShardedTensors:
    1. Non-flat regular ShardedTensors (`not has_flattened_range`)
    2. N-D flattened ShardedTensors (`has_flattened_range`)
    (1) type are saved according to their original shape.
    Type (2) however requires global shape adjustment for efficiency:
    we treat [X, Y, Z] global shape tensor with local shape [x, y, z]
    as a [X // x, Y // y, Z // z, x * y * z] tensor with last axis
    partitioned according to `flattened_range` slices.
    This will need special handling while resharding.
    Args:
        sh_tens (List[ShardedTensor]):
def mcore_to_pyt_state_dict(
    state_dict: Dict[str, List[ShardedBase]],
    is_loading: bool = False,
    init_device: torch.device = torch.device("cpu"),
    load_legacy_1d_flatten_tensors: bool = False,
) -> Dict[str, Union[TorchShardedTensor, io.BytesIO]]:
    """Convert state dict with ShardedTensors and ShardedObjects
    to state dict compatible with PyT Dist format.
    Operates in-place and returns the original state dict.
    Args:
        state_dict (Dict[str, List[ShardedBase]]):
def _unwrap_pyt_sharded_tensor(
    sh_ten: Union[TorchShardedTensor, CheckpointableShardedTensor, LocalShardsContainer, Any]
) -> Union[List[torch.Tensor], Any]:
    """Unwrap tensor from PyT ShardedTensor instance.
    If `prepend_axis_num` was non-zero (which is specific to MCore ShardedTensor)
    then the tensor has additional singleton dimensions which should be squeezed.
    """
    if isinstance(sh_ten, CheckpointableShardedTensor):
def _replace_state_dict_keys_with_sharded_keys(
    sharded_state_dict: ShardedStateDict, keep_only_main_replica: bool = False
) -> Tuple[Dict[str, List[ShardedBase]], FLATTEN_MAPPING, Dict[str, List[str]]]:
    """Group ShardedBase objects by keys and
    return mappings required for recreating the original dict."""
    flat_sd, flat_mapping = flatten_state_dict(sharded_state_dict)
    rename_mapping = defaultdict(list)
    new_flat_sd = defaultdict(list)
    for k, sh_base in flat_sd.items():
def _replace_sharded_keys_with_state_dict_keys(
    state_dict: Dict[str, List[Union[torch.Tensor, io.BytesIO]]],
    flat_mapping: FLATTEN_MAPPING,
    rename_mapping: Dict[str, List[str]],
):
def _restore_dict_types(x: Union[dict, list, Any], keys_template: Union[dict, list, Any]):
def _get_filesystem_reader(
    checkpoint_dir: Union[str, Path], cache_metadata: bool = False
) -> FileSystemReader:
    if MultiStorageClientFeature.is_enabled():

third_party/Megatron-LM/megatron/core/config.py
def set_experimental_flag(flag: bool):
def is_experimental_enabled():

third_party/Megatron-LM/megatron/core/config_logger.py
def get_config_logger_path(config):
def has_config_logger_enabled(config):
def get_path_count(path):
def get_path_with_count(path):
def log_config_to_disk(config, dict_data, prefix='', rank_str=''):

third_party/Megatron-LM/megatron/rl/sequence_packing_utils.py
def load_packed_data_by_index(bin_idx: int, packing_context: PackingContext, logprobs_is_correction: bool):
def log_packing_efficiency(packing_context: PackingContext):
def get_actual_sequence_lengths(sequences: torch.Tensor, pad_token: int) -> List[int]:
    """Get actual sequence lengths for pre-padded sequences.
    Args:
        sequences: Tensor of shape [batch_size, seq_len] with pre-padded sequences
        pad_token: The padding token ID
    Returns:
        List of actual sequence lengths (excluding padding)
    """
    if len(sequences.shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {sequences.shape}")
    actual_lengths = []
    # Find actual length of each sequence by locating where padding starts
    for seq in sequences:
        # Find the last non-padding token
        non_pad_mask = seq != pad_token
        if non_pad_mask.any():
def create_empty_bins(
    num_empty_bins : int,
    bin_size : int,
    packed_trajs : torch.Tensor,
    packed_position_ids : torch.Tensor,
    packed_loss_mask : torch.Tensor,
    packed_attention_mask : torch.Tensor,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Create empty bins for padding to ensure all ranks have the same number of bins.
    Args:
        num_empty_bins: Number of empty bins to create
        bin_size: Size of each bin
        packed_trajs: Packed trajectories tensor (for dtype/device reference)
        packed_position_ids: Packed position IDs tensor (for dtype/device reference)
        packed_loss_mask: Packed loss mask tensor (for dtype/device reference)
        packed_attention_mask: Packed attention mask tensor (can be None)
        tokenizer: Tokenizer for pad token
    Returns:
        Tuple of (empty_trajs, empty_position_ids, empty_loss_mask, empty_attention_mask, empty_packing_info_entries)
    """
    device = packed_trajs.device
    # Create empty bins with proper shape
    empty_bins = []
    empty_position_ids_list = []
    empty_loss_mask_list = []
    empty_attention_mask_list = []
    empty_packing_info_entries = []
    for i in range(num_empty_bins):
def get_default_packed_seq_params(seq_length: int, device: torch.device) -> PackedSeqParams:
    """Create a default PackedSeqParams that acts as no-op for a single sequence.
    This ensures CUDA graph signature consistency when packed_seq_params
    would otherwise be None. A single sequence spanning the full length
    means no actual packing boundaries
    Args:
        seq_length: The sequence length 
        device: Device to create tensors on.
    Returns:
        PackedSeqParams configured as a single unpacked sequence.
    """
    args = get_args()
    # Pad to the maximum number of sequences in the bin for the attention kernel.
    cu_seqlens = torch.full((args.rl_sequence_packing_max_sequences_per_bin,), seq_length, dtype=torch.int32, device=device)
    cu_seqlens[0] = 0
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=seq_length,
        max_seqlen_kv=seq_length,
    )
def create_packed_seq_params(packing_context: PackingContext):
def create_packed_seq_params_for_bin(
    packing_info: PackingInfo,
    bin_idx: int,
    bin_size: int,
    max_sequences_per_bin: int,
    device: torch.device
) -> Optional[PackedSeqParams]:
    """Create PackedSeqParams for a single bin to enable proper attention masking in TE.
    When using Transformer Engine with sequence packing, we need to provide cu_seqlens
    (cumulative sequence lengths) so that TE knows the boundaries between sequences
    within a packed bin. This prevents attention leakage between unrelated sequences.
    Args:
        packing_info: PackingInfo object containing packing metadata from SequencePacker
        bin_idx: Index of the bin to create params for
        bin_size: Size of the bin (padded sequence length)
        max_sequences_per_bin: Maximum number of sequences per bin
        device: Device to create tensors on
    Returns:
        PackedSeqParams with cu_seqlens set for proper attention masking (or None if empty)
    """
    seq_indices = packing_info.bin_seq_indices[bin_idx]
    # Handle empty bins (padding bins with no sequences)
    if not seq_indices:
        return None
    # Get actual sequence lengths for sequences in this bin
    seq_lengths_in_bin = [packing_info.seq_lengths[idx] for idx in seq_indices]
    # Build cumulative sequence lengths for actual sequences
    # cu_seqlens should be [0, len(seq1), len(seq1)+len(seq2), ..., total_actual_len]
    cu_seqlens_list = np.append(np.cumsum([0] + seq_lengths_in_bin), bin_size)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    # Pad cu_seqlens to bin_size by repeating the last value (creates zero-length ghost sequences)
    # This ensures a fixed tensor size for CUDA graph compatibility
    if len(cu_seqlens) < max_sequences_per_bin:
        out = cu_seqlens.new_full((max_sequences_per_bin,), bin_size)
        out[:len(cu_seqlens)] = cu_seqlens
        cu_seqlens = out
    max_seqlen = bin_size
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
def pack_inference_logprobs(
    inference_logprobs: List[torch.Tensor],
    packing_info: PackingInfo,
    generation_masks: torch.Tensor,
    bin_size: int,
) -> torch.Tensor:
    """Pack inference logprobs into bins aligned with packed sequences.
    Args:
        inference_logprobs: List of inference logprobs tensors for each sequence
        packing_info: PackingInfo object containing bin assignments and sequence positions
        generation_masks: Tensor indicating which tokens were generated
        bin_size: Size of each bin
    Returns:
        Packed inference logprobs tensor of shape [num_bins, bin_size - 1]
    """
    num_bins = len(packing_info.bin_seq_indices)
    # Create packed inference logprobs tensor (logprobs are 1 token shorter than sequences)
    packed_inference_logprobs = torch.zeros(
        (num_bins, bin_size - 1), dtype=torch.float32, device='cpu'
    )
    # Create mapping from global sequence index to local bin index
    # This is needed because seq_to_bin_idx uses global bin indices,
    # but after distribution each rank only has a subset of bins
    seq_to_local_bin = {}
    for local_bin_idx, seq_indices in enumerate(packing_info.bin_seq_indices):
def compute_packed_inference_logprobs_stats(
    old_logprobs: torch.Tensor,
    packed_inference_logprobs: torch.Tensor,
    packed_loss_mask: torch.Tensor,
    group_stats: Any,
) -> None:
    """Compute statistics for packed inference logprobs for logging purposes.
    Compares packed inference logprobs with old logprobs using the packed loss mask
    to identify valid positions. Updates group_stats with computed metrics.
    Args:
        old_logprobs: Old logprobs tensor in packed format [num_bins, seq_len-1]
        packed_inference_logprobs: Packed inference logprobs [num_bins, seq_len-1]
        packed_loss_mask: Loss mask indicating valid positions [num_bins, seq_len]
        group_stats: Statistics object to update with computed metrics
    """
    # Lazy import to avoid circular dependency (rl_utils imports from this module)
    from megatron.rl.rl_utils import update_inference_logprobs_group_stats
    # Ensure all tensors are on the same device (CPU for stats computation)
    old_logprobs = old_logprobs.cpu()
    packed_inference_logprobs = packed_inference_logprobs.cpu()
    packed_loss_mask = packed_loss_mask.cpu()
    # Use packed_loss_mask to identify valid positions for stats (shift by 1 for logprobs)
    mask = packed_loss_mask[:, 1:].bool()
    # Ensure shapes match
    if mask.shape != old_logprobs.shape:
        return
    # Update group statistics using common helper
    update_inference_logprobs_group_stats(
        old_logprobs=old_logprobs,
        inference_logprobs=packed_inference_logprobs,
        mask=mask,
        group_stats=group_stats,
    )
class SequencePacker:
    """Packs multiple sequences into bins to minimize padding and improve GPU utilization."""
    def __init__(self, bin_size: int, pad_token: int, max_sequences_per_bin: int = 16):
def distribute_packed_bins(
    packed_trajs: torch.Tensor,
    packed_position_ids: torch.Tensor,
    packed_attention_mask: torch.Tensor,
    packed_loss_mask: torch.Tensor,
    packing_info: PackingInfo,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PackingInfo]:
    """Distribute packed bins across the data parallel ranks."""
    rank = mpu.get_data_parallel_rank()
    world_size = mpu.get_data_parallel_world_size()
    tokenizer = get_tokenizer()
    # Distribute packed bins across data parallel ranks
    num_bins, bin_size = packed_trajs.shape
    packing_algo = packing_info.packing_algo
    if packing_algo == 'round-robin':
        # Round-robin assignment: rank i gets bins [i, i+world_size, i+2*world_size, ...]
        my_bin_indices = list(range(rank, num_bins, world_size))
    else:  # fifo (default)
        world_size = world_size if world_size > 0 else 1
        # FIFO assignment: divide bins sequentially across ranks
        bins_per_rank = num_bins // world_size
        extra_bins = num_bins % world_size
        # Calculate start and end indices for this rank
        if rank < extra_bins:
            # Ranks with extra bins
            start_idx = rank * (bins_per_rank + 1)
            end_idx = start_idx + bins_per_rank + 1
        else:
            # Ranks without extra bins
            start_idx = rank * bins_per_rank + extra_bins
            end_idx = start_idx + bins_per_rank
        my_bin_indices = list(range(start_idx, end_idx))
    # Calculate the maximum bins any rank has (for synchronization)
    max_bins_per_rank = (num_bins + world_size - 1) // world_size
    # Extract this rank's bins
    my_packed_trajs = []
    my_packed_position_ids = []
    my_packed_attention_mask = []
    my_packed_loss_mask = []
    my_bin_seq_indices = []
    my_seq_starts = {}
    # Build the local data from the global indices
    for new_idx, old_idx in enumerate(my_bin_indices):
def pack_all_trajectories(trajs, generation_masks, inference_logprobs, global_advantages, bin_size, max_sequences_per_bin, packing_algo):
def get_microbatch_dataloader(packing_context: PackingContext) -> Tuple[DataLoader, int]:
    args = get_args()
    num_bins_this_rank = len(packing_context.packed_trajs)
    dp_world_size = mpu.get_data_parallel_world_size()
    # Ratio of collected sequences to the global batch size
    pct_of_sequences_per_batch = len(packing_context.packing_info.seq_lengths) / args.global_batch_size
    # Ceiling division means we will reuse some bins
    # If we did floor we would leave some behind
    local_bins_per_step = math.ceil(pct_of_sequences_per_batch * num_bins_this_rank)
    effective_global_batch_size = local_bins_per_step * dp_world_size
    # Store packing plan in runtime state for the training loop to use
    optimizer_steps = -(-num_bins_this_rank // local_bins_per_step)
    old_num_microbatches = get_num_microbatches()
    reconfigure_num_microbatches_calculator(
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        rampup_batch_size=args.rampup_batch_size,
        global_batch_size=effective_global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=dp_world_size,
        decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
    )
    new_num_microbatches = get_num_microbatches()
    log_single_rank(
        logger, logging.INFO, f"[Sequence Packing] Multi-step training plan:"
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Target sequences per step: {args.global_batch_size}",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Bins per rank per step: {pct_of_sequences_per_batch}*{num_bins_this_rank}={local_bins_per_step}",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Total optimizer steps: {optimizer_steps}",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"[Sequence Packing]  - Microbatches per step: {new_num_microbatches} (was {old_num_microbatches})",
    )
    bin_seq_indices = packing_context.packing_info.bin_seq_indices
    for step in range(min(3, optimizer_steps)):
def update_sequence_packing_metrics(args):
def get_sequence_packing_log_info(args):
def get_sequence_packing_tensorboard_metrics(args):

third_party/Megatron-LM/megatron/core/inference/communication/torch_symm_triton/multimem_asm.py
def ld_128(ptr, mask, multicast_op: tl.constexpr):
def st_128(ptr, x, y, z, w, mask, multicast_op):
def add_v8_bf16_from_u32(
    a0,
    a1,
    a2,
    a3,  # First vector of 8 bf16s, packed in 4 uint32s
    b0,
    b1,
    b2,
    b3,  # Second vector of 8 bf16s, packed in 4 uint32s
):
def asm_rsqrt(x, eps):

third_party/Megatron-LM/megatron/core/export/trtllm/trtllm_layers.py
def get_layer_name_without_prefix(layer: TRTLLMLayers) -> str:
    """Get TRTLayer name without prefix
    Given a layer e.g TRTLLMLayers.attention_qkv_weight it returns 'attention.qkv.weight'
    Args:
        layer (TRTLLMLayers):

third_party/Megatron-LM/megatron/core/models/common/embeddings/rope_utils.py
def get_pos_emb_on_this_cp_rank(
    pos_emb: Tensor, seq_dim: int, cp_group: torch.distributed.ProcessGroup
) -> Tensor:
    """Get the position embedding on the current context parallel rank.
    Args:
        pos_emb (Tensor):
def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]
    Args:
        x (Tensor):
def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.
    check https://kexue.fm/archives/8265 for detailed formulas
    Args:
        t (Tensor):
def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor, offset: int = 0
) -> Tensor:
    """Get the correct frequency slice for this context parallel rank with optional sequence offset.
    Args:
        cp_rank: Current context parallel rank
        cp_size: Total context parallel size
        x: Input tensor for current sequence
        freqs: Frequency tensor - either full batch positions or max sequence length
        offset: Starting position offset for this sequence in the original batch (default: 0)
    Returns:
        Tensor: Frequency slice corresponding to this CP rank's portion of the sequence
    Note:
        This function supports two modes based on the offset parameter:
        1. offset > 0: Exact mapping mode - freqs contains all positions across all sequences.
           The offset ensures each sequence gets frequencies from its actual position within
           the overall batch. Critical for non-1D RoPE in VLMs where spatial positions matter.
        2. offset = 0: Traditional mode - freqs contains only max sequence length positions.
           All sequences use frequencies starting from position 0, preserving backward
           compatibility.
    """
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        # Apply offset to both forward and backward segments for context parallelism
        # offset=0: traditional behavior, freqs[0:cp_seg] and freqs[...]
        # offset>0: exact mapping, freqs[offset+0:offset+cp_seg] and freqs[offset+...]
        return torch.cat(
            [
                freqs[offset + cp_rank * cp_seg : offset + (cp_rank + 1) * cp_seg],
                freqs[
                    offset
                    + full_seqlen
                    - (cp_rank + 1) * cp_seg : offset
                    + full_seqlen
                    - cp_rank * cp_seg
                ],
            ]
        )
    else:
        # For single context parallel rank:
        # offset=0: use freqs[0:x.size(0)] (traditional)
        # offset>0: use freqs[offset:offset+x.size(0)] (exact mapping)
        return freqs[offset : offset + x.size(0)]
def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.
    Args:
        t (Tensor):
def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config: TransformerConfig,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
):
def apply_rotary_pos_emb_with_cos_sin(
    t: Tensor, cos: Tensor, sin: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """
    This function applies rotary positional embedding to the target tensor t
    using precomputed cos and sin of size (seq_len, d_rot / 2)
    """
    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)
    if apply_rotary_emb_flash is None:
        # Combine cos and sin into freqs
        freqs = torch.stack([cos, sin], dim=-1).flatten(start_dim=-2)
        # Expand freqs to match t's shape
        while freqs.dim() < t.dim():

third_party/Megatron-LM/megatron/training/utils.py
def calc_params_l2_norm(model, force_create_fp32_copy=False):
def calc_dtensor_params_l2_norm(params):
def average_losses_across_data_parallel_group(losses):
def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.
    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        stat, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        return None
    else:
        return stat.item()
def logical_and_across_model_parallel_group(input: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group
    """
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        input, op=torch.distributed.ReduceOp.MIN, group=mpu.get_model_parallel_group()
    )
    return bool(input.item())
def report_memory(name):
def print_params_min_max_norm(optimizer, iteration):
def check_adlr_autoresume_termination(iteration, model, optimizer, opt_param_scheduler):
def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    pad_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    pad_mask_loss):
def print_rank_0(message, rank=None):
def warn_rank_0(message, rank=None):
def is_rank0():
def is_last_rank():
def print_rank_last(message):
def is_first_or_last_pipeline_stage(vp_stage):
def get_device_arch_version():
def append_to_progress_log(string, barrier=True):
def get_blend_and_blend_per_split(args):
def get_batch_on_this_tp_rank(data_iterator, mtp_on_this_rank: bool = False):
def update_use_dist_ckpt(args):
def to_empty_if_meta_device(module: torch.nn.Module, *, device: torch.device, recurse=True):
def get_nvtx_range():

third_party/Megatron-LM/megatron/rl/__init__.py
def import_class(class_path: str) -> Type:
    """Import a class from a string path.
    Args:
        class_path: String path to the class (e.g. 'examples.rl.environments.countdown.countdown_agent.CountdownAgent' or '../environments.countdown.py:CountdownAgent')
    Returns:
        The class object
    """
    if '.py:' in class_path:
        # filepath.py:Classname branch.
        module_path, class_name = class_path.split(':')
        abs_path = os.path.abspath(module_path)
        spec = importlib.util.spec_from_file_location('acemath_agent', abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path, package=__package__)
    return getattr(module, class_name)
class TypeLookupable(BaseModel, extra='allow'):

third_party/Megatron-LM/megatron/core/models/multimodal/context_parallel.py
def get_padding(
    seq_len,
    cp_size,
    tp_size,
    has_sp,
    decoder_tp_comm_overlap=False,
    decoder_seq_len=None,
    fp8_enabled=False,
    fp8_recipe=None,
):
def get_packed_seq_params(tokens, img_seq_len, padding_needed, cp_size, use_packed_sequence=False):

third_party/Megatron-LM/megatron/core/dist_checkpointing/strategies/fully_parallel.py
def distribute_main_replicas_with_precomputed_distribution(
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    precomputed_distribution: Optional[ShardDistribution],
):
def _defer_loading_sharded_items(
    sharded_state_dict: ShardedStateDict, item_type: type, shard_id_func: Callable[[T], _ShardId]
) -> Tuple[ShardedStateDict, ShardedStateDict, Dict[_ShardId, T], Dict[_ShardId, T]]:
    """Divides state dict into parts loaded by this vs other ranks.
    Args:
        sharded_state_dict (ShardedStateDict):
def _fill_in_deferred_sharded_items(
    sharded_state_dict: ShardedStateDict,
    loaded_items: Dict[_ShardId, Any],
    item_type: type,
    shard_id_func: Callable[[T], _ShardId],
) -> None:
    """Helper function to fill in items not loaded by current rank."""
    def fill_in_sharded_item(x: Any) -> Any:
        if isinstance(x, item_type):

third_party/Megatron-LM/megatron/training/theoretical_memory_usage.py
def compute_weight_and_optimizer_memory(args, verbose=False):
def compute_activation_memory(args, num_microbatches, verbose=False):
def compute_activation_memory_without_sp(args, num_microbatches, verbose=False):
def report_theoretical_memory(args, num_microbatches=None, verbose=False):

third_party/Megatron-LM/megatron/core/pipeline_parallel/fine_grained_activation_offload.py
def debug_rank(message):
def print_offload_summary_table(total_offload_bytes: Dict[str, int]):
def fine_grained_offloading_disable_offload():
def fine_grained_offloading_enable_offload():
def fine_grained_offloading_group_commit(
    tensor, name, forced_released_tensors=None, delay_offload=False
):
def fine_grained_offloading_group_flush_delayed_groups():
def fine_grained_offloading_group_start(tensor, name=None):
def fine_grained_offloading_forward_record(event: torch.cuda.Event) -> None:
    """Record the forward event for cuda graph capture."""
    d2h_stream = PipelineOffloadManager.get_instance().d2h_stream
    torch.cuda.current_stream().record_event(event)
    torch.cuda.current_stream().wait_stream(d2h_stream)
class FineGrainedOffloadingBackwardRecordFunction(torch.autograd.Function):
def fine_grained_offloading_backward_record(tensor, event: torch.cuda.Event) -> torch.Tensor:
    """Record the backward event for cuda graph capture."""
    return FineGrainedOffloadingBackwardRecordFunction.apply(tensor, event)
class FineGrainedActivationOffloadingInterface:
    """Interface for fine-grained activation offloading."""
    def __init__(self, offload: bool, tensor: torch.Tensor, name: str):

third_party/Megatron-LM/megatron/core/num_microbatches_calculator.py
def get_num_microbatches() -> int:
    """Get number of microbatches."""
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()
def get_current_global_batch_size() -> int:
    """Get current global batch size."""
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()
def get_micro_batch_size() -> int:
    """Get micro batch size."""
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_micro_batch_size()
def get_current_running_global_batch_size() -> int:
    """Get current running global batch size, taking into account number of DP replicas might be
    incompatible with true global batch size if `decrease_batch_size_if_needed` is True."""
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_running_global_batch_size()
def update_num_microbatches(
    consumed_samples: int, consistency_check: bool = True, verbose: bool = False
) -> None:
    """Update number of microbatches.
    Args:
        consumed_samples (int):
def unset_num_microbatches_calculator():
def init_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool = False,
) -> None:
    """Initialize number of microbatches calculator. Supporting backward compatibility.
    Args:
        rank (int):
def destroy_num_microbatches_calculator():
def reconfigure_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool = False,
) -> None:
    """Reconfigure number of microbatches calculator. Supporting backward compatibility.
    Args:
        rank (int):
def _configure_global_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool = False,
    init: bool = False,
) -> None:
    """Configure number of microbatches calculator. Can be used for initialization and
    reconfiguration.
    Args:
        rank (int):
def _build_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
    decrease_batch_size_if_needed: bool,
) -> Union['ConstantNumMicroBatchesCalculator', 'RampupBatchsizeNumMicroBatchesCalculator']:
    """Build number of microbatches calculator. Internal helper method.
    Args:
        rank (int):
def _round(batch_size: int, divisor: int) -> int:
    """Round `batch_size` down to nearest batch size divisible by `divisor`."""
    return (batch_size // divisor) * divisor
class NumMicroBatchesCalculator(ABC):

third_party/Megatron-LM/megatron/core/models/T5/t5_model.py
def t5_extended_attention_mask(attention_mask_list: List[Tensor]) -> List[Tensor]:
    """Creates the extended attention mask
    Converts the attention mask of dimension [batch size, seq_len, seq_len]
    to [batch size, 1, seq_len, seq_len]
    Args:
        attention_mask (Tensor):
def t5_position_ids(token_ids: Tensor) -> Tensor:
    """Calculate position ids from token ids
    Args:
        token_ids (Tensor):

third_party/Megatron-LM/megatron/rl/server/agent/fastapi_env_server.py
def run(agent_cls: type[Agent], cls_args: dict, port: int):

third_party/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/mixed_precision.py
def is_te_min_version(vers, check_equality=True):
def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a FP8 tensor."""
    return HAVE_TE and isinstance(tensor, FP8_TENSOR_CLASS)
def is_blockwise_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Blockwise FP8 tensor."""
    return HAVE_TE_BLOCKWISE_FP8TENSOR and isinstance(tensor, Float8BlockwiseQTensor)
def fp8_need_transpose_data(tensor: torch.Tensor) -> bool:
    """Check if a FP8 tensor needs transpose data."""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)
def fp8_need_transpose_data_for_meta_device_init(module: TransformerEngineBaseModule) -> bool:
    """Check if a FP8 tensor needs transpose data, for meta device init scenario."""
    return HAVE_TE_MXFP8TENSOR and module.fp8_meta["recipe"].mxfp8()
def fp8_discard_transpose_cache(tensor: torch.Tensor) -> None:
    """Discard the transpose cache of a FP8 tensor."""
    assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"
    if hasattr(tensor, "_transpose_invalid"):
def fp8_create_transpose_cache(tensors: List[torch.Tensor]) -> None:
    """Create the transpose cache of a FP8 tensor."""
    if HAVE_TE_POST_ALL_GATHER_PROCESSING:
        post_all_gather_processing(tensors)
    else:
        _fp8_create_transpose_cache_fallback(tensors)
def _fp8_create_transpose_cache_fallback(tensors: List[torch.Tensor]) -> None:
    if not isinstance(tensors, list):
def fp8_set_raw_data(tensor: torch.Tensor, data: torch.Tensor, set_transpose: bool = False) -> None:
    """Set the raw data of a Transformer Engine Float8Tensor."""
    assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"
    if set_transpose:
        assert fp8_need_transpose_data(tensor), f"Type {type(tensor)} does not need transpose data"
        data_attr = "_columnwise_data"
    else:
        data_attr = "_rowwise_data" if hasattr(tensor, "_rowwise_data") else "_data"
    old_data = getattr(tensor, data_attr)
    if old_data is not None:
        assert (
            old_data.dtype == data.dtype
        ), f"The data types of raw data don't match {old_data.dtype} vs {data.dtype}"
        assert (
            old_data.shape == data.shape
        ), f"Shape {old_data.shape} of old_data doesn't match {data.shape} of new_data"
    setattr(tensor, data_attr, data)
def fp8_get_raw_data(tensor: torch.Tensor, get_transpose: bool = False) -> torch.Tensor:
    """Get the underlying raw storage of a FP8 tensor."""
    assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"
    if get_transpose:
        assert fp8_need_transpose_data(tensor), f"Type {type(tensor)} does not need transpose data"
        data_attr = "_columnwise_data"
    else:
        data_attr = "_rowwise_data" if hasattr(tensor, "_rowwise_data") else "_data"
    return getattr(tensor, data_attr)
def fp8_dequantize(tensor: torch.Tensor) -> torch.Tensor:
    """Dequantize a FP8 tensor to a higher precision."""
    assert is_float8tensor(tensor), f"Type {type(tensor)} is not a FP8 tensor"
    assert is_te_min_version(
        "2.0"
    ), "Transformer Engine >= 2.0 is required for dequantizing parameters."
    return tensor.dequantize()
def fp8_quantize(
    model_params: List[torch.Tensor],
    main_params: List[torch.Tensor],
    start_offsets: List[int],
    data_parallel_group: torch.distributed.ProcessGroup,
    fsdp_shard_model_params: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
) -> None:
    """Quantize sharded parameters to FP8."""
    if len(model_params) == 0:
        return
    fsdp_shard_model_params = [x[0] if x[1] is None else x for x in fsdp_shard_model_params]
    if HAVE_TE_CAST_MASTER_WEIGHTS_TO_FP8:
        cast_master_weights_to_fp8(
            model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params
        )
    else:
        _fp8_quantize_fallback(
            model_params, main_params, start_offsets, data_parallel_group, fsdp_shard_model_params
        )
def _fp8_quantize_fallback(
    model_params: List[torch.Tensor],
    main_params: List[torch.Tensor],
    start_offsets: List[int],
    data_parallel_group: torch.distributed.ProcessGroup,
    fsdp_shard_model_params: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
) -> None:
    for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
        model_params, main_params, start_offsets, fsdp_shard_model_params
    ):

third_party/Megatron-LM/megatron/rl/agent/pass_at_evaluation_agent.py
def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """Lower variance estimator of pass@k."""
    assert n_samples >= 0, "n_samples should be non-negative"
    assert n_correct >= 0, "n_correct should be non-negative"
    assert k <= n_samples, "k should be less than or equal to n_samples"
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n_samples - n_correct + 1, n_samples + 1))
class PassAtEvaluationResult(RewardEvaluationResult):

third_party/Megatron-LM/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py
def _yarn_find_correction_dim(
    num_rotations: float, dim: int, rotary_base: float = 10000, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(rotary_base)
    )
# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
    round_to_int: bool = True,
) -> tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings)
    if round_to_int:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case
def _yarn_linear_ramp_mask(min: float, max: float, dim: int, device: torch.device) -> Tensor:
    if min == max:
        max += 0.001  # Prevent singularity
    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func
def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
@lru_cache(maxsize=8)
def _yarn_get_concentration_factor(
    scaling_factor: float, mscale: Optional[float], mscale_all_dim: Optional[float]
) -> float:
    """
    Get the concentration factor (factor multiplied to the sine and cosine components of the
    embedding). This factor is also known as attention factor, and sometimes homonymously known as
    "mscale"
    """
    if mscale is None or mscale_all_dim is None:
        return _yarn_get_mscale(scaling_factor)
    return float(
        _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim)
    )
def _yarn_get_concentration_factor_from_config(config: TransformerConfig) -> float:
    if hasattr(config, "yarn_rotary_scaling_factor"):

third_party/Megatron-LM/megatron/rl/logging.py
def log(message):

third_party/Megatron-LM/megatron/core/dist_checkpointing/strategies/state_dict_saver.py
def _compare_dataclasses(obj1, obj2):
def save_state_dict_async_plan(
    state_dict: STATE_DICT_TYPE,
    storage_writer: 'FileSystemWriterAsync',
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    planner: Optional[Union[SavePlanner, 'MCoreSavePlanner']] = None,
    cached_ckpt_structure: Optional[Tuple[SavePlan, SavePlan, bool]] = None,
    loaded_all_plans: Optional[List[SavePlan]] = None,
) -> Tuple[Tuple['FileSystemWriterAsync', Union[Metadata, None], _DistWrapper], SavePlan, bool]:
    """
    First stage of saving a state dict to storage.
    This is an async adjustment of torch.distributed.checkpoint.state_dict_saver.
    In order to support async save, saving should be split into three parts:
    1. Planning
    2. Actual saving
    3. Finalization
    Out of these, step (2) *must* happen asynchronously.
    The first step is realized with this function.
    The planning part consists of several steps, described here:
    https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.SavePlanner
    Args:
        state_dict (STATE_DICT_TYPE):
def verify_global_md_reuse(
    loaded_all_plans: List[SavePlan], local_plan: SavePlan, rank: int, dist_wrapper: _DistWrapper
) -> bool:
    """
    Verifies that global metadata reuse is possible by checking the loaded plans from the
     checkpoint are consistent, which means we have the same settings when resuming training.
    Args:
        loaded_all_plans: List[SavePlan], The loaded plans from the checkpoint
         (stored in checkpoint metadata).
        local_plan: SavePlan, The local save plan.
        rank: Current process rank.
        dist_wrapper (_DistWrapper):
def save_state_dict_async_finalize(
    storage_writer: 'FileSystemWriterAsync', global_metadata: Metadata, dist_wrapper: _DistWrapper
) -> None:
    """
    Finalization of save_state_dict_async_plan.
    The input arguments are the same as the save_state_dict_async_plan output,
    the `write_results` are retrieved from the storage_writer.
    Args:
        storage_writer (FileSystemWriterAsync):

third_party/Megatron-LM/megatron/core/transformer/custom_layers/batch_invariant_kernels.py
def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Build launch metadata for Triton matmul kernels used in BIK matmul."""
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
def get_compute_units():
def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
def _log_softmax_kernel(
    input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.
    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)
    >> Stashed changes
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")
    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()
    n_rows, n_cols = input_2d.shape
    # Allocate output tensor
    output = torch.empty_like(input_2d)
    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024
    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d, output, input_2d.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
def mean_dim(
    input: torch.Tensor, dim: int, keepdim: bool = False, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.
    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)
    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert (
        -input.ndim <= dim < input.ndim
    ), f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"
    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim
    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype
    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)
    # Get input shape and strides
    shape = list(input.shape)
    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
def mm_batch_invariant(a, b):
def addmm_batch_invariant(bias, a, b):
def _log_softmax_batch_invariant(input, dim, _half_to_float):
def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    """Return the (block_m, block_n) tiling used for batch-invariant attention."""
    return AttentionBlockSize(block_m=16, block_n=16)
_batch_invariant_MODE = False
_batch_invariant_LIB = None
_TE_GENERAL_GEMM_ORIG = None
_TE_RMSNORM_ORIG_FWD = None
_MEG_TE_GENERAL_GEMM_ORIG = None
_TE_RMSNORM_FUNC_ORIGS: Dict[str, Any] = {}
_TE_GEMM_FUNC_ORIGS: Dict[str, Any] = {}
def _import_module_if_available(name: str):
def _te_patch_for_batch_invariant():
def _te_unpatch_for_batch_invariant():
def _extract_te_gemm_args(args: tuple, kwargs: Dict[str, Any]):
def _is_supported_dtype_for_bik(t: torch.dtype) -> bool:
    return t in {torch.float16, torch.bfloat16, torch.float32}
class BatchInvariantTEGemmFn(torch.autograd.Function):
def _te_general_gemm_patched(*args, **kwargs) -> List[torch.Tensor]:
    """
    Batch-invariant replacement for TE general_gemm.
    Returns a list of tensors to match TE's API: (gemm_out, bias_grad, gelu_input, extra_output)
    """
    global _TE_GENERAL_GEMM_ORIG
    # If original not captured, do nothing
    if _TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError("TE general_gemm original not captured; patching order issue")
    A, B, out_dtype, layout, out, bias, grad = _extract_te_gemm_args(args, kwargs)
    extra_output = kwargs.get("extra_output", None)
    ub = kwargs.get("ub", None)
    ub_type = kwargs.get("ub_type", None)
    bulk_overlap = kwargs.get("bulk_overlap", False)
    # Guardrails: validate inputs
    if A is None or B is None:
        raise ValueError("Batch-invariant GEMM requires A and B tensors.")
    if (not A.is_cuda) or (not B.is_cuda):
def rmsnorm_batch_invariant(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Batch-invariant RMSNorm wrapper that delegates to autograd-aware implementation.
    This provides a simple functional interface while using the optimized BatchInvariantRMSNormFn
    which has better numerics (fp32 precision in forward/backward).
    """
    # Delegate to the autograd function with zero_centered_gamma=False (standard RMSNorm)
    return BatchInvariantRMSNormFn.apply(x, weight, eps, False)
def _te_rmsnorm_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    """Patched TE RMSNorm.forward that routes to batch-invariant
    implementation with autograd support.
    """
    weight = getattr(self, "weight", None)
    if weight is None:
        raise RuntimeError("Batch-invariant RMSNorm requires affine weight.")
    eps = getattr(self, "eps", 1e-5)
    zero_centered_gamma = getattr(self, "zero_centered_gamma", False)
    return BatchInvariantRMSNormFn.apply(x, weight, eps, zero_centered_gamma)
def is_batch_invariant_mode_enabled():
def enable_batch_invariant_mode():
def disable_batch_invariant_mode():
def set_batch_invariant_mode(enabled: bool = True):

third_party/Megatron-LM/megatron/core/models/multimodal/llava_model.py
def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
def _load_state_dict_hook_ignore_extra_state(
    module: torch.nn.Module, incompatible_keys: namedtuple
):
def pixel_shuffle(x, scale_factor=0.5, version=2):

third_party/Megatron-LM/megatron/core/dist_checkpointing/strategies/base.py
def get_default_strategy(action: StrategyAction, backend: str, version: int):
def register_default_strategy(
    action: StrategyAction,
    backend: str,
    version: int,
    strategy: Union['SaveStrategyBase', 'LoadStrategyBase'],
):

third_party/Megatron-LM/megatron/core/dist_checkpointing/strategies/common.py
def register_default_common_strategies():

third_party/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py
def experimental_api(func: Callable) -> Callable:
    """
    Mark a function or class as experimental API in Megatron CI/CD.
    TODO(@cspades):
def fully_shard_model(
    module: torch.nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    dp_shard_dim: Optional[str] = None,
    dp_outer_dim: Optional[str] = None,
    tp_dim: Optional[str] = None,
    hybrid_fsdp_group: Optional[torch.distributed.ProcessGroup] = None,
    expt_device_mesh: Optional[DeviceMesh] = None,
    fsdp_unit_modules: Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]] = None,
    zero_dp_strategy: str | int = 3,
    outer_dp_sharding_strategy: str | int = 0,
    device: Optional[torch.device] = None,
    init_model_with_meta_device: bool = False,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = True,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    sync_model_each_microbatch: bool = True,
    preproc_state_dict_for_dcp_ckpt: bool = True,
    check_for_nan_in_grad: bool = True,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
    disable_symmetric_registration: bool = False,
) -> torch.nn.Module:
    """
    Fully-shard the model for Megatron-FSDP. This wraps the model in a MegatronFSDP
    class that schedules the sharding lifecycle of the model parameters and gradients
    during training and inference.
    The original `torch.nn.Module` can be accessed at `MegatronFSDP.module`.
    Args:
        module (torch.nn.Module):
def fully_shard_optimizer(
    optimizer: torch.optim.Optimizer, preproc_state_dict_for_dcp_ckpt: bool = True
) -> torch.optim.Optimizer:
    """
    Fully shard the optimizer for Megatron-FSDP. This is an in-place operation on the optimizer
    instance, which modifies the optimizer to call methods exposed by the MegatronFSDP model API.
    The optimizer should be registered on the MegatronFSDP distributed model parameters:
    ```
        # Fully-shard the model.
        mfsdp_model = fully_shard_model(model, ...)
        # Register the fully-sharded parameters with the optimizer.
        # Use MegatronFSDP._replace_param_with_distributed_if_needed()
        # to swap to the distributed optimizer state parameters.
        optimizer = fully_shard_optimizer(Adam(params=mfsdp_model.parameters()))
    ```
    Args:
        optimizer (torch.optim.Optimizer):

third_party/Megatron-LM/megatron/core/models/bert/bert_model.py
def get_te_version():

third_party/Megatron-LM/megatron/core/dist_checkpointing/strategies/async_utils.py
def _disable_gc():

third_party/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py
def _replace_module_parameter(module, name, new_param):

third_party/Megatron-LM/megatron/core/models/bert/bert_layer_specs.py
def get_bert_layer_with_transformer_engine_spec():
def __getattr__(name):

third_party/Megatron-LM/megatron/core/dist_checkpointing/state_dict_utils.py
def save_preprocess(
    sharded_state_dict: ShardedStateDict,
    validate_access_integrity: bool = True,
    preprocess_common_before_consistancy_check: Callable[[CommonStateDict], StateDict] = None,
):
def load_preprocess(sharded_state_dict: ShardedStateDict):
def filter_out_empty_flatten_tensor(sharded_state_dict: Union[dict, list]):

third_party/Megatron-LM/megatron/core/distributed/fsdp/mcore_fsdp_adapter.py
def _get_hsdp_tp_mesh(outer_fsdp_dp_group, dp_cp_group, tp_group):
def _get_dp_tp_mesh(dp_cp_group, tp_group, ep_size=1):
def _check_mesh_ranks_and_group_ranks_are_consistent(mesh_ranks, group_ranks):
def _get_rng_state_dict():
def _load_rng_state_dict(rng_state_dict):

third_party/Megatron-LM/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py
def reduce_scatter_with_fp32_accumulation(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    group: torch.distributed.ProcessGroup,
    async_op: bool,
):

third_party/Megatron-LM/megatron/core/models/vision/radio.py
def fp8_pad_hook(
    module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):

third_party/Megatron-LM/megatron/core/models/T5/t5_spec.py
def encoder_model_with_transformer_engine_default_spec() -> ModuleSpec:
    """T5 encoder TE spec (uses Transformer Engine components)."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
def decoder_model_with_transformer_engine_default_spec() -> ModuleSpec:
    """T5 decoder TE spec (uses Transformer Engine components)."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_cross_attn_layernorm=TENorm,
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params={"attn_mask_type": AttnMaskType.padding},
                submodules=CrossAttentionSubmodules(
                    linear_q=TEColumnParallelLinear,
                    linear_kv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
def encoder_model_with_local_spec() -> ModuleSpec:
    """T5 encoder local spec (uses Megatron-Core components)."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.arbitrary},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )
def decoder_model_with_local_spec() -> ModuleSpec:
    """T5 decoder local spec (uses Megatron-Core components)."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_cross_attn_layernorm=LNImpl,
            cross_attention=ModuleSpec(
                module=CrossAttention,
                params={"attn_mask_type": AttnMaskType.arbitrary},
                submodules=CrossAttentionSubmodules(
                    linear_q=ColumnParallelLinear,
                    linear_kv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            cross_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )
def get_t5_encoder_with_transformer_engine_block_spec(
    num_layers: int,
) -> TransformerBlockSubmodules:
    """T5 encoder block spec for Transformer Engine
    Args:
      config (TransformerConfig):
def get_t5_decoder_with_transformer_engine_block_spec(
    num_layers: int,
) -> TransformerBlockSubmodules:
    """T5 decoder block spec for Transformer Engine
    Args:
      config (TransformerConfig):
def get_t5_encoder_with_local_block_spec(num_layers: int) -> TransformerBlockSubmodules:
    """T5 encoder block spec for local (uses Megatron-Core components)
    Args:
      num_layers (int):
def get_t5_decoder_with_local_block_spec(num_layers: int) -> TransformerBlockSubmodules:
    """T5 decoder block spec for local (uses Megatron-Core components)
    Args:
      num_layers (int):

third_party/Megatron-LM/megatron/core/models/vision/clip_vit_model.py
def get_num_image_embeddings(
    img_h,
    img_w,
    patch_dim,
    vision_model_type,
    disable_vision_class_token,
    class_token_len,
    pixel_shuffle,
    use_tile_tags=False,
    max_num_tiles=0,
    tokenizer_type=None,
):

third_party/Megatron-LM/megatron/core/models/huggingface/module.py
def get_hf_model_type(model_path):
def build_hf_model(config, model_path):

third_party/Megatron-LM/megatron/core/models/gpt/experimental_attention_variant_module_specs.py
def get_gated_delta_net_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Build module spec for GatedDeltaNet attention."""
    if backend is None:
        backend = _get_backend_spec_provider(config=config)
    rms_norm = config.normalization == "RMSNorm"
    attention = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj=backend.column_parallel_layer_norm_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": True},
    )
    return attention
def get_dsa_module_spec_for_backend(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Helper function to get module spec for Sparse Attention."""
    assert config.multi_latent_attention, "Currently only MLA supports sparse attention."
    assert config.qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
    linear_q_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if config.qk_layernorm
        else backend.column_parallel_linear()
    )
    linear_kv_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if config.qk_layernorm
        else backend.column_parallel_linear()
    )
    # Because TransformerEngine does not support sparse attention yet, we use local
    # implementation whether the backend is TransformerEngine or not.
    core_attention = ModuleSpec(
        module=DSAttention,
        submodules=DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=backend.linear(),
                    linear_wk=backend.linear(),
                    k_norm=backend.layer_norm(rms_norm=False, for_qk=True),
                    linear_weights_proj=backend.linear(),
                ),
            )
        ),
    )
    attention = ModuleSpec(
        module=MLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=linear_q_up_proj,
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=linear_kv_up_proj,
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
        ),
    )
    return attention
def get_experimental_attention_variant_module_spec(
    config: TransformerConfig, backend: BackendSpecProvider = None
) -> ModuleSpec:
    """Helper function to get module spec for experimental attention variant"""
    if backend is None:
        backend = _get_backend_spec_provider(config=config)
    if config.experimental_attention_variant == "gated_delta_net":
        return get_gated_delta_net_module_spec(config=config, backend=backend)
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )
##########
# Experimental GPT Decoder Block Spec
##########
def get_transformer_block_with_experimental_attention_variant_spec(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> TransformerBlockSubmodules:
    """Build transformer block spec with experimental attention variants (e.g., linear attention).
    This function constructs a heterogeneous transformer block that supports mixing different
    attention mechanisms (experimental vs standard) and MLP types (MoE vs dense) across layers.
    **Note that, this API is a experimental API in the short term, and might be deprecated in the
    future. In the long run, we will move to a new design that better support hybrid models.**
    Key Design:
        1. Attention and MLP patterns: The attention pattern and MLP pattern are orthogonal
           and determined independently. This allows flexible combinations (e.g., linear attention
           with MoE, or standard attention with dense MLP).
           - Attention pattern: derived from `config.linear_attention_freq` or
             `config.experimental_attention_variant`.
           - MLP pattern: derived from `config.moe_layer_freq`.
        2. Per-Layer Spec Construction: Iterates through layers, constructing transformer
           layer specs based on attention and MLP patterns.
        3. Pipeline Slicing: Extracts layer specs for the current pipeline stage.
    Args:
        config: Transformer configuration containing model hyperparameters and feature flags.
        vp_stage: Virtual pipeline stage index for interleaved pipeline parallelism.
        pp_rank: Pipeline model parallel rank.
    Returns:
        TransformerBlockSubmodules containing per-layer specs and final layer norm.
    Note:
        Currently only supports transformer_engine backend. Kitchen backend can be used as a
        wrapper with TE fallback for unsupported operations.
    """
    backend = _get_backend_spec_provider(config=config)
    # Get attention patterns and specs
    experimental_attention_pattern = [0] * config.num_layers
    if is_linear_attention_variant(config.experimental_attention_variant):
def is_linear_attention_variant(experimental_attention_variant: Optional[str]) -> bool:
    """Check if the experimental attention variant is a linear attention variant."""
    linear_attention_variants = ["gated_delta_net"]
    return experimental_attention_variant in linear_attention_variants
def get_moe_layer_pattern(config: TransformerConfig) -> List[int]:
    """Parse config.moe_layer_freq to get per-layer MoE pattern (1=MoE, 0=dense).
    - int N: one MoE layer every N layers (e.g., N=2 -> [1,0,1,0,...])
    - list: use directly as the pattern."""
    if isinstance(config.moe_layer_freq, int):
def get_linear_attention_pattern(config: TransformerConfig) -> List[int]:
    """Parse config.linear_attention_freq to get per-layer attention pattern (1=LA, 0=SDPA).
    - int N: one SDPA layer every N layers (e.g., N=4 -> [1,1,1,0,1,1,1,0,...])
    - list: use directly as the pattern."""
    if isinstance(config.linear_attention_freq, int):

third_party/Megatron-LM/megatron/core/fusions/fused_cross_entropy.py
def calculate_logits_max(vocab_parallel_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the maximum logits of the predicted tokens.
    """
    vocab_parallel_logits, logits_max = VocabParallelCrossEntropy.calculate_logits_max(
        vocab_parallel_logits
    )
    return vocab_parallel_logits, logits_max
@jit_fuser
def calculate_predicted_logits(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    logits_max: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the predicted logits for the tokens.
    """
    (target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits) = (
        VocabParallelCrossEntropy.calculate_predicted_logits(
            vocab_parallel_logits, target, logits_max, vocab_start_index, vocab_end_index
        )
    )
    predicted_logits_sum_exp_logits = torch.cat((predicted_logits, sum_exp_logits))
    return target_mask, masked_target_1d, predicted_logits_sum_exp_logits, exp_logits
@jit_fuser
def calculate_cross_entropy_loss(
    exp_logits: torch.Tensor, predicted_logits_sum_exp_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the final cross entropy loss for the tokens.
    """
    split_val = predicted_logits_sum_exp_logits.size()[0] // 2
    predicted_logits, sum_exp_logits = torch.split(predicted_logits_sum_exp_logits, split_val)
    exp_logits, loss = VocabParallelCrossEntropy.calculate_cross_entropy_loss(
        exp_logits, predicted_logits, sum_exp_logits
    )
    return exp_logits, loss
@jit_fuser
def calculate_gradients(
    softmax: torch.Tensor,
    grad_output: torch.Tensor,
    target_mask: torch.Tensor,
    masked_target_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the logits gradients scaled based on the CE loss
    """
    (grad_2d, arange_1d, softmax_update, grad_input) = (
        VocabParallelCrossEntropy.prepare_gradient_calculation_operands(softmax, target_mask)
    )
    grad_input = VocabParallelCrossEntropy.calculate_gradients(
        grad_2d, arange_1d, masked_target_1d, softmax_update, grad_input, grad_output
    )
    grad_input = grad_input.to(torch.bfloat16)
    return grad_input
class _VocabParallelCrossEntropy(torch.autograd.Function):
def fused_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group):

third_party/Megatron-LM/megatron/core/pipeline_parallel/schedules.py
def get_forward_backward_func(pp_size: Optional[int] = None, vp_size: Optional[int] = None):
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
def custom_backward(output, grad_output):
def set_current_microbatch(model, microbatch_id):
def forward_step_calc_loss(
    model,
    output_tensor,
    loss_func,
    config,
    vp_stage,
    collect_non_loss_data,
    num_microbatches,
    forward_data_store,
    cp_group_size=None,
    is_last_stage=None,
):
def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    cp_group_size,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    vp_stage=None,
    is_last_stage=True,
):
def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
def check_first_val_step(first_val_step, forward_only, cond):
def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: Optional[int] = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,  # unused
    pg_collection: Optional[ProcessGroupCollection] = None,
):
def clear_embedding_activation_buffer(config, model, is_last_stage):
def finish_embedding_wgrad_compute(config, embedding_module, is_last_stage, tp_group):
def get_pp_rank_microbatches(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    forward_only=False,
    overlap_moe_expert_parallel_comm=False,
    p2p_communicator: Optional[P2PCommunicator] = None,
):
def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
def get_tensor_shapes(
    *,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    tp_group: torch.distributed.ProcessGroup,
    cp_group: torch.distributed.ProcessGroup,
):
def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):

third_party/Megatron-LM/megatron/core/transformer/mlp.py
def apply_swiglu_sharded_factory(
    original_sh_ten, sharded_offsets, singleton_local_shards: bool = False
):

third_party/Megatron-LM/megatron/core/dist_checkpointing/utils.py
def zip_strict(*args):
def _sharded_tensor_shard_id(sharded_tensor: ShardedTensor) -> _ShardId:
    """Unique id of the sharded tensor data.
    Should yield the same value for same data replicated on different ranks.
    Args:
        sharded_tensor (ShardedTensor):
def _sharded_object_id(sharded_object: ShardedObject) -> _ShardId:
    """Unique id of the sharded object data.
    Should yield the same value for same data replicated on different ranks.
    Args:
        sharded_object (ShardedObject):
def extract_sharded_tensors(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only ShardedTensor objects
    from a given state dict with any objects.
    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor objects
    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor (keeping the original state dict structure)
            - state dict with all objects other than ShardedTensor
              (keeping the original state dict structure)
    """
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedTensor))
def extract_sharded_tensors_and_factories(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only ShardedTensor and ShardedTensorFactory objects
    from a given state dict with any objects.
    Args:
        sharded_state_dict:
            state dict possibly containing ShardedTensor and ShardedTensorFactory objects
    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor and ShardedTensorFactory
              (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, (ShardedTensor, ShardedTensorFactory))
    )
def extract_sharded_tensors_or_nonpersistent(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only ShardedTensor, ShardedTensorFactory
    and LocalNonpersistentObject objects from a given state dict with any objects.
    Args:
        sharded_state_dict: state dict possibly containing ShardedTensor, ShardedTensorFactory
        and LocalNonpersistentObject objects
    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedTensor, ShardedTensorFactory and LocalNonpersistentObject
              (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(
        sharded_state_dict,
        lambda v: isinstance(v, (ShardedTensor, LocalNonpersistentObject, ShardedTensorFactory)),
    )
def extract_sharded_base(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only ShardedBase from a given state dict with any objects.
    Args:
        sharded_state_dict: state dict possibly containing ShardedBase objects
    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all ShardedBase objects (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(sharded_state_dict, lambda v: isinstance(v, ShardedBase))
def extract_nonpersistent(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, StateDict]:
    """Extract a dict consisting of only LocalNonpersistentObjects from a given state dict.
    Args:
        sharded_state_dict: state dict possibly containing LocalNonpersistentObjects
    Returns:
        Tuple[ShardedStateDict, StateDict]: tuple of:
            - state dict with all LocalNonpersistentObjects
              (keeping the original state dict structure)
            - state dict with all other objects (keeping the original state dict structure)
    """
    return extract_matching_values(
        sharded_state_dict, lambda v: isinstance(v, LocalNonpersistentObject)
    )
def add_prefix_for_sharding(sharded_state_dict: ShardedStateDict, prefix: str):
def replace_prefix_for_sharding(
    sharded_state_dict: ShardedStateDict, old_prefix: str, new_prefix: str
):
def apply_prefix_mapping(sharded_state_dict: ShardedStateDict, prefix_map: Dict[str, str]):
def force_all_tensors_to_non_fp8(sharded_state_dict: ShardedStateDict):
def logger_stack(name: Optional[str] = None, current_logger: Optional[logging.Logger] = None):
def debug_time(
    name: str, logger: Optional[logging.Logger] = None, threshold: float = float("-inf"), level=None
):
def debug_msg(msg: str):

third_party/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py
def gather_and_compute_chunk_metadata(dtensor: DTensor) -> ChunkStorageMetadata:
    """
    Gather chunk metadata for a DTensor across all ranks and compute the
    offsets and sizes of each chunk. This is necessary for handling uneven
    sharding in distributed tensors.
    """
    local_tensor = dtensor.to_local()
    local_shape = local_tensor.shape
    device_mesh = dtensor.device_mesh
    offsets = [0] * len(local_shape)
    cumulative_shape = list(local_shape).copy()
    def _update_offsets_and_cumulative_shape(
        mesh_dim: int, offsets: List[int], cumulative_shape: List[int]
    ):
def update_uneven_dtensor_chunk_metadata(dtensor: DTensor) -> dict:
    """
    Update the DTensor's chunk metadata to handle uneven sharding.
    This function modifies the DTensor in-place to include chunk metadata
    and write items closures for saving and loading.
    """
    def _chunk_list_closure(chunk_meta):
def validate_uneven_dtensor(dtensor: DTensor) -> None:
    """
    Validates the chunk metadata of an uneven DTensor to ensure correctness and boundary coverage.
    Notes:
    - `gather_and_compute_chunk_metadata` will ensure that all chunks do not overlap.
    This function performs the following checks:
      - All chunk offsets and sizes are within the tensor shape bounds.
      - All boundaries of each dimension are actually covered by shard placements.
    Args:
        dtensor (DTensor):
def filter_unflattened_state_dict(state_dict, key_chain=[], visit_condition=lambda x: False):
def get_unflattened_state_dict(state_dict, key_chain=[]):
def preprocess_state_dict_for_uneven_dtensor(state_dict: dict) -> dict:
    """
    Preprocess the state_dict to prepare it for saving or loading unevenly sharded DTensors.
    This function modifies the DTensors in the state_dict to include chunk metadata
    and write items closures.
    """
    visit_dtensor = filter_unflattened_state_dict(
        state_dict, visit_condition=lambda x: isinstance(x, DTensor)
    )
    for key_chain in visit_dtensor:
        # Get the DTensor at the key chain
        dtensor = get_unflattened_state_dict(state_dict, key_chain)
        update_uneven_dtensor_chunk_metadata(dtensor)
    return state_dict
def gather_uneven_dtensor_to_full_tensor(
    dtensor: DTensor, target_device: Optional[torch.device] = None
) -> DTensor:
    """
    Gather an unevenly sharded DTensor distributed across multiple ranks,
    reconstructing the full (unsharded) tensor on each rank.
    This function handles uneven chunk sizes and offsets by collecting
    chunk metadata from all ranks, performing all-gather operations,
    and assembling the full tensor accordingly. The returned tensor
    is fully replicated across the given device mesh.
    Args:
        dtensor (DTensor):
def _assemble_full_tensor_from_uneven_chunks(
    dtensor: DTensor,
    all_chunk_info: List[dict],
    process_group: torch.distributed.ProcessGroup,
    target_device: Optional[torch.device],
) -> DTensor:
    """
    Assemble the full tensor from unevenly sized chunks gathered from all ranks.
    Args:
        dtensor (DTensor):
def _intersection(s1, s2):
def _offset_slice(s, offset):
def split_dtensor(
    dtensor: DTensor,
    split_size_or_sections: Union[int, List[int]],
    dim: int = 0,
    update_uneven_dtensor_chunk_meta: bool = False,
) -> Iterable[DTensor]:
    """
    Splits a DTensor into smaller DTensors along a specified dimension.
    This function manages uneven sharding by accurately assigning chunk metadata
    for each split. Unlike the native PyTorch DTensor split functionality,
    it does not redistribute `Replicate` placements, which helps avoid Out-Of-Memory (OOM) issues.
    Args:
        dtensor (DTensor):

third_party/Megatron-LM/tests/test_utils/python_scripts/download_coverage_results.py
def main(pipeline_id: int):

third_party/Megatron-LM/megatron/training/training.py
def set_startup_timestamps(program_start=None, main_entry=None):
def destroy_global_state():
def print_datetime(string, override_timestamp=None):
def num_floating_point_operations(args, batch_size):
def get_start_time_from_progress_log():
def preprocess_common_state_dict(common_state_dict):
def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    non_loss_data_func=None,
    store=None,
    inprocess_call_wrapper: Optional[CallWrapper] = None,
):
def update_train_iters(args):
def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True, config=None, pg_collection=None):
def get_optimizer_param_scheduler(optimizer):
def get_megatron_optimizer_config(args: Any) -> OptimizerConfig:
    """Return a Megatron optimizer config object from Megatron's arguments."""
    config = None
    if args.optimizer == 'adam' or 'muon' in args.optimizer:
        # TODO(deyuf):
def setup_model_and_optimizer(
    model_provider_func,
    model_type,
    checkpointing_context=None,
):
def dummy_train_step(data_iterator):
def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config, forward_backward_func):
def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    max_attention_logit,
    pg_collection=None,
    is_first_iteration=False,
):
def compute_throughputs_and_append_to_progress_log(iteration, num_floating_point_operations_so_far):
def enable_forward_pre_hook(model_chunks):
def disable_forward_pre_hook(model_chunks, param_sync=True):
def force_param_sync(model_chunks: list[DDP]) -> None:
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.start_param_sync(force_sync=True)
def save_checkpoint_and_time(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
    checkpointing_context,
    non_persistent_ckpt=False,
    train_data_iterator=None,
):
def post_training_step_callbacks(
    model,
    optimizer,
    opt_param_scheduler,
    iteration,
    prof,
    num_floating_point_operations_since_last_log_event,
    nsys_nvtx_context = None,
):
def checkpoint_and_decide_exit(
    model,
    optimizer,
    opt_param_scheduler,
    iteration,
    num_floating_point_operations_so_far,
    checkpointing_context,
    train_data_iterator,
):
def train(
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    train_data_iterator,
    valid_data_iterator,
    process_non_loss_data_func,
    config,
    checkpointing_context,
    non_loss_data_func,
    inference_model=None,
):
def evaluate(
    forward_step_func,
    data_iterator,
    model,
    process_non_loss_data_func,
    config,
    verbose=False,
    non_loss_data_func=None,
    eval_iters=None,
):
def evaluate_and_print_results(
    prefix,
    forward_step_func,
    data_iterator,
    model,
    iteration,
    process_non_loss_data_func,
    config,
    verbose=False,
    write_to_tensorboard=True,
    non_loss_data_func=None,
):
def cyclic_iter(iter):
def get_train_valid_test_num_samples():
def build_train_valid_test_datasets(build_train_valid_test_datasets_provider, train_valid_test_num_samples=None):
def build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider):
def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
def should_disable_forward_pre_hook(args):

third_party/Megatron-LM/megatron/core/transformer/moe/moe_utils.py
def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
    fused: bool = False,
):
def z_loss_func(logits, z_loss_coeff):
def sinkhorn(cost: torch.Tensor, tol: float = 0.0001):
def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
def sort_chunks_by_idxs(
    input: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_idxs: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    fused: bool = False,
):
def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
def pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Pad the routing map to ensure each expert has a multiple of pad_multiple tokens.
    This function ensures that each expert has a number of tokens that is a multiple of
    pad_multiple by converting some 0s to 1s in the routing map. The padding is done by
    selecting the first N zero elements in each row, where N is the number needed to reach
    the next multiple of pad_multiple.
    Args:
        routing_map (torch.Tensor):
def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    fused: bool = False,
):
def compute_routing_scores_for_aux_loss(
    logits: torch.Tensor, topk: int, score_function: str, fused: bool = False
):
def apply_router_token_dropping(
    routing_probs: torch.Tensor,
    routing_map: torch.Tensor,
    router_topk: int,
    capacity_factor: float,
    drop_policy: str = "probs",
    pad_to_capacity: bool = False,
):
def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: Optional[torch.distributed.ProcessGroup] = None,
    avg_group: Optional[torch.distributed.ProcessGroup] = None,
    reduce_group_has_dp: bool = False,
):
def clear_aux_losses_tracker():
def reduce_aux_losses_tracker_across_ranks(
    track_names: Optional[List[str]] = None, pg_collection: Optional[ProcessGroupCollection] = None
):
def track_moe_metrics(
    loss_scale: float,
    iteration: int,
    writer,
    wandb_writer=None,
    total_loss_dict=None,
    per_layer_logging=False,
    force_initialize: bool = False,
    track_names: Optional[List[str]] = None,
    num_layers: Optional[int] = None,
    moe_layer_freq: Optional[Union[int, List[int]]] = None,
    mtp_num_layers: Optional[int] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
def get_updated_expert_bias(tokens_per_expert, expert_bias, expert_bias_update_rate):
def maybe_move_tensor_to_cpu(tensor, as_numpy=False, record_stream=False):
def get_moe_layer_wise_logging_tracker():
def apply_random_logits(logits):
def router_gating_linear(
    inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, router_dtype: torch.dtype
):
def get_align_size_for_quantization(config: TransformerConfig):
def get_default_pg_collection():
def maybe_skip_or_early_return_by_cudagraph(step_condition):

third_party/Megatron-LM/megatron/core/fusions/fused_weighted_squared_relu.py
def weighted_squared_relu(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Element-wise weight applied after Squared-ReLU.
    Args:
        x (torch.Tensor):
def _squared_relu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gradient of Squared-ReLU.
    The derivative of ``(ReLU(x))^2`` w.r.t ``x`` is ``2 * ReLU(x)``.
    """
    return g * 2 * F.relu(x)
@jit_fuser
def weighted_squared_relu_back(g: torch.Tensor, x: torch.Tensor, weights: torch.Tensor):
def weighted_squared_relu_impl(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Token-wise weighted Squared-ReLU fusion with optional FP8 storage.
    Args:
        input (torch.Tensor):

third_party/Megatron-LM/megatron/core/fusions/fused_bias_gelu.py
def bias_gelu(bias, y):
def bias_gelu_back(g, bias, y):

third_party/Megatron-LM/megatron/core/pipeline_parallel/hybrid_cp_schedule.py
def hybrid_context_parallel_forward_backward(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    output_tensor_grad,
    forward_data_store,
    config,
    collect_non_loss_data,
    first_val_step,
    forward_only,
    no_sync_func,
    total_num_tokens,
    check_first_val_step,
    model_type,
):

third_party/Megatron-LM/megatron/core/dist_checkpointing/dict_utils.py
def extract_matching_values(
    x: Union[dict, list], predicate: Callable[[Any], bool], return_lists_as_dicts: bool = False
) -> Tuple[Union[dict, list], Union[dict, list]]:
    """Return matching and nonmatching values. Keeps hierarchy.
    Args:
        x (Union[dict, list]) :
def diff(x1: Any, x2: Any, prefix: Tuple = ()) -> Tuple[list, list, list]:
    """Recursive diff of dicts.
    Args:
        x1 (object):
def inspect_types(x: Any, prefix: Tuple = (), indent: int = 4):
def nested_values(x: Union[dict, list]):
def nested_items_iter(x: Union[dict, list]):
def dict_map(f: Callable, d: dict):
def dict_map_with_key(f: Callable, d: dict):
def dict_list_map_inplace(f: Callable[[U], V], x: Union[Dict, List, U]):
def dict_list_map_outplace(f: Callable[[U], V], x: Union[Dict, List, U]) -> Union[Dict, List, V]:
    """Maps dicts and lists *out-of-place* with a given function."""
    if isinstance(x, dict):
def merge(x1: Union[dict, list], x2: Union[dict, list], key: Tuple[Union[str, int], ...] = ()):

third_party/Megatron-LM/megatron/core/pipeline_parallel/p2p_communication.py
def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
def is_single_shape(x) -> bool:
    """Check if the input is a single shape."""
    if isinstance(x, torch.Size):

third_party/Megatron-LM/megatron/core/transformer/moe/upcycling_utils.py
def _get_keys_endswith(model, suffix):
def _find_submodule(model, submodule_name):
def _get_config(moe_model, dense_model):
def _convert_to_moe_state_dict(moe_model, dense_model):
def upcycle_state_dict(moe_model, dense_model):
def load_and_upcycle_model(
    load_dense_ckpt_func, moe_model, dense_model, strict=True, load_args=(), load_kwargs={}
):

third_party/Megatron-LM/tests/functional_tests/test_cases/gpt/gpt_dynamic_inference_tp1_pp1_583m_cuda_graphs_validation/cuda_graphs.py
def clear_output_dir() -> None:
    """Clear output directory."""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.mkdir(OUTPUT_DIR)
def get_output_path(num_cuda_graphs: int, trial_idx: int) -> str:
    """Get output path for a given test.
    Args:
        num_cuda_graphs (int):
def run_tests(args: argparse.Namespace) -> None:
    """Run all tests, iterating over `num_cuda_graphs` and `NUM_TRIALS`."""
    # Iterate `num_cuda_graphs` and `NUM_TRIALS`.
    for num_cuda_graphs_idx, num_cuda_graphs in enumerate(NUM_CUDA_GRAPHS_LIST):
def load_results(num_cuda_graphs: int) -> dict:
    """Load all trial outputs for a given `num_cuda_graphs`.
    Args:
        num_cuda_graphs (int):
def validate_cuda_graph_request_counts(result_map: dict) -> None:
    """Validate `cuda_graph_request_count` usage across tests.
    For each test (i.e., each `num_cuda_graphs`), we validate how many times each
    cuda graph was used within the test.
    Args:
        result_map (dict):
def validate_step_counts(result_map: dict) -> None:
    """Validate engine step counts.
    This value should be identical across all tests, regardless of whether cuda
    graphs are enabled or not.
    Args:
        result_map (dict):
def validate_latencies(result_map: dict) -> None:
    """Validate that latency decreases as we increase the number of cuda graphs.
    *Note*: This test is disabled for now, since the latency difference between
    these small tests is small, rendering this check unstable.
    Args:
        result_map (dict):
def validate_latency_proxies(result_map: dict) -> None:
    """Validate that the latency 'proxy' decreases as we increase the number of cuda graphs.
    Latency proxy is computed as the sum of `cuda_graph_request_count` *
    `cuda_graph_usage` for a given test.
    Args:
        result_map (dict):
def validate_logprobs(result_map: dict) -> None:
    """Validate the logprob tensors.
    The logprobs should remain bitwise equal whether cuda graphs are enabled or not.
    Args:
        result_map (dict):

third_party/Megatron-LM/megatron/core/transformer/moe/shared_experts.py
def set_tensor_grad_fn_sequence_sr(tensor, value):

third_party/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/utils.py
def get_te_version():
def is_te_min_version(vers, check_equality=True):
def is_submodule(module, parent_module, strict=True):
def get_mesh_names(
    device_mesh: Optional[DeviceMesh] = None, only_submesh_dims: bool = False
) -> list[str]:
    """
    Get all the sub-mesh ("dp", "cp", etc.) and flattened-mesh ("dp_cp", etc.) names
    in the DeviceMesh. When only_submesh_dims=True, only checks for sub-mesh dimensions.
    """
    if device_mesh is None:
        # Device mesh does not exist.
        return []
    # Sub-mesh dimension names.
    submesh_dim_names = (
        list(device_mesh.mesh_dim_names) if device_mesh.mesh_dim_names is not None else []
    )
    # Flattened mesh dimension names.
    try:
        # Retrieve all flattened meshes associated with DeviceMesh.
        # The flattened DeviceMesh are all located in the _flatten_mapping
        # dictionary of the root DeviceMesh.
        flatten_mesh_names = [
            flat_dim
            for flat_dim, flat_mesh in device_mesh._get_root_mesh()._flatten_mapping.items()
        ]
    except AttributeError:
        # Fallback to the DeviceMesh global state to retrieve flattened
        # meshes associated with the DeviceMesh.
        from torch.distributed.device_mesh import _mesh_resources
        flatten_mesh_names = [
            child_mesh_dim_name
            for child_mesh, root_mesh in _mesh_resources.child_to_root_mapping.items()
            for child_mesh_dim_name in (child_mesh.mesh_dim_names or [])
            if root_mesh == device_mesh and child_mesh_dim_name not in submesh_dim_names
        ]
    # Order of the returned list of mesh dimension names must match the index
    # of the root mesh dimension names followed by flattened sub-meshes:
    # [<root mesh dimension names>, <flattened mesh dimension names>]
    if only_submesh_dims:
        return submesh_dim_names
    else:
        return submesh_dim_names + flatten_mesh_names
def contains_submesh(
    device_mesh: Optional[DeviceMesh], submesh_names: Optional[str | Sequence[str]]
) -> bool:
    """
    Check if a sub-mesh exists in the device mesh by name.
    """
    if device_mesh is None or submesh_names is None:
        # Device mesh does not exist.
        return False
    if isinstance(submesh_names, str):
def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda", clone: bool = False, graph_safe: bool = False
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU.
    Arguments:
        device (int):
def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    force_reset: bool = False,
):
def get_cuda_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
def get_global_memory_buffer():
def create_updated_function_signature(original_function, **extended_kwargs: dict):
def is_mcore_tensor_model_parallel(param: torch.Tensor) -> bool:
    """
    Check if the given parameter is Megatron-Core tensor model parallel.
    """
    return getattr(param, "_mcore_tp", False) or getattr(param, "tensor_model_parallel", False)
def is_mcore_tensor_parallel_duplicated(param: torch.Tensor) -> bool:
    """
    Check if the given parameter is Megatron-Core tensor model parallel and duplicated.
    """
    return getattr(param, "_tp_duplicated", False)
def get_mcore_tensor_parallel_partition_dim(param: torch.Tensor) -> Optional[int]:
    """
    Get the partition dimension for a Megatron-Core tensor model parallel parameter.
    """
    if is_mcore_tensor_model_parallel(param):

third_party/Megatron-LM/tests/test_utils/python_scripts/dashboard.py
def get_gitlab_handle():
def get_build_analytics(pipeline_id: int) -> pd.DataFrame:
    pipeline_jobs = (
        get_gitlab_handle()
        .projects.get(PROJECT_ID)
        .pipelines.get(pipeline_id)
        .jobs.list(get_all=True)
    )
    return pd.DataFrame(
        [
            {
                "name": pipeline_job.name,
                "started_at": pipeline_job.attributes['started_at'],
                "finished_at": pipeline_job.attributes['finished_at'],
            }
            for pipeline_job in pipeline_jobs
            if pipeline_job.name.startswith("test:build_image: [CI")
        ]
    )
def get_unit_test_analytics(pipeline_id: int) -> pd.DataFrame:
    pipeline = get_gitlab_handle().projects.get(PROJECT_ID).pipelines.get(pipeline_id)
    unit_test_pipeline_bridges = [
        pipeline_bridge
        for pipeline_bridge in pipeline.bridges.list()
        if pipeline_bridge.name.startswith("test:unit_tests")
        and pipeline_bridge.downstream_pipeline is not None
    ]
    return pd.DataFrame(
        [
            {
                "name": pipeline_bridge.name,
                "started_at": pipeline_bridge.attributes['started_at'],
                "finished_at": pipeline_bridge.attributes['finished_at'],
            }
            for pipeline_bridge in unit_test_pipeline_bridges
        ]
    )
def get_functional_test_analytics(pipeline_id: int, type: str) -> pd.DataFrame:
    pipeline = get_gitlab_handle().projects.get(PROJECT_ID).pipelines.get(pipeline_id)
    functional_test_pipeline_bridges = [
        pipeline_bridge
        for pipeline_bridge in pipeline.bridges.list()
        if pipeline_bridge.name.startswith(type) and pipeline_bridge.downstream_pipeline is not None
    ]
    return pd.DataFrame(
        [
            {
                "name": pipeline_bridge.name,
                "started_at": pipeline_bridge.attributes['started_at'],
                "finished_at": pipeline_bridge.attributes['finished_at'],
            }
            for pipeline_bridge in functional_test_pipeline_bridges
        ]
    )
def get_failed_stage(pipeline_id: int) -> pd.DataFrame:
    pipeline_bridges = (
        get_gitlab_handle().projects.get(PROJECT_ID).pipelines.get(pipeline_id).bridges.list()
    )
    pipeline_jobs = (
        get_gitlab_handle()
        .projects.get(PROJECT_ID)
        .pipelines.get(pipeline_id)
        .jobs.list(get_all=True)
    ) + [
        pipeline_bridge
        for pipeline_bridge in pipeline_bridges
        if pipeline_bridge.downstream_pipeline is not None
    ]
    failed_stages = list(
        set(
            [
                job.stage
                for job in pipeline_jobs
                if job.status == "failed" and job.allow_failure is False
            ]
        )
    )
    if "build" in failed_stages:
        return "build"
    elif "test" in failed_stages:
        return "test"
    elif "integration_tests" in failed_stages:
        return "integration_tests"
    elif "functional_tests" in failed_stages:
        return "functional_tests"
    return
def get_analytics_per_pipeline(pipeline_id: int) -> pd.DataFrame:
    build_analytics = get_build_analytics(pipeline_id)
    unit_tests_analytics = get_unit_test_analytics(pipeline_id)
    integration_tests_analytics = get_functional_test_analytics(pipeline_id, "integration")
    functional_tests_analytics = get_functional_test_analytics(pipeline_id, "functional")
    failed_stage = get_failed_stage(pipeline_id)
    analytics = {"mcore_analytics": "v0.2", "pipeline_id": pipeline_id}
    analytics["build_stage_failed"] = int(failed_stage == "build")
    analytics["unit_tests_stage_failed"] = int(failed_stage == "test")
    analytics["integration_tests_stage_failed"] = int(failed_stage == "integration_tests")
    if not build_analytics.empty:
        analytics["ci_started_at"] = build_analytics['started_at'].min()
        analytics["build_started_at"] = build_analytics['started_at'].min()
        analytics["build_finished_at"] = build_analytics['finished_at'].max()
        analytics["build_duration_total"] = (
            pd.Timestamp(build_analytics['finished_at'].max())
            - pd.Timestamp(build_analytics['started_at'].min())
        ).total_seconds()
        analytics["functional_tests_stage_failed"] = int(failed_stage == "functional_tests")
    if not unit_tests_analytics.empty:
        analytics["unit_tests_started_at"] = unit_tests_analytics['started_at'].min()
        analytics["unit_tests_finished_at"] = unit_tests_analytics['finished_at'].max()
        analytics["unit_tests_duration_total"] = (
            pd.Timestamp(unit_tests_analytics['finished_at'].max())
            - pd.Timestamp(unit_tests_analytics['started_at'].min())
        ).total_seconds()
    if not integration_tests_analytics.empty:
        analytics["integration_tests_started_at"] = integration_tests_analytics['started_at'].min()
        analytics["integration_tests_finished_at"] = integration_tests_analytics[
            'finished_at'
        ].max()
        analytics["integration_tests_duration_total"] = (
            pd.Timestamp(integration_tests_analytics['finished_at'].max())
            - pd.Timestamp(integration_tests_analytics['started_at'].min())
        ).total_seconds()
    if not functional_tests_analytics.empty:
        analytics["functional_tests_started_at"] = functional_tests_analytics['started_at'].min()
        analytics["functional_tests_finished_at"] = functional_tests_analytics['finished_at'].max()
        analytics["functional_tests_duration_total"] = (
            pd.Timestamp(functional_tests_analytics['finished_at'].max())
            - pd.Timestamp(functional_tests_analytics['started_at'].min())
        ).total_seconds()
    return pd.DataFrame([analytics])
@click.command()
@click.option("--pipeline-id", required=True, type=int, help="PipelineID")
def upload_statistics(pipeline_id: int):

third_party/Megatron-LM/megatron/core/dist_checkpointing/mapping.py
def is_main_replica(replica_id: ReplicaId):
def apply_factories(sharded_state_dict: ShardedStateDict):
def apply_factory_merges(
    x1: StateDict, x2: ShardedStateDict, key: Tuple[str, ...] = ()
) -> StateDict:
    """Apply merges defined by ShardedTensorFactories *in-place*.
    Args:
        x1 (StateDict):

third_party/Megatron-LM/megatron/training/async_utils.py
def init_persistent_async_worker():
def schedule_async_save(async_request: AsyncRequest):
def maybe_finalize_async_save(blocking: bool = False, terminate=False):
def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.
    Returns:
        bool: True if there is any ongoing async call.
    """
    return _async_calls_queue.get_num_unfinalized_calls() == 0
def reset_persistent_async_worker():

third_party/Megatron-LM/megatron/core/fusions/fused_mla_yarn_rope_apply.py
def _get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size):
def rotary_fwd_q_kernel(
    Q,
    COS,
    SIN,
    qk_head_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
def rotary_bwd_q_kernel(
    DO,
    COS,
    SIN,
    qk_head_dim,
    emb_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_q,
    stride_x_seq,
    stride_x_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
def fused_apply_mla_rope_for_q(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qk_head_dim: int,
    emb_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
):
def rotary_fwd_kv_kernel(
    KV,
    K_POS_EMB,
    O_KEY,
    O_VALUE,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_kv_seq,
    stride_kv_nheads,
    stride_emb_seq,
    stride_k_seq,
    stride_k_nheads,
    stride_v_seq,
    stride_v_nheads,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
def rotary_bwd_kv_kernel(
    dK,
    dV,
    dKV,
    dEMB,
    COS,
    SIN,
    emb_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    head_num: tl.constexpr,
    batch_size,
    seq_num,
    cu_seqlens_kv,
    stride_dk_seq,
    stride_dk_nheads,
    stride_dv_seq,
    stride_dv_nheads,
    stride_dkv_seq,
    stride_dkv_nheads,
    stride_demb_seq,
    cp_rank,
    cp_size,
    BLOCK_H: tl.constexpr,
):
def fused_apply_mla_rope_for_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    emb_dim: int,
    k_dim: int,
    v_dim: int,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    rotary_interleaved: bool = False,
):

third_party/Megatron-LM/megatron/training/inprocess_restart.py
def destroy_state():
def inprocess_restart(train, args):
def maybe_wrap_for_inprocess_restart(pretrain):
def maybe_force_nccl_backend_init(device_id):

third_party/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py
def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error
    message ``s`` since otherwise, it is swallowed.
    """
    if not cond:
        logger.error(s)
        logger.error(''.join(traceback.format_stack()))
        if raise_assertion_error:
            raise AssertionError(s)
def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.
    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
def _free_storage(tensor: torch.Tensor):
def _pad(number_to_be_padded: int, divisor: int) -> int:
    return int(math.ceil(number_to_be_padded / divisor) * divisor)
def build_data_parallel_buffer_index(
    elements: List[torch.Size],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    is_data_distributed: bool,
    ddp_config: DistributedDataParallelConfig,
    bucket_id: int = 0,
    chunk_size_factor: int = 1,
) -> Tuple[List[tuple], BucketIndex, ShardBucketIndex]:
    """
    Assuming that all input tensor elements contiguously compose a global
    buffer, give the index range of every tensor, the bucket in the buffer,
    and the (distributed) shard within the bucket. Note that the global bucket
    buffer is only temporarily allocated, but is abstractly tracked via indices
    deduced from the number of raw parameters assigned to this buffer / bucket.
    Args:
        elements (List[torch.Size]):
def _get_dp_buffer_shard_bucket_index(
    bucket_index: BucketIndex,
    is_data_distributed: bool,
    data_parallel_world_size: int,
    data_parallel_rank: int,
) -> ShardBucketIndex:
    """
    Build the data parallel buffer shard bucket index from the bucket index.
    Args:
        bucket_index (BucketIndex):
def _get_parameter_groups(
    module: torch.nn.Module,
    policy: BucketingPolicy,
    meta_device_init_fp8_params: dict,
    bucket_group_by_fsdp_unit: bool = True,
):
def gradient_reduce_preprocessing(grad_data, scaling_factor, ddp_config):
def check_gpu_memory(threshold=0.9):
def override_sharded_param_methods_with_safety_checks(params, all_gather_pipeline):
def _dtype_size(dtype: torch.dtype) -> int:
    """
    Get the size of the dtype.
    Args:
        dtype (torch.dtype):
def to_local_if_dtensor(tensor):
def _get_fsdp_tensor_spec(
    param, dist_index: FSDPDistributedIndex, is_sharded_param, is_expert_param
):
def make_fsdp_dtensor(
    local_tensor: torch.Tensor,
    param: torch.nn.Parameter,
    dist_index: FSDPDistributedIndex,
    is_sharded_param: bool = True,
    is_expert_param: bool = False,
    run_check: bool = False,
    update_uneven_dtensor_chunk_meta: bool = False,
    force_sync_tp_duplicated_param: bool = False,
):

third_party/Megatron-LM/tests/test_utils/python_scripts/launch_nemo_run_workload.py
def is_flaky_failure(concat_allranks_logs: str) -> bool:
    """Assumes that certain keywords hint towards intermittent failures"""
    return (
        "The server socket has failed to listen on any local network address."
        in concat_allranks_logs
        or "Some NCCL operations have failed or timed out." in concat_allranks_logs
        or "uncorrectable ECC error encountered" in concat_allranks_logs
        or "illegal memory access" in concat_allranks_logs
        or "illegal instruction" in concat_allranks_logs
        or "torch.distributed.DistNetworkError" in concat_allranks_logs
        or "Segmentation fault" in concat_allranks_logs
        or "found NaN in" in concat_allranks_logs
        or "For debugging consider passing CUDA_LAUNCH_BLOCKING=1" in concat_allranks_logs
        or "double free or corruption" in concat_allranks_logs
        or "Call to CUDA function failed." in concat_allranks_logs
        or "Connection reset by peer" in concat_allranks_logs
        or "invalid pointer" in concat_allranks_logs
        or "malloc():
def main(
    scope,
    model,
    test_case,
    environment,
    platform,
    container_image,
    data_dir: Optional[str] = None,
    tag: Optional[str] = None,
    enable_lightweight_mode: Optional[bool] = False,
):

third_party/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py
def _bias_dropout_add_func(x_with_bias, residual, prob, training):
def bias_dropout_add_unfused(training):
def bias_dropout_add_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, True)
@jit_fuser
def bias_dropout_add_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return _bias_dropout_add_func(x_with_bias, residual, prob, False)
def get_bias_dropout_add(training, fused):

third_party/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py
def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.
    Args:
        x (torch.Tensor):
def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
def init_hybrid_ep_buffer(
    group: torch.distributed.ProcessGroup,
    hidden_dim: int,
    seq_len: int,
    num_local_experts: int,
    num_sms_dispatch_api: int,
    num_sms_combine_api: int,
    fp8_dispatch: bool,
) -> None:
    '''
    Initialize the HybridEP buffer, including buffer allocation and metadata
    initialization.
    If a runtime dispatch/combine requires a larger buffer than the one
    initialized, the buffer will be reallocated at runtime,
    incuring extra run-time overhead.
    Args:
        group (torch.distributed.ProcessGroup):
def reset_hybrid_ep_buffer():

third_party/Megatron-LM/tests/test_utils/python_scripts/notify.py
def get_gitlab_handle():
def get_jobs_per_bridge(pipeline_id: int, type_of_job: str):
def main(pipeline_id: int, check_for: str, pipeline_context: str, pipeline_created_at: str):

third_party/Megatron-LM/megatron/core/dist_checkpointing/optimizer.py
def get_optim_param_to_id_map(optim_params_iter: Iterable[torch.nn.Parameter]) -> Dict[int, int]:
    """Generate mapping from optimizer param to optimizer state id."""
    param_mappings = {}
    for i, param in enumerate(optim_params_iter):
def get_param_id_to_sharded_param_map(
    model_sharded_state_dict: ShardedStateDict, optim_params_iter: Iterable[torch.nn.Parameter]
) -> Dict[int, Union[ShardedTensor, ShardedTensorFactory]]:
    """Generate mapping from optimizer state ids to model sharded parameters.
    Args:
        model_sharded_state_dict: sharded state dict with all model sharded tensors
            (can have any structure)
        optim_params_iter: iterable which iterates over model parameters tracked by the optimizer.
            The iteration must be in the same order as in the optimizer parameters.
    Returns:
        Dict[int, Union[ShardedTensor, ShardedTensorFactory]]: mapping from optimizer state ids
            to model sharded parameters.
    """
    model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(model_sharded_state_dict)
    id_to_sharded_param_map = {}
    param_to_id_map = get_optim_param_to_id_map(optim_params_iter)
    # If using PyTorch FSDP2 the values in model_sharded_state_dict would
    # have been converted to local tensors during initialization.
    # See the make_(tp)_sharded_tensor_for_checkpoint functions.
    for ten in nested_values(model_sharded_state_dict):
def make_sharded_optimizer_tensor(
    model_param: Union[ShardedTensor, ShardedTensorFactory], optim_param: torch.Tensor, prefix: str
) -> Union[ShardedTensor, ShardedTensorFactory]:
    """Build a ShardedTensor or ShardedTensorFactory for optimizer param based on model param
    Args:
        model_param (Union[ShardedTensor, ShardedTensorFactory]):
def optim_state_to_sharding_state(
    optim_state_dict: StateDict,
    id_to_sharded_param_map: Dict[int, ShardedTensor],
    exclude_keys: Tuple[str] = (),
):

third_party/Megatron-LM/megatron/core/dist_checkpointing/validation.py
def parse_strict_flag(strict: Union[str, StrictHandling]) -> StrictHandling:
    """Parse user passed strict flag from a string to StrictHandling instance.
    Args:
        strict (str, StrictHandling):
def validate_integrity_and_strict_load(
    sharded_state_dict: ShardedStateDict,
    strict: StrictHandling,
    validate_access_integrity: bool,
    local_metadata: Optional[_LocalMetadata] = None,
    global_metadata: Optional[_GlobalMetadata] = None,
    ckpt_sharded_metadata: Optional["CkptShardedMetadata"] = None,
) -> Tuple[ShardedStateDict, Set[str], Set[str]]:
    """Validates sharding integrity and potential mismatches with the checkpoint.
    `validate_access_integrity` controls sharding integrity check (orthogonal
    to strictness checking) which verifies `sharded_state_dict` runtime completeness
    (in isolation from the actual checkpoint).
    `strict` flag controls handling of mismatches between the requested
    sharded state dict to load and the actual checkpoint. See `StrictHandling`
    docs for details regarding flag behavior and performance implications
    (disk interactions or inter-rank communication).
    Args:
        sharded_state_dict (ShardedStateDict):
def verify_checkpoint_and_load_strategy(
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
) -> Tuple[LoadShardedStrategy, LoadCommonStrategy]:
    """Verifies if checkpoint metadata exists and matches given strategies.
    If no strategies are passed, they are determined based on the checkpoint metadata.
    Args:
        checkpoint_dir (str):
def adjust_non_strict_load(
    sharded_state_dict: ShardedStateDict, sharded_keys_to_remove: Set[str]
) -> ShardedStateDict:
    """Adjusts sharded state dict removing keys not existing in the checkpoint.
    Args:
        sharded_state_dict (ShardedStateDict):
def _determine_missing_and_unexpected_keys(
    ckpt_sharded_metadata: "CkptShardedMetadata",
    local_metadata: _LocalMetadata,
    global_metadata: Optional[_GlobalMetadata] = None,
) -> Tuple[Set[str], Set[str]]:
    """Determines load mismatches based on metadata.
    There is an asymmetry between "unexpected" and "missing" keys.
    Unexpected keys can be determined based only on local metadata.
    Missing keys must be based on global metadata, since other ranks might access
    different keys than the current rank.
    In consequence, the return value of this function is different on each rank:
    "missing_keys" are equal, but "unexpected_keys" might differ across ranks.
    Args:
        ckpt_sharded_metadata (CkptShardedMetadata):
def maybe_report_missing_and_unexpected_keys(
    missing_keys: Set[str], unexpected_keys: Set[str], raise_error: bool = True
) -> None:
    """Raises or logs an error in case missing or unexpected keys are non-empty.
    Args:
        missing_keys (Set[str]):
def _validate_common_state_dict(common_state_dict: CommonStateDict) -> None:
    """Validate consistancy across ranks for the common state dict
    We save the common state dict only on rank 0. We validate to make sure that the common dict is consistent across ranks before saving.
    Args:
        common_state_dict: The common state dict present in all ransk
    """
    if not torch.distributed.is_initialized():
def validate_sharding_integrity(
    global_metadata: _GlobalMetadata, common_state_dict: CommonStateDict = None
) -> None:
    """Validate if the ShardedTensors and ShardedObjects from multiple processes define correct sharding.
    Local ShardedTensors and ShardedObject metadata is exchanged with `torch.distributed.all_gather_object`
    and then process with global rank 0 checks if main replicas of the shards:
    - cover the whole global tensors
    - don't overlap
    Args:
        global_metadata (_GlobalMetadata):
def _validate_sharding_for_key(rank_sharding: List[Tuple[int, ShardedTensor]]):
def _compute_shards_access(rank_sharding):
def _validate_objects_for_key(sharded_objects: List[ShardedObject]):
def determine_global_metadata(
    sharded_state_dict: ShardedStateDict,
) -> Tuple[_LocalMetadata, _GlobalMetadata]:
    """Exchanges local metadata with `all_gather_object` to determine global metadata.
    Args:
        sharded_state_dict (ShardedStateDict):
def validate_sharded_objects_handling(
    sharded_strategy: Union[SaveShardedStrategy, LoadShardedStrategy],
    common_strategy: Union[SaveCommonStrategy, LoadCommonStrategy],
) -> None:
    """Checks if either of the passed strategies can handle sharded objects.
    Args:
        sharded_strategy (Union[SaveShardedStrategy, LoadShardedStrategy]):

third_party/Megatron-LM/megatron/core/transformer/moe/grouped_gemm_util.py
def grouped_gemm_is_available():
def assert_grouped_gemm_is_available():

third_party/Megatron-LM/tests/test_utils/python_scripts/download_golden_values.py
def main(pipeline_id: int, only_failing: bool):

third_party/Megatron-LM/megatron/core/fusions/fused_bias_swiglu.py
def swiglu(y):
def bias_swiglu(y, bias):
def weighted_swiglu(y, weights):
def swiglu_back(g, y):
def bias_swiglu_back(g, y, bias):
def weighted_swiglu_back(g, y, weights):
def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False):

third_party/Megatron-LM/megatron/core/fusions/fused_bias_geglu.py
def geglu(y):
def bias_geglu(bias, y):
def geglu_back(g, y):
def bias_geglu_back(g, y, bias):
def bias_geglu_impl(input, bias):
def quick_gelu(y: torch.Tensor) -> torch.Tensor:
    """Sigmoid approximation of gelu"""
    return y * torch.sigmoid(1.702 * y)
@jit_fuser
def quick_geglu(y: torch.Tensor, linear_offset: float = 0.0) -> torch.Tensor:
    """Performs Quick-GELU-based GEGLU activation : quick_gelu(y1) * (y2 + offset).
    Args:
        y: Input tensor split into two halves on the last dimension.
        linear_offset: Optional linear offset added to the second half before gating.
    Returns:
        Tensor after applying the GEGLU activation.
    """
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return quick_gelu(y_1) * (y_2 + linear_offset)
@jit_fuser
def weighted_quick_geglu(
    y: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0
) -> torch.Tensor:
    """Token-wise-weighted Quick-GEGLU activation.
    The weights tensor is expected to have the same first-dimension length as ``y`` and a trailing
    singleton dimension so that it broadcasts over the feature dimension.
    """
    dtype = y.dtype
    res = quick_geglu(y, linear_offset) * weights
    return res.to(dtype)
# gradient of sigmoid approximation of gelu
@jit_fuser
def quick_geglu_back(g, y, linear_offset: float = 0.0) -> torch.Tensor:
    """Backward helper for Quick-GEGLU.
    Args:
        g (torch.Tensor):
def weighted_quick_geglu_back(g, y, weights, linear_offset: float = 0.0):
def weighted_bias_quick_geglu(
    y: torch.Tensor, bias: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0
) -> torch.Tensor:
    """Token-wise weighted Quick-GEGLU activation with bias.
    Args:
        y: Input tensor before bias addition.
        bias: Bias tensor broadcastable to `y`.
        weights: Weight tensor with shape `[tokens, 1]` broadcasting over feature dim.
        linear_offset: Optional linear offset for the second half before gating.
    Returns:
        Activated tensor with same dtype as `y`.
    """
    dtype = y.dtype
    res = quick_geglu(y + bias, linear_offset) * weights
    return res.to(dtype)
@jit_fuser
def weighted_bias_quick_geglu_back(g, y, bias, weights, linear_offset: float = 0.0):
def weighted_bias_quick_geglu_impl(
    input, bias, weights, fp8_input_store=False, linear_offset=0.0, clamp_value=None
):

third_party/Megatron-LM/tests/test_utils/python_scripts/wait_for_resources.py
def get_gitlab_handle():
def ci_is_busy(pipeline, target_branch: str):
def main(pipeline_id, target_branch):

third_party/Megatron-LM/megatron/core/models/gpt/heterogeneous/heterogeneous_layer_specs.py
def _get_layer_norm(config: AttentionConfig | MLPConfig, use_te: bool, normalization: str):
def _get_qk_layernorm(use_te: bool, normalization: str):
def _get_heterogenous_attention_spec(
    attn_config: AttentionConfig, use_te: bool, qk_layernorm: bool, normalization: str
):
def _get_heterogenous_mlp_spec(mlp_config: MLPConfig, use_te: bool):
def _get_sharded_state_dict_keys_map(block_config: TransformerBlockConfig, use_te: bool):
def get_gpt_heterogeneous_layer_spec(
    config: HeterogeneousTransformerConfig,
    use_te: bool = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):

third_party/Megatron-LM/megatron/training/ft_integration.py
def get_rank_monitor_client() -> Optional[Any]:
    """Returns the underlying fault tolerance client instance
    Returns:
        RankMonitorClient: rank monitor client instance, or None if FT was not initialized
    """
    return _GLOBAL_RANK_MONITOR_CLIENT
def setup() -> None:
    """Initialize fault tolerance before initialize_megatron"""
    args = arguments.parse_args(ignore_unknown_args=True)
    if not args.enable_ft_package:
        return
    # Initialize fault tolerance
    from nvidia_resiliency_ext.fault_tolerance import RankMonitorClient
    if os.environ.get("RANK") == "0":
        print("FT: initializing...", flush=True)
    checkpoint_dir = args.save
    if not checkpoint_dir:
        raise ValueError("checkpointing save dir must be set to enable fault tolerance")
    if not os.path.exists(checkpoint_dir):
def on_training_step_start() -> None:
    """Should be called before each training step"""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        global _is_setup_section_open
        if _is_setup_section_open:
            rmon_cli.end_section("setup")
            _is_setup_section_open = False
        if _seen_tr_iters_cnt >= _NUM_WARMUP_ITERS:
            rmon_cli.start_section("step")
        # reset eval step index. we started training, so evaluation is done
        global _curr_eval_iter_idx
        _curr_eval_iter_idx = 0
def on_training_step_end() -> None:
    """Should be called after each training step"""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        global _seen_tr_iters_cnt
        if _seen_tr_iters_cnt >= _NUM_WARMUP_ITERS:
            rmon_cli.end_section("step")
        _seen_tr_iters_cnt += 1
def on_eval_step_start() -> None:
    """Should be called before each validation step"""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        global _is_setup_section_open
        if _is_setup_section_open:
            # setup section can be open if there were no training iters before evaluation
            rmon_cli.end_section("setup")
            _is_setup_section_open = False
        if _curr_eval_iter_idx >= _NUM_WARMUP_ITERS:
            rmon_cli.start_section("step")
def on_eval_step_end() -> None:
    """Should be called after each validation step"""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        global _curr_eval_iter_idx
        if _curr_eval_iter_idx >= _NUM_WARMUP_ITERS:
            rmon_cli.end_section("step")
        _curr_eval_iter_idx += 1
def on_checkpointing_start() -> None:
    """Should be called before each checkpoint-saving-related operation."""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        rmon_cli.start_section("checkpointing")
def on_checkpointing_end(is_async_finalization: bool) -> None:
    """Should be called after each checkpoint-saving-related operation.
    Args:
        is_async_finalization (bool):
def on_checkpoint_loaded(is_local_chkpt: bool) -> None:
    """Should be called after a checkpoint was loaded
    Args:
        is_local_chkpt (bool):
def shutdown() -> None:
    """Shutdowns fault folerance, updates the FT timeouts if possible"""
    global _GLOBAL_RANK_MONITOR_CLIENT
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        print_rank_0("FT: closing...")
        _maybe_update_timeouts(is_closing_ft=True)
        rmon_cli.shutdown_workload_monitoring()
        print_rank_0("FT: closed.")
    _GLOBAL_RANK_MONITOR_CLIENT = None
def _load_state_if_exists():
def _update_timeouts(selected_sections, calc_out_of_section):
def _maybe_update_timeouts(is_closing_ft=False):
def maybe_setup_simulated_fault() -> None:
    """Sets a simulated fault, based on `FT_SIM_FAULT_DESC` env variable.
    Simulated fault description format:
    rank_hung|rank_killed;rank_to_fail|"";base_delay
    NOTE: This if for FT testing only
    """
    simulated_fault_desc = os.environ.get('FT_SIM_FAULT_DESC', None)
    if not simulated_fault_desc:
        return
    fault_type: Any  # silence mypy
    rank_to_fail: Any  # silence mypy
    base_delay: Any  # silence mypy
    fault_type, rank_to_fail, base_delay = simulated_fault_desc.split(';')
    fault_type = fault_type.strip()
    rank_to_fail = rank_to_fail.strip()
    rank_to_fail = int(rank_to_fail) if rank_to_fail else None
    base_delay = float(base_delay.strip())
    rng = random.Random()
    print_rank_0(
        f"FT: Initializing simulated fault: {fault_type},"
        + f"rank to fail: {rank_to_fail}, base delay: {base_delay}"
    )
    # rank that simulates a fault can be explicitly specified in the `rank_to_fail` field
    # if not specified, it just picks a random rank
    rank = torch.distributed.get_rank()
    rand_rank = rng.randint(0, torch.distributed.get_world_size() - 1)
    rank_to_fail = rank_to_fail if rank_to_fail is not None else rand_rank
    rank_to_fail = torch.tensor([rank_to_fail], device=torch.cuda.current_device())
    torch.distributed.broadcast(rank_to_fail, 0)
    rank_to_fail = int(rank_to_fail.item())
    if rank != rank_to_fail:
        # this rank is not going to simulate a fault, nothing more to do
        return
    if fault_type == 'random':
        fault_type = rng.choice(['rank_killed', 'rank_hung'])
    if fault_type == 'rank_killed':
        target_pid = os.getpid()
    elif fault_type == 'rank_hung':
        target_pid = os.getpid()
    else:
        raise Exception(f"Unknown fault type {fault_type} expected one of: rank_killed, rank_hung.")
    # add some randomness to the delay
    delay = base_delay + 0.2 * rng.random() * base_delay
    print_rank_0(f"FT: Selected fault={fault_type}; target rank={rank_to_fail}; delay={delay}")
    def __fault_thread():

third_party/Megatron-LM/megatron/core/transformer/cuda_graphs.py
def is_graph_capturing():
def _set_capture_start():
def _set_capture_end():
def _check_supported_type(meta):
def _determine_if_transformer_decoder_layer(base_module):
def _determine_if_first_last_layer_of_this_vp_chunk(base_module):
def _clone_nested_tensors(value: Any) -> Any:
    """Recursively clone tensors inside nested containers."""
    if torch.is_tensor(value):
def _ensure_generator_state_is_cudagraph_safe(gen: torch.Generator) -> torch.Generator:
    """Make generator state safe for CUDA graph capture/replay.
    Generator state tensors can become inference tensors if created under `torch.inference_mode()`.
    CUDA graph capture may later attempt in-place updates on that state; this fails for inference
    tensors. Fix the generator *in-place* (preserving identity) by cloning its state outside
    inference mode and setting it back.
    """
    with torch.inference_mode(mode=False):
def create_cudagraphs():
def delete_cuda_graphs():
def _layer_is_graphable(layer, config):
def convert_schedule_table_to_order(num_warmup_microbatches, num_model_chunks, schedule_table):
def get_overlap_moe_expert_parallel_comm_order(order, num_layers_per_chunk, capture_wgrad_graph):

third_party/Megatron-LM/megatron/core/models/gpt/gpt_layer_specs.py
def get_gpt_layer_with_inference_spec(
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec to use inference optimized linear layers.
    Args:
        qk_layernorm (bool, optional):
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_kitchen: bool = False,
    use_te_activation_func: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).
    Args:
        num_experts (int, optional):
def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    use_kitchen: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.
    Args:
        num_experts (int, optional):
def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
):
def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )
    if use_te_op_fuser:
        if not is_te_min_version("1.13.0"):
def get_mlp_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func() if use_te_activation_func else None
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        module = TEFusedMLP if use_te_op_fuser else MLP
        if backend.fuse_layernorm_and_linear():
def get_gpt_decoder_layer_specs(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    if use_transformer_engine:
        layer_norm_impl = TENorm
        dense_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
        moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
    else:
        layer_norm_impl = LNImpl
        dense_layer_spec = get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
        moe_layer_spec = get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    layer_specs = get_gpt_decoder_layer_specs(
        config, use_transformer_engine, normalization, qk_l2_norm
    )
    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    if config.pipeline_model_parallel_layout is not None:
        layout = config.pipeline_model_parallel_layout
        assert isinstance(layout, PipelineParallelLayerLayout)
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]
    if use_transformer_engine:
        layer_norm_impl = TENorm
    else:
        layer_norm_impl = LNImpl
    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )
    return block_spec
def get_gpt_mtp_block_spec(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    use_transformer_engine: bool,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    if use_transformer_engine:
        backend: BackendSpecProvider = (
            KitchenSpecProvider(
                fallback=TESpecProvider(),
                use_kitchen_attention=config.use_kitchen_attention,
                kitchen_attention_backend=config.kitchen_attention_backend,
            )
            if config.use_kitchen
            else TESpecProvider()
        )
    else:
        backend = (
            KitchenSpecProvider(
                fallback=LocalSpecProvider(),
                use_kitchen_attention=config.use_kitchen_attention,
                kitchen_attention_backend=config.kitchen_attention_backend,
            )
            if config.use_kitchen
            else LocalSpecProvider()
        )
    return get_gpt_mtp_block_spec_for_backend(
        config=config, spec=spec, backend=backend, vp_stage=vp_stage, pp_rank=pp_rank
    )
def get_gpt_mtp_block_spec_for_backend(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    backend: BackendSpecProvider,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = get_mtp_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    if num_layers_to_build == 0:
        return None
    if isinstance(spec, TransformerBlockSubmodules):

third_party/Megatron-LM/megatron/core/fusions/fused_pad_routing_map.py
def _pad_routing_map_kernel(
    routing_map_ptr, output_ptr, num_tokens, pad_multiple: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
def fused_pad_routing_map(routing_map: torch.Tensor, pad_multiple: int) -> torch.Tensor:
    """Fused version of pad_routing_map.
    Args:
        routing_map (torch.Tensor):

third_party/Megatron-LM/megatron/core/dist_checkpointing/core.py
def check_is_distributed_checkpoint(checkpoint_dir):
def maybe_load_config(checkpoint_dir: str) -> Optional[CheckpointingConfig]:
    """Returns checkpoint config if `checkpoint_dir` is a distributed checkpoint and None otherwise
    Args:
        checkpoint_dir: checkpoint directory
    Returns:
        CheckpointingConfig (optional):
def save_config(config: CheckpointingConfig, checkpoint_dir: str):

third_party/Megatron-LM/tests/test_utils/python_scripts/check_status_of_main.py
def get_gitlab_handle():
def most_recent_pipeline(target_branch: str):
def is_pending(target_branch: str):
def main(target_branch: str, continuous: bool):

third_party/Megatron-LM/megatron/core/models/retro/decoder_spec.py
def get_retro_decoder_layer_te_spec(
    encoder_block_spec: typing.Union[ModuleSpec, TransformerBlockSubmodules, None] = None
) -> ModuleSpec:
    """Retro decoder TE spec (uses Transformer Engine components).
    A Retro decoder layer uses custom attention and bias-dropout-add operators
    to perform chunked-cross attention. Additionally, the first Retro decoder
    layer instantiates an entire encoder transformer block. As such, the decoder
    cross attention module takes an optional encoder block spec, which is only
    provided for the first Retro decoder layer.
    Args:
        encoder_block_spec (ModuleSpec):
def get_retro_decoder_layer_local_spec(
    encoder_block_spec: typing.Optional[ModuleSpec] = None,
) -> ModuleSpec:
    """Retro decoder local spec (uses Megatron-Core components).
    A Retro decoder layer uses custom attention and bias-dropout-add operators
    to perform chunked-cross attention. Additionally, the first Retro decoder
    layer instantiates an entire encoder transformer block. As such, the decoder
    cross attention module takes an optional encoder block spec, which is only
    provided for the first Retro decoder layer.
    Args:
        encoder_block_spec (ModuleSpec):
def get_retro_decoder_block_spec(
    config: RetroConfig,
    use_transformer_engine: bool,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Retro decoder block spec.
    Retro decoder block implementation details:
    - The retro decoder block consists of interleaved GPT layers
        and customized Retro decoder layers.
    - The Retro decoder layers are spaced three layers apart,
        and start on layer 6 or 9 (depending on the total number of layers).
    - The first decoder layer instantiates an encoder block,
        and it therefore passes in an encoder_block_spec.
    Args:
        config (RetroConfig):

third_party/Megatron-LM/megatron/core/fusions/fused_indices_converter.py
def _indices_to_multihot_kernel(
    indices_ptr,
    probs_in_indices_ptr,
    multihot_indices_ptr,  # bool
    probs_in_multihot_ptr,
    position_map_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
def _multihot_to_indices_kernel(
    probs_in_multihot_ptr,
    position_map_ptr,
    probs_indices_ptr,
    num_of_local_experts: tl.constexpr,
    num_of_local_experts_next_power_of_2: tl.constexpr,
    topk: tl.constexpr,
    topk_next_power_of_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
def fused_indices_to_multihot(indices, probs_indices, num_of_local_experts):

third_party/Megatron-LM/tests/test_utils/python_scripts/download_unit_tests_dataset.py
def download_and_extract_asset(assets_dir: Path) -> bool:
    """
    Download and extract an asset to the assets directory.
    Args:
        asset_url: URL to download the asset from
        asset_name: Name of the asset file
        assets_dir: Directory to extract the asset to
    Returns:
        bool: True if successful, False otherwise
    """
    for asset in ASSETS:
        asset_name, asset_url = asset.values()
        try:
            # Download the asset
            logger.info(f"  Downloading {asset_name}...")
            response = requests.get(asset_url, stream=True)
            response.raise_for_status()
            # Save to temporary file
            temp_file = assets_dir / asset_name
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
def main(repo, assets_dir):

third_party/Megatron-LM/megatron/core/transformer/fsdp_dtensor_checkpoint.py
def get_ep_layer_offset(num_experts: int | None = None) -> int:
    """
    Get the expert layer offset for the current model.
    Args:
        num_experts: Total number of experts in the model. If None, returns 0.
    Returns:
        The expert layer offset for the current EP rank.
    """
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    num_local_experts = num_experts // ep_size if num_experts else 0
    local_expert_offset = ep_rank * num_local_experts
    return local_expert_offset
def get_expert_index_from_key(key):
def handle_experts_in_state_dict(state_dict, num_experts: int | None = None):
def expert_param_local_key(key: str, num_experts: int | None = None) -> str:
    """Get the module parameter corresponding to the key.
    Args:
        key: The parameter key to process.
        num_experts: Total number of experts in the model. If None, no expert processing occurs.
    Returns:
        The local parameter key with adjusted expert indices.
    """
    local_expert_offset = get_ep_layer_offset(num_experts)
    expert_index = get_expert_index_from_key(key)
    if expert_index is not None:
        new_expert_index = expert_index - local_expert_offset
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
        if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
            new_key = key.replace(f'weight{expert_index}', f'weight{new_expert_index}')
        # SequentialMLP: index is between 'local_experts.' and next '.'
        elif 'mlp.experts.local_experts' in key:
            new_key = key.replace(
                f'local_experts.{expert_index}.', f'local_experts.{new_expert_index}.'
            )
        else:
            raise ValueError(f"Unexpected expert key format: {key}")
        key = new_key
    return key
def handle_swiglu_in_state_dict(model, model_state_dict, optimizer_state_dict):
def handle_fp8_extra_state_case(model_state_dict):
def flatten_state_dict(obj, parent_key="", sep="."):
def print_diff_in_state_dicts(state_dict_metadata, load_state_dict, limit=100):
def validate_loaded_state_dict(state_dict, checkpoint_path):
def get_global_unique_param_name(model_chunks, param):

third_party/Megatron-LM/megatron/core/transformer/module.py
def param_is_not_shared(param):
def conversion_helper(val, conversion):
def fp32_to_float16(val, float16_convertor):
def float16_to_fp32(val):

third_party/Megatron-LM/megatron/core/models/retro/utils.py
def get_config_path(project_dir: str) -> str:
    """Config copy stored within retro project dir."""
    return os.path.join(project_dir, "config.json")
def get_gpt_data_dir(project_dir: str) -> str:
    """Get project-relative directory of GPT bin/idx datasets."""
    return os.path.join(project_dir, "data")
# ** Note ** : Retro's compatibility between cross attention and Flash/Fused
#   Attention is currently a work in progress. We default to returning None for
#   now.
# def get_all_true_mask(size, device):
def get_all_true_mask(size, device):

third_party/Megatron-LM/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py
def _param_generator(cpu_optimizer):

third_party/Megatron-LM/megatron/training/one_logger_utils.py
def get_timestamp_in_ms():
def on_train_start(iteration, consumed_train_samples, train_samples, seq_length,
                   train_iters, save, async_save, log_throughput,
                   num_floating_point_operations_so_far):
def _produce_e2e_metrics(log_throughput=False, throughput=None):
def track_e2e_metrics(log_throughput=False, throughput=None):
def on_save_checkpoint_start(async_save):
def on_pretrain_start():
def track_config_flags(train_iters, skip_train, do_train, do_valid, do_test,
                           dataloader_type, retro_project_dir, retro_cyclic_train_iters):
def on_save_checkpoint_success(productive_metrics, async_save):
def on_save_checkpoint_end(save_checkpoint_duration, current_iteration, async_save):
def track_app_tag(batch_size, world_size, seq_length):
def finish():

third_party/Megatron-LM/megatron/core/models/gpt/fine_grained_callables.py
def weak_method(method):
def should_free_input(name, is_moe, config):
def build_transformer_layer_callables(layer: TransformerLayer):
def build_mtp_layer_callables(layer):
def build_layer_callables(layer):

third_party/Megatron-LM/tests/test_utils/python_scripts/auto_reminder.py
def retry_with_backoff(func, max_retries=5, initial_delay=1):
def get_gitlab_handle():
def get_recent_milestones(project):
def get_current_review_stage(mr):
def get_mcore_reviewers():
def get_days_in_stage(mr, stage):
def get_days_since(dt_str):
def get_required_reviewers(mr):
def get_priority(days_in_current_stage):
def get_slack_user_id(email):
def send_to_slack(message, dry_run=False):
def process_mrs(project, milestones, labels, dry_run=False):
def main(dry_run):

third_party/Megatron-LM/megatron/training/dist_signal_handler.py
def get_world_size():
def get_device(local_rank=None):
def all_gather_item(item, dtype, group=None, async_op=False, local_rank=None):

third_party/Megatron-LM/megatron/core/dist_checkpointing/exchange_utils.py
def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor"""
    return HAVE_TE_FLOAT8TENSOR and isinstance(tensor, Float8Tensor)
logger = logging.getLogger(__name__)
class ShardDistribution(NamedTuple):
def _shard_size(sh_ten: ShardedTensor):
def _get_empty_tensor_for_exchange(
    shard_id: _ShardId,
    needed_shards: Dict[_ShardId, ShardedTensor],
    unneeded_shards: Dict[_ShardId, ShardedTensor],
    loaded_tensors: Dict[_ShardId, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.device]]:
    """Determines the empty tensor to use for exchange.
    If shard_id is needed by this rank, it will be in the `unloaded_shards`.
    Otherwise, the metadata for this tensor can be found in `shard_to_metadata`
    Args:
        shard_id (_ShardId):
def distribute_shards_to_ranks(
    shard_to_ranks: Dict[T, List[int]],
    shard_to_size: Dict[T, int],
    num_ranks: int,
    cross_parallelization_group_loads: Set[T],
) -> Dict[T, int]:
    """Computes uniform distribution of workload across ranks, based on sizes.
    Currently, the assignment is greedy, based on:
    1. Cross-parallelization group dependencies (shards with main rank in another group
       are assigned at the end to make sure the distribution for load and save
       is as similar as possible).
    2. Secondly, the coverage of each shard
        (how many ranks the shard is available on; lower coverage is assigned first)
    3. Then, the size of each shard (larger size is assigned first)
    4. Finally, shard id for differentiation.
    Last step is added because we rely on the fact that
    the assignment is deterministic on all ranks.
    Args:
        shard_to_ranks (Dict[T, List[int]]):
def determine_main_replica_uniform_distribution(
    sharded_state_dict: ShardedStateDict,
    parallelization_group: torch.distributed.ProcessGroup,
    ignore_groups: bool = False,
) -> Optional[ShardDistribution]:
    """Computes the save distribution.
    Should be used in conjunction with `distribute_main_replicas_with_precomputed_distribution`
    which applies the computed save distribution.
    We rely on the fact that the assignment algorithm is deterministic on all ranks,
    so there is no extra communication needed after metadata exchange.
    Args:
        sharded_state_dict (ShardedStateDict):
def exchange_loaded_tensors_gather_rounds(
    loaded_tensors: Dict[_ShardId, torch.Tensor],
    unloaded_shards: Dict[_ShardId, ShardedTensor],
    shard_distribution: ShardDistribution = None,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Dict[_ShardId, torch.Tensor]:
    """Exchange the tensors loaded by different ranks with several all_gather calls.
    Groups tensors by dtype, divide tensors that will be exchanged into rounds
    and execute all_gather for tensors from each round.
    Note: the loading is distributed across ranks based on total loaded size
    in bytes, so there is no guarantee that number of rounds needed for each
    rank will be similar, which might result in a lot of almost empty
    all_gathers. The solution would be to group all tensors into a one
    bytes tensor and do a single all_gather (with similarly sized messages).
    Args:
        loaded_tensors (Dict[_ShardId, torch.Tensor]):
def exchange_loaded_tensors_gather_object(
    loaded_tensors: Dict[_ShardId, torch.Tensor],
    unloaded_shards: Dict[_ShardId, ShardedTensor],
    shard_distribution: ShardDistribution,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Dict[_ShardId, torch.Tensor]:
    """Exchange the tensors loaded by different ranks with a simple all_gather_object call.
    This version can be used for debugging purposes do to its simplistic
    implementation. Shouldn't be used if performance is important.
    Args:
        loaded_tensors (Dict[_ShardId, torch.Tensor]):
def exchange_loaded_objects_gather_object(
    loaded_objects: Dict[_ShardId, Any]
) -> Dict[_ShardId, Any]:
    """Exchange the objects loaded by different ranks with a simple all_gather_object call.
    Args:
        loaded_objects (Dict[_ShardId, Any]):
def exchange_loaded_tensors_broadcast(
    loaded_tensors: Dict[_ShardId, torch.Tensor],
    unloaded_shards: Dict[_ShardId, ShardedTensor],
    shard_distribution: ShardDistribution,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Dict[_ShardId, torch.Tensor]:
    """Exchange the tensors loaded by different ranks by a series of broadcasts.
    For each rank for each loaded tensor do a broadcast to the whole group.
    A reasonable tradeoff in terms of performance and simplicity.
    Args:
        loaded_tensors (Dict[_ShardId, torch.Tensor]):
def exchange_by_distribution(
    loaded_tensors: Dict[_ShardId, torch.Tensor],
    unloaded_shards: Dict[_ShardId, ShardedTensor],
    shard_distribution: ShardDistribution,
    parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
    exchange_algo="broadcast",
) -> Dict[_ShardId, torch.Tensor]:
    """Exchange tensors loaded by different ranks using the specified exchange_algo.
    Args:
        loaded_tensors (Dict[_ShardId, torch.Tensor]):

third_party/Megatron-LM/megatron/core/optimizer/optimizer.py
def _zero_grad_group_helper(
    group: List[torch.nn.Parameter], set_to_none: bool, use_decoupled_grad: bool = False
):
def _multi_tensor_copy_this_to_that(
    this: List[torch.Tensor], that: List[torch.Tensor], overflow_buf: Optional[torch.Tensor] = None
):

third_party/Megatron-LM/megatron/core/transformer/heterogeneous/linear_replacements.py
def _gather_from_tensor_parallel_region(x: Tensor, config: TransformerConfig) -> Tensor:
    if get_tensor_model_parallel_world_size() > 1:
        if config.sequence_parallel:
            # pad hidden dimension (last dimension) with zeros such that the valid data is placed in
            # indices [tp_rank * hidden/tp_size, (tp_rank+1) * hidden/tp_size),
            # and zeros fill the other parts.
            output_size = config.hidden_size
            output_size_per_partition = divide(output_size, get_tensor_model_parallel_world_size())
            pad_before = get_tensor_model_parallel_rank() * output_size_per_partition
            pad_after = output_size - pad_before - output_size_per_partition
            pad_shape = [0] * (x.ndim - 1) * 2 + [pad_before, pad_after]
            x = F.pad(x, pad_shape, "constant", 0)
            x = reduce_scatter_to_sequence_parallel_region(x)
        else:
            x = gather_from_tensor_model_parallel_region(x)
    return x
if HAVE_TE:
    class TELayerNormColumnParallelLinearGathered(TELayerNormColumnParallelLinear):

third_party/Megatron-LM/megatron/core/models/retro/encoder_spec.py
def get_retro_encoder_layer_te_spec() -> ModuleSpec:
    """Retro encoder TE spec (uses Transformer Engine components).
    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.
    Returns:
        A module spec if Transformer Engine modules.
    """
    spec = get_gpt_layer_with_transformer_engine_spec()
    spec.submodules.pre_cross_attn_layernorm = TENorm
    spec.submodules.cross_attention = ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={"attn_mask_type": AttnMaskType.padding},
        submodules=CrossAttentionSubmodules(
            linear_q=TEColumnParallelLinear,
            linear_kv=TEColumnParallelLinear,
            core_attention=TEDotProductAttention,
            linear_proj=TERowParallelLinear,
        ),
    )
    spec.submodules.cross_attn_bda = ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm = ModuleSpec(module=RetroEncoderLayerNorm, submodules=TENorm)
    spec.submodules.mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )
    return spec
def get_retro_encoder_layer_local_spec() -> ModuleSpec:
    """Retro encoder local spec (uses Megatron-Core components).
    A Retro encoder layer uses custom attention, bias-dropout-add, and layernorm
    operators to encode neighboring chunks that are retrieved from the chunk
    database. Each operator is responsible for iterating the retrieved chunks
    and processing them individually.
    Returns:
        A module spec if local modules.
    """
    spec = get_gpt_layer_local_spec()
    spec.submodules.pre_cross_attn_layernorm = LNImpl
    spec.submodules.cross_attention = ModuleSpec(
        module=RetroEncoderCrossAttention,
        params={"attn_mask_type": AttnMaskType.padding},
        submodules=CrossAttentionSubmodules(
            linear_q=ColumnParallelLinear,
            linear_kv=ColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=RowParallelLinear,
        ),
    )
    spec.submodules.cross_attn_bda = ModuleSpec(module=RetroEncoderBiasDropoutAdd)
    spec.submodules.pre_mlp_layernorm = ModuleSpec(module=RetroEncoderLayerNorm, submodules=LNImpl)
    spec.submodules.mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
    )
    spec.submodules.sharded_state_dict_keys_map = {
        'input_layernorm.': 'self_attention.linear_qkv.layer_norm_'
    }  # pre_mlp_layernorm doesn't need remapping
    return spec
def get_retro_encoder_block_spec(
    config: RetroConfig, use_transformer_engine: bool
) -> TransformerBlockSubmodules:
    """Retro encoder block spec.
    The retro encoder block consists of one customized Retro encoder layer
    (layer 1), and all of the following layers are standard GPT layers.
    Args:
      config (RetroConfig):

third_party/Megatron-LM/megatron/training/yaml_arguments.py
def env_constructor(loader, node):
def validate_yaml(args, defaults={}):
def _print_args(title, args):
def core_config_from_args(args, dataclass=TransformerConfig):
def _check_arg_is_not_none(args, arg):
def core_transformer_config_from_yaml(args, transfomer_key = "language_model"):
def load_yaml(yaml_path):

third_party/Megatron-LM/megatron/core/dist_checkpointing/serialization.py
def load(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[LoadCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
    strict: Union[str, StrictHandling] = StrictHandling.ASSUME_OK_UNEXPECTED,
) -> Union[StateDict, Tuple[StateDict, Set[str], Set[str]]]:
    """Loading entrypoint.
    In the steps below, the following verbs refer to corresponding objects:
    - load = load from checkpoint
    - extract = extract from sharded_state_dict
    - add = add to the final state dict
    Steps:
    1. Load common state dict and form the base of the result state dict
    2. Apply factories to sharded_state_dict
    3. Extract LocalNonPersistentObject and add
    4. (optional) Extract ShardedObjects, load and add
    5. Extract ShardedBase, load, apply factory merges and add
    Args:
        sharded_state_dict (ShardedStateDict):
def load_common_state_dict(checkpoint_dir: Union[str, Path]) -> StateDict:
    """Load common (non-sharded) objects state dict from the checkpoint.
    Args:
        checkpoint_dir (str):
def load_tensors_metadata(
    checkpoint_dir: str, sharded_strategy: Union[LoadShardedStrategy, None] = None
) -> CkptShardedMetadata:
    """Load tensors metadata from the checkpoint.
    Returns a dictionary similar to a sharded state dict, but note that
    the dictionary keys are simply ShardedTensor keys (contrary to the
    actual sharded state dicts where keys correspond to state dict keys).
    Dict values are ShardedTensors without any sharding (so, the only useful
    information is tensors global shape and dtype).
    Concrete implementation depends on the loading strategy. If no strategy is
    given, a default for a given backend is used.
    Args:
        checkpoint_dir (str):
def load_sharded_metadata(
    checkpoint_dir: str,
    sharded_strategy: Union[LoadShardedStrategy, None] = None,
    common_strategy: Union[LoadCommonStrategy, None] = None,
) -> CkptShardedMetadata:
    """Load sharded metadata from the checkpoint.
    Similar to `load_tensors_metadata`, but includes also ShardedObjects.
    Returns a dictionary similar to a sharded state dict, but note that
    the dictionary keys are simply ShardedTensor keys (contrary to the
    actual sharded state dicts where keys correspond to state dict keys).
    Dict values are ShardedTensors without any sharding (so, the only useful
    information is tensors global shape and dtype).
    Concrete implementation depends on the loading strategy. If no strategy is
    given, a default for a given backend is used.
    Args:
        checkpoint_dir (str):
def load_plain_tensors(checkpoint_dir: str) -> StateDict:
    """Load checkpoint tensors without any sharding and plain structure.
    NOTE: common state dict is NOT included.
    Args:
        checkpoint_dir (str):
def load_content_metadata(
    checkpoint_dir: Optional[str] = None, *, preloaded_state_dict: Optional[StateDict] = None
) -> Optional[dict]:
    """Load content metadata stored in the checkpoint with `save(..., content_metadata=...)`.
    Args:
        checkpoint_dir (str, optional):
def remove_sharded_tensors(checkpoint_dir: str, key_prefix: str):
def save(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Union[SaveShardedStrategy, Tuple[str, int], None] = None,
    common_strategy: Union[SaveCommonStrategy, Tuple[str, int], None] = None,
    validate_access_integrity: bool = True,
    async_sharded_save: bool = False,
    preprocess_common_before_consistancy_check: Optional[
        Callable[[CommonStateDict], StateDict]
    ] = None,
    content_metadata: Optional[dict] = None,
) -> Optional[AsyncRequest]:
    """Saving entrypoint.
    Extracts ShardedTensors from the given state dict. Rank 0 saves the
    "regular" part of the checkpoint to common torch file.
    The ShardedTensors are saved according to a strategy specified by the
    config.
    Steps:
    1. Apply factories
    2. Extract and discard LocalNonPersistentObject
    3. Extract all ShardedBase object
    4. Save all other objects to common.pt
    5. (optional) Extract and save ShardedObjects
    6. Save all ShardedBase objects
    7. Write metadata.json file with backend and version metadata.
    Step (6) can be performed asynchronously (see `async_sharded_save`), in this
    case the actual save is embodied in the returned async request and can be
    scheduled by the external caller. For async request, step (7) is added as
    one of the finalization functions, so that metadata.json is written only
    if the checkpoint is complete.
    Args:
        sharded_state_dict (ShardedStateDict):

third_party/Megatron-LM/megatron/training/arguments.py
def add_megatron_arguments(parser: argparse.ArgumentParser):
def parse_args(extra_args_provider=None, ignore_unknown_args=False):
def validate_model_config_args_from_heterogeneous_config(args):
def load_retro_config(retro_project_dir):
def load_retro_args(args):
def _eval_pattern(pattern):
def no_rope_freq_type(x):
def moe_freq_type(x):
def la_freq_type(x):
def tuple_type(x):
def validate_args(args, defaults={}):
def _print_args(title, args):
def _check_arg_is_not_none(args, arg):
def core_transformer_config_from_args(args, config_class=None):
def _add_transformer_engine_args(parser):
def _add_inference_args(parser):
def _add_retro_args(parser):
def _add_network_size_args(parser):
def _add_straggler_detector_args(parser):
def _add_workload_inspector_server_args(parser):
def _add_inprocess_restart_args(parser):
def _add_one_logger_args(parser):
def _add_ft_package_args(parser):
def _add_config_logger_args(parser):
def _add_logging_args(parser):
def _add_regularization_args(parser):
def _add_rl_args(parser):
def _add_training_args(parser):
def _add_rerun_machine_args(parser):
def _add_initialization_args(parser):
def _add_learning_rate_args(parser):
def _add_checkpointing_args(parser):
def _add_mixed_precision_args(parser):
def _add_distributed_args(parser):
def _add_validation_args(parser):
def _add_tokenizer_args(parser):
def _add_data_args(parser):
def _add_autoresume_args(parser):
def _add_biencoder_args(parser):
def _add_vision_args(parser):
def _add_moe_args(parser):
def _add_mla_args(parser):
def _add_experimental_attention_variant_args(parser):
def _add_heterogeneous_args(parser):
def _add_experimental_args(parser):
def _add_msc_args(parser):
def _add_kitchen_quantization_arguments(parser: argparse.ArgumentParser):
def _add_sft_args(parser):

third_party/Megatron-LM/megatron/training/wandb_utils.py
def _get_wandb_artifact_tracker_filename(save_dir: str) -> Path:
    """Wandb artifact tracker file records the latest artifact wandb entity and project"""
    return Path(save_dir) / "latest_wandb_artifact_path.txt"
def _get_artifact_name_and_version(save_dir: Path, checkpoint_path: Path) -> Tuple[str, str]:
    return save_dir.stem, checkpoint_path.stem
def on_save_checkpoint_success(checkpoint_path: str, tracker_filename: str, save_dir: str, iteration: int) -> None:
    """Function to be called after checkpointing succeeds and checkpoint is persisted for logging it as an artifact in W&B
    Args:
        checkpoint_path (str):
def on_load_checkpoint_success(checkpoint_path: str, load_dir: str) -> None:
    """Function to be called after succesful loading of a checkpoint, for aggregation and logging it to W&B
    Args:
        checkpoint_path (str):

third_party/Megatron-LM/megatron/core/optimizer/__init__.py
def get_standard_config_overrides(config: OptimizerConfig) -> Dict[ParamKey, ParamGroupOverride]:
    """Get standard config overrides for the optimizer, handling decoupled LR and common wd skips.
    Args:
        config (OptimizerConfig):
def _get_param_groups(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
) -> List[Dict]:
    """Create parameter groups for optimizer.
    Creates parameter groups from provided optimizer config object.
    NOTE There can be more than one match between a ParamKey and a parameter.
        What we do is merge all of the matching ParamKey overrides into a single ParamGroupOverride
        for that parameter and use that as the key for that parameter. Any parameters that get
        the same set of merged overrides will be mapped into the same parameter group.
    Args:
        model_chunks (List[MegatronModule]):
def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
    filter_fn: Callable,
    buffer_name: str,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.
    Args:
        model_chunks (List[MegatronModule]):
def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
    distributed_optimizer_instance_id: Optional[int] = 0,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.
    Args:
        config (OptimizerConfig):
def check_config_overrides_consistency(
    config: OptimizerConfig, config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]]
):
def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
    dump_param_to_param_group_map: Optional[str] = None,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.
    We use separate optimizers for expert parameters and non-expert parameters.
    Args:
        config (OptimizerConfig):

third_party/Megatron-LM/megatron/core/transformer/utils.py
def get_linear_layer(rows, columns, init_method, perform_initialization=True):
def get_default_causal_mask(sq: int) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input."""
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()
def get_sliding_window_causal_mask(sq, skv, window_size):
def attention_mask_func(attention_scores, attention_mask):
def gelu_impl(x):
def openai_gelu(x):
def erf_gelu(x):
def make_sharded_tensors_for_checkpoint(
    state_dict: StateDict,
    prefix: str,
    tensor_parallel_layers_axis_map: Optional[Dict[str, int]] = None,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    extra_state_suffix: str = '_extra_state',
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    dp_cp_group: Optional[torch.distributed.ProcessGroup] = None,
):
def make_sharded_object_for_checkpoint(
    obj: Any,
    key: str,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    replica_id: Union[None, int, Tuple[int, ...]] = None,
    **kwargs,
):
def _get_extra_state_offsets(
    sharded_offsets: Iterable[Tuple[int, int, int]]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Turns ShardedTensor offsets into offsets suitable for ShardedObject."""
    if sharded_offsets:
        sharded_offsets = sorted(sharded_offsets, key=itemgetter(0))  # sort by axis
        axis, extra_state_offset, extra_state_shape = zip(*sharded_offsets)
        assert list(axis) == list(
            range(len(axis))
        ), f'Expected contiguous axis for offsets: {sharded_offsets}'
    else:
        extra_state_shape = (1,)
        extra_state_offset = (0,)
    return extra_state_shape, extra_state_offset
def ensure_metadata_has_dp_cp_group(metadata: Optional[dict]) -> dict:
    """Ensure `metadata` is a dict containing `dp_cp_group` entry.
    If `metadata` is None, a new dict is returned with `dp_cp_group` set.
    If `metadata` is a dict and missing `dp_cp_group`, it is updated in-place.
    Otherwise, asserts that `dp_cp_group` exists.
    """
    if metadata is None:
        return {'dp_cp_group': parallel_state.get_data_parallel_group(with_context_parallel=True)}
    assert isinstance(metadata, dict), "metadata must be a dict with dp_cp_group as key"
    if 'dp_cp_group' not in metadata:
        metadata['dp_cp_group'] = parallel_state.get_data_parallel_group(with_context_parallel=True)
    return metadata
def sharded_state_dict_default(
    module: torch.nn.Module,
    prefix: str = '',
    sharded_offsets: Tuple[Tuple[int, int, int]] = (),
    metadata: Optional[dict] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> ShardedStateDict:
    """Provides implementation for sharded_state_dict method for non-MegatronModules.
    Tries to call `module.sharded_state_dict` when possible,
    otherwise uses regular state dict and assumes tensors are replicated across TP and DP.
    `keep_vars=True` is passed to module.state_dict so that optimizer states
    can be sharded later on.
    Args:
        module (torch.nn.Module):
def _init_sequence_parallel_cache(model, exclude_modules):
def set_model_to_sequence_parallel(model, set_to=False, exclude_modules=None):
def init_cuda_graph_cache(model):
def toggle_cuda_graphs(model, set_to="none", reset_cuda_graphs=True):
def is_layer_window_attention(
    window_size: Optional[Tuple[int, int]], window_attn_skip_freq: int | list, layer_number: int
) -> bool:
    # layer_number is 1-indexed
    if not window_size:
        return False
    if window_attn_skip_freq is None:
        return True
    if isinstance(window_attn_skip_freq, int):

third_party/Megatron-LM/megatron/core/ssm/triton_cache_manager.py
def _version_no_greater_than(version, version_limit):
def default_cache_dir():

third_party/Megatron-LM/megatron/training/checkpointing.py
def set_checkpoint_version(value):
def get_checkpoint_version():
def check_checkpoint_args(checkpoint_args):
def isfile(filename) -> bool:
    if MultiStorageClientFeature.is_enabled():
def ensure_directory_exists(filename, check_parent=True):
def get_checkpoint_name(checkpoints_path, iteration, release=False,
                        pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None,
                        expert_parallel=None, expert_rank=None,
                        return_base_dir=False, basename="model_optim_rng.pt"):
def get_load_checkpoint_path_by_args(args, load_arg="load"):
def get_distributed_optimizer_checkpoint_name(model_checkpoint_name):
def find_checkpoint_rank_0(checkpoints_path, iteration, release=False):
def get_checkpoint_tracker_filename(checkpoints_path):
def checkpoint_exists(checkpoints_path):
def read_metadata(tracker_filename):
def get_rng_state(ckpt_format: str, tp_group: torch.distributed.ProcessGroup, pp_group: torch.distributed.ProcessGroup) -> Union[List[Dict[str, Any]], ShardedObject]:
    """Collect rng state across data parallel ranks."""
    args = get_args()
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}
    rng_state_list = None
    if args.data_parallel_random_init and torch.distributed.is_initialized() and \
            mpu.get_data_parallel_world_size() > 1:
        rng_state_list = \
            [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list,
            rng_state,
            group=mpu.get_data_parallel_group())
    else:
        rng_state_list = [rng_state]
    if ckpt_format == "torch_dist":
        pp_rank = get_pg_rank(pp_group)
        pp_size = get_pg_size(pp_group)
        tp_rank = get_pg_rank(tp_group)
        tp_size = get_pg_size(tp_group)
        ep_size = mpu.get_expert_model_parallel_world_size()
        if ep_size > 1:
            # Shard RNG by PP, TP, DP when using expert parallelism.
            dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
            rng_state_list = ShardedObject(
                'rng_state',
                rng_state_list,
                (pp_size, tp_size, dp_size),
                (pp_rank, tp_rank, dp_rank),
                replica_id=0,
            )
        else:
            rng_state_list = ShardedObject(
                'rng_state',
                rng_state_list,
                (pp_size, tp_size),
                (pp_rank, tp_rank),
                replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
            )
    elif ckpt_format == "fsdp_dtensor":
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        rng_state_list = {
            f"({pp_rank}, {tp_rank})": rng_state_list
        }
    return rng_state_list
class CheckpointType(Enum):
def _build_sharded_state_dict_metadata(args: Namespace, dp_cp_group: Optional[torch.distributed.ProcessGroup] = None) -> dict:
    """Builds metadata used for sharded_state_dict versioning.
    The whole content metadata is passed to ``shared_state_dict`` model and optimizer methods
    and therefore affects only the logic behind sharded_state_dict creation.
    The content metadata should be minimalistic, ideally flat (or with a single nesting level)
    and with semantically meaningful flag names (e.g. `distrib_optim_sharding_type`).
    In particular, a simple integer (or SemVer) versioning flag (e.g. `metadata['version'] = 3.4`)
    is discouraged, because the metadata serves for all models and optimizers and it's practically
    impossible to enforce a linearly increasing versioning for this whole space.
    Args:
        args: Arguments namespace
        dp_cp_group: Data parallel + context parallel group (default: None, falls back to mpu API)
    """
    metadata = {}
    if args.use_distributed_optimizer and args.ckpt_format == "fsdp_dtensor":
        metadata['distrib_optim_sharding_type'] = 'fsdp_dtensor'
    if args.use_distributed_optimizer and args.ckpt_format != "fsdp_dtensor":
        if args.dist_ckpt_optim_fully_reshardable:
            metadata['distrib_optim_sharding_type'] = 'fully_reshardable'
            metadata['distrib_optim_fully_reshardable_mem_efficient'] = args.distrib_optim_fully_reshardable_mem_efficient
        else:
            metadata['distrib_optim_sharding_type'] = 'dp_reshardable'
    metadata['singleton_local_shards'] = False
    metadata['chained_optim_avoid_prefix'] = True
    # Add dp_cp_group to metadata. If not provided, fallback to global parallel state.
    if dp_cp_group is None:
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    metadata['dp_cp_group'] = dp_cp_group
    return metadata
def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far,
                    checkpointing_context=None, pipeline_rank=None, expert_rank=None, tensor_rank=None, pipeline_parallel=None, expert_parallel=None, non_persistent_ckpt=False,
                    train_data_iterator=None, preprocess_common_state_dict_fn = None, release=False, tp_group: Optional[torch.distributed.ProcessGroup] = None, pp_group: Optional[torch.distributed.ProcessGroup] = None, dp_cp_group: Optional[torch.distributed.ProcessGroup] = None):
def cleanup_old_non_persistent_checkpoint(save_dir, leave_ckpt_num=1, do_async=False):
def maybe_save_dataloader_state(train_iterator, iteration, dataloader_save_path):
def generate_state_dict(
    args,
    model,
    optimizer,
    opt_param_scheduler,
    rng_state,
    iteration=None,
    optim_sd_kwargs=None,
    model_sd_kwargs=None,
    rerun_state=None,
):
def preprocess_fsdp_dtensor_state_dict(args, raw_state_dict, model):
def _transpose_first_dim(t, num_splits, num_splits_first, model):
def fix_query_key_value_ordering(model, checkpoint_version):
def _get_non_persistent_iteration(non_persistent_global_dir, args, checkpointing_context=None):
def _load_non_persistent_base_checkpoint(
    non_persistent_global_dir,
    args,
    rank0,
    sharded_state_dict,
    non_persistent_iteration,
    checkpointing_context=None,
):
def _load_global_dist_base_checkpoint(
    load_dir, args, rank0, sharded_state_dict, iteration, release, checkpointing_context=None
):
def _get_checkpoint_format(checkpoint_name, args):
def _load_base_checkpoint(
    load_dir,
    args,
    rank0=False,
    sharded_state_dict=None,
    checkpointing_context=None,
):
def load_args_from_checkpoint(
    args, load_arg='load', checkpointing_context=None
):
def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, load_arg='load', strict=True,
                    checkpointing_context=None, skip_load_to_model_and_opt=False, tp_group: Optional[torch.distributed.ProcessGroup] = None, pp_group: Optional[torch.distributed.ProcessGroup] = None, dp_cp_group: Optional[torch.distributed.ProcessGroup] = None):
def _to_dtensor(wrapped_model, model_state_dict):
def load_biencoder_checkpoint(model, only_query_model=False,
                              only_context_model=False, custom_load_path=None):

third_party/Megatron-LM/megatron/core/datasets/object_storage_utils.py
def _remove_s3_prefix(path: str) -> str:
    """Remove the S3 prefix from a path
    Args:
        path (str):
def _is_s3_path(path: str) -> bool:
    """Ascertain whether a path is in S3
    Args:
        path (str):
def _remove_msc_prefix(path: str) -> str:
    """
    Remove the MSC prefix from a path
    Args:
        path (str):
def _is_msc_path(path: str) -> bool:
    """Checks whether a path is in MSC path (msc://profile/path/to/file)
    Args:
        path (str):
def _s3_download_file(client: S3Client, s3_path: str, local_path: str) -> None:
    """Download the object at the given S3 path to the given local file system path
    Args:
        client (S3Client):
def _s3_object_exists(client: S3Client, path: str) -> bool:
    """Ascertain whether the object at the given S3 path exists in S3
    Args:
        client (S3Client):
def is_object_storage_path(path: str) -> bool:
    """Ascertain whether a path is in object storage
    Args:
        path (str):
def get_index_cache_path(idx_path: str, object_storage_config: ObjectStorageConfig) -> str:
    """Get the index cache path for the given path
    Args:
        idx_path (str):
def parse_s3_path(path: str) -> Tuple[str, str]:
    """Parses the given S3 path returning correspsonding bucket and key.
    Args:
        path (str):
def get_object_storage_access(path: str) -> str:
    """Get the object storage access"""
    return "s3" if _is_s3_path(path) else "msc"
def dataset_exists(path_prefix: str, idx_path: str, bin_path: str) -> bool:
    """Check if the dataset exists on object storage
    Args:
        path_prefix (str):
def cache_index_file(remote_path: str, local_path: str) -> None:
    """Download a file from object storage to a local path with distributed training support.
    The download only happens on Rank 0, and other ranks will wait for the file to be available.
    Note that this function does not include any barrier synchronization. The caller (typically
    in blended_megatron_dataset_builder.py) is responsible for ensuring proper synchronization
    between ranks using torch.distributed.barrier() after this function returns.
    Args:
        remote_path (str):

third_party/Megatron-LM/megatron/core/optimizer/muon.py
def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.
    Args:
        config (OptimizerConfig):

third_party/Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa.py
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428
    Args:
        x: Input tensor (must be bfloat16).
    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)
class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""
    tracker = {}
    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.
    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.
    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.
    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)
    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)
    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)
    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask
    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)
    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )
    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()
    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff
    return indexer_loss
class DSAIndexerLossAutoScaler(torch.autograd.Function):
def unfused_dsa_fn(query, key, value, topk_indices, softmax_scale):

third_party/Megatron-LM/megatron/core/tokenizers/text/utils/build_tokenizer.py
def build_tokenizer(args):

third_party/Megatron-LM/tests/test_utils/python_scripts/recipe_parser.py
def resolve_cluster_config(cluster: str) -> str:
    if cluster == "dgxh100_eos":
        return "eos"
    if cluster == "dgxgb200_oci-hsg":
        return "oci-hsg"
    if cluster == "dgxa100_dracooci":
        return "draco-oci-iad"
    if cluster == "dgxa100_dracooci-ord":
        return "draco-oci-ord"
    if cluster == "dgxh100_coreweave":
        return "coreweave"
    if cluster == "ghci":
        return "ghci"
    raise ValueError(f"Unknown cluster {cluster} provided.")
def flatten_products(workload_manifest: dotdict) -> dotdict:
    """Flattens a nested dict of products"""
    workload_manifest.products = [
        dict(**dict(zip(inp.keys(), values)), **{"test_case": product["test_case"][0]})
        for product in (workload_manifest.products or [])
        if "products" in product
        for inp in product["products"]
        for values in itertools.product(*inp.values())
    ]
    return workload_manifest
def flatten_workload(workload_manifest: dotdict) -> List[dotdict]:
    """Flattens a workload with products into a list of workloads that don't have products."""
    workload_manifest = dict(workload_manifest)
    products = workload_manifest.pop("products")
    workload_manifests = []
    for product in products:
        workload = copy.deepcopy(workload_manifest)
        workload["spec"] = {k: v for k, v in workload["spec"].items() if k not in product.keys()}
        workload["spec"] = dict(**dict(workload["spec"].items()), **product)
        workload_manifests.append(dotdict(**workload))
    return workload_manifests
def set_build_dependency(workload_manifests: List[dotdict]) -> List[dotdict]:
    for workload_manifest in workload_manifests:
        workload_manifest.spec["build"] = workload_manifest.spec["build"].format(
            **dict(workload_manifest.spec)
        )
    return workload_manifests
def load_config(config_path: str) -> dotdict:
    """Loads and parses a yaml file into a JETWorkloadManifest"""
    with open(config_path) as stream:
        try:
            return dotdict(**yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            raise exc
def load_and_flatten(config_path: str) -> List[dotdict]:
    """Wrapper function for doing all the fun at once."""
    return set_build_dependency(
        flatten_workload(flatten_products(load_config(config_path=config_path)))
    )
def filter_by_test_case(workload_manifests: List[dotdict], test_case: str) -> Optional[dotdict]:
    """Returns a workload with matching name. Raises an error if there no or more than a single workload."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest["spec"]["test_case"] == test_case
    )
    if len(workload_manifests) > 1:
        logger.info("Duplicate test_case found!")
        return None
    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return None
    return workload_manifests[0]
def filter_by_scope(workload_manifests: List[dotdict], scope: str) -> List[dotdict]:
    """Returns all workload with matching scope."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest.spec["scope"] == scope
    )
    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []
    return workload_manifests
def filter_by_environment(workload_manifests: List[dotdict], environment: str) -> List[dotdict]:
    workload_manifests_copy = list(
        workload_manifest
        for workload_manifest in workload_manifests.copy()
        if (
            hasattr(dotdict(**workload_manifest["spec"]), "environment")
            and workload_manifest["spec"]["environment"] == environment
        )
    )
    if len(workload_manifests_copy) == 0:
        logger.info("No test_case found!")
        return []
    return workload_manifests_copy
def filter_by_platform(workload_manifests: List[dotdict], platform: str) -> List[dotdict]:
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if (
            hasattr(dotdict(**workload_manifest["spec"]), "platforms")
            and workload_manifest.spec["platforms"] == platform
        )
    )
    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []
    return workload_manifests
def filter_by_model(workload_manifests: List[dotdict], model: str) -> List[dotdict]:
    """Returns all workload with matching model."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if workload_manifest.spec["model"] == model
    )
    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []
    return workload_manifests
def filter_by_tag(workload_manifests: List[dotdict], tag: str) -> List[dotdict]:
    """Returns all workload with matching tag."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        if hasattr(dotdict(**workload_manifest["spec"]), "tag")
        and workload_manifest["spec"]["tag"] == tag
    )
    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []
    return workload_manifests
def filter_by_test_cases(workload_manifests: List[dotdict], test_cases: str) -> List[dotdict]:
    """Returns a workload with matching name. Raises an error if there no or more than a single workload."""
    workload_manifests = list(
        workload_manifest
        for workload_manifest in workload_manifests
        for test_case in test_cases.split(",")
        if workload_manifest["spec"]["test_case"] == test_case
    )
    if len(workload_manifests) == 0:
        logger.info("No test_case found!")
        return []
    return workload_manifests
def load_workloads(
    container_tag: str,
    n_repeat: int = 1,
    time_limit: int = 1800,
    tag: Optional[str] = None,
    environment: Optional[str] = None,
    platform: Optional[str] = None,
    test_cases: str = "all",
    scope: Optional[str] = None,
    model: Optional[str] = None,
    test_case: Optional[str] = None,
    container_image: Optional[str] = None,
    record_checkpoints: Optional[str] = None,
) -> List[dotdict]:
    """Return all workloads from disk that match scope and platform."""
    recipes_dir = BASE_PATH / ".." / "recipes"
    local_dir = BASE_PATH / ".." / "local_recipes"
    workloads: List[dotdict] = []
    build_workloads: List = []
    for file in list(recipes_dir.glob("*.yaml")) + list(local_dir.glob("*.yaml")):
def main(model: Optional[str], test_case: Optional[str]):

third_party/Megatron-LM/megatron/training/initialize.py
def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    parsed_args=None,
    store=None,
):
def _compile_dependencies():
def _initialize_tp_communicators():
def _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, store):
def _init_autoresume():
def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
def write_args_to_tensorboard():
def set_jit_fusion_options():
def _warmup_jit_function():

third_party/Megatron-LM/megatron/core/transformer/transformer_layer.py
def get_transformer_layer_offset(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
):

third_party/Megatron-LM/megatron/core/ssm/gated_delta_net.py
def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference
    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )
    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))
    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):

third_party/Megatron-LM/megatron/core/optimizer/qk_clip.py
def clip_qk(model, log_max_only=False) -> float:
    """
    Clip the QK attention logits to the threshold, recommended for Muon optimizer.
    Args:
        model: The model to clip the QK attention logits, a list of model chunks.
        log_only: Whether to only log the max attention logit, without updating the weights.
    Returns:
        The maximum attention logit, a float.
    """
    with torch.no_grad():

third_party/Megatron-LM/tests/test_utils/python_scripts/approve_merge_gate.py
def main():

third_party/Megatron-LM/megatron/core/transformer/spec_utils.py
def import_module(module_path: Tuple[str]):
def get_module(spec_or_module: Union[ModuleSpec, type], **additional_kwargs):
def build_module(spec_or_module: Union[ModuleSpec, type], *args, **kwargs):

third_party/Megatron-LM/megatron/core/ssm/mamba_hybrid_layer_allocation.py
def _allocate_auto(
    total_layers_count: int, target_attention_ratio: float, target_mlp_ratio: float
) -> list:
    # First, allocate attention (evenly spaced, starting and ending with mamba)
    attention_layers_count: int = round(total_layers_count * target_attention_ratio)
    mamba_layers_count: int = total_layers_count - attention_layers_count
    mamba_sections_count: int = attention_layers_count + 1
    mamba_section_length: float = mamba_layers_count / mamba_sections_count
    layer_type_list = [Symbols.MAMBA] * total_layers_count
    x: float = mamba_section_length
    for l in range(total_layers_count):
def _allocate_override(total_layers_count: int, override_pattern: str) -> list:
    layer_type_list = list(override_pattern)
    override_pattern_length = len(layer_type_list)
    if override_pattern_length != total_layers_count:
        raise ValueError(
            "The hybrid override pattern is the wrong "
            f"length: got {override_pattern_length}, expected "
            f"{total_layers_count}"
        )
    for l in layer_type_list:
        if l not in Symbols.VALID:
            raise ValueError(f"In hybrid override pattern, '{l}' is not one of {Symbols.VALID}")
    return layer_type_list
def _layer_counts_match(a: list, b: list) -> bool:
    for s in Symbols.VALID:
        if a.count(s) != b.count(s):
def allocate_layers(
    total_layers_count: int,
    target_attention_ratio: float,
    target_mlp_ratio: float,
    override_pattern: str = None,
) -> list:
    """Allocates layers according to the requested distribution of layer types."""
    assert total_layers_count > 0
    assert target_attention_ratio >= 0.0 and target_attention_ratio <= 1.0
    assert target_mlp_ratio >= 0.0 and target_mlp_ratio <= 1.0
    assert target_attention_ratio + target_mlp_ratio <= 1.0
    # Note: target_mamba_ratio = 1.0 - target_attention_ratio - target_mlp_ratio
    layer_type_list = _allocate_auto(total_layers_count, target_attention_ratio, target_mlp_ratio)
    if override_pattern is not None:
        layer_type_list_override = _allocate_override(total_layers_count, override_pattern)
        log_single_rank(logger, logging.INFO, "Using hybrid override pattern")
        if (target_attention_ratio > 0.0 or target_mlp_ratio > 0.0) and not _layer_counts_match(
            layer_type_list_override, layer_type_list
        ):
def get_layer_maps_from_layer_type_list(
    layer_type_list: List[str],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Returns maps from global layer index to the corresponding layer index
    for each layer type in [Attention, Mamba, MLP, MoE] given a layer type list.
    """
    layer_types = [Symbols.ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE]
    layer_maps = {layer_type: {} for layer_type in layer_types}
    for global_layer_idx, layer_type in enumerate(layer_type_list):

third_party/Megatron-LM/megatron/core/datasets/gpt_dataset.py
def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents
    Args:
        documents (numpy.ndarray):
def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle
    Args:
        num_samples (int):
def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):

third_party/Megatron-LM/tests/test_utils/python_scripts/auto_reminder_github.py
def main():

third_party/Megatron-LM/tests/test_utils/python_scripts/swap_pr_labels.py
def main():

third_party/Megatron-LM/megatron/core/tokenizers/text/libraries/tiktoken_tokenizer.py
def reload_mergeable_ranks(
    path: str, max_vocab: Optional[int] = None, num_special_tokens: Optional[int] = None
) -> Dict[bytes, int]:
    """
    Reload the tokenizer JSON file and convert it to Tiktoken format.
    Args:
        path (str):

third_party/Megatron-LM/megatron/core/resharding/utils.py
def _get_rank_in_group(global_rank: int, group_ranks: list[int]) -> int:
    try:
        return group_ranks.index(global_rank)
    except ValueError:
        raise ValueError(
            f"Rank {global_rank} not found in process group {group_ranks}. "
            f"This likely indicates a configuration mismatch."
        )
def _detect_expert_index_from_param_name(param_name: str) -> Optional[int]:
    """Extract expert index from parameter name for TEGroupedMLP per-expert tensors."""
    for part in param_name.split('.'):
def assign_ep_resolved_name_inplace(
    meta: ParameterMetadata, *, base_name: str | None = None
) -> None:
    """
    EP-only canonicalization for per-expert parameters.
    Under Expert Parallelism (EP), each rank owns a subset of experts with local indices
    (e.g., rank 1 has "weight0" locally, but it's actually global expert 4). The raw param
    name can't be used to match across source/destination because the same local name refers
    to different global experts on different ranks. This function remaps local expert indices
    to global indices in `resolved_name` and sets `global_expert_index`.
    Effects:
    - Sets meta.resolved_name (defaults to base_name/meta.name for non-EP).
    - Sets meta.global_expert_index for per-expert parameters; otherwise leaves it as None.
    """
    base = meta.name if base_name is None else base_name
    meta.resolved_name = base
    meta.global_expert_index = None
    if not meta.is_ep:
        return
    local_idx = _detect_expert_index_from_param_name(base)
    if local_idx is None:
        # Fused experts tensor: leave name as-is; TP planner will handle slicing
        return
    ep_group = meta.expert_parallel_group_ranks
    ep_size = len(ep_group)
    ep_local_rank = ep_group.index(meta.owner_rank)
    experts_per_rank = meta.num_experts // ep_size
    global_idx = ep_local_rank * experts_per_rank + local_idx
    meta.global_expert_index = global_idx
    # Replace trailing integer in "weightK"/"biasK" with global_idx
    parts = base.split('.')
    new_parts = []
    for p in parts:
        if p.startswith('weight') and len(p) > len('weight') and p[len('weight') :
def assign_resolved_name_inplace(
    meta: ParameterMetadata,
    *,
    layer_module_prefix_map: Mapping[str, str] | None = None,
    base_name: str | None = None,
) -> None:
    """Set meta.resolved_name so the planner can match the same weights across models.
    It rewrites PP layer indices to global layer indices (when layer_module_prefix_map is
    provided) and
    rewrites EP per-expert indices (weightK/biasK) to global expert indices.
    """
    name = meta.name if base_name is None else base_name
    if layer_module_prefix_map:
        name = _resolve_global_layer_number_in_name(name, layer_module_prefix_map)
    assign_ep_resolved_name_inplace(meta, base_name=name)
def _build_layer_module_prefix_map(module: torch.nn.Module) -> dict[str, str]:
    """Build a mapping local_module_prefix -> global_module_prefix for PP layer modules.
    Megatron assigns a global, 1-indexed layer_number to each transformer layer module at
    construction time (including PP/VPP/layout offsets). We convert that to the 0-indexed naming
    convention used in parameter names and build a map such as:
    - "decoder.layers.0" → "decoder.layers.16"  (if layer_number == 17)
    """
    prefix_map: dict[str, str] = {}
    for module_name, submodule in module.named_modules():
def _resolve_global_layer_number_in_name(
    name: str, layer_module_prefix_map: Mapping[str, str]
) -> str:
    """Rewrite a parameter name to use global layer indices (PP-aware).
    Given a parameter name like decoder.layers.0.self_attention..., this function rewrites
    the decoder.layers.0 prefix to the corresponding global layer index using the owning
    layer module's layer_number.
    Implementation:
    - Build a {local_prefix -> global_prefix} map once (outside the per-parameter loop).
    - Perform a longest-prefix match replacement so we only rewrite the module path portion.
    """
    if not layer_module_prefix_map:
        return name
    parts = name.split('.')
    for i in range(len(parts), 0, -1):

third_party/Megatron-LM/megatron/core/ssm/mamba_context_parallel.py
def _all_to_all_cp2hp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Perform AlltoAll communication on a context parallel group, transform the
    input tensor from shape
    [global-sequence/context-parallel-size, batch, local-hidden] to
    [global-sequence, batch, local-hidden/context-parallel-size].
    Args:
        input_ (torch.Tensor):
def _all_to_all_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Perform AlltoAll communication on a context parallel group, transform the
    input tensor from shape
    [global-sequence, batch, local-hidden/context-parallel-size] to
    [global-sequence/context-parallel-size, batch, local-hidden].
    Args:
        input_ (torch.Tensor):
def _undo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, packed_seq_params: Optional[PackedSeqParams] = None
) -> torch.Tensor:
    """
    Undoes the context parallel attention load balancing.
    For example (non-packed), for cp_size=3, converts 162534 to 123456 for
    sequential processing by the convolution and SSM.
    """
    if packed_seq_params is None:
        num_chunks_div_2 = cp_size
        num_chunks = num_chunks_div_2 * 2
        chunks = torch.chunk(input_, chunks=num_chunks, dim=0)
        order = [2 * i for i in range(num_chunks_div_2)] + [
            num_chunks - 2 * i - 1 for i in range(num_chunks_div_2)
        ]
        reordered_chunks = [chunks[i] for i in order]
        return torch.cat(reordered_chunks, dim=0)
    else:
        assert tex is not None and is_te_min_version("1.10.0"), (
            "Please update Transformer Engine to >= 1.10 to use "
            "Context Parallel with THD format data"
        )
        if packed_seq_params.cu_seqlens_q_padded is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        else:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        total_tokens = input_.size(0)
        assert total_tokens % cp_size == 0
        seqlen_per_rank = total_tokens // cp_size
        output = torch.empty_like(input_)
        for cp_rank in range(cp_size):
def _redo_attention_load_balancing(
    input_: torch.Tensor, cp_size: int, packed_seq_params: Optional[PackedSeqParams] = None
) -> torch.Tensor:
    """
    Redo the context parallel attention load balancing.
    For example (non-packed), for cp_size=3, converts 123456 to 162534 for
    efficient processing by attention.
    """
    if packed_seq_params is None:
        num_chunks_div_2 = cp_size
        num_chunks = num_chunks_div_2 * 2
        chunks = torch.chunk(input_, chunks=num_chunks, dim=0)
        order = [None] * num_chunks
        order[::2] = range(num_chunks_div_2)  # order[even]
        order[1::2] = reversed(range(num_chunks_div_2, num_chunks))  # order[odd]
        reordered_chunks = [chunks[i] for i in order]
        return torch.cat(reordered_chunks, dim=0)
    else:
        assert tex is not None and is_te_min_version("1.10.0"), (
            "Please update Transformer Engine to >= 1.10 to use "
            "Context Parallel with THD format data"
        )
        if packed_seq_params.cu_seqlens_q_padded is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q_padded
        else:
            cu_seqlens = packed_seq_params.cu_seqlens_q
        total_tokens = input_.size(0)
        assert total_tokens % cp_size == 0
        seqlen_per_rank = total_tokens // cp_size
        index = torch.empty(total_tokens, device=input_.device, dtype=torch.int32)
        for cp_rank in range(cp_size):

third_party/Megatron-LM/megatron/core/optimizer/clip_grads.py
def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    grad_stats_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Calculate the norm of gradients in fp32.
    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters.
    Arguments:
        grads_for_norm (Iterable[Tensor] or Tensor):
def clip_grad_by_total_norm_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    max_norm: Union[int, float],
    total_norm: float,
    use_decoupled_grad: bool = False,
):
def count_zeros_fp32(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grad_stats_parallel_group: torch.distributed.ProcessGroup,
    use_decoupled_grad: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Counts the number of zeros in gradients associated with the passed-in list of
    parameters.
    Args:
        parameters (Iterable[Tensor] or Tensor):

third_party/Megatron-LM/megatron/core/resharding/refit.py
def swap_model_weights(
    src_model: LanguageModule,
    target_model: LanguageModule,
    refit_method: Union[RefitBackendName, CopyService],
):
def reshard_model_weights(
    src_model: LanguageModule, target_model: LanguageModule, service: CopyService
):

third_party/Megatron-LM/tests/test_utils/python_scripts/generate_jet_trigger_job.py
def main(
    scope: str,
    environment: str,
    n_repeat: int,
    time_limit: int,
    test_cases: str,
    platform: Optional[str],
    cluster: Optional[str],
    partition: Optional[str],
    output_path: str,
    container_image: str,
    container_tag: str,
    dependent_job: str,
    record_checkpoints: str,
    slurm_account: str,
    tag: Optional[str] = None,
    run_name: Optional[str] = None,
    wandb_experiment: Optional[str] = None,
    enable_lightweight_mode: bool = False,
    enable_warmup: Optional[bool] = None,
):

third_party/Megatron-LM/megatron/core/datasets/utils.py
def compile_helpers():
def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization
    Args:
        weights (List[float]):
def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    # pylint: disable=line-too-long
    """Get the blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend
    from the blend list
    Args:
        blend (Optional[List[str]]):

third_party/Megatron-LM/megatron/core/resharding/execution.py
def execute_reshard_plan(
    plan: ReshardPlan,
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    service: CopyService,
) -> None:
    """
    Execute a reshard plan (from centralized controller).
    A communication service must be provided to abstract transport.
    Expected service API: submit_send(tensor, dest_rank), submit_recv(tensor, src_rank), run().
    """
    src_params = {name: p for name, p in src_module.named_parameters(recurse=True)}
    dst_params = {name: p for name, p in dst_module.named_parameters(recurse=True)}
    submit_send_with_id = getattr(service, "submit_send_with_id", None)
    submit_recv_with_id = getattr(service, "submit_recv_with_id", None)
    # Submit sends
    for op in plan.send_ops:
        src_param = src_params.get(op.param_name)
        if src_param is not None:
            src_view = src_param.data[op.my_slice].contiguous()
            if submit_send_with_id is not None and op.task_id is not None:
                submit_send_with_id(op.task_id, src_view, op.peer_rank)
            else:
                service.submit_send(src_view, op.peer_rank)
    # Submit recvs
    recv_writebacks: List[Tuple[torch.Tensor, torch.nn.Parameter, tuple[slice, ...]]] = []
    for op in plan.recv_ops:
        dst_param = dst_params.get(op.param_name)
        if dst_param is not None:
            dst_slice_view = dst_param.data[op.my_slice]
            recv_buffer = torch.empty_like(dst_slice_view.contiguous())
            if submit_recv_with_id is not None and op.task_id is not None:
                submit_recv_with_id(op.task_id, recv_buffer, op.peer_rank)
            else:
                service.submit_recv(recv_buffer, op.peer_rank)
            recv_writebacks.append((recv_buffer, dst_param, op.my_slice))
    # Execute
    logger.info(f"Executing {len(plan.send_ops)} sends + {len(plan.recv_ops)} recvs")
    service.run()
    dist.barrier()
    # Write back received buffers into their destination parameter slices
    for recv_buffer, dst_param, dst_slice in recv_writebacks:
        with torch.no_grad():

third_party/Megatron-LM/megatron/core/datasets/helpers.py
def build_sample_idx(
    sizes: numpy.ndarray,
    document_indices: numpy.ndarray,
    sequence_length: int,
    num_epochs: int,
    tokens_per_epoch: int,
    drop_last_partial_sequence: bool = True,
    add_extra_token_to_sequence: bool = True,
):

third_party/Megatron-LM/megatron/core/tokenizers/megatron_tokenizer.py
def _get_metadata_path(tokenizer_path: str) -> str:
    """
    Returns metadata file path.
    Args:
        tokenizer_path (str):

third_party/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_config.py
def parse_and_normalize_split(split: str) -> List[float]:
    """Parse the dataset split ratios from a string
    Args:
        split (str):
def convert_split_vector_to_split_matrix(
    vector_a: List[float], vector_b: Optional[List[float]] = None
) -> List[Optional[Tuple[float, float]]]:
    """Build the split matrix from one or optionally two contributing split vectors.
    Ex. a standard conversion:
    [0.99, 0.01, 0.0] -> [(0, 0.99), (0.99, 1.0), None]
    Ex. a conversion for Retro when Retro pretraining uses a [0.99, 0.01, 0.0] split and Retro
    preprocessing used a [0.98, 0.02, 0.0] split:
    [0.99, 0.01, 0.0], [0.98, 0.02, 0.0] -> [(0, 0.98), (0.99, 1.0), None]
    Args:
        vector_a (List[float]):

third_party/Megatron-LM/tests/test_utils/python_scripts/generate_local_jobs.py
def load_script(config_path: str) -> str:
    with open(config_path) as stream:
        try:
            return yaml.safe_load(stream)["spec"]["script"]
        except yaml.YAMLError as exc:
            raise exc
@click.command()
@click.option("--model", required=False, type=str, help="Filters all tests by matching model")
@click.option(
    "--scope", required=False, type=str, default="mr", help="Filters all tests by matching scope"
)
@click.option(
    "--test-case", required=False, type=str, help="Returns a single test-case with matching name."
)
@click.option(
    "--environment",
    required=True,
    type=str,
    help="Pass 'lts' for PyTorch 24.01 and 'dev' for a more recent version.",
)
@click.option(
    "--output-path",
    required=True,
    type=str,
    help="Directory where the functional test will write its artifacts to (Tensorboard logs)",
    default="/opt/megatron-lm",
)
@click.option(
    "--enable-lightweight-mode",
    is_flag=True,
    show_default=True,
    required=False,
    type=bool,
    default=False,
    help="Run 2-step smoke tests instead of full training",
)
@click.option(
    "--record-checkpoints",
    is_flag=True,
    show_default=True,
    required=False,
    type=bool,
    default=False,
    help="Save checkpoints, do not run pytest",
)
def main(
    model: Optional[str],
    scope: Optional[str],
    test_case: Optional[str],
    environment: str,
    output_path: str,
    enable_lightweight_mode: bool = False,
    record_checkpoints: bool = False,
):

third_party/Megatron-LM/megatron/core/datasets/retro/query/gpt_chunk_dataset.py
def build_gpt_chunk_datasets_from_gpt_datasets(
    project_dir: str, gpt_datasets: dict, sample_length: int, chunk_length: int
) -> dict:
    """Get train, valid, test GPT chunk datasets.
    Args:
        project_dir (str):

third_party/Megatron-LM/megatron/core/datasets/indexed_dataset.py
def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix
    Args:
        path_prefix (str):
def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix
    Args:
        path_prefix (str):

third_party/Megatron-LM/megatron/core/datasets/blended_megatron_dataset_builder.py
def _get_size_per_split_per_dataset(
    normalized_weights: List[float], target_size_per_split: List[int], surplus: float = 0.0
) -> List[List[int]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits
    Args:
        normalized_weights (List[float]):

third_party/Megatron-LM/megatron/core/ssm/mamba_mixer.py
def _split_tensor_factory(
    orig_sh_ten: ShardedTensor, split_sections: List[int], split_names: List[str], split_dim: int
) -> ShardedTensorFactory:
    """Builds a factory that splits a given ShardedTensor into several independent chunks."""
    assert isinstance(orig_sh_ten, ShardedTensor), type(orig_sh_ten)
    orig_sh_ten_no_data = orig_sh_ten.without_data()  # remove `data` reference
    if sum(split_sections) != orig_sh_ten_no_data.local_shape[split_dim]:
        raise ValueError(
            f"Split sections must cover the whole dimension size, "
            f"got {split_sections=} vs dimensions size "
            f"{orig_sh_ten_no_data.local_shape[split_dim]}"
        )
    assert not isinstance(
        split_sections, int
    ), "Splitting into predefined section sizes is supported (`split_sections` must be a list)"
    assert len(split_sections) == len(split_names), (len(split_sections), len(split_names))
    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
def _check_mamba_sequence_packing_support(
    for_inference_not_training: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Checks whether `causal_conv1d` and `mamba_ssm` support sequence packing."""
    if for_inference_not_training:
        # https://github.com/Dao-AILab/causal-conv1d/commit/d87608f78f87d1288a7821d9e6ff4b10a8d5bf07
        conv1d_min = "1.5.3.post1"
        # https://github.com/state-spaces/mamba/commit/4f77d5306e19f5c7ae37665a44c3e61e24cafcb5
        mamba_min = "2.2.6.post3"
    else:
        conv1d_min = "1.4.0"
        mamba_min = "2.0.0"
    if not is_causal_conv1d_min_version(conv1d_min):

third_party/Megatron-LM/megatron/core/datasets/retro/query/utils.py
def get_query_dir(project_dir: str) -> str:
    """Get root directory of all saved query data.
    Args:
        project_dir (str):
def get_neighbor_dir(project_dir: str, key: str, dataset: MegatronDataset) -> str:
    """Get directory containing neighbor IDs for a dataset (i.e., train, valid, or test).
    Args:
        project_dir (str):

third_party/Megatron-LM/megatron/core/datasets/retro/utils.py
def log_retro_rank_0(message: str) -> None:
    """Log on rank 0.
    Args:
        message (str):
def retro_makedir(config: RetroPreprocessingConfig, path: str) -> None:
    """Make a directory, conditional on not being in validation mode.
    Args:
        config (RetroPreprocessingConfig):
def extract_data_config(config: RetroPreprocessingConfig) -> MultiSplitGPTDatasetConfig:
    """Extract data config from dataset.
    Args:
        config (RetroPreprocessingConfig):
def get_num_chunks_per_sample(sample_length: int, chunk_length: int) -> int:
    """Compute seq_length // chunk_length.
    Args:
        sample_length (int):
def get_blocks(
    dirname: str, n_samples: int, block_size: int, validate: Optional[Callable] = None
) -> SimpleNamespace:
    """Divide range [0, num_samples) to sequence of block ranges.
    This is a core method within the concept of block processing. The idea
    is to divide a range (size n_samples) into a sequence of blocks. Each
    block corresponds to a file within 'dirname' with name
    '{start_idx}-{end_idx}.hdf5'. This method checks for the existence of
    these files, and returns two lists, one for existing blocks and one for
    missing blocks.
    Args:
        dirname (str):
def get_blocks_by_rank(
    dirname: str,
    n_samples: int,
    block_size: int,
    validate: Optional[Callable] = None,
    sample: Optional[float] = None,
    process_group: Optional[ProcessGroup] = None,
) -> SimpleNamespace:
    """Divide existing and missing blocks evenly across all ranks.
    See 'get_blocks()' above for description. The returned lists of existing and
    missing blocks are split evenly across ranks via interleaving. This way,
    each rank has a roughly equal number of blocks to process for a
    downstream operation.
    Args:
        dirname (str):

third_party/Megatron-LM/megatron/core/datasets/retro/index/validate.py
def validate_training_embeddings(config: RetroPreprocessingConfig) -> None:
    """Validate training embeddings.
    Steps:
    - Randomly sample subset of text dataset blocks.
    - Embed each block.
    - Compare against saved embeddings.
    Args:
        config (RetroPreprocessingConfig):
def validate_added_encodings(config: RetroPreprocessingConfig) -> None:
    """Validate added encodings.
    Steps:
    - Randomly sample subset of text dataset blocks.
    - Encode each block.
    - Compare against saved encodings.
    Args:
        config (RetroPreprocessingConfig):
def validate_index(config: RetroPreprocessingConfig) -> None:
    """Validate index.
    Validating index involves sequentially running stages above:
    - Validate trained index.
    - Validate filled index.
    Args:
        config (RetroPreprocessingConfig):

third_party/Megatron-LM/megatron/core/datasets/retro/query/query.py
def get_index(config: RetroPreprocessingConfig, ondisk: bool = False) -> "faiss.Index":
    """Read index from disk.
    Args:
        config (RetroPreprocessingConfig):
def embed_block(
    config: RetroPreprocessingConfig, gpt_dataset: GPTChunkDataset, block: dict
) -> np.ndarray:
    """Embed block of chunks.
    Args:
        config (RetroPreprocessingConfig):
def query_embeddings(
    config: RetroPreprocessingConfig,
    db_dataset: DBDataset,
    index: Index,
    embeddings: np.ndarray,
    chunk_id_range: range,
    sample_map: dict,
    n_chunks_per_sample: int,
    verbose: bool = True,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Query neighbors of a block of embeddings.
    Querying includes:
      - Query index for neighbor chunk IDs.
      - Filter chunk IDs that have the same document ID as the queried embedding.
    Args:
        config (RetroPreprocessingConfig):
def query_embedding_block(
    config: RetroPreprocessingConfig,
    db_dataset: DBDataset,
    index: Index,
    embeddings: np.ndarray,
    chunk_id_range: range,
    sample_map: dict,
    n_chunks_per_sample: int,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Query a block of embeddings.
    The block is broken into smaller sub-blocks, for easier tracking of progress.
    Both the raw neighbor IDs and the filtered neighbor IDs (i.e., chunks with the
    same document ID are removed) are collected.
    Args:
        config (RetroPreprocessingConfig):
def query_block_neighbors(
    config: RetroPreprocessingConfig,
    db_dataset: DBDataset,
    query_dataset: GPTChunkDataset,
    index: Index,
    block: dict,
) -> None:
    """Query neighbors of a dataset block (i.e., range).
    Args:
        config (RetroPreprocessingConfig):
def query_dataset_neighbors(
    config: RetroPreprocessingConfig,
    db_dataset: DBDataset,
    query_dataset: GPTChunkDataset,
    num_active_chunks: int,
    prefix: str,
    neighbor_dir: str,
    index: Index,
) -> None:
    """Query neighbors of each chunk within a dataset.
    Args:
        config (RetroPreprocessingConfig):
def query_neighbors(config: RetroPreprocessingConfig) -> None:
    """Query pretraining datasets (train & valid).
    Args:
        config (RetroPreprocessingConfig):

third_party/Megatron-LM/megatron/core/parallel_state.py
def get_nccl_options(pg_name, nccl_comm_cfgs):
def update_pg_timeout(
    timeout: timedelta, pg: Optional[torch._C._distributed_c10d.ProcessGroup] = None
):
def create_group(
    ranks=None,
    timeout=None,
    backend=None,
    pg_options=None,
    use_local_synchronization=False,
    group_desc=None,
):
def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    r"""Generate orthogonal parallel groups based on the parallel size and mask.
    Arguments:
        world_size (int):
def create_hierarchical_groups(
    rank,
    ranks,
    hierarchical_group_sizes,
    create_gloo_process_groups=False,
    pg_options=None,
    timeout=None,
    group_desc=None,
):
def create_hybrid_dp_cp_groups(rank, ranks, pg_options):
def default_embedding_ranks(pp_ranks):
def default_position_embedding_ranks(pp_ranks):
def overwrite_nccl_comm_cfgs(nccl_comm_cfgs, pg_name, key_value_pair):
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_comm_backend: Optional[str] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    hybrid_context_parallel: bool = False,
    expert_model_parallel_size: int = 1,
    num_distributed_optimizer_instances: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    create_gloo_process_groups: bool = True,
    high_priority_stream_groups: Optional[List[str]] = None,
    sharp_enabled_group: Optional[str] = None,
) -> None:
    """Initialize model data parallel groups.
    Args:
        tensor_model_parallel_size (int, default = 1):
def is_initialized():
def is_unitialized() -> bool:
    """Check if parallel state has been initialized
    Deprecated. Use is_initialized instead.
    """
    warnings.warn("is_unitialized is deprecated, use is_initialized instead", DeprecationWarning)
    return not is_initialized()
def model_parallel_is_initialized():
def get_model_parallel_group(check_initialized=True):
def get_tensor_model_parallel_group(check_initialized=True):
def get_pipeline_model_parallel_group(check_initialized=True):
def get_data_parallel_group(with_context_parallel=False, partial_data_parallel=False):
def get_data_parallel_group_gloo(with_context_parallel=False, partial_data_parallel=False):
def get_context_parallel_group(check_initialized=True):
def get_context_parallel_global_ranks(check_initialized=True):
def get_hierarchical_context_parallel_groups(check_initialized=True):
def get_hybrid_data_context_parallel_groups(check_initialized=True, group_size=None):
def get_embedding_group(check_initialized=True):
def get_position_embedding_group(check_initialized=True):
def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
def get_tensor_and_data_parallel_group(check_initialized=True, with_context_parallel=False):
def get_tensor_and_context_parallel_group(check_initialized=True):
def set_tensor_model_parallel_world_size(world_size):
def set_pipeline_model_parallel_world_size(world_size):
def set_virtual_pipeline_model_parallel_world_size(world_size):
def get_tensor_model_parallel_world_size():
def get_pipeline_model_parallel_world_size():
def set_tensor_model_parallel_rank(rank):
def set_pipeline_model_parallel_rank(rank):
def get_tensor_model_parallel_rank():
def get_pipeline_model_parallel_rank():
def is_pipeline_first_stage(ignore_virtual=True, vp_stage=None):
def is_pipeline_last_stage(ignore_virtual=True, vp_stage=None):
def is_rank_in_embedding_group(ignore_virtual=True, vp_stage=None):
def is_rank_in_position_embedding_group():
def get_virtual_pipeline_model_parallel_rank():
def set_virtual_pipeline_model_parallel_rank(rank):
def get_virtual_pipeline_model_parallel_world_size():
def get_tensor_model_parallel_src_rank():
def get_model_parallel_src_rank():
def get_data_parallel_src_rank(with_context_parallel=False):
def get_pipeline_model_parallel_first_rank():
def get_pipeline_model_parallel_last_rank():
def get_pipeline_model_parallel_next_rank():
def get_pipeline_model_parallel_prev_rank():
def get_data_parallel_world_size(with_context_parallel=False, partial_data_parallel=False):
def set_data_parallel_rank(rank):
def get_data_parallel_rank(with_context_parallel=False, partial_data_parallel=False):
def get_context_parallel_world_size():
def get_context_parallel_rank():
def get_tensor_and_context_parallel_world_size():
def get_tensor_and_context_parallel_rank():
def get_expert_model_parallel_group(check_initialized=True):
def get_expert_model_parallel_src_rank():
def get_expert_model_parallel_world_size():
def set_expert_model_parallel_world_size(world_size):
def get_expert_model_parallel_rank():
def set_expert_model_parallel_rank(rank):
def get_expert_tensor_parallel_group(check_initialized=True):
def get_expert_tensor_parallel_world_size():
def set_expert_tensor_parallel_world_size(world_size):
def get_expert_tensor_parallel_rank():
def set_expert_tensor_parallel_rank(rank):
def get_expert_tensor_and_model_parallel_group(check_initialized=True):
def get_expert_tensor_and_model_parallel_world_size():
def get_expert_tensor_and_model_parallel_rank():
def get_expert_tensor_model_pipeline_parallel_group(check_initialized=True):
def get_expert_data_parallel_group(check_initialized=True, partial_expert_data_parallel=False):
def get_data_modulo_expert_parallel_group(partial_expert_data_parallel=False):
def get_expert_data_parallel_group_gloo(partial_expert_data_parallel=False):
def get_expert_data_parallel_rank(partial_expert_data_parallel=False):
def get_expert_data_parallel_world_size(partial_expert_data_parallel=False):
def get_intra_distributed_optimizer_instance_group(check_initialized=True):
def get_inter_distributed_optimizer_instance_group(check_initialized=True):
def _set_global_memory_buffer():
def _set_global_symmetric_memory_buffer():
def get_global_memory_buffer():
def get_global_symmetric_memory_buffer():
def destroy_global_memory_buffer():
def destroy_global_symmetric_memory_buffer():
def get_all_ranks():
def destroy_model_parallel():

third_party/Megatron-LM/megatron/core/datasets/retro/index/utils.py
def get_index_dir(config: RetroPreprocessingConfig) -> str:
    """Create sub-directory for this index.
    Args:
        config (RetroPreprocessingConfig):
def num_samples_to_block_ranges(
    config: RetroPreprocessingConfig, num_samples: int
) -> List[Tuple[int, int]]:
    """Split a range (length num_samples) into sequence of block ranges
    of size block_size.
    Args:
        config (RetroPreprocessingConfig):
def get_training_data_root_dir(config: RetroPreprocessingConfig) -> str:
    """Get root directory for embeddings (blocks and merged data).
    Args:
        config (RetroPreprocessingConfig):
def get_training_data_block_dir(config: RetroPreprocessingConfig) -> str:
    """Get directory for of saved embedding blocks.
    Args:
        config (RetroPreprocessingConfig):
def get_training_data_block_paths(config: RetroPreprocessingConfig) -> List[str]:
    """Get paths to saved embedding blocks.
    Args:
        config (RetroPreprocessingConfig):
def get_training_data_merged_path(config: RetroPreprocessingConfig) -> str:
    """Get path to merged training embeddings.
    Args:
        config (RetroPreprocessingConfig):
def get_added_codes_dir(config: RetroPreprocessingConfig) -> str:
    """Get directory of saved encodings.
    Args:
        config (RetroPreprocessingConfig):
def get_added_code_paths(config: RetroPreprocessingConfig) -> List[str]:
    """Get paths to all saved encodings.
    Args:
        config (RetroPreprocessingConfig):

third_party/Megatron-LM/megatron/core/datasets/retro/query/retro_dataset.py
def get_retro_datasets(
    config: RetroConfig, gpt_datasets: dict, sample_length: int, eod_token_id: int
) -> Tuple[Optional[RetroDataset], Optional[RetroDataset], Optional[RetroDataset]]:
    """Get train, valid, test retro datasets.
    Args:
        config (RetroConfig):

third_party/Megatron-LM/tests/test_utils/python_scripts/launch_jet_workload.py
def register_pipeline_terminator(pipeline: jetclient.JETPipeline):
def launch_and_wait_for_completion(
    test_case: str,
    environment: str,
    n_repeat: int,
    time_limit: int,
    scope: str,
    container_image: Optional[str],
    container_tag: str,
    cluster: str,
    platform: str,
    account: str,
    record_checkpoints: str,
    partition: Optional[str],
    tag: Optional[str],
    run_name: Optional[str],
    wandb_experiment: Optional[str],
    enable_lightweight_mode: bool,
) -> jetclient.JETPipeline:
    cluster_config = {"account": account}
    if partition is not None:
        cluster_config["partition"] = partition
    n_submission_attempts = 0
    while n_submission_attempts < 3:
        try:
            pipeline = jetclient.JETClient(
                customer="mcore", gitlab_ci_token=os.getenv("RO_API_TOKEN"), env="prod"
            ).workloads.submit(
                workloads=[
                    jetclient.JETWorkloadManifest(**workload)
                    for workload in recipe_parser.load_workloads(
                        test_case=test_case,
                        n_repeat=n_repeat,
                        time_limit=(1200 if enable_lightweight_mode else time_limit),
                        tag=tag,
                        scope=scope,
                        container_image=container_image,
                        container_tag=container_tag,
                        platform=platform,
                        environment=environment,
                        record_checkpoints=record_checkpoints,
                    )
                ],
                config_id=f"mcore/{recipe_parser.resolve_cluster_config(cluster)}",
                custom_config={
                    "launchers": {cluster: cluster_config},
                    "executors": {
                        "jet-ci": {
                            "environments": {
                                cluster: {
                                    "variables": {
                                        "RUN_NAME": run_name or "",
                                        "ENABLE_LIGHTWEIGHT_MODE": str(
                                            enable_lightweight_mode
                                        ).lower(),
                                        "WANDB_API_KEY": os.getenv("WANDB_API_KEY") or "",
                                        "WANDB_EXPERIMENT": wandb_experiment or "",
                                        "RECORD_CHECKPOINTS": str(
                                            record_checkpoints == "true"
                                        ).lower(),
                                        "RO_API_TOKEN": os.getenv("RO_API_TOKEN") or "",
                                        "MCORE_REPO": os.getenv("CI_REPOSITORY_URL") or "",
                                        "MCORE_MR_COMMIT": os.getenv("MCORE_MR_COMMIT") or "",
                                        "MCORE_BACKWARDS_COMMIT": (
                                            os.getenv("MCORE_BACKWARDS_COMMIT") or ""
                                        ),
                                        "HF_HUB_CACHE": "/lustre/fsw/coreai_dlalgo_mcore/hf_hub",
                                        "TRANSFORMERS_OFFLINE": "1",
                                        "CLUSTER": cluster,
                                        "RUN_ID": str(uuid.uuid4()),
                                    }
                                }
                            }
                        }
                    },
                },
                wait_for_validation=True,
                max_wait_time=(60 * 60),
            )
        except (
            jetclient.clients.gitlab.GitlabAPIError,
            jetclient.facades.objects.util.WaitTimeExceeded,
        ) as e:
            logger.error(f"Faced {str(e)}. Waiting and retrying...")
            n_submission_attempts += 1
            time.sleep(2**n_submission_attempts * 5)
            continue
        if pipeline.get_status() == PipelineStatus.SUBMISSION_FAILED:
            n_submission_attempts += 1
            logger.info("Submission failed, attempt again (%s/3)", str(n_submission_attempts))
            continue
        break
    if n_submission_attempts == 3:
        sys.exit(1)
    register_pipeline_terminator(pipeline=pipeline)
    logger.info(
        "Pipeline triggered; inspect it here: https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/%s",
        pipeline.jet_id,
    )
    pipeline.wait(max_wait_time=60 * 60 * 24 * 7, interval=60 * 1, retries_on_error=3)
    logger.info(f"Pipeline terminated; status: {pipeline.get_status()}")
    return pipeline
def download_job_assets(logs: List[jet_log.JETLog], iteration: int = 0) -> Optional[pathlib.Path]:
    if not logs:
        logger.info("No logs found for download.")
        return None
    assets_base_path = (
        BASE_PATH / ".." / ".." / ".." / "results" / f"iteration={iteration}"
    ).resolve()
    for restart_idx, log in enumerate(logs):
def extract_torchrunlogs_to_string(logs_path: pathlib.Path) -> Dict[int, List[str]]:
    logs_dict = {}
    # Iterate through all restart folders
    for restart_dir in logs_path.glob("restart=*"):
def extract_main_log_to_string(logs_path: pathlib.Path) -> List[str]:
    logs = []
    # Iterate through all restart folders
    for restart_dir in logs_path.glob("restart=*"):
def parse_failed_job(logs: List[str]) -> Optional[bool]:
    for log_row in logs[::-1]:
        match = re.search(r"Job finished with status 'FAILED'", log_row)
        if match is not None:
            return True
    return False
def telemetrics_and_exit(
    success: bool, test_case: str, environment: str, pipeline_id: int, is_integration_test: bool
):
def is_flaky_failure(concat_allranks_logs: str) -> bool:
    return (
        "The server socket has failed to listen on any local network address."
        in concat_allranks_logs
        or "Some NCCL operations have failed or timed out." in concat_allranks_logs
        or "uncorrectable ECC error encountered" in concat_allranks_logs
        or "illegal memory access" in concat_allranks_logs
        or "illegal instruction" in concat_allranks_logs
        or "torch.distributed.DistNetworkError" in concat_allranks_logs
        or "Segmentation fault" in concat_allranks_logs
        or "found NaN in" in concat_allranks_logs
        or "For debugging consider passing CUDA_LAUNCH_BLOCKING=1" in concat_allranks_logs
        or "double free or corruption" in concat_allranks_logs
        or "Call to CUDA function failed." in concat_allranks_logs
        or "Connection reset by peer" in concat_allranks_logs
        or "invalid pointer" in concat_allranks_logs
        or "malloc():
def main(
    model: str,
    test_case: str,
    environment: str,
    n_repeat: int,
    time_limit: int,
    scope: str,
    account: str,
    partition: Optional[str],
    cluster: str,
    platform: str,
    container_tag: str,
    record_checkpoints: str,
    tag: Optional[str] = None,
    container_image: Optional[str] = None,
    run_name: Optional[str] = None,
    wandb_experiment: Optional[str] = None,
    enable_lightweight_mode: bool = False,
):

third_party/Megatron-LM/megatron/core/resharding/planner.py
def _build_descriptors_for_param(
    src_metadata: ParameterMetadata, dst_metadata: ParameterMetadata
) -> list[ShardingDescriptor]:
    """Construct sharding descriptors (currently TP) for this parameter based on actual layout.
    Guard TP descriptor with size conservation so we don't mis-classify replicated tensors.
    """
    descriptors: list[ShardingDescriptor] = []
    # TP descriptor: allow when either side participates in TP
    if src_metadata.is_tp or dst_metadata.is_tp:
        # Prefer destination partition_dim, else source
        tp_dim = dst_metadata.partition_dim if dst_metadata.is_tp else src_metadata.partition_dim
        src_tp_ranks = src_metadata.tensor_parallel_group_ranks
        dst_tp_ranks = dst_metadata.tensor_parallel_group_ranks
        if src_tp_ranks is None or dst_tp_ranks is None:
            # Not enough context to build TP descriptor
            return descriptors
        src_stride = src_metadata.partition_stride if src_metadata.is_tp else 1
        dst_stride = dst_metadata.partition_stride if dst_metadata.is_tp else 1
        # Size conservation check on partition dim
        src_world = len(src_tp_ranks)
        dst_world = len(dst_tp_ranks)
        src_local = src_metadata.shape[tp_dim]
        dst_local = dst_metadata.shape[tp_dim]
        if src_world * src_local != dst_world * dst_local:
            raise RuntimeError(
                f"Cannot build TP descriptor for {dst_metadata.name} dim{tp_dim}: "
                f"src_world*src_local={src_world}*{src_local} != {dst_world}*{dst_local}. "
                "This usually means the param is marked TP but is effectively replicated on that "
                "dim or partition_dim/metadata is inconsistent between source and destination."
            )
        descriptors.append(
            ShardingDescriptor(
                name="tp",
                dim=tp_dim,
                src_stride=src_stride,
                dst_stride=dst_stride,
                src_dim_ranks=src_tp_ranks,
                dst_dim_ranks=dst_tp_ranks,
            )
        )
    return descriptors
def _plan_multi_dim_lcm(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    descriptors: list[ShardingDescriptor],
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """
    TP-only planner using LCM tiling to support strides on source/destination.
    - Requires exactly one TP descriptor
    - Supports arbitrary integer strides (contiguous micro-tiles)
    """
    if not descriptors:
        return []
    if len(descriptors) != 1:
        raise NotImplementedError(
            f"{param_name}: _plan_multi_dim_lcm supports TP-only (one descriptor)"
        )
    if descriptors[0].name != "tp":
        raise NotImplementedError(f"{param_name}: _plan_multi_dim_lcm expects TP descriptor")
    d = descriptors[0]
    if my_global_rank not in d.dst_dim_ranks:
        return []
    src_shape = tuple(src_metadata.shape)
    dst_shape = tuple(dst_metadata.shape)
    dim = d.dim
    src_world = len(d.src_dim_ranks)
    dst_world = len(d.dst_dim_ranks)
    src_local = src_shape[dim]
    dst_local = dst_shape[dim]
    if src_world * src_local != dst_world * dst_local:
        raise RuntimeError(
            f"{param_name}: size mismatch on TP dim{dim} "
            f"(src_world={src_world}, src_local={src_local}, "
            f"dst_world={dst_world}, dst_local={dst_local})"
        )
    # LCM tiling with strides
    Ns = src_world * max(1, d.src_stride)
    Nd = dst_world * max(1, d.dst_stride)
    full_len = dst_local * dst_world
    g = math.gcd(Ns, Nd)
    L = (Ns // g) * Nd
    if full_len % L != 0:
        raise RuntimeError(
            f"{param_name}: TP dim{dim} full_len {full_len} not divisible by LCM {L} "
            f"(Ns={Ns}, Nd={Nd})"
        )
    unit = full_len // L  # micro-tile length
    cps = L // Ns  # micro-tiles per source segment
    cpd = L // Nd  # micro-tiles per destination segment
    seg_src = cps * unit  # contiguous length per source segment
    seg_dst = cpd * unit  # contiguous length per destination segment
    dst_local_rank = _get_rank_in_group(my_global_rank, d.dst_dim_ranks)
    ops: list[tuple[int, tuple[slice, ...], tuple[slice, ...]]] = []
    # Sweep destination segments owned by this rank (handle destination stride)
    for k in range(max(1, d.dst_stride)):
def _finalize_dp_transfers(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Return receiver-side transfer for a parameter that is not TP-sharded.
    This is reached when we cannot build a TP sharding descriptor for the parameter
    (i.e., it is effectively replicated with respect to sharding).  We use this when the
    destination and source mode have no TP or the parameter is replicted on all ranks
    such as layernorm. If the source and destination DP groups match, we return a local
    full-tensor copy; otherwise we pick a source rank from the source DP group in a
    deterministic round-robin manner based on the receiver's index in its destination DP group.
    """
    dst_dp_ranks = dst_metadata.data_parallel_group_ranks
    src_dp_ranks = src_metadata.data_parallel_group_ranks
    if my_global_rank not in dst_dp_ranks:
        return []
    my_dst_dp_rank = _get_rank_in_group(my_global_rank, dst_dp_ranks)
    dst_shape = dst_metadata.shape
    # Same DP layout - local copy
    if src_dp_ranks == dst_dp_ranks:
        full_slice = tuple(slice(None) for _ in range(len(dst_shape)))
        return [(my_global_rank, full_slice, full_slice)]
    # Different DP groups - use round-robin for load balancing
    src_global_rank = src_dp_ranks[my_dst_dp_rank % len(src_dp_ranks)]
    full_slice = tuple(slice(None) for _ in range(len(dst_shape)))
    return [(src_global_rank, full_slice, full_slice)]
def _determine_source_ranks_for_dst_param(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Route to dimension-specific planner based on parameter sharding type."""
    # Regular TP/DP planning with EP-resolved metadata
    descriptors = _build_descriptors_for_param(src_metadata=src_metadata, dst_metadata=dst_metadata)
    if descriptors:
        return _plan_multi_dim_lcm(
            param_name=param_name,
            src_metadata=src_metadata,
            dst_metadata=dst_metadata,
            descriptors=descriptors,
            my_global_rank=my_global_rank,
        )
    # DP / replicated fallback
    return _finalize_dp_transfers(param_name, src_metadata, dst_metadata, my_global_rank)
def build_centralized_reshard_plan(
    src_module: torch.nn.Module, dst_module: torch.nn.Module, num_experts: int = None
) -> ReshardPlan:
    """
    Centralized planning: Rank 0 builds complete plan for all ranks, then scatters.
    """
    my_global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Get process groups
    src_pg = getattr(src_module, "pg_collection", None)
    dst_pg = getattr(dst_module, "pg_collection", None)
    if src_pg is None or dst_pg is None:
        raise ValueError("Both modules must have pg_collection")
    # Gather param metadata from all ranks
    my_src_params = {name: p for name, p in src_module.named_parameters(recurse=True)}
    my_dst_params = {name: p for name, p in dst_module.named_parameters(recurse=True)}
    # Build PP layer prefix maps to be used for parameter name rewriting
    src_layer_prefix_map = _build_layer_module_prefix_map(src_module)
    dst_layer_prefix_map = _build_layer_module_prefix_map(dst_module)
    my_src_metadata = [
        extract_param_metadata(
            p,
            name,
            my_global_rank,
            src_pg,
            num_experts=num_experts,
            layer_module_prefix_map=src_layer_prefix_map,
        )
        for name, p in my_src_params.items()
    ]
    my_dst_metadata = [
        extract_param_metadata(
            p,
            name,
            my_global_rank,
            dst_pg,
            num_experts=num_experts,
            layer_module_prefix_map=dst_layer_prefix_map,
        )
        for name, p in my_dst_params.items()
    ]
    all_src_metadata_by_rank = [None] * world_size
    all_dst_metadata_by_rank = [None] * world_size
    dist.all_gather_object(all_src_metadata_by_rank, my_src_metadata)
    dist.all_gather_object(all_dst_metadata_by_rank, my_dst_metadata)
    # Parameter to metadata maps keyed by resolved_name
    src_param_metadata_by_rank = {}
    dst_param_metadata_by_rank = {}
    src_param_metadata: dict[str, list[ParameterMetadata]] = {}
    for rank_id, rank_metadata_list in enumerate(all_src_metadata_by_rank):

third_party/Megatron-LM/megatron/core/datasets/retro/db/utils.py
def get_db_dir(project_dir: str) -> str:
    """Sub-directory for DB data.
    Args:
        project_dir (str):
def init_indexed_dataset_infos(config: RetroPreprocessingConfig) -> List[Dict]:
    """Gather meta-info about each indexed dataset.
    The returned info array allows for easy access to the configuration, and
    helps remove ambiguity.
    Args:
        config (RetroPreprocessingConfig):
def get_indexed_dataset_infos_path(project_dir: str) -> str:
    """Path to indexed dataset meta-infos.
    Args:
        project_dir (str):
def save_indexed_dataset_infos(project_dir: str, indexed_dataset_infos: List[Dict]) -> None:
    """Save dataset order & meta-info.
    Args:
        project_dir (str):
def load_indexed_datasets(project_dir: str, indexed_dataset_infos: List[Dict]) -> None:
    """Loaded indexed datasets into memory-mapped datasets.
    Args:
        project_dir (str):
def get_indexed_dataset_infos(project_dir: str) -> List[Dict]:
    """Load indexed dataset meta-infos.
    Args:
        project_dir (str):
def get_individual_db_dir(project_dir: str, prefix: str) -> str:
    """Individual DB's directory.
    Args:
        project_dir (str):
def get_individual_db_paths(project_dir: str, prefix: str) -> List[str]:
    """Get paths of all database blocks of an individual dataset.
    Args:
        project_dir (str):
def get_individual_chunk_db(project_dir: str, ds_id: int, ds_info: dict) -> np.ndarray:
    """Load individual dataset's chunk DB.
    Args:
        project_dir (str):
def get_individual_doc_offsets(project_dir: str, ds_id: int, ds_info: dict) -> np.ndarray:
    """Load individual dataset's document offsets.
    Args:
        project_dir (str):
def get_merged_db_path_map(project_dir: str) -> dict:
    """Paths to merged datasets.
    Args:
        project_dir (str):
def get_merged_dataset(
    project_dir: str,
    chunk_length: int,
    eod_token_id: int,
    db_type: str,
    indexed_dataset_infos: Optional[List[Dict]] = None,
) -> DBDataset:
    """Get merged dataset.
    Args:
        project_dir (str):
def get_merged_sampled_dataset(
    project_dir: str,
    chunk_length: int,
    eod_token_id: int,
    indexed_dataset_infos: Optional[List[Dict]] = None,
) -> DBDataset:
    """Get sampled dataset (for training the vector index).
    Args:
        project_dir (str):
def get_merged_train_dataset(
    project_dir: str,
    chunk_length: int,
    eod_token_id: int,
    indexed_dataset_infos: Optional[List[Dict]] = None,
) -> DBDataset:
    """Get training dataset (for adding to the vector index).
    Args:
        project_dir (str):
def get_merged_valid_dataset(
    project_dir: str,
    chunk_length: int,
    eod_token_id: int,
    indexed_dataset_infos: Optional[List[Dict]] = None,
) -> DBDataset:
    """Get validation dataset (for testing the vector index).
    Args:
        project_dir (str):
def get_merged_datasets(project_dir: str, chunk_length: int, eod_token_id: int) -> dict:
    """Get all merged datasets.
    Args:
        project_dir (str):

third_party/Megatron-LM/megatron/core/datasets/retro/index/build.py
def get_empty_index_path(config: RetroPreprocessingConfig) -> str:
    """Path of empty index.
    Args:
        config (RetroPreprocessingConfig):
def get_block_nload(block_path: str, load_fraction: float) -> int:
    """Compute number of blocks to load.
    This is computed by multiplying the total number of available blocks with the
    fraction of blocks to load.
    Args:
        block_path (str):
def merge_embedding_blocks(config: RetroPreprocessingConfig) -> None:
    """Merge individual embedding blocks into a single binary mmap file.
    The embeddings are initially stored in block-sized (e.g., ~100k embeddings per
    block) HDF5 files. These individual block files must be merged into a single
    file before training, to be based as a numpy mmap array to the index.
    Args:
        config (RetroPreprocessingConfig):
def get_text_dataset_for_training(config: RetroPreprocessingConfig) -> GPTToTextDataset:
    """Convert GPT token chunk dataset to a text dataset for passing to the
    embedder.
    Args:
        config (RetroPreprocessingConfig):
def embed_training_chunks(config: RetroPreprocessingConfig) -> None:
    """Embed DB chunks.
    Store chunks in blocks on disk. These blocks will later be merged into
    a single dataset for training the index.
    Args:
        config (RetroPreprocessingConfig):
def train_on_embeddings(config: RetroPreprocessingConfig) -> None:
    """Train index on embedded DB chunks.
    Args:
        config (RetroPreprocessingConfig):
def remove_embeddings(config: RetroPreprocessingConfig) -> None:
    """Remove embeddings after training.
    Args:
        config (RetroPreprocessingConfig):
def _train_index(config: RetroPreprocessingConfig) -> None:
    """Train index on DB chunks.
    Args:
        config (RetroPreprocessingConfig):
def train_index(config: RetroPreprocessingConfig) -> None:
    """Entry point for training the index.
    We select whether to train a new index, or validate an existing index.
    Args:
        config (RetroPreprocessingConfig):
def get_text_dataset_for_adding(config: RetroPreprocessingConfig) -> GPTToTextDataset:
    """Convert GPT token chunk dataset to a text dataset for passing to the
    embedder.
    Args:
        config (RetroPreprocessingConfig):
def _add_to_index(config: RetroPreprocessingConfig) -> str:
    """Add DB chunks to index.
    Args:
        config (RetroPreprocessingConfig):
def add_to_index(config: RetroPreprocessingConfig) -> None:
    """Entry point for adding to the index.
    We select whether to add to a new index, or validate an existing index.
    Args:
        config (RetroPreprocessingConfig):
def build_index(config: RetroPreprocessingConfig) -> None:
    """Build index.
    Building index involves sequentially running stages above:
    - Train index (on sampled training chunks).
    - Add to index (on all training chunks).
    Args:
        config (RetroPreprocessingConfig):

third_party/Megatron-LM/megatron/core/datasets/retro/db/build.py
def build_partial_db(
    config: types.SimpleNamespace,
    dataset_idx: int,
    n_datasets: int,
    indexed_dataset: IndexedDataset,
    block_id: int,
    n_blocks: int,
    block: dict,
    proc_id: int,
    n_procs: int,
) -> Tuple[int, list, list, dict]:
    """Process a document index range of the indexed dataset.
    The chunk database is built in parallel blocks, since de-tokenizing &
    re-tokenizing for Bert-length computation is expensive. This method
    iterates each document and extracts sequential 'chunk-length' sequences
    from each document.
    Args:
        config (types.SimpleNamespace):
def build_block_db(
    config: RetroPreprocessingConfig,
    dataset_idx: int,
    n_datasets: int,
    indexed_dataset: IndexedDataset,
    n_procs: int,
    executor: ProcessPoolExecutor,
    n_missing_blocks: int,
    block_idx: int,
    block: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each document within block into consecutive retro_gpt_chunk_length size chunks.
    Args:
        config (RetroPreprocessingConfig):
def save_block_db(
    block: dict, chunk_db_valid: np.ndarray, chunk_db_invalid: np.ndarray, doc_offsets: np.ndarray
) -> None:
    """Save block of chunked tokens to disk. These blocks are later used for
    training and adding to the vector index.
    Args:
        block (dict):
def build_individual_db(
    config: RetroPreprocessingConfig, dataset_idx: int, n_datasets: int, dataset_info: dict
) -> None:
    """Process a single indexed dataset & extract chunks.
    Args:
        config (RetroPreprocessingConfig):
def build_individual_dbs(
    config: RetroPreprocessingConfig, indexed_dataset_infos: List[Dict]
) -> None:
    """Iterate each indexed dataset & process its chunks.
    Args:
        config (RetroPreprocessingConfig):
def update_chunk_counts(
    config: RetroPreprocessingConfig, indexed_dataset_infos: List[Dict]
) -> None:
    """Set n_chunks_train & n_chunks sampled for each individual DB.
    Args:
        config (RetroPreprocessingConfig):
def merge_dbs(project_dir: str, indexed_dataset_infos: List[Dict], db_type: str) -> None:
    """Merge individual DBs into single DB.
    Args:
        project_dir (str):
def build_merged_dbs(project_dir: str, indexed_dataset_infos: List[Dict]) -> None:
    """Merge individual dataset components into single database.
    This method merges databases for DB types:
    - 'sampled': used for training the vector index.
    - 'train': used for adding to the trained vector index.
    - 'valid': can be used for validating/testing the vector index.
    Args:
        project_dir (str):
def build_db(config: RetroPreprocessingConfig) -> None:
    """Extract token chunks from each indexed dataset.
    Iterate each document of each indexed dataset, extract that document's chunks,
        and save to a 'DB' (hdf5 file).
    Args:
        config (RetroPreprocessingConfig):
