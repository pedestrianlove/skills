third_party/nvidia-resiliency-ext/nvrx_build.py
def find_file_in_dir(cuda_path, sfile):
def _skip_ext_build():
def get_cuda_path():
def _compile_protos(proto_dir, proto_filenames):
def build(setup_kwargs):

third_party/nvidia-resiliency-ext/examples/inprocess/advanced_example.py
def parse_args():
def train(
    base_store,
    model,
    opt,
    backend,
    device,
    timeout,
    args,
    call_wrapper: Optional[inprocess.CallWrapper] = None,
):
def main():

third_party/nvidia-resiliency-ext/examples/inprocess/basic_example.py
def parse_args():
def main(call_wrapper: Optional[inprocess.CallWrapper] = None):

third_party/nvidia-resiliency-ext/examples/checkpointing/async_writer.py
def print_on_rank0(msg):
def parse_args():
def init_distributed_backend(backend="nccl"):
def get_save_and_finalize_callbacks(writer, save_state_dict_ret) -> AsyncRequest:
    """Creates an async save request with a finalize function."""
    save_fn, preload_fn, save_args = writer.get_save_function_and_args()
    def finalize_fn():
def save_checkpoint(checkpoint_dir, async_queue, model, thread_count, enable_msc):
def load_checkpoint(checkpoint_dir, model, thread_count, enable_msc):
def main():

third_party/nvidia-resiliency-ext/examples/checkpointing/async_ckpt.py
def parse_args():
def init_distributed_backend(backend="nccl"):
def cleanup(ckpt_dir):
def main():

third_party/nvidia-resiliency-ext/examples/checkpointing/local_ckpt.py
def parse_args():
def init_distributed_backend(backend="nccl"):
def create_checkpoint_manager(args):
def save(args, ckpt_manager, async_queue, model, iteration):
def load(args, ckpt_manager):
def main():

third_party/nvidia-resiliency-ext/examples/straggler/example.py
def train(args) -> None:
    print(args)
    straggler.Detector.initialize(gather_on_rank0=True)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    print(f"Running basic straggler det. DDP example on device {device}.")
    model = Model().to(device)
    ddp_model = DDP(model, device_ids=[local_rank])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    optim = torch.optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.5)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_num = 0
    ddp_model.train()
    total_iters_made = 0
    training_start_time = time.monotonic()
    while epoch_num < args.num_epochs:
        for batch_idx, (data, target) in enumerate(loader):

third_party/nvidia-resiliency-ext/examples/attribution/nvrx_attrsvc.py
def get_slack_user_email(userID: str, token: str) -> str | None:
    client = WebClient(token=token)
    try:
        # Fetch all users (pagination may be needed for very large teams)
        user_id = client.users_lookupByEmail(email=f"{userID}@nvidia.com").get("user")["id"]
        return user_id  # User not found
    except SlackApiError as e:
        logger.error(f"Error fetching user email: {e.response['error']}")
        return None
def send_slack_notification(data: dict, slack_bot_token: str, slack_channel: str):
def setup() -> "Settings":
    """
    Group environment configuration and logging setup for nvrx_attrsvc.
    Returns a configured Settings instance.
    """
    try:
        cfg = Settings()  # type: ignore[call-arg]
    except Exception as e:
        # Fail fast if required settings are missing or invalid
        raise SystemExit(f"nvrx_attrsvc configuration error: {e}")
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL_NAME, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    # Validate ALLOWED_ROOT
    allowed_root = os.path.realpath(cfg.ALLOWED_ROOT)
    if not os.path.isabs(allowed_root):
def create_app(cfg: Settings) -> FastAPI:
    """
    Construct and return the FastAPI app for the NVRX Attribution Service (nvrx_attrsvc).
    """
    app = FastAPI(
        title="NVRX Attribution Service",
        summary="nvrx_attrsvc - NVRX attribution service for artifact/log analysis",
        contact={
            "name": "NVRX Attribution Service",
            "email": "nvrx@nvidia.com",
        },
        root_path=cfg.FAST_API_ROOT_PATH,
        debug=cfg.DEBUG,
    )
    # Global exception handlers to standardize error bodies
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_, exc: HTTPException):
def _get_server_command() -> list[str]:
    """
    Resolve and return the server launcher command for the MCP client.
    """
    pkg = "nvidia_resiliency_ext.attribution.mcp_integration"
    try:
        resource = pkg_files(pkg).joinpath("server_launcher.py")
    except Exception as e:
        raise FileNotFoundError(f"failed to locate server_launcher.py in package {pkg}: {e}")
    if not resource.exists():
def _create_mcp_client() -> NVRxMCPClient:
    """
    Create and return an initialized NVRxMCPClient.
    """
    return NVRxMCPClient(_get_server_command())
def _normalize_and_validate_path(
    user_path: str, cfg: Settings, *, require_regular_file: bool
) -> str:
    """
    Normalize and validate an input path:
      - Must be absolute
      - Resolve realpath under allowed root (no traversal outside)
      - Must not be a symlink
      - If require_regular_file=True, must be a regular readable file; otherwise allow directories too
    Returns the normalized absolute path or raises HTTPException(400/404).
    """
    if not os.path.isabs(user_path):
def main():

third_party/nvidia-resiliency-ext/examples/fault_tolerance/log_utils.py
def setup_logging(log_all_ranks=True, filename=os.devnull, filemode='w'):

third_party/nvidia-resiliency-ext/examples/utils/node_health_check_example.py
def main():

third_party/nvidia-resiliency-ext/examples/fault_tolerance/dist_utils.py
def broadcast(tensor, src, group=torch.distributed.group.WORLD, async_op=False):
def all_gather(tensor_list, tensor, group=torch.distributed.group.WORLD, async_op=False):
def sync_workers():
def barrier():
def get_rank():
def all_reduce_item(value, op='sum'):
def get_world_size(group=None):
def init_distributed_with_tcp_store(device):
def init_distributed_with_file_store(device, store_file_dir="/tmp"):
def gather_tensors(tensor, device):
def gather_objects(obj, device):
def get_device_for_backend(group):
def is_true_on_any_rank(flag: bool, group=None) -> bool:
    """Check if a boolean flag is true in any processes in the group."""
    ret = flag
    if get_world_size(group) > 1:
        device = get_device_for_backend(group)
        flag_tensor = torch.tensor([1.0 if flag else 0], dtype=torch.float32, device=device)
        torch.distributed.all_reduce(flag_tensor, op=torch.distributed.ReduceOp.MAX, group=group)
        ret = bool(flag_tensor.item() > 0)
    return ret
class ResumableDistributedSampler(torch.utils.data.DistributedSampler):

third_party/nvidia-resiliency-ext/examples/fault_tolerance/train_ddp_heartbeats_api.py
def parse_args():
def load_checkpoint(path):
def save_checkpoint(
    progress,
    model,
    optimizer,
    ft_client,
    output_dir,
    checkpoint_fname,
):
def training_loop(
    ft_client,
    para_model,
    model,
    optimizer,
    device,
    dataloader,
    sampler,
    progress,
    args,
):
def validation_loop(ft_client, model, val_dataloader, epoch_idx, device):
def _cancel_simulated_fault():
def _setup_simulated_fault(ft_client, fault_desc, device):
def _sig_handler(*args, **kwargs):
def main():

third_party/nvidia-resiliency-ext/examples/fault_tolerance/basic_ft_example.py
def print_on_rank0(msg):
def main(rank, world_size):

third_party/nvidia-resiliency-ext/examples/fault_tolerance/train_ddp_sections_api.py
def parse_args():
def load_checkpoint(path):
def save_checkpoint(
    progress,
    model,
    optimizer,
    ft_client,
    output_dir,
    checkpoint_fname,
):
def maybe_load_ft_state(path):
def save_ft_state(ft_client, path):
def update_ft_section_timeouts(ft_client, selected_sections, calc_out_of_section, ft_state_path):
def training_loop(
    ft_client,
    para_model,
    model,
    optimizer,
    device,
    dataloader,
    progress,
    args,
):
def validation_loop(ft_client, model, val_dataloader, epoch_idx, device):
def _cancel_simulated_fault():
def _setup_simulated_fault(fault_desc, device):
def _sig_handler(*args, **kwargs):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/attribution.py
def format_interruption_records(records):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/compose.py
def find_common_ancestor(*instances):

third_party/nvidia-resiliency-ext/examples/fault_tolerance/in_job_and_in_process_example.py
def _get_last_sim_fault_iter_path(rank, target_dir="/tmp/") -> str:
    # Returns the path of the file that stores the last simulated fault iteration for this rank
    return os.path.join(target_dir, f"_injob_inproc_example_rank{rank}_failed_iter.txt")
def _save_last_sim_fault_iter(rank, iteration, target_dir="/tmp/"):
def _get_last_sim_fault_iter(rank, target_dir="/tmp/") -> int:
    file_path = _get_last_sim_fault_iter_path(rank=rank, target_dir=target_dir)
    if os.path.exists(file_path):
def _parse_fault_desc_arg(value) -> Mapping[Tuple[int, int], _SimFaultDesc]:
    # Returns a mapping of (rank, iteration) to the simulated fault that should occur at that point.
    rank_iter_to_fault = dict()
    if value:
        for str_desc in value.split(','):
def _maybe_simulate_fault(rank, iteration, rank_iter_to_fault):
def parse_args():
def train(
    ft_client,
    base_store,
    model,
    opt,
    backend,
    device,
    timeout,
    args,
    call_wrapper: Optional[inprocess.CallWrapper] = None,
):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/utils.py
def logger_stack(name: Optional[str] = None, current_logger: Optional[logging.Logger] = None):
def debug_time(
    name: str, logger: Optional[logging.Logger] = None, threshold: float = float("-inf"), level=None
):
def debug_msg(msg: str):
def preload_tensors(state_dict: Dict, non_blocking=True):
def _disable_gc():
def wrap_for_async(fn):
def diff(x1: Any, x2: Any, prefix: Tuple = ()) -> Tuple[list, list, list]:
    """Recursive diff of dicts.
    Args:
        x1 (object):
def dict_list_map_outplace(f: Callable[[U], V], x: Union[Dict, List, U]) -> Union[Dict, List, V]:
    """Maps dicts and lists *out-of-place* with a given function."""
    if isinstance(x, dict):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/replication/group_utils.py
def batched(iterable, n):
def parse_group_sequence(replication_jump, replication_factor, world_size):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/state.py
def freeze_dataclass(cls):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/replication/_torch_future.py
def call_with_only_valid_kwargs(fn, **kwargs):
def object_to_tensor(obj, current_device=None, group=None):
def tensor_to_object(tensor, tensor_size, group=None):
def send_object_list(object_list, dst, group=None, device=None):
def recv_object_list(object_list, src=None, group=None, device=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/replication/torch_device_utils.py
def get_default_device_from_type(device_type: str) -> torch.device:
    """Returns the default PyTorch device based on the specified device type.
    This function maps a device type string to the corresponding PyTorch device.
    It supports both "cpu" and "cuda" types, raising an error for unsupported types.
    Args:
        device_type (str):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/replication/utils.py
def zip_strict(*args):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/basic_state_dict.py
def nested_values(x: Union[dict, list]):
def dict_list_map_inplace(f, x):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/wrap.py
def reserve_fn(state, store, progress_watchdog, progress_watchdog_interval):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/async_ckpt/filesystem_async.py
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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/tools/inject_fault.py
def register_fault(fault_name_or_enum: Union[str, Fault], handler: Callable):
def dispatch_fault_injection(fault, delay, callback):
def async_raise(tid, exc_type):
def termination_signal_handler(signum, frame):
def workload_exception(delay, callback):
def maybe_raise_workload_exception():
def clear_workload_exception():
def async_raise_exception(tid, delay, callback):
def raise_gpu_error(delay, callback):
def gpu_sleep(delay, device, callback):
def lock_gil(delay, callback):
def segfault(delay, callback):
def send_signal(pid, signal, delay, callback):
def abort(delay, callback):
def inject_fault(
    faults: tuple[Fault],
    num_faults: int | tuple[int, int],
    keep_alive: int,
    delay: float | tuple[float, float],
    seed: int,
    callback: Optional[Callable[[], Any]] = None,
):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/utils.py
def torch_older_than(version):
def format_exc(exc: BaseException):
def format_rank_set_verbose(ranks):
def format_rank_set_brief(ranks, max_show=8):
def format_rank_set(ranks):
def log_exc(rank_or_state, exc, name):
def _log_exec(target, offset=3):
def log_exec(target):
def find_nearest_handler(logger, handler_cls):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/rank_assignment.py
def bounded_activate(node, counter, path=None, current_state=None):
def propagate_terminations(node, terminated_ranks):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/log_analyzer/nvrx_logsage.py
def lines_after(lines, needle):
def chunk_logs_strict(lines):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/monitor_process.py
def is_process_active(process):
def terminate_process(
    process: psutil.Process, termination_grace_time: datetime.timedelta, log: logging.Logger
):
def daemonize_fn(fn, fn_args=(), fn_kwargs=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/monitor_thread.py
def async_raise(tid, exc_type, event=None):
def delayed_async_raise(tid, exc_type):
def reraise_if_unraisable(exc_type):
def async_abort_main_thread(msg=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/utils.py
def capture_logs(logger_name=None):
def capture_stdout(logger_name=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/async_ckpt/state_dict_saver.py
def _compare_dataclasses(obj1, obj2):
def init_checkpoint_metadata_cache(cached_global_metadata: Metadata = None):
def get_metadata_caching_status():
def save_state_dict_async_plan(
    state_dict: STATE_DICT_TYPE,
    storage_writer: 'FileSystemWriterAsync',
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    planner: Optional[Union[SavePlanner, DefaultSavePlanner]] = None,
    enable_cache: bool = False,
    metadata_cache: Optional[CheckpointMetadataCache] = None,
) -> Tuple['FileSystemWriterAsync', Union[Metadata, None], _DistWrapper]:
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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/param_utils.py
def check_type(annotation, cls):
def count_type_in_params(fn, cls):
def substitute_param_value(fn, args, kwargs, subs):
def enforce_subclass(argument, class_or_tuple):
def enforce_type(argument, class_or_tuple):
def enforce_value(condition):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/straggler/dist_utils.py
def all_gather_object(obj, group):
def get_world_size(group):
def get_rank(group):
def get_device_for_backend(group):
def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
def gather_on_rank0(tensor, group=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/ft_rendezvous_barrier.py
def _rdzv_signal_exception_handler(sig: int, frame: Optional[FrameType]) -> None:
    del frame
    raise SignalException(f"Received signal {sig} during rendezvous", signal.Signals(sig))
def _install_rdzv_signal_handlers() -> Dict[signal.Signals, Any]:
    prev_handlers: Dict[signal.Signals, Any] = {}
    for sig_to_handle in (signal.SIGTERM, signal.SIGINT):
def _restore_rdzv_signal_handlers(prev_handlers: Dict[signal.Signals, Any]) -> None:
    for sig_to_handle, handler in prev_handlers.items():
def get_method_name(depth=2):
def _parse_domain_id_from_nvidia_smi() -> str:
    """Parse domain ID from GPU using nvidia-smi to query ClusterUUID.
    The ClusterUUID serves as the domain identifier.
    All GPUs in the same NVLink domain share the same ClusterUUID.
    Returns:
        The ClusterUUID as the domain ID string.
    Raises:
        RuntimeError: If ClusterUUID cannot be retrieved.
    Example:
        >>> domain_id = _parse_domain_id_from_nvidia_smi()
        >>> # domain_id is "abc9829a-d4c8-491c-8da5-ad28fb34876b"
    """
    import subprocess
    try:
        # Run nvidia-smi to query ClusterUUID
        result = subprocess.run(
            ['nvidia-smi', '-q'],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi command failed with return code {result.returncode}. "
                f"stderr: {result.stderr}"
            )
        # Parse output to find ClusterUUID
        cluster_uuid = None
        for line in result.stdout.split('\n'):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/health_check.py
def _run_shell(cmd: str, timeout: int = 55) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        return -9, e.stdout or "", e.stderr or "timeout"
    except Exception as e:
        return -1, "", str(e)
# Adds basic thread safety, allowing to run health checks from multiple threads.
# This is needed for rendezvous unit tests. NOTE: It will work as long as each
# function/method that uses NVML performs NVML initialization and shutdown.
# Please follow this pattern when adding new code.
_nvml_lock = threading.RLock()
def with_pynvml_lock(func):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/trace_analyzer/fr_attribution.py
def eprint(*args, **kwargs):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/log_aggregator.py
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/dict_utils.py
def extract_matching_values(
    x: Union[dict, list],
    predicate: Callable[[Any], bool],
    return_lists_as_dicts: bool = False,
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
def dict_list_map_inplace(f: Callable, x: Union[dict, list]):
def dict_list_map_outplace(f: Callable, x: Union[dict, list]):
def merge(x1: dict, x2: dict, key: Tuple[str, ...] = ()):
def map_reduce(
    xs: Iterable,
    key_fn: Callable = lambda x: x,
    value_fn: Callable = lambda x: x,
    reduce_fn: Callable = lambda x: x,
) -> dict:
    """Simple map-reduce implementation following `more_itertools.map_reduce` interface."""
    res = defaultdict(list)
    for x in xs:
        res[key_fn(x)].append(value_fn(x))
    for k in res:
        res[k] = reduce_fn(res[k])
    return dict(res)
def merge_state_dicts_(current, incoming) -> None:
    # Recursively add new keys to `current`
    # Keys that already exists in the `current` will be overwritten
    for key, value in incoming.items():
def merge_namespaces_(ns1, ns2) -> None:
    """Merge attributes of ns2 into ns1."""
    for key, value in vars(ns2).items():
def compare_namespaces(ns1, ns2):
def merge_namespace_changes(original_ns, changes):
def compare_state_dicts_and_get_new_values(curr_state, new_state):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/async_ckpt/core.py
def abort_nvrx_checkpoint():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/combined_log_fr/combined_log_fr.py
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/mcp_integration/module_definitions.py
def register_all_modules():
def create_args_from_dict(module_name: str, config: dict) -> argparse.Namespace:
    """
    Create an argparse.Namespace from a configuration dictionary.
    Args:
        module_name: Name of the module
        config: Configuration dictionary
    Returns:
        argparse.Namespace with module configuration
    """
    metadata = global_registry.get_module_metadata(module_name)
    if not metadata:
        raise ValueError(f"Module '{module_name}' not found in registry")
    # Get schema defaults
    schema = metadata.input_schema
    properties = schema.get("properties", {})
    # Build args with defaults
    args_dict = {}
    for prop_name, prop_schema in properties.items():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/storage_probe.py
def _storage_path_probe(paths: Optional[list[str]] = None) -> dict:
    """
    Probe a list of paths and return a dict with keys: invalid, missing, unreadable.
    This function is invoked inside a short-lived subprocess to avoid hangs on
    remote filesystems access.
    """
    invalid: list[str] = []
    missing: list[str] = []
    unreadable: list[str] = []
    if not paths:
        logger.debug("storage probe invoked with no paths; treating as success")
        return {"invalid": invalid, "missing": missing, "unreadable": unreadable}
    for p in paths:
        # Skip None and empty strings
        if not p:
            continue
        # Ensure p is a string (defensive check)
        if not isinstance(p, str):

third_party/nvidia-resiliency-ext/tests/inprocess/test_app.py
def launch(fn, **kwargs):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/mcp_integration/registry.py
def serialize_result(result: Any) -> str:
    """Serialize attribution result to JSON string."""
    if result is None:
        return json.dumps(None)
    if is_dataclass(result):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/per_cycle_logs.py
def _should_filter_line(line: str) -> bool:
    """
    Filter out noisy lines that clutter logs but provide no useful information.
    Currently filters:
    - Nvidia driver /proc/devices dumps on process exit (Character/Block devices listings)
    Optimized for minimal overhead:
    - Pre-compiled regex at module level
    - Fast early-exit checks before expensive operations
    - Defers strip() until actually needed (avoids string allocation on fast path)
    Args:
        line: Log line to check (without rank prefix)
    Returns:
        True if line should be filtered out (dropped), False if it should be kept
    """
    # Fast path #1: Most log lines are much longer than device dumps
    # Device dump lines are typically < 50 chars: "NNN device-name" or "Character devices:"
    # Check original line length (no strip needed yet)
    # ~99% of lines exit here in typical workloads
    if len(line) > 65:  # Extra buffer for trailing newlines/spaces
        return False
    # Fast path #2: Empty lines - keep them (they're intentional formatting)
    if not line:
        return False
    # Now we know it's a short line that's not empty - strip for pattern matching
    # Only ~1% of lines reach this point in typical workloads
    stripped = line.strip()
    # Whitespace-only lines (e.g., "   \n", "\t\n") - keep them too
    # These happen when workers output blank lines for readability
    if not stripped:
        return False
    # Check for section headers (very fast frozenset lookup)
    if stripped in _DEVICE_SECTION_HEADERS:
        return True
    # Check for device number entries: "NNN device-name"
    # Pattern: 1-3 digits, space(s), device name
    # Examples: "252 device-mapper", "1 mem", "195 nvidia"
    # After stripping, device entries always start with a digit (fast pre-check before regex)
    first_char = stripped[0]
    if first_char.isdigit() and _DEVICE_ENTRY_PATTERN.match(stripped):
def _patch_subprocess_handler_once():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/c10d_monkey_patch.py
def _patched_create_tcp_store(params: "RendezvousParameters") -> "TCPStore":  # noqa: F821
    """
    Patched version of _create_tcp_store that supports use_libuv parameter.
    This function is identical to the original _create_tcp_store except it
    extracts and uses the use_libuv parameter from RendezvousParameters.
    """
    import os
    from datetime import timedelta
    from typing import cast
    from torch.distributed import TCPStore
    from torch.distributed.elastic.events import NodeState, construct_and_record_rdzv_event
    from torch.distributed.elastic.rendezvous.api import RendezvousConnectionError
    from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import (
        _matches_machine_hostname,
        parse_rendezvous_endpoint,
    )
    # Default port for TCP store (29400) - defined locally for PyTorch 2.3.1 compatibility
    DEFAULT_PORT = 29400
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=DEFAULT_PORT)
    cfg_is_host = params.get_as_bool("is_host")
    # If the user has explicitly specified whether our process should host the
    # the store, respect it.
    if cfg_is_host is not None:
        is_host = cfg_is_host
    # Otherwise try to determine whether we are the host based on our hostname
    # and IP address.
    else:
        is_host = _matches_machine_hostname(host)
    # The timeout
    read_timeout = cast(int, params.get_as_int("read_timeout", 60))
    if read_timeout <= 0:
        raise ValueError("The read timeout must be a positive integer.")
    # The use_libuv parameter - NEW FUNCTIONALITY
    use_libuv = params.get_as_bool("use_libuv", True)
    # In specific cases we attempt to instantiate the store twice. For details
    # see the explanation in the except clause below.
    for is_server in [is_host, False]:
        try:
            store = TCPStore(
                host,
                port,
                is_master=is_server,
                multi_tenant=True,
                timeout=timedelta(seconds=read_timeout),
                use_libuv=use_libuv,  # NEW PARAMETER
            )
            if is_server:
                msg = f"Process {os.getpid()} hosts the TCP store for the C10d rendezvous backend."
                construct_and_record_rdzv_event(
                    run_id=params.run_id, message=msg, node_state=NodeState.INIT
                )
                logger.info(msg)
            break
        except (ValueError, RuntimeError, TimeoutError) as exc:
            # If we heuristically inferred the value of is_host as True and our
            # first attempt to instantiate the TCP store has failed, try it one
            # more time with is_host set to False. As an edge case there can be
            # more than one process that is part of the same rendezvous on this
            # machine and only one of them will eventually host the store.
            if not is_server or cfg_is_host is not None:
                raise RendezvousConnectionError(
                    "The connection to the C10d store has failed. See inner exception for details."
                ) from exc
    return store  # type: ignore[possibly-undefined]
def apply_c10d_patch():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/grpc_log_server.py
def serve(host: str, port: int, max_workers: int = 100, graceful_shutdown_timeout: float = 60.0):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/os_utils.py
def validate_directory(dir_path: str) -> None:
    """
    Validate that a directory is safe for file operations.
    This function performs comprehensive security checks to ensure the directory
    is safe from symlink attacks and has appropriate permissions.
    Args:
        dir_path: Path to the directory to validate
    Raises:
        OSError: If the directory is unsafe or inaccessible
    """
    if not os.path.exists(dir_path):
def validate_filepath(file_path: str) -> None:
    """
    Validate that a file path is safe for file operations.
    This function checks that the file (if it exists) is not a symlink and is a regular file.
    Args:
        file_path: Path to the file to validate
    Raises:
        OSError: If the file is unsafe or inaccessible
    """
    if os.path.exists(file_path):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/mcp_integration/server_launcher.py
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/utils.py
def get_infrastructure_rank(skip_nodename_logic: bool = False) -> int:
    """Get infrastructure rank from environment variables with SLURM validation.
    Returns infrastructure rank with the following precedence:
    1. NVRX_INFRA_RANK_FROM_NODENAME (if set and not skipped) - calculate rank by extracting all digits from SLURMD_NODENAME
       - Example: "nvl72134-T01" -> rank 7213401
    2. CROSS_SLURM_PROCID (for multi-job coordination)
    3. SLURM_TOPOLOGY_ADDR with block awareness (if SLURM_TOPOLOGY_ADDR_PATTERN is "block.node" and not skipped)
       - Parses format "blockX.nodeY" and calculates rank as X * multiplier + Y
       - Default multiplier is 10^10 (10 billion), reserving 10 digits for node numbers
       - This keeps block index in MSB for proper ordering with 64-bit integers
       - Can be overridden with SLURM_TOPOLOGY_NODES_PER_BLOCK env var
       - Raises ValueError if node number >= 10^10 (exceeds 10 digits)
       - Examples with default multiplier=10^10:
         * "block5.node3"   -> rank 5*10^10 + 3 = 50000000003
         * "block5.node9"   -> rank 5*10^10 + 9 = 50000000009
         * "block5.node10"  -> rank 5*10^10 + 10 = 50000000010
         * "block6.node2"   -> rank 6*10^10 + 2 = 60000000002
    4. SLURM_PROCID (set by SLURM), with job array support
    5. GROUP_RANK (fallback, set by launcher)
    For SLURM job arrays with one task per node, the infrastructure rank is calculated as:
        array_task_id * nnodes_per_array_task + slurm_procid
    This ensures unique ranks across all nodes in all array tasks.
    If none are set, returns -1 to indicate it should be assigned deterministically.
    Args:
        skip_nodename_logic: If True, skip the NVRX_INFRA_RANK_FROM_NODENAME and SLURM_TOPOLOGY_ADDR logic
                           and fall through to SLURM array task ID calculation. Default is False.
    Returns:
        int: Infrastructure rank (>=0) or -1 if not set
    Raises:
        RuntimeError: If SLURM_JOB_ID is set but neither CROSS_SLURM_PROCID nor SLURM_PROCID is defined
        ValueError: If NVRX_INFRA_RANK_FROM_NODENAME is set (and not skipped) but SLURMD_NODENAME
                   is not set or contains no digits
        ValueError: If SLURM_TOPOLOGY_ADDR_PATTERN is "block.node" (and not skipped) but SLURM_TOPOLOGY_ADDR
                   does not match expected format or parts contain no digits
        ValueError: If node number in SLURM_TOPOLOGY_ADDR exceeds 10 digits (>= 10^10)
    """
    # Check NVRX_INFRA_RANK_FROM_NODENAME first (for nodename-based rank calculation)
    if not skip_nodename_logic and os.getenv('NVRX_INFRA_RANK_FROM_NODENAME') is not None:
        nodename = os.getenv('SLURMD_NODENAME')
        if nodename is None:
            raise ValueError(
                "NVRX_INFRA_RANK_FROM_NODENAME is set but SLURMD_NODENAME environment variable is not set"
            )
        # Extract all digits from nodename
        digits = ''.join(c for c in nodename if c.isdigit())
        if not digits:
            raise ValueError(
                f"NVRX_INFRA_RANK_FROM_NODENAME is set but SLURMD_NODENAME '{nodename}' contains no digits"
            )
        infra_rank = int(digits)
        logger.debug(f"Using infrastructure rank {infra_rank} from SLURMD_NODENAME '{nodename}'")
        return infra_rank
    # Check CROSS_SLURM_PROCID second (for multi-job scenarios)
    cross_slurm_procid = os.getenv('CROSS_SLURM_PROCID')
    if cross_slurm_procid is not None:
        infra_rank = int(cross_slurm_procid)
        logger.debug(f"Using infrastructure rank {infra_rank} from CROSS_SLURM_PROCID")
        return infra_rank
    # Check SLURM_TOPOLOGY_ADDR with block awareness third
    if not skip_nodename_logic:
        topology_addr = os.getenv('SLURM_TOPOLOGY_ADDR')
        topology_pattern = os.getenv('SLURM_TOPOLOGY_ADDR_PATTERN')
        if (
            topology_addr is not None
            and topology_pattern is not None
            and topology_pattern.lower() == 'block.node'
        ):
def is_slurm_job_array() -> bool:
    """Check if the current job is running in a SLURM job array.
    Returns:
        bool: True if running in a SLURM job array (SLURM_ARRAY_TASK_ID is set), False otherwise
    """
    return os.getenv('SLURM_ARRAY_TASK_ID') is not None
def is_process_alive(pid):
def wait_until_process_terminated(pid, timeout=0):
def wait_for_mp_events(events, timeout=60):
def terminate_mp_processes(allowed_ppids, allowed_pgids):
def set_ipc_socket_timeouts(fileno, timeout):
def recv_all(sock, n):
def read_obj_from_ipc_socket(sock, raise_exc=False):
def write_object_to_ipc_socket(obj, sock):
def get_rank():
def reduce_cuda_ctx_size():
def get_processes_by_pgids(pgids, exclude_launcher=True):
def patched_method(obj, method_name, new_method):
def install_exception_handler():

third_party/nvidia-resiliency-ext/tests/inprocess/test_torch.py
def abort_process_group():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/_ft_rendezvous.py
def get_method_name(depth=2):
def _is_final_workers_state(state: WorkerState) -> bool:
    # Final worker group state once reached will not be changed
    return state in {WorkerState.SUCCEEDED, WorkerState.FAILED, WorkerState.UNKNOWN}
def _remove_participant_epilogue(state: _RendezvousState, settings: RendezvousSettings) -> None:
    if state.complete:
        # If we do not have any participants left, move to the next round.
        if not state.participants:
            msg = "No participants left in the rendezvous, marking rendezvous as incomplete"
            log.debug(msg)
            state.complete = False
            state.round += 1
    else:
        if len(state.participants) < settings.min_nodes:
            msg = (
                f"Number of participants {len(state.participants)}) less than"
                f"min_nodes {settings.min_nodes}, clearning deadline in state"
            )
            log.debug(msg)
            state.deadline = None
class _RendezvousStateHolder(ABC):
def _should_keep_alive(ctx: _RendezvousContext) -> bool:
    """Determine whether a keep-alive heartbeat should be sent."""
    try:
        last_heartbeat = ctx.state.last_heartbeats[ctx.node]
    except KeyError:
        return False
    return last_heartbeat <= datetime.utcnow() - ctx.settings.keep_alive_interval
class _SetWorkersStateOp:
    def __init__(self, target_state: WorkerState):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/ptl_resiliency/_utils.py
def is_module_available(module: str) -> bool:
    import importlib
    return importlib.util.find_spec(module) is not None
@dataclass
class SimulatedFaultParams:
    """
    Description of a simulated rank fault, used for FT testing and debugging.
    Simulated fault types are:
    - 'rank_killed' a rank is killed with SIGKILL
    - 'rank_hung' a rank is stopped with SIGSTOP
    - 'random' randomly selects one of the above faults.
    Fault delay is computed as:
    - `base_delay` + RAND_FLOAT_FROM_0.0_to_1.0 * `rand_delay`
    Attributes:
        fault_type (str):
def parse_simulated_fault_params(simulated_fault_params) -> Optional[SimulatedFaultParams]:
    if simulated_fault_params is None:
        return None
    if isinstance(simulated_fault_params, SimulatedFaultParams):
def setup_simulated_fault(fault_desc: SimulatedFaultParams):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/wait_daemon.py
def wait_for_pids(pids: List[int]) -> None:
    """Wait for all specified PIDs to finish."""
    print(f"Monitoring {len(pids)} PIDs: {pids}")
    while pids:
        finished_pids = []
        for pid in pids:
            try:
                # Check if process exists by sending signal 0
                os.kill(pid, 0)
            except OSError:
                # Process has finished or doesn't exist
                finished_pids.append(pid)
        # Remove finished/invalid PIDs from the monitoring list
        for pid in finished_pids:
            pids.remove(pid)
            print(f"PID {pid} has finished or is invalid. {len(pids)} PIDs remaining: {pids}")
        if pids:
            time.sleep(1)
def read_pids_from_file(pidfile: str) -> List[int]:
    """Read PIDs from a file."""
    pids = []
    try:
        with open(pidfile, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
def wait_daemon(pidfile: str) -> None:
    """Main function to wait for daemon processes."""
    # If PID file doesn't exist, exit immediately
    if not os.path.exists(pidfile):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/testing_utils/health_check_injector.py
def set_current_cycle(cycle: int) -> None:
    """
    Set the current rendezvous cycle for injection checking.
    This should be called by the rendezvous handler before health checks.
    Args:
        cycle: The current rendezvous round/cycle number.
    """
    global _current_cycle
    _current_cycle = cycle
class HealthCheckInjector:
    """Manages GPU health check failure injection for testing purposes."""
    def __init__(self):
def _get_injector() -> Optional[HealthCheckInjector]:
    """Get the global health check injector instance, or None if not enabled."""
    global _injector
    if _injector is None and os.environ.get("NVRX_INJECT_GPU_FAILURE"):
def _monkey_patch_gpu_health_check():

third_party/nvidia-resiliency-ext/tests/shared_utils/test_logger.py
def create_test_workspace(clean: bool = True):
def setup_vars(global_id, local_id, file_size, max_file_num=None, dbg_on="0"):
def gen_log_msg(logger, num_msg, log_type="info"):
def worker_process(id, num_msg, file_size):

third_party/nvidia-resiliency-ext/tests/ptl_resiliency/unit/test_local_ckpt_callback.py
def _run_trainining(
    max_steps=100_000,
    max_epochs=3,
    max_time=None,
    val_check_interval=None,
    custom_callbacks=None,
    custom_checkpoint_io_fn=None,
):
def inspect_checkpoints(dirpath, expected_paths: Iterable[str] = None, verbose: bool = True):
def test_local_ckpt_callback_called_every_n_train_steps(tmp_path):
def test_local_ckpt_callback_called_every_time_interval(tmp_path):
def test_local_ckpt_restoration(tmp_path, caplog):

third_party/nvidia-resiliency-ext/tests/ptl_resiliency/unit/test_straggler_det_callback.py
def tmp_path():
def _create_test_logger(logger_name, log_file_path):
def test_prints_perf_scores_when_fitting(tmp_path):
def test_logs_perf_scores_when_fitting(tmp_path):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/func/_launcher_mode_test_worker.py
def _sigusr_handler(signum, frame, world_size_file):
def update_run_cnt(run_cnt_file):
def increment_world_size(world_size_file):
def decrement_world_size(world_size_file):
def parse_arguments():

third_party/nvidia-resiliency-ext/tests/inprocess/app.py
def unixTimestamp():
def parse_args(namespace=None, allow_extras=True):
def maybe_trigger_fault(rank, world_size, args):
def training_main():
def train(
    base_store: torch.distributed.TCPStore = None,
    call_wrapper: inprocess.CallWrapper = None,
    namespace: Optional[argparse.Namespace] = None,
):
def main():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/func/nodehc_service.py
def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
def _cleanup_socket(path: str) -> None:
    try:
        if os.path.exists(path) and not os.path.isdir(path):
def parse_args(argv):
def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    # Determine success
    success = True
    if args.fail:
        success = False
    elif args.success:
        success = True
    # Prepare socket
    uds_path = args.socket
    _ensure_parent_dir(uds_path)
    _cleanup_socket(uds_path)
    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = HealthCheckService(
        success=success,
        exit_code=args.exit_code,
        output=args.output,
        error=args.error,
        echo_args=bool(args.echo_args),
        sleep_seconds=int(args.sleep_seconds or 0),
    )
    nvhcd_pb2_grpc.add_HealthCheckServiceServicer_to_server(servicer, server)
    target = f"unix://{uds_path}"
    server.add_insecure_port(target)
    # Graceful shutdown on SIGTERM/SIGINT
    stopped = {"flag": False}
    def handle_signal(signum, frame):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_launcher_grpc_integration.py
def tmp_dir():
def _run_launcher(cmd_to_run, timeout=DEFAULT_TIMEOUT):
def _save_ft_cfg(cfg, dirpath):
def _get_util_script_path():
def test_grpc_server_can_start_and_shutdown():
def test_grpc_server_health_check():
def _wait_for_grpc_server_ready(port, timeout=10.0, poll_interval=0.2):
def _wait_for_file_content(path, required_substrings, timeout=5.0, poll_interval=0.1):
def test_grpc_client_can_connect_and_stream_logs(tmp_dir):
def test_launcher_starts_grpc_server_on_correct_port(tmp_dir):
def test_launcher_creates_grpc_server_log(tmp_dir):
def test_launcher_without_grpc_flag_does_not_start_server(tmp_dir):
def test_launcher_custom_server_log_path(tmp_dir):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_process_utils.py
def _sleeping_process(time_to_sleep):
def test_is_process_alive():
def test_wait_until_process_terminated():

third_party/nvidia-resiliency-ext/tests/ptl_resiliency/unit/test_ft_callback_hb.py
def tmp_path():
def _get_ft_test_config():
def run_rank_monitors():
def _create_test_logger(logger_name, log_file_path):
def _run_trainining(
    tmp_path,
    max_steps=100_000,
    max_epochs=4,
    max_time=None,
    val_check_interval=None,
    custom_callbacks=None,
    expects_fit_exception=False,
):
def _run_eval(tmp_path, which='not set'):
def test_finished_fit_with_iter_limit(tmp_path, run_rank_monitors):
def test_finished_fit_after_all_read(tmp_path, run_rank_monitors):
def test_finished_fit_with_time_limit(tmp_path, run_rank_monitors):
def test_timeouts_updated_when_graceful_stop(tmp_path, run_rank_monitors):
def test_timeouts_updated_when_exc(tmp_path, run_rank_monitors):
def test_timeouts_updated_when_sys_exit(tmp_path, run_rank_monitors):
def test_simulated_fault(tmp_path):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/func/_workload_ctrl_test_worker.py
def _sigusr1_handler(*args, **kwargs):
def _sigusr2_handler(*args, **kwargs):
def _sigterm_handler(*args, **kwargs):
def get_cmd_file_content(ctl_file):
def get_random_rank(this_rank, is_agent_rdzv_host):
def maybe_handle_control_cmd(this_rank, ctl_file, is_agent_rdzv_host):
def update_run_cnt(run_cnt_file):
def parse_arguments():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/log_manager.py
def setup_logger(
    node_local_tmp_dir=None,
    force_reset=False,
    node_local_tmp_prefix: str = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup the distributed logger.
    This function configures the standard Python logger "nvrx" with appropriate
    handlers for distributed logging. It's safe to call multiple times - if the
    logger is already configured, it won't be reconfigured unless force_reset=True.
    The expectation is that this function is called once at the start of the program,
    and then the logger is used throughout the program i.e. its a singleton.
    The logger automatically adapts to distributed or regular mode based on
    whether NVRX_NODE_LOCAL_TMPDIR is set. If set, enables distributed logging
    with aggregation. If not set, logs go directly to stderr/stdout.
    The logger is fork-safe: all ranks use file-based message passing to ensure
    child processes can log even when they don't inherit the aggregator thread.
    Args:
        node_local_tmp_dir: Optional directory path for temporary files. If None, uses NVRX_NODE_LOCAL_TMPDIR env var.
        force_reset: If True, force reconfiguration even if logger is already configured.
                    Useful for subprocesses that need fresh logger setup.
        node_local_tmp_prefix: Optional prefix for log files (e.g. "ftlauncher").
        log_file: Optional path to log file. When specified, logs are written to this file
                 with rank prefixes (like srun -l) instead of console. All processes write
                 to the same file using append mode for safe concurrent writes.
    Returns:
        logging.Logger: Configured logger instance
    Example:
        # In main script (launcher.py) or training subprocess
        from nvidia_resiliency_ext.shared_utils.log_manager import setup_logger
        logger = setup_logger()
        # With log file for consolidated logging across all ranks/nodes
        logger = setup_logger(log_file="/path/to/base.log")
        # In subprocesses that need fresh logger setup
        logger = setup_logger(force_reset=True)
        # In other modules
        import logging
        logger = logging.getLogger(LogConfig.name)
        logger.info("Some message")
    """
    # Check if the nvrx logger is already configured
    logger = logging.getLogger(LogConfig.name)
    # If force_reset is True or the logger has no handlers, configure it
    if force_reset or not logger.handlers:
        # Clear existing handlers if force_reset is True
        if force_reset:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            # Clear any stored log manager to force fresh creation
            if hasattr(setup_logger, '_log_manager'):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/conftest.py
def pytest_configure():
def pytest_sessionfinish(session, exitstatus):

third_party/nvidia-resiliency-ext/tests/ptl_resiliency/unit/test_ft_callback_sections.py
def tmp_path():
def _get_ft_test_config():
def run_rank_monitors(request):
def ipc_connector(tmp_path):
def _create_test_logger(logger_name, log_file_path):
def _run_trainining(
    tmp_path,
    max_steps=100_000,
    max_epochs=4,
    max_time=None,
    val_check_interval=None,
    chkpt_save_interval=None,
    custom_callbacks=None,
    expects_fit_exception=False,
    ft_exc_handling_policy=None,
):
def _get_ft_state(exp_dir):
def _get_timeout_updates_count(log_file_path):
def _run_eval(tmp_path, which='not set'):
def test_finished_fit_with_iter_limit(tmp_path, run_rank_monitors):
def test_eval_with_callback(tmp_path, run_rank_monitors):
def test_finished_fit_after_all_read(tmp_path, run_rank_monitors):
def test_finished_fit_with_time_limit(tmp_path, run_rank_monitors):
def test_timeouts_updated_when_graceful_stop(tmp_path, run_rank_monitors):
def test_timeouts_updated_when_exc(tmp_path, run_rank_monitors):
def test_timeouts_updated_when_sys_exit(tmp_path, run_rank_monitors):
def test_simulated_fault(tmp_path):
def _get_ft_test_config_no_step_section():
def _get_ft_test_config_unknown_sections():
def test_invalid_config(tmp_path, run_rank_monitors):
def test_workload_ctrl_messages(tmp_path, run_rank_monitors, ipc_connector):

third_party/nvidia-resiliency-ext/tests/checkpointing/unit/__init__.py
def empty_dir(path: Path):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/_launcher_test_util.py
def test_rank_not_send_initial_hb(args):
def test_rank_failed(args):
def test_ranks_exit_gracefully(args):
def test_with_sigterm(args):
def test_dump_ft_config(args):
def _get_restart_cnt(tmp_dir, update=False):

third_party/nvidia-resiliency-ext/tests/inprocess/common.py
def silence_deprecation_warnings():
def torch_older_than(version):
def find_free_port():
def apply_all_tests(decorator):
def retry(max_attempts=5, delay=1):
def is_port_available(port, host='localhost'):
def wait_for_port(port, host='localhost', timeout=5, check_interval=0.1):
def wrap_store(store, *args):
def unroll_generators(params):
def subtests(params):
def instantiate_parametrized_tests(generic_cls):
def compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn):
def dtype_name(dtype):

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_name_mapper.py
def _get_summary(timings):
def test_name_mapper_gather():
def test_name_mapper_no_gather():
def _test_mapping_consitiency_no_gather_on_rank0(ret_queue):
def test_mapping_consitiency_no_gather_on_rank0():
def _test_mapping_consitiency_with_gather_on_rank0(ret_queue):
def test_mapping_consitiency_with_gather_on_rank0():

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_data_shared.py
def _get_summary(timings):
def _test_num_of_all_gather_object_calls(*args, **kwargs):
def test_all_gather_object_calls_num():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_init.py
def _get_ft_test_config():
def _install_signal_handler():
def _set_rmon_socket_env_var_for_this_rank():
def _run_rank_monitors_fixture():
def _rank_main_all_ranks_initialized(*args, **kwargs):
def _rank_main_no_initial_heartbeat_from_rank_0(*args, **kwargs):
def _rank_main_no_initial_heartbeats(*args, **kwargs):
def test_init(rank_main, is_chkpt_expected, expected_ret_codes):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_config.py
def test_from_kwargs():
def test_from_args():
def test_signal_field_with_valid_values():
def test_signal_field_with_invalid_values():
def test_log_level_field_with_valid_values(level_to_set, expected_level):
def test_log_level_field_with_invalid_values():
def tmp_yaml_file(lines):
def test_read_from_yaml():
def test_read_from_yaml_nested():
def test_fails_when_missing_ft_section():
def test_fails_when_ft_section_has_invald_items():
def test_to_yaml_file():

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_wrap_callables.py
def _rank_main(*args, test_scenario, ret_queue, **kwargs):
def test_straggler_sections_detected(test_scenario):

third_party/nvidia-resiliency-ext/tests/checkpointing/unit/conftest.py
def tmp_path_dist_ckpt(tmp_path_factory) -> Path:
    """Common directory for saving the checkpoint.
    Can't use pytest `tmp_path_factory` directly because directory must be shared between processes.
    """
    tmp_dir = tmp_path_factory.mktemp('ignored', numbered=False)
    tmp_dir = tmp_dir.parent.parent / 'tmp_dist_ckpt'
    if Utils.rank == 0:
        with TempNamedDir(tmp_dir, sync=False):
def async_queue():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_barrier_rendezvous.py
def _test_segment_check_interval(seconds=TEST_SEGMENT_CHECK_INTERVAL_SECS):
def _assert_threads_finished(test_case, threads, timeout_secs):

third_party/nvidia-resiliency-ext/tests/inprocess/test_timeout.py
def kwargs():

third_party/nvidia-resiliency-ext/tests/ptl_resiliency/func/nemo20/test_local_ckpt_llama3.py
def get_trainer(args, callbacks, async_save=True):
def print_rank0(msg):
def get_parser():
def main():

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_cupti_manager.py
def test_cupti_manager_start_stop():
def test_cupti_manager_captures_all_started_kernels():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_timeouts_calc.py
def tmp_dir():
def test_with_heartbeats():
def test_with_custom_sections():
def test_with_sections_subset():
def test_section_timeouts_merging():
def test_get_merged_section_timeouts():
def test_end_all_sections():
def _rank_main(*args, tmp_dir, **kwargs):
def test_distributed(tmp_dir):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_timeouts_sections.py
def _install_signal_handler():
def tmp_dir():
def _set_rmon_socket_env_var_for_this_rank():
def rank_monitors_running(ft_cfg, mp_ctx):
def _rank_main_1st_run(*args, tmp_dir, **kwargs):
def _rank_main_2nd_run(*args, tmp_dir, **kwargs):
def test_timeouts(tmp_dir):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_launcher.py
def tmp_dir():
def _get_func_name():
def _run_launcher(cmd_to_run, timeout):
def _save_ft_cfg(cfg, dirpath):
def _get_util_script_path():
def test_rank_not_send_initial_hb(tmp_dir):
def test_rank_failed(tmp_dir):
def test_ranks_exit_gracefully(tmp_dir):
def test_launcher_sigterm_graceful_exit(tmp_dir):
def test_launcher_sigterm_ignored(tmp_dir):
def test_ranks_restart(tmp_dir):
def test_missing_cfg(tmp_dir):
def test_config_provided_via_cli(tmp_dir):
def test_config_provided_via_cli_overwrites_yaml(tmp_dir):

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_relative_gpu_scores.py
def _get_summary(timings):
def test_relative_gpu_scores_one_rank():
def _rank_main_gpu_relative_test(*args, gather_on_rank0, ret_queue, **kwargs):
def test_relative_gpu_scores_gather_on_rank0():
def test_relative_gpu_scores_no_gather():
def _rank_main_gpu_relative_test_some_common_kernels(*args, gather_on_rank0, ret_queue, **kwargs):
def test_relative_gpu_scores_some_common_kernels():
def _rank_main_gpu_relative_test_no_common_kernels_in_rank(
    *args,
    ranks_with_unique_kernels=(),
    ranks_without_kernels=(),
    gather_on_rank0=True,
    ret_queue,
    **kwargs,
):
def test_relative_gpu_scores_ranks_with_unique_kernels():
def test_relative_gpu_scores_ranks_without_kernels():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_dynamic_rendezvous.py
def _extract_rendezvous_info(rendezvous_result) -> Tuple[int, int]:
    """Extract rank and world_size from either RendezvousInfo or tuple.
    Args:
        rendezvous_result: Either a RendezvousInfo object or a tuple (store, rank, world_size)
    Returns:
        Tuple of (rank, world_size)
    """
    try:
        # Try to access as RendezvousInfo object
        return rendezvous_result.rank, rendezvous_result.world_size
    except AttributeError:
        # Fall back to tuple format (store, rank, world_size)
        return rendezvous_result[1], rendezvous_result[2]
def _extract_store(rendezvous_result) -> Store:
    """Extract store from either RendezvousInfo or tuple.
    Args:
        rendezvous_result: Either a RendezvousInfo object or a tuple (store, rank, world_size)
    Returns:
        Store object
    """
    try:
        # Try to access as RendezvousInfo object
        return rendezvous_result.store
    except AttributeError:
        # Fall back to tuple format (store, rank, world_size)
        return rendezvous_result[0]
class CustomAssertMixin:
    assertDictEqual: Callable
    def assert_state_equal(self, actual: _RendezvousState, expected: _RendezvousState) -> None:
        self.assertDictEqual(vars(actual), vars(expected))
    def assert_state_empty(self, actual: _RendezvousState) -> None:
        self.assertDictEqual(vars(actual), vars(_RendezvousState()))
# Basic classes like RendezvousTimeout, NodeDesc, NodeDescGenerator are unchanged from upstream PyTorch
# and are well-tested there. We focus on FT-specific functionality.
class AssignRanksTest(TestCase):
def _ignore_exception(exception_type: Exception, fn: Callable):
def _wait_for(condition, timeout=30, interval=1, name=None):

third_party/nvidia-resiliency-ext/tests/ptl_resiliency/func/nemo20/check_straggler_log.py
def cmd_check_status(log_file_lines, args):
def cmd_num_reports(log_file_lines, args):
def _check_gpu_stragglers(log_file_lines, pattern, expected_stragglers):
def cmd_relative_gpu_stragglers(log_file_lines, args):
def cmd_individual_gpu_stragglers(log_file_lines, args):
def cmd_check_terminating(log_file_lines, args):
def read_log_file(log_file):
def main():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_reconnect.py
def _get_ft_test_config():
def _set_rmon_socket_env_var_for_this_rank():
def rank_monitors_fixture():
def _rank_main(*args, rank_ready_events, **kwargs):
def _send_sig(sig, pids):
def test_reconnect(rank_monitors_fixture):

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_det_section_api.py
def _straggler_init_shutdown():
def _straggler_shutdown_at_exit():
def test_fail_if_not_initialized():
def test_unique_name_is_enforced(_straggler_init_shutdown):
def test_default_names_are_unique(_straggler_init_shutdown):
def test_with_block_location_is_used(_straggler_init_shutdown):
def test_periodic_capture(_straggler_shutdown_at_exit):
def test_cuda_profiling_disabled(_straggler_shutdown_at_exit):
def test_can_handle_empty_elapseds(_straggler_init_shutdown):

third_party/nvidia-resiliency-ext/tests/checkpointing/unit/test_async_writer.py
def mock_open(
    self,
    path: str,
    mode: str = "rb",
) -> IO[Any]:
    """Function matching the system open() signature that always raises an error."""
    raise OSError('worker critical failure during open()')
class TestAsyncSave:
    def get_async_save_request(self, writer, save_state_dict_ret) -> AsyncRequest:
        """Creates an async save request with a finalization step."""
        save_fn, preload_fn, save_args = writer.get_save_function_and_args()
        def finalize_fn():

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_interval_tracker.py
def test_estimate():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_shutdown_sections.py
def _get_ft_test_config():
def _set_rmon_socket_env_var_for_this_rank():
def _run_rank_monitors_fixture():
def wait_for_process(pid, timeout):
def _send_sig(sig, pids):
def _rank_main_with_step(*args, rank_ready_events, **kwargs):
def test_shutdown_due_to_step_section_timeout():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_shutdown.py
def _get_ft_test_config():
def _set_rmon_socket_env_var_for_this_rank():
def _run_rank_monitors_fixture():
def _rank_main(*args, rank_ready_events, **kwargs):
def wait_for_process(pid, timeout):
def _send_sig(sig, pids):
def test_shutdown(test_scenario):
def _rank_main_explicit_shutdown(*args, rank_ready_events, **kwargs):
def test_explicit_shutdown():

third_party/nvidia-resiliency-ext/tests/ptl_resiliency/func/nemo20/ft_test_llama3.py
def get_ft_callback(args):
def get_trainer(args, callbacks):
def print_rank0(msg):
def get_parser():
def main():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_ipc_connector.py
def _sender_process(rank):
def test_ipc_connector_send_receive():

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/utils.py
def assert_fn(arg):
def is_port_open(host, port):
def find_free_port(host="127.0.0.1"):
def setup_distributed_worker_tcp_store(
    rank,
    world_size,
    queue,
    barrier,
    backend,
):
def setup_distributed_worker_file_store(
    rank,
    world_size,
    queue,
    barrier,
    backend,
):
def distributed_worker(
    rank,
    world_size,
    dist_store_type,
    queue,
    barrier,
    ready_flag,
    worker_fn,
    backend,
    **kwargs,
):
def multiprocessing_execute_start(
    worker_fn,
    mp_ctx,
    world_size=2,
    dist_store_type="file",
    backend="nccl",
    **kwargs,
):
def multiprocessing_execute_join(processes, timeout):

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_reporting_elapsed.py
def parse_args():
def setup_distributed():
def skip_if_condition(test_scenario):
def test_report_elapsed_wrap_callables(test_scenario):
def test_report_elapsed_det_section(test_scenario):
def test_report_min_interval_is_profiling_interval():

third_party/nvidia-resiliency-ext/tests/straggler/func/ddp_test.py
def parse_args():
def round_float_values(d: Dict) -> Dict:
    return {k: round(v, 2) for k, v in d.items()}
def print_gpu_scores(report, rank):
def print_section_scores(report, rank):
def print_stragglers(stragglers):
def _all_reduce_bool_flag(flag):
def print_rank0(msg):
def main():

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_cupti_ext.py
def test_basic_kernel_tracking():
def test_tracking_start_stop():
def test_reset():
def test_max_stats_per_kernel():
def test_profiler_is_singleton():
def test_with_cuda_graph():

third_party/nvidia-resiliency-ext/tests/straggler/unit/_utils.py
def assert_fn(arg):
def is_port_open(host, port):
def find_free_port(host="127.0.0.1"):
def setup_distributed_worker_tcp_store(
    rank,
    world_size,
    queue,
    barrier,
    backend,
):
def setup_distributed_worker_file_store(
    rank,
    world_size,
    queue,
    barrier,
    backend,
):
def distributed_worker(
    rank,
    world_size,
    dist_store_type,
    queue,
    barrier,
    ready_flag,
    worker_fn,
    backend,
    **kwargs,
):
def multiprocessing_execute_start(
    worker_fn,
    mp_ctx,
    world_size=2,
    dist_store_type="file",
    backend="nccl",
    **kwargs,
):
def multiprocessing_execute_join(processes, timeout):

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_individual_gpu_scores.py
def _get_summary(timings):
def test_individual_gpu_scores_one_rank():
def _rank_main_gpu_indiv_test(*args, gather_on_rank0, ret_queue, **kwargs):
def test_individual_gpu_scores_gather_on_rank0():
def test_individual_gpu_scores_no_gather():

third_party/nvidia-resiliency-ext/tests/ptl_resiliency/func/nemo20/straggler_test_llama3.py
def straggler_det_callback(
    report_time_interval=20, stop_if_detected=False
) -> run.Config[StragglerDetectionCallback]:
    return run.Config(
        StragglerDetectionCallback,
        report_time_interval=report_time_interval,
        calc_relative_gpu_perf=True,
        calc_individual_gpu_perf=True,
        num_gpu_perf_scores_to_print=5,
        gpu_relative_perf_threshold=0.7,
        gpu_individual_perf_threshold=0.7,
        stop_if_detected=stop_if_detected,
        enable_ptl_logging=True,
    )
def step_timing_callback() -> run.Config[TimingCallback]:
    return run.Config(TimingCallback)
def local_executor(
    nodes=1, devices=8, ft=False, retries=0, container_image="", time=""
) -> run.LocalExecutor:
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }
    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)
    return executor
def print_rank0(msg):
def get_parser():
def main():

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_reporting.py
def parse_args():
def setup_distributed():
def skip_if_condition(test_scenario):
def test_reporting_options(test_scenario, monkeypatch):
def test_no_gather_called(monkeypatch):

third_party/nvidia-resiliency-ext/tests/straggler/func/check_log.py
def cmd_num_reports(log_file_lines, args):
def _check_gpu_stragglers(log_file_lines, pattern, expected_stragglers):
def cmd_relative_gpu_stragglers(log_file_lines, args):
def cmd_individual_gpu_stragglers(log_file_lines, args):
def read_log_file(log_file):
def main():

third_party/nvidia-resiliency-ext/tests/straggler/unit/test_sections.py
def _dummy_section_work(section_name, rank, test_scenario, is_slow_iter=False):
def _rank_main(*args, test_scenario, ret_queue, **kwargs):
def test_straggler_sections_detected(test_scenario):

third_party/nvidia-resiliency-ext/tests/fault_tolerance/unit/test_timeouts.py
def _install_signal_handler():
def tmp_dir():
def _set_rmon_socket_env_var_for_this_rank():
def rank_monitors_running(ft_cfg, mp_ctx):
def _rank_main_1st_run(*args, tmp_dir, **kwargs):
def _rank_main_2nd_run(*args, tmp_dir, **kwargs):
def _rank_main_3nd_run(*args, tmp_dir, **kwargs):
def test_timeouts(tmp_dir):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/launcher.py
def init_node_health_check(endpoint: Optional[str]) -> None:
    global _NODE_HEALTH_CHECK_INSTANCE
    if endpoint:
        _NODE_HEALTH_CHECK_INSTANCE = NodeHealthCheck(endpoint=endpoint)
    else:
        _NODE_HEALTH_CHECK_INSTANCE = None
def get_node_health_check() -> Optional[NodeHealthCheck]:
    return _NODE_HEALTH_CHECK_INSTANCE
def _register_ft_rdzv_handler(impl_type: str = "legacy"):
def _wrap_entrypoint_with_numactl(
    entrypoint: str,
    args: Tuple,
    local_rank: int,
    numa_bind_strict: bool = False,
) -> Tuple:
    """
    Wrap a binary entrypoint with numactl command for NUMA binding.
    This function should only be called when NVRX_GPUS_PER_NUMA is set and
    entrypoint is a binary/script path (string).
    Args:
        entrypoint: The worker entrypoint binary/script path
        args: Original arguments for the worker
        local_rank: Local rank of the worker
        numa_bind_strict: If True, use strict binding with --membind. If False, use --localalloc.
    Returns:
        Tuple of wrapped arguments with numactl prepended
    """
    gpus_per_numa = int(os.getenv("NVRX_GPUS_PER_NUMA"))
    numa_node = local_rank // gpus_per_numa
    # Choose memory binding strategy based on numa_bind_strict
    if numa_bind_strict:
        memory_args = ("--membind", str(numa_node))
        logger.debug(
            f"Wrapping rank {local_rank} with numactl (strict mode):
def _get_entrypoint_name(entrypoint: Union[Callable, str, None], args: List[Any]) -> str:
    """Retrieve entrypoint name with the rule:
    1. If entrypoint is a function, use ``entrypoint.__qualname__``.
    2. If entrypoint is a string, check its value:
        2.1 if entrypoint equals to ``sys.executable`` (like "python"), use the first element from ``args``
            which does not start with hifen letter (for example, "-u" will be skipped).
        2.2 otherwise, use ``entrypoint`` value.
    3. Otherwise, return empty string.
    """
    if isinstance(entrypoint, Callable):
def _get_addr_and_port(
    rdzv_parameters: RendezvousParameters,
) -> Tuple[Optional[str], Optional[int]]:
    if rdzv_parameters.backend != "static":
        return (None, None)
    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError(
            "Endpoint is missing in endpoint. Try to add --master-addr and --master-port"
        )
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    if master_port == -1:
        raise ValueError(f"port is missing in endpoint: {endpoint}. Try to specify --master-port")
    return (master_addr, master_port)
def _is_store_host(params: RendezvousParameters) -> bool:
    """Returns true if this agent is hosting the TCP store"""
    host, _ = parse_rendezvous_endpoint(params.endpoint, default_port=0)
    cfg_is_host = params.get_as_bool("is_host")
    if cfg_is_host is not None:
        return bool(cfg_is_host)
    return _matches_machine_hostname(host)
def launch_agent(
    config: LaunchConfig,
    entrypoint: Union[Callable, str, None],
    args: List[Any],
) -> Dict[int, Any]:
    if not config.run_id:
        run_id = str(uuid.uuid4().int)
        logger.warning("config has no run_id, generated a random run_id: %s", run_id)
        config.run_id = run_id
    entrypoint_name = _get_entrypoint_name(entrypoint, args)
    # with min-healthy restarting policy, if the rendezvous is completed (workers are running),
    # we dont want to replace missing/dead nodes with spares nor to upscale the rendezvous with new arrivals
    config.rdzv_configs['upscaling_enabled'] = config.restart_policy != "min-healthy"
    logger.info(
        "Starting elastic_operator with launch configs:\n"
        "  entrypoint       : %(entrypoint)s\n"
        "  min_nodes        : %(min_nodes)s\n"
        "  max_nodes        : %(max_nodes)s\n"
        "  nproc_per_node   : %(nproc_per_node)s\n"
        "  run_id           : %(run_id)s\n"
        "  rdzv_backend     : %(rdzv_backend)s\n"
        "  rdzv_endpoint    : %(rdzv_endpoint)s\n"
        "  rdzv_configs     : %(rdzv_configs)s\n"
        "  max_restarts     : %(max_restarts)s\n"
        "  restart_policy   : %(restart_policy)s\n"
        "  monitor_interval : %(monitor_interval)s\n"
        "  log_dir          : %(log_dir)s\n"
        "  metrics_cfg      : %(metrics_cfg)s\n",
        {
            "entrypoint": entrypoint_name,
            "min_nodes": config.min_nodes,
            "max_nodes": config.max_nodes,
            "nproc_per_node": config.nproc_per_node,
            "run_id": config.run_id,
            "rdzv_backend": config.rdzv_backend,
            "rdzv_endpoint": config.rdzv_endpoint,
            "rdzv_configs": config.rdzv_configs,
            "max_restarts": config.max_restarts,
            "restart_policy": config.restart_policy,
            "monitor_interval": config.monitor_interval,
            "log_dir": config.logs_specs.root_log_dir,  # type: ignore[union-attr]
            "metrics_cfg": config.metrics_cfg,
        },
    )
    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        local_addr=config.local_addr,
        **config.rdzv_configs,
    )
    master_addr, master_port = _get_addr_and_port(rdzv_parameters)
    is_store_host = _is_store_host(rdzv_parameters)
    # Add is_store_host to rdzv_parameters
    rdzv_parameters.config["is_store_host"] = is_store_host
    # Add nproc_per_node so the rendezvous handler can restore local_world_size for standby->active transitions
    rdzv_parameters.config["nproc_per_node"] = config.nproc_per_node
    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(args),
        rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        master_addr=master_addr,
        master_port=master_port,
        local_addr=config.local_addr,
    )
    agent = LocalElasticAgent(
        spec=spec,
        fault_tol_cfg=config.fault_tol_cfg,
        logs_specs=config.logs_specs,  # type: ignore[arg-type]
        start_method=config.start_method,
        log_line_prefix_template=config.log_line_prefix_template,
        term_timeout=config.term_timeout,
        workers_stop_timeout=config.workers_stop_timeout,
        restart_policy=config.restart_policy,
        is_store_host=is_store_host,
        rank_monitors=config.rank_monitors,  # Pass pre-created rank monitors
    )
    # Set agent reference in rendezvous handler for callbacks
    # This allows the handler to directly sync agent state (like progress tracker)
    # when important events occur (e.g., rendezvous round updates)
    spec.rdzv_handler.set_agent(agent)
    shutdown_rdzv = True
    try:
        metrics.initialize_metrics(metrics.MetricsConfig(config.metrics_cfg))
        result = agent.run()
        # records that agent.run() has succeeded NOT that workers have succeeded
        events.record(agent.get_event_succeeded())
        if result is None:
            logger.info("Agent .run() result is None. Agent was waiting at the rendezvous.")
            return None
        if result.is_failed():
def get_args_parser() -> ArgumentParser:
    """Parse the command line options."""
    parser = ArgumentParser(description="Torch Distributed Elastic Training Launcher")
    #
    # Worker/node size related arguments.
    #
    parser.add_argument(
        "--nnodes",
        action=env,
        type=str,
        default="1:1",
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        action=env,
        type=str,
        default="1",
        help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
    )
    #
    # Rendezvous related arguments
    #
    parser.add_argument(
        "--rdzv-backend",
        "--rdzv_backend",
        action=env,
        type=str,
        default="c10d",
        help="Rendezvous backend. Currently only c10d is supported.",
    )
    parser.add_argument(
        "--rdzv-endpoint",
        "--rdzv_endpoint",
        action=env,
        type=str,
        default="",
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv-id",
        "--rdzv_id",
        action=env,
        type=str,
        default="none",
        help="User-defined group id.",
    )
    parser.add_argument(
        "--rdzv-conf",
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
        "on a free port. Useful when launching single-node, multi-worker job. If specified "
        "--rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values "
        "are ignored.",
    )
    #
    # User-code launch related arguments.
    #
    parser.add_argument(
        "--max-restarts",
        "--max_restarts",
        action=env,
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--term-timeout",
        "--term_timeout",
        action=env,
        type=float,
        default=1800,
        help="Interval, in seconds, between initial SIGTERM and rank termination with SIGKILL, when the launcher forwards a received signal to ranks.",
    )
    parser.add_argument(
        "--workers-stop-timeout",
        "--workers_stop_timeout",
        action=env,
        type=float,
        default=15,
        help="Interval, in seconds, between initial SIGTERM and rank termination with SIGKILL, when the launcher stops its ranks in order to restart them.",
    )
    parser.add_argument(
        "--monitor-interval",
        "--monitor_interval",
        action=env,
        type=float,
        default=0.3,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--start-method",
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method to use when creating workers.",
    )
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    parser.add_argument(
        "-m",
        "--module",
        action=check_env,
        help="Change each process to interpret the launch script as a Python module, executing "
        "with the same behavior as 'python -m'.",
    )
    parser.add_argument(
        "--no-python",
        "--no_python",
        action=check_env,
        help="Skip prepending the training script with 'python' - just execute it directly. Useful "
        "when the script is not a Python script.",
    )
    parser.add_argument(
        "--run-path",
        "--run_path",
        action=check_env,
        help="Run the training script with runpy.run_path in the same interpreter."
        " Script must be provided as an abs path (e.g. /abs/path/script.py)."
        " Takes precedence over --no-python.",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
        "directory is re-used for multiple runs (a unique job-level sub-directory is created with "
        "rdzv_id as the prefix).",
    )
    parser.add_argument(
        "--ft-per-cycle-applog-prefix",
        "--ft_per_cycle_applog_prefix",
        action=env,
        type=str,
        default=None,
        dest="ft_per_cycle_applog_prefix",
        help="Prefix for per-cycle application log files (must be absolute path, e.g. /lustre/logs/job_12345.log). "
        "Creates training worker logs per cycle: /lustre/logs/job_12345_cycle0.log, job_12345_cycle1.log, etc. "
        "All ranks/nodes capture logs with automatic rank prefixes (like 'srun -l'). "
        "Without --ft-enable-log-server: Each node writes directly to Lustre with O_APPEND (safe concurrent writes). "
        "With --ft-enable-log-server (recommended):
def parse_args(args):
def parse_min_max_nnodes(nnodes: str):
def determine_local_world_size(nproc_per_node: str):
def get_rdzv_endpoint(args):
def get_use_env(args) -> bool:
    """
    Retrieve ``use_env`` from the args.
    ``use_env`` is a legacy argument, if ``use_env`` is False, the
    ``--node-rank`` argument will be transferred to all worker processes.
    ``use_env`` is only used by the ``torch.distributed.launch`` and will
    be deprecated in future releases.
    """
    if not hasattr(args, "use_env"):
def _get_logs_specs_class(logs_specs_name: Optional[str]) -> Type[LogsSpecs]:
    """
    Attemps to load `torchrun.logs_spec` entrypoint with key of `logs_specs_name` param.
    Provides plugin mechanism to provide custom implementation of LogsSpecs.
    Returns `DefaultLogsSpecs` when logs_spec_name is None.
    Raises ValueError when entrypoint for `logs_spec_name` can't be found in entrypoints.
    Built-in options:
    - None (default):
def config_from_args(args, launcher_pipe_read_fd=None, launcher_log_file=None) -> Tuple[LaunchConfig, Union[Callable, str], List[str]]:
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    assert args.max_restarts >= 0
    if hasattr(args, "master_addr") and args.rdzv_backend not in ["static", "c10d"] and not args.rdzv_endpoint:
        logger.warning(
            "master_addr is only used for static and c10d rdzv_backend when rdzv_endpoint "
            "is not specified."
        )
    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        logger.warning(
            "\n*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process to be "
            "%s in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************",
            omp_num_threads,
        )
        # This env variable will be passed down to the subprocesses
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
    log_line_prefix_template = os.getenv("TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE")
    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)
    # Add use_libuv=False for c10d backend with legacy rendezvous only
    if args.rdzv_backend == 'c10d' and getattr(args, 'ft_rdzv_impl', 'legacy') == 'legacy':
        rdzv_configs['use_libuv'] = False
    # Node health check endpoint is consumed by launcher to init singleton; not passed via rdzv configs
    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank
    rdzv_endpoint = get_rdzv_endpoint(args)
    if args.rdzv_backend.lower() != 'c10d':
        raise ValueError(
            f"Current ft_launcher version supports only rdzv_backend=c10d. Got {args.rdzv_backend}"
        )
    fault_tol_cfg = FaultToleranceConfig.from_args(args)
    # Pass segment-related configs to rendezvous config
    rdzv_configs['segment'] = fault_tol_cfg.segment
    # Pass NIC health check configs to rendezvous config
    rdzv_configs['enable_nic_healthcheck'] = fault_tol_cfg.enable_nic_healthcheck
    rdzv_configs['link_state_path_template'] = fault_tol_cfg.link_state_path_template
    # Pass enable_nic_healthcheck and link_state_path_template from fault tolerance config to rendezvous config
    rdzv_configs['enable_nic_healthcheck'] = fault_tol_cfg.enable_nic_healthcheck
    rdzv_configs['link_state_path_template'] = fault_tol_cfg.link_state_path_template
    # Pass attribution service configuration if provided
    if getattr(fault_tol_cfg, 'attrsvc_host', None):
def run_script_path(training_script: str, *training_script_args: str):
def _start_grpc_log_server(args, base_log_file: str) -> Optional[subprocess.Popen]:
    """
    Start gRPC log aggregation server as subprocess.
    This should only be called on the TCP store host (rank 0 node).
    The server accepts log chunks from clients, with each chunk specifying its target file_path.
    Args:
        args: Parsed command-line arguments (Namespace)
        base_log_file: Base log file path (used to derive server log file name if not specified)
    Returns:
        subprocess.Popen object for the server process, or None if failed
    Notes:
        max_workers is dynamically sized: min(4096, max(100, num_nodes + 10))
        - Each node runs one GrpcWriterThread (long-lived streaming connection)
        - Floor of 100: Handles small clusters and parsing failures gracefully
        - Cap at 4096: Safety limit for extremely large clusters
        - Threads are I/O-bound (blocked on queue), so 2048 threads ≈ 20MB memory
        - Insufficient workers cause clients beyond Nth to queue indefinitely!
    """
    # Determine server log file path
    grpc_server_log = getattr(args, 'ft_log_server_log', None)
    if grpc_server_log is None:
        # Derive from base_log_file: replace .log with _grpc_server.log
        if base_log_file.endswith('.log'):
def run(args):
def main(args=None):
