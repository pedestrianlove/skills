---
name: nvidia-resiliency-ext
description: Skills for agents to consume for nvidia-resiliency-ext, contain only
  the function signature.
---
third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/utils.py
def load_nvidia_api_key() -> str:
    """Load NVIDIA API key from environment or file.
    Checks in order:
    1. NVIDIA_API_KEY environment variable
    2. NVIDIA_API_KEY_FILE environment variable (path to key file)
    3. ~/.nvidia_api_key
    4. ~/.config/nvrx/nvidia_api_key
    Returns:
        API key string, or empty string if not found.
    """
    # Check direct env var first
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        return api_key.strip()
    # Check file path from env var
    key_file = os.getenv("NVIDIA_API_KEY_FILE")
    if key_file and os.path.isfile(key_file):
def capture_logs(logger_name=None):
def capture_stdout(logger_name=None):

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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/ft_rendezvous_barrier.py
def _rdzv_signal_exception_handler(sig: int, frame: Optional[FrameType]) -> None:
    del frame
    global _current_joined_state
    if _current_joined_state is not None:
        _current_joined_state._withdraw_on_unwind = True
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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/wrap.py
def reserve_fn(state, store, progress_watchdog, progress_watchdog_interval):

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
def get_log_aggregator_shard_index(num_aggregators: int) -> int:
    """Shard index in ``[0, num_aggregators)`` for first-level log aggregator selection.
    Static topology from minimal infra signals only (same spirit as common SLURM layouts):
def is_process_alive(pid):
def wait_until_process_terminated(pid, timeout=0):
def wait_for_mp_events(events, timeout=60):
def terminate_mp_processes(allowed_pgids):
def set_ipc_socket_timeouts(fileno, timeout):
def recv_all(sock, n):
def read_obj_from_ipc_socket(sock, raise_exc=False):
def write_object_to_ipc_socket(obj, sock):
def get_rank():
def reduce_cuda_ctx_size():
def get_processes_by_pgids(pgids, exclude_launcher=True):
def patched_method(obj, method_name, new_method):
def install_exception_handler():
def _parse_hostname_prefix_suffix(name: str) -> Optional[tuple[str, int, str]]:
    """If name is 'prefix' + digits (e.g. node001, nvl73111-T01), return (prefix, num, raw_suffix). Else None."""
    m = re.match(r"^(.*?)(\d+)$", name)
    if not m:
        return None
    return (m.group(1), int(m.group(2)), m.group(2))
def _numbers_to_slurm_ranges(numbers: list[int], pad: int) -> list[str]:
    """Turn sorted unique numbers into SLURM range parts, e.g. [1,2,3,5] -> ['001-003', '005']."""
    if not numbers:
        return []
    ranges: list[str] = []
    start = end = numbers[0]
    for n in numbers[1:]:
        if n == end + 1:
            end = n
        else:
            s, e = str(start).zfill(pad), str(end).zfill(pad)
            ranges.append(f"{s}-{e}" if start != end else s)
            start = end = n
    s, e = str(start).zfill(pad), str(end).zfill(pad)
    ranges.append(f"{s}-{e}" if start != end else s)
    return ranges
def hostnames_to_slurm_nodelist(addrs: list) -> str:
    """Convert a list of node addresses (hostnames or FQDNs) to SLURM node range format.
    Expects hostnames like "node001", "node002", "node005" or "node001.cluster.com".
    Uses the first component (before '.') as the node name. If all names match
    pattern prefix + numeric suffix, produces e.g. "node[001-002,005]". Otherwise
    returns comma-separated list of (short) hostnames.
    Args:
        addrs: List of participant addresses (hostname or FQDN).
    Returns:
        SLURM-style node list string (range format or comma-separated).
    """
    if not addrs:
        return ""
    short_names = [a.split(".")[0].strip() for a in addrs if a]
    if not short_names:
        return ""
    # Parse each name as prefix + numeric suffix; if any fail, return comma-separated list.
    parsed: list[tuple[str, int, str] | None] = [
        _parse_hostname_prefix_suffix(n) for n in short_names
    ]
    if any(p is None for p in parsed):

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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/c10d_monkey_patch.py
def _wait_for_tcp_store_server(
    host: str,
    port: int,
    max_wait_seconds: float,
    probe_timeout_seconds: float = _DEFAULT_STORE_PROBE_TIMEOUT_SECONDS,
    probe_interval_seconds: float = _DEFAULT_STORE_PROBE_INTERVAL_SECONDS,
    probe_interval_jitter_fraction: float = _DEFAULT_STORE_PROBE_INTERVAL_JITTER_FRACTION,
) -> None:
    """
    Wait until a TCP server is accepting connections on host:port using short
    probes. Use this before creating a TCPStore client so the server has time
    to come up without consuming the full read_timeout on a single connect.
    Each probe is a single TCP connect then close (low cost per client). At
    scale (e.g. 10K workers), jitter is applied to the retry interval to avoid
    thundering herd on the TCPStore server.
    Raises:
        TimeoutError: If the server is not reachable within max_wait_seconds.
    """
    deadline = time.monotonic() + max_wait_seconds
    # Stagger first probe to spread load when many workers start together.
    # Cap stagger so a short max_wait_seconds still leaves room for actual probes.
    max_stagger = min(probe_interval_seconds * 0.5, max_wait_seconds * 0.2)
    initial_stagger = random.uniform(0, max(0.0, max_stagger))
    time.sleep(min(initial_stagger, max(0.0, deadline - time.monotonic())))
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with socket.create_connection((host, port), timeout=probe_timeout_seconds):
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
    # Optional: wait for TCPStore server with short probes before connecting as client.
    # This avoids burning the full read_timeout on a single connect when the server
    # is not up yet (e.g. store host starts after workers).
    store_connect_wait_seconds = params.get_as_int("store_connect_wait_seconds", 0) or 0
    if not is_host and store_connect_wait_seconds > 0:
        logger.debug(
            "Waiting up to %ds for TCPStore server at %s:%s (short probes).",
            store_connect_wait_seconds,
            host,
            port,
        )
        _wait_for_tcp_store_server(host, port, max_wait_seconds=store_connect_wait_seconds)
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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/cycle_info_writer.py
def _cycle_info_filename(job_id: str, attempt_index: int, cycle_number: int) -> str:
    return f"cycle_info.{job_id}.{attempt_index}.{cycle_number}"
def _current_symlink_name(job_id: str) -> str:
    return f"cycle_info.{job_id}.current"
class _CycleInfoTask:
    """Single task for the writer thread."""
    CREATE = "create"
    UPDATE = "update"
    SHUTDOWN = "shutdown"
    def __init__(self, op: str, **kwargs: Any):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/log_aggregator.py
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/trace_analyzer/fr_attribution.py
def eprint(*args, **kwargs):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/replication/utils.py
def zip_strict(*args):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/async_ckpt/filesystem_async.py
def _compute_data_structure_key_from_plan(items: List[WriteItem]) -> str:
    """Compute a hash key based on plan items only (no data resolution needed).
    This creates a deterministic key from plan metadata that's available without
    resolving the actual tensor data.
    Args:
        items: List of WriteItem from the plan
    Returns:
        Hex-digest string key representing the data structure
    """
    structure_info = []
    for item in items:
        # Include item metadata that defines the structure
        item_info = (
            item.index.fqn,  # Fully qualified name
            item.type,  # WriteItemType (BYTE_IO or TENSOR)
        )
        # Include metadata from plan (available without resolving data)
        if item.tensor_data is not None:
            # Use tensor metadata from the plan
            data_info = (
                tuple(item.tensor_data.chunk.sizes),  # Tensor chunk shape
                str(item.tensor_data.properties.dtype),  # Data type
            )
        else:
            # For non-tensor data (BYTE_IO), use placeholder
            data_info = (("BYTE_IO",), "BYTE_IO")
        structure_info.append((item_info, data_info))
    # Use SHA-256 for collision resistance and cross-process stability
    # (Python's built-in hash() is randomized per-process and collision-prone)
    return hashlib.sha256(str(structure_info).encode()).hexdigest()
@_disable_gc()
def get_write_results_queue(mp_mode: str = 'spawn') -> mp.Queue:
    """Get or create a multiprocessing queue for write results.
    Args:
        mp_mode (str):
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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/fault_tolerance/launcher.py
def init_node_health_check(endpoint: Optional[str]) -> None:
    global _NODE_HEALTH_CHECK_INSTANCE
    if endpoint:
        _NODE_HEALTH_CHECK_INSTANCE = NodeHealthCheck(endpoint=endpoint)
    else:
        _NODE_HEALTH_CHECK_INSTANCE = None
def get_node_health_check() -> Optional[NodeHealthCheck]:
    return _NODE_HEALTH_CHECK_INSTANCE
def _register_ft_rdzv_handler(impl_type: str = "barrier"):
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
    cycle_info_writer = None
    if is_store_host and config.fault_tol_cfg.cycle_info_dir:
        cycle_info_writer = CycleInfoWriter(config.fault_tol_cfg.cycle_info_dir)
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
        cycle_info_writer=cycle_info_writer,
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
            logger.info("Agent .run() returned None (hot spare or standby exited successfully).")
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
def _parse_bool_env(var_name: str) -> bool:
    """Parse a boolean-like env var. Unset means False."""
    raw = os.getenv(var_name)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(
        f"Invalid value for {var_name}: {raw!r}. "
        "Expected one of: 1/0, true/false, yes/no."
    )
def _validate_slurm_single_launcher_per_node() -> None:
    """
    Validate Slurm launch shape to prevent accidental multiple ft_launchers per node.
    Override for intentional simulation:
      NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE=1
    """
    if os.getenv("SLURM_JOB_ID") is None:
        return
    if _parse_bool_env("NVRX_ENABLE_MULTI_LAUNCHERS_PER_NODE"):
def _validate_args(args: Any) -> None:
    """Centralized validation of CLI args (cross-flag consistency). Raises ValueError if invalid."""
    n_log_agg = int(getattr(args, "ft_log_aggregator_count", 2))
    if n_log_agg < 1:
        raise ValueError(
            f"--ft-log-aggregator-count must be >= 1, got {n_log_agg}"
        )
    _validate_slurm_single_launcher_per_node()
    if getattr(args, "ft_cycle_info_dir", None) and not getattr(args, "ft_per_cycle_applog_prefix", None):
def config_from_args(args, launcher_pipe_read_fd=None, launcher_log_file=None) -> Tuple[LaunchConfig, Union[Callable, str], List[str]]:
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    _validate_args(args)
    if getattr(args, "ft_nvrx_logfile", None):
def run_script_path(training_script: str, *training_script_args: str):
def _grpc_log_path(base_log_file: str, suffix: str) -> str:
    if base_log_file.endswith('.log'):
def _start_grpc_log_servers(
    args: Any, base_log_file: str, funnel_ports: LogFunnelPorts
) -> List[subprocess.Popen]:
    """
    On TCP store host: start root log server (Lustre writer), and optionally N leaf
    aggregators that forward to the root. In two-level mode the root is given
    ``2 * graceful_shutdown_timeout`` so it outlasts leaf drain after SIGTERM.
    Returns:
        Non-empty list of Popen [root] or [root, leaf0, ...] on success, [] on failure.
    """
    graceful_shutdown_timeout = getattr(args, 'ft_log_server_graceful_shutdown_timeout', 60.0)
    _, max_nodes = parse_min_max_nnodes(args.nnodes)
    def _open_log(path: str):
def _grpc_child_wait_timeout_after_terminate(graceful_shutdown_timeout: float) -> float:
    """Seconds to wait after SIGTERM before SIGKILL for a gRPC funnel child.
    Matches each server's graceful client-wait, ``server.stop(grace=5)``, and
    ``wait_for_termination`` budget (see ``grpc_log_server`` / ``grpc_log_leaf_server``).
    """
    return float(graceful_shutdown_timeout) + 5.0 + 65.0
def _wait_grpc_subprocess_after_terminate(p: subprocess.Popen, wait_timeout: float) -> None:
    """Wait for one gRPC funnel child after SIGTERM; force-kill if still alive past timeout."""
    try:
        p.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            f"gRPC server PID={p.pid} did not shut down within {wait_timeout}s, killing..."
        )
        with contextlib.suppress(Exception):
def run(args):
def main(args=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/state.py
def freeze_dataclass(cls):

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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/inject_fault.py
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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/grpc_log_server.py
def _format_grpc_peer(raw: Optional[str]) -> str:
    """Decode URL-encoded characters in ``context.peer()`` for readable log lines."""
    if not raw:
        return "unknown_peer"
    return urllib.parse.unquote(raw)
class LogAggregationServicer(log_aggregation_pb2_grpc.LogAggregationServiceServicer):
def serve(host: str, port: int, max_workers: int = 100, graceful_shutdown_timeout: float = 60.0):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/replication/torch_device_utils.py
def get_default_device_from_type(device_type: str) -> torch.device:
    """Returns the default PyTorch device based on the specified device type.
    This function maps a device type string to the corresponding PyTorch device.
    It supports both "cpu" and "cuda" types, raising an error for unsupported types.
    Args:
        device_type (str):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/async_ckpt/core.py
def _set_process_qos(cpu_priority: int, io_priority: Optional[int]) -> None:
    """
    Set QoS (Quality of Service) for the current checkpoint writer process.
    This ensures checkpoint writing doesn't interfere with training.
    Args:
        cpu_priority: Nice value for CPU scheduling (0-19, higher = lower priority).
                     Default 10 is moderately deprioritized.
        io_priority: ionice scheduling class. If None, I/O priority is unchanged.
                    Valid values (0-3):
def abort_nvrx_checkpoint():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/shared_utils/grpc_log_leaf_server.py
def _copy_chunk(chunk: log_aggregation_pb2.LogChunk) -> log_aggregation_pb2.LogChunk:
    return log_aggregation_pb2.LogChunk(
        node_id=chunk.node_id,
        data=chunk.data,
        file_path=chunk.file_path,
    )
def _format_grpc_peer(raw: Optional[str]) -> str:
    """Decode URL-encoded characters in ``context.peer()`` for readable log lines."""
    if not raw:
        return "unknown_peer"
    return urllib.parse.unquote(raw)
class _LeafChunkQueue:
    """Per-leaf buffer from many StreamLogs handlers to one upstream forwarder."""
    def __init__(self, max_chunks: int):
def serve(
    host: str,
    port: int,
    upstream: str,
    max_workers: int,
    max_queue_chunks: int,
    graceful_shutdown_timeout: float,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(process)5s] %(filename)s:%(lineno)d %(message)s',
    )
    chunk_queue = _LeafChunkQueue(max_queue_chunks)
    stop_forwarder = threading.Event()
    reject_new = threading.Event()
    addr = f'{host}:{port}'
    forwarder = _UpstreamForwarder(upstream, chunk_queue, stop_forwarder)
    forwarder.start()
    logger.info(
        f"Leaf waiting for upstream root {upstream!r} to become healthy "
        f"(timeout={int(_LEAF_UPSTREAM_ROOT_READY_TIMEOUT_S)}s); downstream port {addr} "
        "will not accept connections until upstream is ready — clients may get ECONNREFUSED."
    )
    if not forwarder.upstream_ready.wait(timeout=_LEAF_UPSTREAM_ROOT_READY_TIMEOUT_S):

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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/monitor_thread.py
def async_raise(tid, exc_type, event=None):
def delayed_async_raise(tid, exc_type):
def reraise_if_unraisable(exc_type):
def async_abort_main_thread(msg=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/rank_assignment.py
def bounded_activate(node, counter, path=None, current_state=None):
def propagate_terminations(node, terminated_ranks):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/param_utils.py
def check_type(annotation, cls):
def count_type_in_params(fn, cls):
def substitute_param_value(fn, args, kwargs, subs):
def enforce_subclass(argument, class_or_tuple):
def enforce_type(argument, class_or_tuple):
def enforce_value(condition):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/log_analyzer/utils.py
def parse_llm_response(raw_text: str) -> ParsedLLMResponse:
    """
    Parse raw LLM response text to extract structured fields.
    The expected format from log_analyzer is:
        <auto_resume_decision>
        <auto_resume_explanation>
        ...
        Attribution: <attribution_text>
        <checkpoint_saved>
    Args:
        raw_text: Raw text from LLM response
    Returns:
        ParsedLLMResponse with extracted fields
    """
    # Extract auto_resume (first line) and explanation (second line)
    lines = raw_text.split("\n")
    auto_resume = lines[0] if lines else ""
    if len(lines) > 1:
        auto_resume_explanation = lines[1]
    else:
        auto_resume_explanation = ""
        logger.warning("Failed to extract auto_resume_explanation: insufficient lines in response")
    # Extract text after 'Attribution:' marker
    attribution_parts = raw_text.split("Attribution:")
    if len(attribution_parts) > 1:
        attribution_section = attribution_parts[1].strip()
        parts = attribution_section.split("\n\n")
        attribution_text = parts[0].replace('"\\', "").replace('\\"', "")
        if len(parts) > 1:
            checkpoint_saved = parts[1]
        else:
            checkpoint_saved = "false"
            logger.debug("No checkpoint_saved field in attribution response")
    else:
        attribution_text = ""
        checkpoint_saved = "false"
        # For ERRORS NOT FOUND, missing Attribution: marker is expected
        if "ERRORS NOT FOUND" in auto_resume:
            logger.debug("No 'Attribution:' marker in LLM response (expected for ERRORS NOT FOUND)")
        else:
            logger.warning("No 'Attribution:' marker found in LLM response")
    # Normalize checkpoint_saved to int flag
    checkpoint_saved_flag = 0
    if isinstance(checkpoint_saved, str) and checkpoint_saved.strip().lower() != "false":
        checkpoint_saved_flag = 1
    return ParsedLLMResponse(
        auto_resume=auto_resume,
        auto_resume_explanation=auto_resume_explanation,
        attribution_text=attribution_text,
        checkpoint_saved_flag=checkpoint_saved_flag,
    )
@dataclass
class JobMetadata:
    """Metadata extracted from log file path."""
    job_id: str
    cycle_id: int
def extract_job_metadata(log_path: str, warn_on_missing_job_id: bool = True) -> JobMetadata:
    """
    Extract job ID and cycle ID from log file path.
    Job ID: tried in order of specificity; see JOB_ID_PATTERNS (inline comments).
    Cycle ID: from ..._cycle<N>.log (see CYCLE_LOG_PATTERN).
    Args:
        log_path: Path to the log file
        warn_on_missing_job_id: If True, log warning when job ID extraction fails.
            Set to False when job_id is provided externally (e.g., from POST request).
    Returns:
        JobMetadata with extracted fields (empty/zero if not found)
    """
    # Try each job ID pattern in order (most specific first)
    job_id = ""
    for pattern in JOB_ID_PATTERNS:
        match = re.search(pattern, log_path)
        if match:
            job_id = match.group(1)
            break
    if not job_id and warn_on_missing_job_id:
        logger.warning(f"Failed to extract job ID from path: {log_path}")
    # Extract cycle ID from path pattern: _cycle<N>.log
    match = re.search(CYCLE_LOG_PATTERN, log_path)
    if match:
        cycle_id = int(match.group(1))
    else:
        cycle_id = 0
        logger.debug(f"No cycle ID in path (not a per-cycle log):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/replication/group_utils.py
def batched(iterable, n):
def parse_group_sequence(replication_jump, replication_factor, world_size):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/combined_log_fr/combined_log_fr.py
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/postprocessing/slack.py
def get_slack_stats() -> SlackStats:
    """Get current Slack statistics."""
    return _slack_stats
def get_slack_user_id(user_id: str, token: str) -> str | None:
    """Look up Slack user ID from NVIDIA email.
    Args:
        user_id: NVIDIA username (will be converted to {user_id}@nvidia.com)
        token: Slack bot token
    Returns:
        Slack user ID if found, None otherwise
    """
    if not HAS_SLACK:
        logger.warning("slack-sdk not installed, cannot look up user")
        return None
    _slack_stats.user_lookups += 1
    client = WebClient(token=token)
    try:
        result = client.users_lookupByEmail(email=f"{user_id}@nvidia.com")
        return result.get("user", {}).get("id")
    except SlackApiError as e:
        _slack_stats.user_not_found += 1
        logger.error(f"Error fetching Slack user for {user_id}: {e.response['error']}")
        return None
def send_slack_notification(
    data: dict,
    slack_bot_token: str,
    slack_channel: str,
) -> bool:
    """Send attribution result to Slack channel.
    Args:
        data: Attribution result dict with keys:
            - s_job_id: Job ID
            - s_user: Username
            - s_attribution: Attribution text
            - s_auto_resume_explanation: Explanation of why job shouldn't restart
        slack_bot_token: Slack bot OAuth token
        slack_channel: Slack channel name or ID
    Returns:
        True if notification sent successfully, False otherwise
    """
    if not HAS_SLACK:
        logger.warning("slack-sdk not installed, cannot send notification")
        return False
    if not slack_bot_token:
        logger.debug("Slack notification skipped: no bot token configured")
        return False
    if not slack_channel:
        logger.warning("Slack notification skipped: no channel configured")
        return False
    client = WebClient(token=slack_bot_token)
    # Try to mention the user
    slack_user_id = get_slack_user_id(data.get("s_user", ""), slack_bot_token)
    mention = f"\n<@{slack_user_id}>" if slack_user_id else ""
    if not slack_user_id and data.get("s_user"):
def should_notify_slack(auto_resume: str) -> bool:
    """Check if this attribution result should trigger a Slack notification.
    Args:
        auto_resume: The auto_resume field from attribution result
    Returns:
        True if should notify (terminal failure), False otherwise
    """
    return auto_resume == AUTO_RESUME_TERMINAL
def maybe_send_slack_notification(data: dict) -> None:
    """If Slack is configured and this result is terminal, send notification.
    Called from post_results after the custom post_fn. No-op if token/channel
    unset or not a terminal result.
    """
    if (
        config.slack_bot_token
        and config.slack_channel
        and should_notify_slack(data.get("s_auto_resume", ""))
    ):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/mcp_integration/server_launcher.py
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/monitor_process.py
def is_process_active(process):
def terminate_process(
    process: psutil.Process, termination_grace_time: datetime.timedelta, log: logging.Logger
):
def daemonize_fn(fn, fn_args=(), fn_kwargs=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/log_analyzer/splitlog.py
def _escape_glob(s: str) -> str:
    """Escape glob/fnmatch metacharacters so the string is matched literally."""
    # Order: [ and ] first so we don't escape the brackets we add for * and ?
    s = s.replace("[", "[[]")
    s = s.replace("]", "[]]")
    s = s.replace("*", "[*]")
    s = s.replace("?", "[?]")
    return s
# Default polling interval for splitlog mode jobs
DEFAULT_POLL_INTERVAL_SECONDS = 300.0  # 5 minutes
# TTL for terminated jobs (1 hour after termination)
DEFAULT_TERMINATED_JOB_TTL_SECONDS = 3600.0  # 1 hour
# TTL for non-terminated jobs (6 months max age)
DEFAULT_MAX_JOB_AGE_SECONDS = 180 * 24 * 3600.0  # 6 months
class SplitlogTracker:
    """
    Background tracker for split logging mode jobs.
    This class manages jobs that write separate log files to a LOGS_DIR for each
    scheduler restart. It does NOT own job storage - it uses callbacks to access
    Job objects stored in LogAnalyzer._jobs.
    Key responsibilities:
    - Background polling thread that runs every poll_interval seconds
    - Detects new scheduler restarts by parsing slurm output for << START PATHS >> markers
    - Discovers log files in LOGS_DIR using configurable glob patterns
    - Triggers analysis for complete files (all files except the last active one)
    - Cleans up terminated jobs after TTL expiration
    Thread safety:
    - All job state access is protected by self._lock
    - Analysis is triggered via ThreadPoolExecutor to avoid blocking async event loop
    Callbacks (set by LogAnalyzer during initialization):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/basic_state_dict.py
def nested_values(x: Union[dict, list]):
def dict_list_map_inplace(f, x):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/checkpointing/local/replication/_torch_future.py
def call_with_only_valid_kwargs(fn, **kwargs):
def object_to_tensor(obj, current_device=None, group=None):
def tensor_to_object(tensor, tensor_size, group=None):
def send_object_list(object_list, dst, group=None, device=None):
def recv_object_list(object_list, src=None, group=None, device=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/mcp_integration/registry.py
def serialize_result(result: Any) -> str:
    """Serialize attribution result to JSON string."""
    if result is None:
        return json.dumps(None)
    if is_dataclass(result):

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

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/log_analyzer/slurm_parser.py
def parse_slurm_output(content: str) -> SlurmOutputInfo:
    """
    Parse a SLURM output file to extract LOGS_DIR and cycle information.
    Args:
        content: Full content of the SLURM output file
    Returns:
        SlurmOutputInfo with extracted information
    """
    # Count cycles by counting lines that ARE the marker (not just contain it)
    # This avoids false positives from log output containing the marker text
    cycle_count = _count_marker_lines(content, START_PATHS_MARKER)
    # Extract LOGS_DIR from the LAST << START PATHS >> block
    # (spec Section 13.4: use latest LOGS_DIR if it changes between restarts)
    logs_dir = _extract_logs_dir(content)
    # Check for Requeue=1 indicator (can restart)
    has_requeue = _check_requeue(content)
    return SlurmOutputInfo(
        logs_dir=logs_dir,
        cycle_count=cycle_count,
        has_requeue=has_requeue,
    )
def _count_marker_lines(content: str, marker: str) -> int:
    """
    Count lines that ARE the marker (with optional surrounding whitespace).
    This avoids false positives from log output that happens to contain
    the marker text as part of a longer line.
    Args:
        content: File content to search
        marker: Marker string to match
    Returns:
        Number of lines that match the marker exactly
    """
    count = 0
    for line in content.splitlines():
def _extract_logs_dir(content: str) -> Optional[str]:
    """
    Extract LOGS_DIR from the SLURM output content.
    Looks for LOGS_DIR= within << START PATHS >> blocks.
    Uses the LAST occurrence per spec Section 13.4.
    Uses line-based parsing to avoid false positives from log output
    that contains the marker text as part of a longer line.
    Args:
        content: SLURM output content
    Returns:
        LOGS_DIR path or None if not found
    """
    logs_dir = None
    lines = content.splitlines()
    in_block = False
    block_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == START_PATHS_MARKER:
            in_block = True
            block_count += 1
            logger.debug(f"_extract_logs_dir: entered START PATHS block #{block_count}")
            continue
        if stripped == END_PATHS_MARKER:
            in_block = False
            logger.debug(f"_extract_logs_dir: exited END PATHS block #{block_count}")
            continue
        # Look for LOGS_DIR= within a block
        if in_block:
            match = LOGS_DIR_PATTERN.match(stripped)
            if match:
                logs_dir = match.group(1).strip().rstrip("/")
                logger.debug(
                    f"_extract_logs_dir: found LOGS_DIR={logs_dir} in block #{block_count}"
                )
    if logs_dir:
        logger.debug(f"_extract_logs_dir: final LOGS_DIR={logs_dir} (from {block_count} blocks)")
    elif block_count > 0:
        logger.debug(f"_extract_logs_dir: no LOGS_DIR found in {block_count} START PATHS blocks")
    return logs_dir
def _check_requeue(content: str) -> bool:
    """
    Check if the job has Requeue=1 (can be restarted).
    This can appear in the SLURM output as part of job info or
    in the environment variables.
    Uses line-based matching to reduce false positives, but allows
    these patterns to appear as part of a line (e.g., in scontrol output).
    Args:
        content: SLURM output content
    Returns:
        True if Requeue=1 is found
    """
    for line in content.splitlines():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/compose.py
def find_common_ancestor(*instances):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/log_analyzer/nvrx_logsage.py
def lines_after(lines, needle):
def chunk_logs_strict(lines):
def main():

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/inprocess/attribution.py
def format_interruption_records(records):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/straggler/dist_utils.py
def all_gather_object(obj, group):
def get_world_size(group):
def get_rank(group):
def get_device_for_backend(group):
def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
def gather_on_rank0(tensor, group=None):

third_party/nvidia-resiliency-ext/src/nvidia_resiliency_ext/attribution/mcp_integration/mcp_client.py
def get_server_command() -> List[str]:
    """
    Resolve and return the server launcher command for the MCP client.
    Returns:
        Command list to launch the MCP server subprocess.
    """
    pkg = "nvidia_resiliency_ext.attribution.mcp_integration"
    try:
        resource = pkg_files(pkg).joinpath("server_launcher.py")
    except Exception as e:
        raise FileNotFoundError(f"failed to locate server_launcher.py in package {pkg}: {e}")
    if not resource.exists():
def create_mcp_client() -> "NVRxMCPClient":
    """
    Create and return an NVRxMCPClient with the default server command.
    Returns:
        Configured NVRxMCPClient ready for use as async context manager.
    """
    return NVRxMCPClient(get_server_command())
class NVRxMCPClient:
    """
    Client for interacting with NVRX Attribution MCP servers.
    Supports:
    - Calling individual attribution modules
    - Running multi-module pipelines
    - Cross-server module composition
    - Result caching and retrieval
    """
    def __init__(self, server_command: List[str]):
