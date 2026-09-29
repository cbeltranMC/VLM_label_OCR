"""Background sampler for GPU (NVML) and vLLM server (/metrics) stats during a run."""

import re
import threading
import time
import urllib.request

try:
    import pynvml
except ImportError:
    pynvml = None

MIB = 1024 ** 2


def _metric(text: str, name: str) -> float | None:
    """Sum a Prometheus gauge/counter over all its label sets (None if absent)."""
    values = re.findall(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.eE+-]+)$", text, re.M)
    return sum(float(v) for v in values) if values else None


class Monitor:
    """Samples GPU memory/utilization/power and vLLM KV-cache usage every `interval` s.

    GPU numbers are device-wide: they include whatever else runs on the GPU,
    and vLLM reserves its memory up front, so `baseline` is already most of it.
    """

    def __init__(self, metrics_url: str, gpu_index: int = 0, interval: float = 0.2):
        self.metrics_url = metrics_url
        self.interval = interval
        self.handle = None
        self.samples = []  # (mem_used_mib, util_pct, power_w)
        self.kv_usage = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except pynvml.NVMLError as e:
                print(f"GPU monitoring disabled: {e}")

    def _energy_mj(self):
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except pynvml.NVMLError:
            return None

    def _scrape(self) -> str | None:
        try:
            with urllib.request.urlopen(self.metrics_url, timeout=2) as r:
                return r.read().decode()
        except OSError:
            return None

    def _sample(self):
        if self.handle is not None:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / MIB
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
            except pynvml.NVMLError:
                power = None
            self.samples.append((mem, util, power))
        text = self._scrape()
        if text:
            usage = _metric(text, "vllm:kv_cache_usage_perc")
            if usage is not None:
                self.kv_usage.append(usage)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self.gpu_info = {}
        if self.handle is not None:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.gpu_info = {
                "gpu_name": pynvml.nvmlDeviceGetName(self.handle),
                "gpu_mem_total_mib": round(mem.total / MIB),
                "gpu_mem_baseline_mib": round(mem.used / MIB),
            }
        # The KV cache size is a label of the vllm:cache_config_info gauge.
        capacity = re.search(r'kv_cache_size_tokens="(\d+)"', self._scrape() or "")
        self.kv_capacity = int(capacity[1]) if capacity else None
        self._energy_start = self._energy_mj() if self.handle is not None else None
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        elapsed = time.perf_counter() - self._t0
        stats = dict(self.gpu_info)
        if self.samples:
            mems, utils, powers = zip(*self.samples)
            powers = [p for p in powers if p is not None]
            stats["gpu_mem_peak_mib"] = round(max(mems))
            stats["gpu_util_mean_pct"] = round(sum(utils) / len(utils), 1)
            stats["gpu_util_peak_pct"] = max(utils)
            if powers:
                stats["gpu_power_mean_w"] = round(sum(powers) / len(powers), 1)
                stats["gpu_power_peak_w"] = round(max(powers), 1)
            energy_end = self._energy_mj()
            if self._energy_start is not None and energy_end is not None:
                stats["gpu_energy_j"] = round((energy_end - self._energy_start) / 1000, 1)
            elif powers:
                stats["gpu_energy_j"] = round(stats["gpu_power_mean_w"] * elapsed, 1)
        if self.kv_usage:
            peak = max(self.kv_usage)
            stats["kv_cache_peak_pct"] = round(100 * peak, 1)
            if self.kv_capacity:
                stats["kv_cache_capacity_tokens"] = self.kv_capacity
                stats["kv_cache_peak_tokens"] = round(peak * self.kv_capacity)
        if self.handle is not None:
            pynvml.nvmlShutdown()
        return stats
