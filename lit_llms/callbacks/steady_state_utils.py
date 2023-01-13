import math
from typing import Optional


def is_steady_state(*metrics: float, rtol: float = 0.015, atol: Optional[float] = None) -> bool:
    mean_metric = sum(metrics) / len(metrics)
    min_metric = min(metrics)
    max_metric = max(metrics)
    return _check_tols(max_metric, mean_metric, rtol, atol) and _check_tols(mean_metric, min_metric, rtol, atol)


def _check_atol(val_a: float, val_b: float, atol: Optional[float]) -> bool:
    return (atol is None) or (abs(val_a - val_b) <= atol)


def _check_rtol(val_a: float, val_b: float, rtol: float) -> bool:
    return abs(val_a - val_b) <= (rtol * val_b)


def _check_tols(val_a: float, val_b: float, rtol: float, atol: Optional[float]) -> bool:
    return _check_atol(val_a, val_b, atol) and _check_rtol(val_a, val_b, rtol)


def chinchilla_metric_samples(final_loss: float, num_params: int) -> int:
    # D = \frac{410}{L-1.69-\frac{406.4}{N^{0.34}}^{\frac{1}{0.27}}
    return int(math.ceil((410 / (final_loss - 1.69 - 406.4 / num_params**0.34)) ** (1 / 0.27)))


def calc_total_time_per_node(num_samples: int, num_procs: int, batch_size: int, time_per_batch: float) -> float:
    num_samples_per_proc = num_samples / num_procs
    num_batches = num_samples_per_proc / batch_size
    return time_per_batch * num_batches / 60 / 60  # time from secs to hours
