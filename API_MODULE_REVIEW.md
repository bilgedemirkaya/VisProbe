# API Module Review - Suggested Improvements

## High Priority Issues

### 1. Fix Global Mutable State (`registry.py`)

**Current Problem:**
```python
GIVEN_TESTS: List[Callable] = []
SEARCH_TESTS: List[Callable] = []
```

**Solution:**
```python
import threading
from typing import Callable, List

class TestRegistry:
    """Thread-safe test registry."""
    _local = threading.local()

    @classmethod
    def get_given_tests(cls) -> List[Callable]:
        if not hasattr(cls._local, 'given_tests'):
            cls._local.given_tests = []
        return cls._local.given_tests

    @classmethod
    def get_search_tests(cls) -> List[Callable]:
        if not hasattr(cls._local, 'search_tests'):
            cls._local.search_tests = []
        return cls._local.search_tests

    @classmethod
    def clear(cls):
        """Clear all registered tests."""
        if hasattr(cls._local, 'given_tests'):
            cls._local.given_tests.clear()
        if hasattr(cls._local, 'search_tests'):
            cls._local.search_tests.clear()
```

---

### 2. Fix QueryCounter Hook Safety (`query_counter.py`)

**Current Problem:**
```python
# Unsafe monkey-patching
self.model.forward = _counting_forward
```

**Solution:**
```python
class QueryCounter:
    """Thread-safe context manager using forward hooks."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.count = 0
        self._hook_handle = None

    def _forward_hook(self, module, input, output):
        self.count += 1
        return output

    def __enter__(self):
        self._hook_handle = self.model.register_forward_hook(self._forward_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @property
    def extra(self):
        """Maintain compatibility with existing API."""
        return max(0, self.count - 1)  # Subtract 1 for initial call
```

---

### 3. Remove Duplicate Code (`analysis_utils.py`)

**Delete lines 196-251** - Complete duplicate of `run_corruption_sweep`

---

### 4. Improve Error Handling (`runner.py`, `config.py`)

**Instead of:**
```python
except Exception:
    pass
```

**Use:**
```python
import logging

logger = logging.getLogger(__name__)

except Exception as e:
    logger.warning(f"Failed to configure threading: {e}")
```

---

### 5. Cross-Platform Path Handling (`report.py`)

**Replace:**
```python
results_dir = "/tmp/visprobe_results"
```

**With:**
```python
import tempfile
import os

def get_results_dir() -> str:
    """Get platform-appropriate results directory."""
    env_dir = os.environ.get('VISPROBE_RESULTS_DIR')
    if env_dir:
        return env_dir
    return os.path.join(tempfile.gettempdir(), 'visprobe_results')
```

---

### 6. Optimize Binary Search (`search_modes.py`)

**Current (line 92):**
```python
while step > min_step and runner.query_count < max_queries:
    # Step-halving search
    if not batch_pass:
        level_next = level - step
        step *= 0.5
    else:
        level_next = level + step
```

**Better:**
```python
def perform_binary_search(runner, params: Dict[str, Any], clean_results: Tuple) -> Dict:
    """True binary search for failure threshold."""
    lo, hi = params['initial_level'], params.get('max_level', 1.0)

    while (hi - lo) > params['min_step'] and runner.query_count < params['max_queries']:
        mid = (lo + hi) / 2.0
        passed = _evaluate_at_level(runner, mid, clean_results)

        if passed:
            lo = mid  # Increase perturbation
        else:
            hi = mid  # Decrease perturbation

    return {"failure_threshold": hi, ...}
```

---

### 7. Better Device Detection (`config.py`)

**Add support for:**
```python
def get_default_device() -> torch.device:
    """Smart device selection with broader hardware support."""
    env_device = os.environ.get("VISPROBE_DEVICE", "").lower().strip()

    if env_device and env_device != "auto":
        return torch.device(env_device)

    prefer_gpu = os.environ.get("VISPROBE_PREFER_GPU", "").lower() in ("1", "true", "yes")

    if prefer_gpu:
        # Try CUDA first
        if torch.cuda.is_available():
            return torch.device("cuda")

        # Try MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")

        # Try ROCm (AMD)
        if hasattr(torch, 'hip') and torch.hip.is_available():
            return torch.device("hip")

    return torch.device("cpu")
```

---

### 8. Add Type Hints Throughout

**Example for `decorators.py`:**
```python
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
R = TypeVar('R')

def model(
    model_obj: nn.Module,
    *,
    capture_intermediate_layers: Optional[List[str]] = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attaches a model to a VisProbe test."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func._visprobe_model = model_obj  # type: ignore
        func._visprobe_capture_intermediate_layers = capture_intermediate_layers  # type: ignore
        return func
    return decorator
```

---

### 9. Memory-Efficient Report Generation

**In `report.py`, add streaming support:**
```python
def save_streaming(self, chunk_size: int = 8192):
    """Save report using streaming to reduce memory usage."""
    results_dir = get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{self.test_name}.json")

    with open(file_path, "w") as f:
        # Write images separately, reference by path
        data = self._build_serializable_dict(results_dir=results_dir)
        json.dump(data, f, cls=NumpyEncoder, indent=2)
```

---

### 10. Add Caching for Expensive Operations

**For resolution analysis:**
```python
from functools import lru_cache

@lru_cache(maxsize=32)
def _get_resized_tensor(
    tensor_hash: int,
    resolution: Tuple[int, int]
) -> torch.Tensor:
    """Cache resized tensors to avoid redundant operations."""
    # Implementation
```

---

## Medium Priority Issues

### 11. Add Validation to Decorators

```python
def given(*, strategy: Strategy, ...):
    def decorator(user_func: Callable) -> Callable:
        # Validate function signature
        import inspect
        sig = inspect.signature(user_func)
        if len(sig.parameters) < 2:
            raise ValueError(
                f"Test function {user_func.__name__} must accept "
                f"at least 2 parameters (original, perturbed)"
            )
        # ... rest of decorator
```

---

### 12. Add Progress Callbacks

```python
from typing import Protocol

class ProgressCallback(Protocol):
    def on_query(self, count: int, total: int) -> None: ...
    def on_level_tested(self, level: float, passed: bool) -> None: ...

def given(
    *,
    strategy: Strategy,
    progress_callback: Optional[ProgressCallback] = None,
    ...
):
    # Use callback during execution
```

---

### 13. Improve Strategy Configuration

**Current:**
```python
try:
    perturb_obj.configure(mean=ctx['mean'], std=ctx['std'], seed=ctx['seed'])
except Exception:
    self._configure_strategy(perturb_obj, ctx['mean'], ctx['std'], None)
```

**Better:**
```python
def configure_strategy_safe(
    strategy: Strategy,
    mean: List[float],
    std: List[float],
    seed: Optional[int]
) -> bool:
    """Safely configure strategy with detailed error reporting."""
    try:
        if hasattr(strategy, 'configure'):
            strategy.configure(mean=mean, std=std, seed=seed)
            return True
    except TypeError as e:
        logger.warning(f"Strategy {strategy.__class__.__name__} "
                      f"configure() has incompatible signature: {e}")
        # Fallback to attribute setting
        if hasattr(strategy, 'mean'):
            strategy.mean = mean
        if hasattr(strategy, 'std'):
            strategy.std = std
        return True
    except Exception as e:
        logger.error(f"Failed to configure strategy: {e}")
        return False
```

---

## Summary of Improvements

| Issue | Severity | Lines Affected | Estimated Fix Time |
|-------|----------|----------------|-------------------|
| Global mutable state | High | registry.py:8-9 | 30 min |
| QueryCounter safety | High | query_counter.py:23 | 20 min |
| Duplicate code | Medium | analysis_utils.py:196-251 | 5 min |
| Poor error handling | High | runner.py (multiple) | 2 hours |
| Hard-coded paths | Medium | report.py:145 | 15 min |
| Missing type hints | Medium | All files | 4 hours |
| Inefficient search | Medium | search_modes.py:92-180 | 1 hour |
| Limited device support | Low | config.py:17-40 | 30 min |
| Memory efficiency | Low | report.py (multiple) | 1 hour |

**Total Estimated Effort:** ~10 hours of development + testing
