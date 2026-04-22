# Custom Ops Library

Pre-built postprocessing (and preprocessing) factory functions for the BioImage Model Zoo,
implemented as part of `bioimageio.core`.

---

## How custom ops work

Model contributors ship a custom postprocessing op **inline** with the model package
during development.  Once the op is accepted here, it gets its own named `id` in the
spec and no longer needs `source`/`sha256`/`callable`.

### Development (inline source)

Write a Python file and put it alongside your weights.
Reference it in `rdf.yaml`:

```yaml
postprocessing:
  - id: custom
    callable: my_postprocess        # class or function name
    source: my_postprocess.py       # packaged with the model
    sha256: <sha256 of the file>    # required
    kwargs:                         # all optional
      threshold: 0.5
```

Compute the sha256:
```bash
python -c "import hashlib; print(hashlib.sha256(open('my_postprocess.py','rb').read()).hexdigest())"
```

### Promotion to built-in

1. Open a PR adding `my_postprocess.py` to this folder
2. Once merged, the op gets its own named `id` in `bioimageio.spec.model.v0_5`
3. Models can then drop `source`/`sha256`/`callable` and use the new id directly

---

## Writing an op — two supported styles

### Style 1 — Callable class (recommended for ops with configuration)

```python
# my_postprocess.py
import numpy as np

class my_postprocess:
    def __init__(self, threshold: float = 0.5) -> None:
        """kwargs from rdf.yaml arrive here."""
        self.threshold = threshold

    def __call__(self, *arrays: np.ndarray) -> np.ndarray:
        """
        Model output tensors arrive here in rdf.yaml declaration order.
        Each array is a numpy.ndarray.
        Must return a single numpy.ndarray.
        """
        return (arrays[0] > self.threshold).astype(np.uint8)
```

### Style 2 — Factory function (closure over kwargs)

```python
# my_postprocess.py
import numpy as np

def my_postprocess(threshold: float = 0.5):
    """kwargs from rdf.yaml arrive here. Return the per-image function."""
    def run(*arrays: np.ndarray) -> np.ndarray:
        return (arrays[0] > threshold).astype(np.uint8)
    return run
```

Both styles work identically. The runtime does:
```python
op = callable(**kwargs)   # __init__ or factory called once
result = op(*tensors)     # __call__ or inner function called per image
```

---

## Rules for contributed ops

- **One file per op**, filename = callable name (e.g. `cellpose_flow_dynamics.py`)
- **Imports**: only `numpy`, `scipy`, `scikit-image`, `torch`, `torchvision`,
  `tensorflow`, `onnxruntime`, `bioimageio.core` — no custom packages
- **Signature**: `callable(**kwargs)` returns something that accepts `*arrays` and returns `np.ndarray`
- **Docstring**: explain what the op does, expected tensor order, and what it returns
- **No side effects**: the op must be stateless across images (state held in `self` or closure is fine)

---

## Available built-in ops

| File | Callable | Description |
|------|----------|-------------|
| [`cellpose_flow_dynamics.py`](cellpose_flow_dynamics.py) | `cellpose_flow_dynamics` | Decode Cellpose/Cellpose-SAM flow fields into instance labels |
