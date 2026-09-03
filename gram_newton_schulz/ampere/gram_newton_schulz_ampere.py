import torch
from torch import Tensor

from ..gram_newton_schulz import (
    _TORCH_BACKEND,
    GramNewtonSchulz,
    _make_compiled_gram,
    _make_compiled_standard,
)
from .gns_cutlass_ampere import CutlassBackend, cutlass_is_installed


class GramNewtonSchulzAmpere(GramNewtonSchulz):
    """Gram Newton-Schulz using CUTLASS kernels on NVIDIA Ampere GPUs.

    The class reuses the repository's Newton-Schulz closures and public call path.
    Rectangular matrix products use the same PyTorch backend as the base class.
    """

    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_coefficients: list[list[float]] | None = None,
        gram_newton_schulz_reset_iterations: list[int] | None = None,
        compile_kwargs: dict[str, object] | None = None,
    ) -> None:
        """Initialize the Ampere Gram Newton-Schulz orthogonalizer.

        ``compile_kwargs`` are forwarded to the internal PyTorch closures. The
        CuTe DSL launches remain registered custom-operator boundaries.
        """
        super().__init__(
            ns_epsilon=ns_epsilon,
            ns_use_kernels=False,
            ns_coefficients=ns_coefficients,
            use_gram_newton_schulz=True,
            gram_newton_schulz_reset_iterations=(gram_newton_schulz_reset_iterations),
            compile_kwargs=compile_kwargs,
        )
        backend = CutlassBackend(fallback=_TORCH_BACKEND)
        self._gram_kernel = _make_compiled_gram(
            backend,
            self.ns_coefficients,
            self.gram_newton_schulz_reset_iterations,
            ns_epsilon,
            compile_kwargs,
        )
        self._standard_kernel = _make_compiled_standard(
            backend,
            self.ns_coefficients,
            ns_epsilon,
            compile_kwargs,
        )
        self._kernel_backend = backend
        self.ns_use_kernels = True

    def __call__(self, X: Tensor) -> Tensor:
        if not X.is_cuda:
            raise ValueError("GramNewtonSchulzAmpere requires a CUDA tensor")
        if torch.cuda.get_device_capability(X.device)[0] != 8:
            raise ValueError("GramNewtonSchulzAmpere requires an SM8X GPU")
        if not cutlass_is_installed():
            raise RuntimeError("GramNewtonSchulzAmpere requires nvidia-cutlass-dsl")
        return super().__call__(X)
