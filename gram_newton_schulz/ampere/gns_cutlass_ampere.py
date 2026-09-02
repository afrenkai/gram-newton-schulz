from collections.abc import Callable
from functools import cache
from hashlib import sha256
from importlib.resources import as_file, files
from typing import cast

import torch
from torch import Tensor

from .gns_cutlass_ampere_utils import (
    CutlassJitModule,
    FlashInferJitSpec,
    MatrixBackend,
    cutlass_supports_problem,
)

try:
    from flashinfer.jit import gen_jit_spec  # ty: ignore[unresolved-import]
except ImportError:
    gen_jit_spec: Callable[..., object] | None = None


@cache
def load_cutlass_module() -> CutlassJitModule:
    if gen_jit_spec is None:
        raise RuntimeError(
            "The CUTLASS backend requires flashinfer-python with JIT support"
        )

    source = files(__package__).joinpath("gns_cutlass_ampere.cu")
    source_digest = sha256(source.read_bytes()).hexdigest()[:12]
    with as_file(source) as source_path:
        specification = cast(
            FlashInferJitSpec,
            gen_jit_spec(
                f"gram_newton_schulz_ampere_sm80_{source_digest}",
                (source_path,),
                extra_cuda_cflags=[
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_80,code=compute_80",
                ],
            ),
        )
        module = specification.build_and_load()
    return cast(CutlassJitModule, module)


def cutlass_is_installed() -> bool:
    return gen_jit_spec is not None


def select_cutlass_tactic(output_rows: int) -> int:
    if output_rows <= 128:
        return 1
    if output_rows <= 768:
        return 2
    return 0


def cutlass_baddbmm(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    tactic: int | None = None,
) -> Tensor:
    if not cutlass_supports_problem(
        accumulator,
        left,
        right,
        symmetric=False,
    ):
        raise ValueError(
            "Tensor shape, layout, dtype, or device is unsupported by SM8X CUTLASS"
        )
    selected_tactic = (
        select_cutlass_tactic(left.shape[-2]) if tactic is None else tactic
    )
    if selected_tactic not in {0, 1, 2}:
        raise ValueError(f"CUTLASS tactic must be 0, 1, or 2, got {selected_tactic}")
    output = torch.empty_like(accumulator)
    right_column_major = not right.is_contiguous()
    load_cutlass_module().cutlass_baddbmm(
        accumulator,
        left,
        right,
        output,
        alpha,
        beta,
        selected_tactic,
        right_column_major,
    )
    return output


def cutlass_bmm(
    left: Tensor,
    right: Tensor,
    *,
    tactic: int | None = None,
) -> Tensor:
    output_shape = (*left.shape[:-2], left.shape[-2], right.shape[-1])
    accumulator = torch.empty(output_shape, dtype=left.dtype, device=left.device)
    return cutlass_baddbmm(
        accumulator,
        left,
        right,
        alpha=1.0,
        beta=0.0,
        tactic=tactic,
    )


def cutlass_symmetric_baddbmm(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    if not cutlass_supports_problem(
        accumulator,
        left,
        right,
        symmetric=True,
    ):
        raise ValueError(
            "Tensor shape, layout, dtype, or device is unsupported by symmetric "
            "SM8X CUTLASS"
        )
    output = torch.empty_like(accumulator)
    mirror_tile_size = (
        64
        if left.shape[0] >= 8
        and torch.cuda.get_device_capability(left.device) == (8, 0)
        else 32
    )
    load_cutlass_module().cutlass_symmetric_baddbmm(
        accumulator,
        left,
        right,
        output,
        alpha,
        beta,
        not left.is_contiguous(),
        not right.is_contiguous(),
        mirror_tile_size,
    )
    return output


def cutlass_symmetric_bmm(left: Tensor, right: Tensor) -> Tensor:
    output_shape = (*left.shape[:-2], left.shape[-2], right.shape[-1])
    accumulator = torch.empty(output_shape, dtype=left.dtype, device=left.device)
    return cutlass_symmetric_baddbmm(
        accumulator,
        left,
        right,
        alpha=1.0,
        beta=0.0,
    )


class CutlassBackend:
    def __init__(self, fallback: MatrixBackend) -> None:
        self.fallback = fallback

    def sym_mm(self, left: Tensor, right: Tensor) -> Tensor:
        return cutlass_symmetric_bmm(left, right)

    def sym_baddbmm(
        self,
        left: Tensor,
        right: Tensor,
        C: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Tensor:
        return cutlass_symmetric_baddbmm(
            C,
            left,
            right,
            alpha=alpha,
            beta=beta,
        )

    def mm(self, left: Tensor, right: Tensor) -> Tensor:
        square_problem = (
            left.shape[-2] == left.shape[-1] and left.shape[-1] == right.shape[-1]
        )
        if not square_problem:
            return self.fallback.mm(left, right)
        return cutlass_bmm(left, right)

    def mm_add(
        self,
        left: Tensor,
        right: Tensor,
        C: Tensor,
        beta: float,
    ) -> Tensor:
        square_problem = (
            left.shape[-2] == left.shape[-1] and left.shape[-1] == right.shape[-1]
        )
        if not square_problem:
            return self.fallback.mm_add(left, right, C, beta)
        return cutlass_baddbmm(
            C,
            left,
            right,
            beta=beta,
        )
