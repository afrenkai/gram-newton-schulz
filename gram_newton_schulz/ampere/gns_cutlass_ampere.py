from functools import cache
from typing import cast

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
    make_fake_tensor,
)
from torch import Tensor

from .gns_cutlass_ampere_kernel import AmpereTensorOpGemm, MirrorLowerTriangle
from .gns_cutlass_ampere_utils import (
    CompiledBmmKernel,
    CompiledMirrorKernel,
    MatrixBackend,
    cutlass_supports_problem,
)


def cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    raise TypeError("Ampere CuTe GEMM requires float16 or bfloat16 tensors")


def select_cutlass_tactic(output_rows: int) -> int:
    if output_rows <= 128:
        return 1
    if output_rows <= 768:
        return 2
    return 0


def cutlass_tactic_config(
    tactic: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if tactic == 0:
        return (128, 128, 32), (2, 2, 1)
    if tactic == 1:
        return (64, 64, 32), (2, 2, 1)
    if tactic == 2:
        return (64, 128, 32), (2, 2, 1)
    raise ValueError(f"CUTLASS tactic must be 0, 1, or 2, got {tactic}")


def cutlass_compile_options(device_capability: tuple[int, int]) -> str:
    major, minor = device_capability
    return f"--enable-tvm-ffi --gpu-arch sm_{major}{minor}"


@cache
def compile_cutlass_bmm(
    dtype: torch.dtype,
    left_column_major: bool,
    right_column_major: bool,
    tactic: int,
    symmetric: bool,
    read_accumulator: bool,
    device_capability: tuple[int, int],
) -> CompiledBmmKernel:
    if device_capability[0] != 8:
        raise ValueError("Ampere CuTe GEMM requires an SM8X GPU")

    element_type = cutlass_dtype(dtype)
    output_rows = cute.sym_int(divisibility=8) if left_column_major else cute.sym_int()
    output_columns = cute.sym_int(divisibility=8)
    inner_dimension = cute.sym_int(divisibility=8)
    batch_size = cute.sym_int()
    left_leading_dimension = 0 if left_column_major else 1
    right_leading_dimension = 1 if right_column_major else 0
    left_stride = tuple(
        1 if dimension == left_leading_dimension else cute.sym_int64(divisibility=8)
        for dimension in range(3)
    )
    right_stride = tuple(
        1 if dimension == right_leading_dimension else cute.sym_int64(divisibility=8)
        for dimension in range(3)
    )

    left = make_fake_tensor(
        element_type,
        (output_rows, inner_dimension, batch_size),
        stride=left_stride,
        assumed_align=16,
    )
    right = make_fake_tensor(
        element_type,
        (output_columns, inner_dimension, batch_size),
        stride=right_stride,
        assumed_align=16,
    )
    output = make_fake_compact_tensor(
        element_type,
        (output_rows, output_columns, batch_size),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    accumulator = make_fake_compact_tensor(
        element_type,
        (output_rows, output_columns, batch_size),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    tile_shape, atom_layout = cutlass_tactic_config(tactic)
    operation = AmpereTensorOpGemm(
        element_type,
        element_type,
        cutlass.Float32,
        tile_shape,
        atom_layout,
        symmetric,
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cast(
        CompiledBmmKernel,
        cute.compile(
            operation,
            left,
            right,
            output,
            accumulator,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            read_accumulator,
            stream,
            options=cutlass_compile_options(device_capability),
        ),
    )


@cache
def compile_mirror_kernel(
    dtype: torch.dtype,
    device_capability: tuple[int, int],
) -> CompiledMirrorKernel:
    if device_capability[0] != 8:
        raise ValueError("Ampere CuTe mirror requires an SM8X GPU")

    matrix_size = cute.sym_int(divisibility=8)
    batch_size = cute.sym_int()
    output = make_fake_compact_tensor(
        cutlass_dtype(dtype),
        (batch_size, matrix_size, matrix_size),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cast(
        CompiledMirrorKernel,
        cute.compile(
            MirrorLowerTriangle(),
            output,
            stream,
            options=cutlass_compile_options(device_capability),
        ),
    )


def run_cutlass_bmm(
    left: Tensor,
    right: Tensor,
    accumulator: Tensor,
    output: Tensor,
    alpha: float,
    beta: float,
    tactic: int,
    *,
    symmetric: bool,
) -> None:
    device_capability = torch.cuda.get_device_capability(left.device)
    with torch.cuda.device(left.device):
        compiled_kernel = compile_cutlass_bmm(
            left.dtype,
            not left.is_contiguous(),
            not right.is_contiguous(),
            tactic,
            symmetric,
            beta != 0.0,
            device_capability,
        )
        compiled_kernel(
            left.permute(1, 2, 0),
            right.permute(2, 1, 0),
            output.permute(1, 2, 0),
            accumulator.permute(1, 2, 0),
            alpha,
            beta,
        )


def mirror_symmetric_output(output: Tensor) -> None:
    device_capability = torch.cuda.get_device_capability(output.device)
    with torch.cuda.device(output.device):
        compile_mirror_kernel(output.dtype, device_capability)(output)


def cutlass_is_installed() -> bool:
    return True


@torch.library.custom_op(
    "gram_newton_schulz::cutlass_baddbmm",
    mutates_args=(),
    device_types="cuda",
)
def cutlass_baddbmm_cuda(
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
            "Tensor shape, layout, dtype, or device is unsupported by SM8X CuTe"
        )
    selected_tactic = (
        select_cutlass_tactic(left.shape[-2]) if tactic is None else tactic
    )
    cutlass_tactic_config(selected_tactic)
    output = torch.empty_like(accumulator, memory_format=torch.contiguous_format)
    run_cutlass_bmm(
        left,
        right,
        accumulator,
        output,
        alpha,
        beta,
        selected_tactic,
        symmetric=False,
    )
    return output


@cutlass_baddbmm_cuda.register_fake
def cutlass_baddbmm_fake(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    tactic: int | None = None,
) -> Tensor:
    return torch.empty_like(accumulator, memory_format=torch.contiguous_format)


@torch.library.custom_op(
    "gram_newton_schulz::cutlass_symmetric_baddbmm",
    mutates_args=(),
    device_types="cuda",
)
def cutlass_symmetric_baddbmm_cuda(
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
            "SM8X CuTe"
        )
    output = torch.empty_like(accumulator, memory_format=torch.contiguous_format)
    run_cutlass_bmm(
        left,
        right,
        accumulator,
        output,
        alpha,
        beta,
        tactic=0,
        symmetric=True,
    )
    mirror_symmetric_output(output)
    return output


@cutlass_symmetric_baddbmm_cuda.register_fake
def cutlass_symmetric_baddbmm_fake(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    return torch.empty_like(accumulator, memory_format=torch.contiguous_format)


def cutlass_baddbmm(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    tactic: int | None = None,
) -> Tensor:
    return cast(
        Tensor,
        torch.ops.gram_newton_schulz.cutlass_baddbmm.default(
            accumulator,
            left,
            right,
            alpha=alpha,
            beta=beta,
            tactic=tactic,
        ),
    )


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
    return cast(
        Tensor,
        torch.ops.gram_newton_schulz.cutlass_symmetric_baddbmm.default(
            accumulator,
            left,
            right,
            alpha=alpha,
            beta=beta,
        ),
    )


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
