from typing import Protocol

import torch
from torch import Tensor


class CompiledBmmKernel(Protocol):
    def __call__(
        self,
        left: Tensor,
        right: Tensor,
        output: Tensor,
        accumulator: Tensor,
        alpha: float,
        beta: float,
    ) -> None: ...


class CompiledMirrorKernel(Protocol):
    def __call__(self, output: Tensor) -> None: ...


class MatrixBackend(Protocol):
    def sym_mm(self, left: Tensor, right: Tensor) -> Tensor: ...

    def sym_baddbmm(
        self,
        left: Tensor,
        right: Tensor,
        C: Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Tensor:
        """Return ``alpha * left @ right + beta * C`` for a symmetric result.

        Callers must provide operands whose product and accumulator are symmetric.
        """
        ...

    def mm(self, left: Tensor, right: Tensor) -> Tensor: ...

    def mm_add(
        self,
        left: Tensor,
        right: Tensor,
        C: Tensor,
        beta: float,
    ) -> Tensor: ...


def cutlass_has_supported_layout(tensor: Tensor) -> bool:
    """Return whether a tensor is contiguous row-major or column-major.

    Column-major tensors must have a unit row stride and a packed column stride.

    :param tensor: Tensor whose two matrix dimensions are checked.
    """
    column_major = tensor.stride(-2) == 1 and tensor.stride(-1) == tensor.shape[-2]
    return tensor.is_contiguous() or column_major


def cutlass_supports_problem(
    accumulator: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    symmetric: bool,
) -> bool:
    """Return whether a batched matrix product meets the CUTLASS contract.

    The contract covers SM8X devices, dtype and device agreement, tensor shapes,
    packed matrix layouts, pointer alignment, batch strides, and integer bounds.

    :param accumulator: Contiguous tensor added to the matrix product.
    :param left: Left row-major or column-major matrix batch.
    :param right: Right row-major or column-major matrix batch.
    :param symmetric: Whether the output must use a supported symmetric layout.
    """
    common_problem_is_supported = (
        left.is_cuda
        and torch.cuda.get_device_capability(left.device)[0] == 8
        and left.dtype in {torch.float16, torch.bfloat16}
        and accumulator.dtype == left.dtype == right.dtype
        and accumulator.device == left.device == right.device
        and accumulator.ndim == left.ndim == right.ndim == 3
        and 0 < left.shape[0] <= 65_535
        and min(*accumulator.shape, *left.shape, *right.shape) > 0
        and max(*accumulator.shape[1:], *left.shape[1:], *right.shape[1:])
        <= 2_147_483_647
        and left.shape[0] == right.shape[0] == accumulator.shape[0]
        and left.shape[2] == right.shape[1]
        and left.shape[1] == accumulator.shape[1]
        and right.shape[2] == accumulator.shape[2]
        and accumulator.is_contiguous()
        and left.shape[2] % 8 == 0
        and right.shape[2] % 8 == 0
        and not (
            accumulator.data_ptr() % 16 or left.data_ptr() % 16 or right.data_ptr() % 16
        )
        and not (accumulator.stride(0) % 8 or left.stride(0) % 8 or right.stride(0) % 8)
    )
    if not common_problem_is_supported:
        return False
    if not symmetric:
        return left.is_contiguous() and cutlass_has_supported_layout(right)

    left_column_major = not left.is_contiguous()
    right_column_major = not right.is_contiguous()
    return (
        accumulator.shape[1] == accumulator.shape[2]
        and cutlass_has_supported_layout(left)
        and cutlass_has_supported_layout(right)
        and not (left_column_major and right_column_major)
    )
