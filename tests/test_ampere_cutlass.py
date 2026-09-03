import unittest

import torch

from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS, GramNewtonSchulz
from gram_newton_schulz.ampere import GramNewtonSchulzAmpere
from gram_newton_schulz.ampere.gns_cutlass_ampere import (
    cutlass_baddbmm,
    cutlass_bmm,
    cutlass_symmetric_baddbmm,
    select_cutlass_tactic,
)


def ampere_is_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 8


def column_major_batch(
    batch_size: int,
    rows: int,
    columns: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(
        batch_size,
        columns,
        rows,
        device="cuda",
        dtype=dtype,
    ).transpose(-2, -1)


def padded_column_major_batch(
    batch_size: int,
    rows: int,
    columns: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix_elements = rows * columns
    batch_stride = matrix_elements + 8
    storage = torch.randn(
        batch_size * batch_stride,
        device="cuda",
        dtype=dtype,
    )
    return storage.as_strided(
        (batch_size, columns, rows),
        (batch_stride, rows, 1),
    ).transpose(-2, -1)


class CutlassTacticTest(unittest.TestCase):
    def test_fake_metadata_and_fullgraph(self) -> None:
        left = torch.empty(2, 72, 88, device="meta", dtype=torch.float16)
        right = torch.empty(2, 88, 80, device="meta", dtype=torch.float16)
        compiled_bmm = torch.compile(cutlass_bmm, backend="eager", fullgraph=True)
        output = compiled_bmm(left, right, tactic=1)
        self.assertEqual(output.shape, (2, 72, 80))
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(output.device.type, "meta")
        self.assertTrue(output.is_contiguous())

    def test_thresholds(self) -> None:
        self.assertEqual(select_cutlass_tactic(128), 1)
        self.assertEqual(select_cutlass_tactic(129), 2)
        self.assertEqual(select_cutlass_tactic(768), 2)
        self.assertEqual(select_cutlass_tactic(769), 0)


@unittest.skipUnless(ampere_is_available(), "requires an SM8X GPU")
class AmpereCutlassTest(unittest.TestCase):
    def test_baddbmm_tactics_and_layouts(self) -> None:
        batch_size, rows, columns, inner = 2, 72, 80, 88
        for dtype in (torch.float16, torch.bfloat16):
            left = torch.randn(
                batch_size,
                rows,
                inner,
                device="cuda",
                dtype=dtype,
            )
            accumulator = torch.randn(
                batch_size,
                rows,
                columns,
                device="cuda",
                dtype=dtype,
            )
            for right in (
                torch.randn(
                    batch_size,
                    inner,
                    columns,
                    device="cuda",
                    dtype=dtype,
                ),
                column_major_batch(batch_size, inner, columns, dtype),
            ):
                reference = torch.baddbmm(
                    accumulator,
                    left,
                    right,
                    alpha=1.25,
                    beta=-0.5,
                )
                for tactic in (0, 1, 2):
                    output = cutlass_baddbmm(
                        accumulator,
                        left,
                        right,
                        alpha=1.25,
                        beta=-0.5,
                        tactic=tactic,
                    )
                    torch.testing.assert_close(
                        output,
                        reference,
                        atol=2e-2,
                        rtol=2e-2,
                    )

    def test_orthogonalizer_fullgraph(self) -> None:
        coefficients = POLAR_EXPRESS_COEFFICIENTS[:2]
        reference_operation = GramNewtonSchulz(
            ns_use_kernels=False,
            ns_coefficients=coefficients,
            gram_newton_schulz_reset_iterations=[],
            compile_kwargs=None,
        )
        ampere_operation = GramNewtonSchulzAmpere(
            ns_coefficients=coefficients,
            gram_newton_schulz_reset_iterations=[],
            compile_kwargs={"fullgraph": True, "mode": "reduce-overhead"},
        )
        for rows, columns in ((384, 256), (256, 256)):
            input_tensor = torch.randn(
                1,
                rows,
                columns,
                device="cuda",
                dtype=torch.bfloat16,
            )
            reference = reference_operation(input_tensor)
            output = ampere_operation(input_tensor)
            torch.testing.assert_close(
                output,
                reference,
                atol=5e-2,
                rtol=5e-2,
            )

    def test_padded_column_major_batch_stride(self) -> None:
        batch_size, rows, columns, inner = 2, 72, 80, 88
        left = torch.randn(
            batch_size,
            rows,
            inner,
            device="cuda",
            dtype=torch.float16,
        )
        right = padded_column_major_batch(
            batch_size,
            inner,
            columns,
            torch.float16,
        )
        accumulator = torch.randn(
            batch_size,
            rows,
            columns,
            device="cuda",
            dtype=torch.float16,
        )
        output = cutlass_baddbmm(accumulator, left, right, tactic=1)
        reference = torch.baddbmm(accumulator, left, right)
        torch.testing.assert_close(
            output,
            reference,
            atol=2e-2,
            rtol=2e-2,
        )

    def test_beta_zero_does_not_read_accumulator(self) -> None:
        left = torch.randn(1, 72, 88, device="cuda", dtype=torch.float16)
        right = torch.randn(1, 88, 80, device="cuda", dtype=torch.float16)
        accumulator = torch.full(
            (1, 72, 80),
            torch.nan,
            device="cuda",
            dtype=torch.float16,
        )
        output = cutlass_baddbmm(
            accumulator,
            left,
            right,
            beta=0.0,
            tactic=1,
        )
        torch.testing.assert_close(
            output,
            torch.bmm(left, right),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_fullgraph_and_cuda_graph(self) -> None:
        left = torch.randn(1, 72, 88, device="cuda", dtype=torch.float16)
        right = torch.randn(1, 88, 80, device="cuda", dtype=torch.float16)
        reference = torch.bmm(left, right)

        compiled_bmm = torch.compile(cutlass_bmm, backend="eager", fullgraph=True)
        compiled_output = compiled_bmm(left, right, tactic=1)
        torch.testing.assert_close(
            compiled_output,
            reference,
            atol=2e-2,
            rtol=2e-2,
        )

        cutlass_bmm(left, right, tactic=1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = cutlass_bmm(left, right, tactic=1)
        graph.replay()
        torch.testing.assert_close(
            graph_output,
            reference,
            atol=2e-2,
            rtol=2e-2,
        )

    def test_symmetric_layouts_and_stream(self) -> None:
        batch_size, matrix_size, inner = 2, 72, 88
        for dtype in (torch.float16, torch.bfloat16):
            base = torch.randn(
                batch_size,
                matrix_size,
                inner,
                device="cuda",
                dtype=dtype,
            )
            symmetric_accumulator = torch.randn(
                batch_size,
                matrix_size,
                matrix_size,
                device="cuda",
                dtype=dtype,
            )
            symmetric_accumulator = (
                symmetric_accumulator + symmetric_accumulator.mT
            ).contiguous()
            column_base = torch.randn(
                batch_size,
                inner,
                matrix_size,
                device="cuda",
                dtype=dtype,
            )
            padded_column_base = padded_column_major_batch(
                batch_size,
                matrix_size,
                inner,
                dtype,
            )
            symmetric_operand = torch.randn(
                batch_size,
                matrix_size,
                matrix_size,
                device="cuda",
                dtype=dtype,
            )
            symmetric_operand = (symmetric_operand + symmetric_operand.mT).contiguous()
            operands = (
                (base, base.mT),
                (column_base.mT, column_base),
                (padded_column_base, padded_column_base.mT.contiguous()),
                (symmetric_operand, symmetric_operand),
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for left, right in operands:
                    output = cutlass_symmetric_baddbmm(
                        symmetric_accumulator,
                        left,
                        right,
                        alpha=0.75,
                        beta=0.25,
                    )
                    reference = torch.baddbmm(
                        symmetric_accumulator,
                        left,
                        right,
                        alpha=0.75,
                        beta=0.25,
                    )
                    torch.testing.assert_close(
                        output,
                        reference,
                        atol=2e-2,
                        rtol=2e-2,
                    )
                    self.assertTrue(torch.equal(output, output.mT))
            stream.synchronize()


if __name__ == "__main__":
    unittest.main()
