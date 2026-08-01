#!/usr/bin/env python3
"""Test GatherBlockQuantized with a small embedding weight.

Validates that symmetric_blockwise_quantize output is compatible with
the GatherBlockQuantized operator's expected INT4 format.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import CudaQuantizer
import torch

# ── Create a small embedding table ──────────────────────────────
np.random.seed(42)
vocab, dim = 100, 896  # small test
weight_fp16 = np.random.randn(vocab, dim).astype(np.float16) * 0.02
weight_tensor = torch.from_numpy(weight_fp16.astype(np.float32))

# ── Quantize with symmetric_blockwise_quantize ──────────────────
block_size = 32
qw_uint8, scales_f32 = CudaQuantizer.symmetric_blockwise_quantize(
    weight_tensor, bits=4, block_size=block_size, unsigned_full_range=True
)
# qw_uint8: [vocab, dim/2] packed uint8
# scales_f32: [vocab, dim/block_size] float32
print(f"qw_uint8: shape={list(qw_uint8.shape)}, dtype={qw_uint8.dtype}")
print(f"scales:   shape={list(scales_f32.shape)}, dtype={scales_f32.dtype}")

qw_np = qw_uint8.numpy()
scales_np = scales_f32.numpy()

# ── Create ONNX INT4 tensor from packed uint8 ───────────────────
# ONNX INT4 (data_type=21) stores 2 values per byte, low nibble first
# symmetric_blockwise_quantize also packs 2 values per byte
# We need to create a TensorProto with data_type=INT4 and dims=[vocab, dim]
# The raw_data should be the packed bytes

# For INT4, onnx uses raw_data field with the packed bytes
# dims=[vocab, dim] means vocab*dim 4-bit values = vocab*dim/2 bytes
qw_tensor = TensorProto()
qw_tensor.name = "weight_Q4"
qw_tensor.data_type = 21  # INT4
qw_tensor.dims.extend([vocab, dim])
qw_tensor.raw_data = qw_np.tobytes()

# Scales as FP16
scales_fp16 = scales_np.astype(np.float16)
scales_tensor = numpy_helper.from_array(scales_fp16, name="weight_scales")

# Zero points: symmetric → 8 (midpoint of 0-15 unsigned range)
# Create as INT4 tensor, all 8s
n_blocks = dim // block_size
zp_bytes = np.full(vocab * n_blocks, 8, dtype=np.uint8)
# Actually for symmetric with unsigned_full_range, zero point might be 0
# Let's try 0 first (symmetric means zp=0 in the quantized domain)
zp_tensor = TensorProto()
zp_tensor.name = "weight_zero_points"
zp_tensor.data_type = 21  # INT4
zp_tensor.dims.extend([vocab, n_blocks])
# Pack: each byte holds 2 zero point values (both = value)
zp_packed = np.full(vocab * n_blocks // 2, 0, dtype=np.uint8)
# If zero_point is 8, pack as low=8, high=8 → 0x88
for i in range(len(zp_packed)):
    zp_packed[i] = (8 & 0x0F) | ((8 & 0x0F) << 4)
zp_tensor.raw_data = zp_packed.tobytes()

# ── Build test ONNX model ───────────────────────────────────────
# Model: indices [1, T] → GatherBlockQuantized → output [1, T, dim]
T = 5
indices = np.array([3, 10, 50, 0, 99], dtype=np.int64).reshape(1, T)

# Expected output (reference)
expected = weight_fp16[indices.flatten()]  # [T, dim]

# Create graph
input_indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, T])
output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, T, dim])

gbq_node = helper.make_node(
    "GatherBlockQuantized",
    inputs=["weight_Q4", "indices", "weight_scales", "weight_zero_points"],
    outputs=["output"],
    domain="com.microsoft",
    block_size=block_size,
    gather_axis=0,
    quantize_axis=1,
)

graph = helper.make_graph(
    [gbq_node],
    "test_gbq",
    [input_indices],
    [output],
    initializer=[qw_tensor, scales_tensor, zp_tensor],
)

opset_imports = [
    helper.make_opsetid("", 17),
    helper.make_opsetid("com.microsoft", 1),
]

model = helper.make_model(graph, opset_imports=opset_imports)
model.ir_version = 10  # compatible with INT4 support

# ── Run and compare ─────────────────────────────────────────────
try:
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    result = sess.run(None, {"indices": indices})
    actual = result[0][0]  # [T, dim]

    # Compare
    max_err = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    mean_err = np.mean(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    expected_abs = np.mean(np.abs(expected.astype(np.float32)))

    print(f"\nExpected (FP16 Gather) mean |val|: {expected_abs:.6f}")
    print(f"Max error:  {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    print(f"Error ratio: {mean_err/expected_abs*100:.2f}%")

    if mean_err / expected_abs < 0.15:
        print("\n✅ GatherBlockQuantized format is COMPATIBLE (error < 15%)")
    else:
        print("\n❌ Format mismatch — trying different zero point...")

    # Show a few values for debugging
    print(f"\nSample expected[0, :4]: {expected[0, :4]}")
    print(f"Sample actual[0, :4]:   {actual[0, :4]}")

except Exception as e:
    print(f"\nError running model: {e}")
    import traceback
    traceback.print_exc()
