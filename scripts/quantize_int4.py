#!/usr/bin/env python3
"""INT4 weight-only quantization for Audio8 TTS ONNX models.

Replaces MatMul nodes with MatMulNBits (bits=4, block_size=128),
matching Audio8's official INT4 ONNX format exactly.

Traces through Transpose nodes to find the weight initializer behind each
MatMul B input. Skips attention MatMuls (Q×K^T, probs×V) which have dynamic B.

Usage:
    python quantize_int4.py --input slow_ar_fp32.onnx --output slow_ar_int4.onnx
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.cuda_quantizer import CudaQuantizer


def quantize_model(input_path: str, output_path: str) -> None:
    print(f"Loading {input_path} ...", flush=True)
    model = onnx.load(input_path, load_external_data=False)
    graph = model.graph

    # Build lookup tables
    init_map = {init.name: init for init in graph.initializer}
    out_map = {}  # output_name -> producing node
    for n in graph.node:
        for o in n.output:
            out_map[o] = n

    matmul_nodes = [n for n in graph.node if n.op_type == "MatMul"]
    print(f"Found {len(matmul_nodes)} MatMul nodes", flush=True)

    quantized_count = 0
    skipped = 0
    nodes_to_remove = set()  # Transpose nodes that become dead after replacement
    new_initializers = []
    replacements = {}  # MatMul node id -> (MatMulNBits node, weight_init_name)

    for node in graph.node:
        if node.op_type != "MatMul":
            continue

        a_name, b_name = node.input[0], node.input[1]

        # Trace B back through Transpose to find weight initializer
        weight_init_name = None
        transpose_node = None
        current = b_name

        if current in out_map:
            producer = out_map[current]
            if producer.op_type == "Transpose":
                transpose_node = producer
                # Transpose input should be the weight initializer
                for inp in producer.input:
                    if inp in init_map:
                        weight_init_name = inp
                        break

        if weight_init_name is None:
            skipped += 1
            continue

        b_init = init_map[weight_init_name]
        if b_init.data_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            # Skip non-float initializers
            skipped += 1
            continue

        # Load weight [N, K] (PyTorch nn.Linear stores weight as [out_features, in_features])
        b_array = numpy_helper.to_array(b_init, base_dir=os.path.dirname(input_path))
        if b_array.ndim != 2:
            skipped += 1
            continue

        N, K = b_array.shape  # [out_features, in_features]
        # Convert to FP32 just for quantization math (model may be FP16)
        b_tensor = torch.from_numpy(np.ascontiguousarray(b_array.astype(np.float32)))

        # Quantize: matmulnbits_blockwise_quantize expects [N, K] and returns
        # qweight [N, K/block_size, block_size*bits/8] when flatten=False
        result = CudaQuantizer.matmulnbits_blockwise_quantize(
            b_tensor,
            bits=4,
            block_size=128,
            symmetric=False,
            return_zero_points=True,
            flatten_qweight=False,
        )
        qweight, scales, zero_points = result

        qweight_np = qweight.numpy().astype(np.uint8)
        scales_np = scales.numpy().astype(np.float16)
        # Zero points are already packed (2x4-bit per uint8) by matmulnbits_blockwise_quantize
        zp_np = zero_points.numpy().astype(np.uint8)

        # Create initializer names
        qw_name = f"{weight_init_name}_Q4"
        scales_name = f"{weight_init_name}_scales"
        zp_name = f"{weight_init_name}_zero_points"

        new_initializers.append(numpy_helper.from_array(qweight_np, name=qw_name))
        new_initializers.append(numpy_helper.from_array(scales_np, name=scales_name))
        new_initializers.append(numpy_helper.from_array(zp_np, name=zp_name))

        # Create MatMulNBits node (replaces both Transpose + MatMul)
        # MatMulNBits is a com.microsoft custom domain op
        mnb_node = helper.make_node(
            "MatMulNBits",
            inputs=[a_name, qw_name, scales_name, zp_name],
            outputs=node.output,
            name=node.name + "_int4" if node.name else None,
            K=K,
            N=N,
            bits=4,
            block_size=128,
            accuracy_level=4,
            domain="com.microsoft",
        )
        replacements[id(node)] = mnb_node
        if transpose_node is not None:
            nodes_to_remove.add(id(transpose_node))
        quantized_count += 1

    # Rebuild node list: replace MatMuls, remove dead Transposes
    new_nodes = []
    for node in graph.node:
        nid = id(node)
        if nid in replacements:
            new_nodes.append(replacements[nid])
        elif nid in nodes_to_remove:
            continue  # skip dead Transpose
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    # Remove quantized weight initializers, add new ones
    quantized_weight_names = set()
    for n in new_nodes:
        if n.op_type == "MatMulNBits":
            # B input is {weight_name}_Q4, so weight_name = B_input[:-3]
            quantized_weight_names.add(n.input[1][:-3])

    kept_inits = [init for init in graph.initializer if init.name not in quantized_weight_names]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)
    graph.initializer.extend(new_initializers)

    print(f"Quantized {quantized_count} MatMul → MatMulNBits, skipped {skipped}", flush=True)

    # Add com.microsoft opset import for MatMulNBits
    has_ms_domain = any(opi.domain == "com.microsoft" for opi in model.opset_import)
    if not has_ms_domain:
        model.opset_import.append(
            helper.make_opsetid("com.microsoft", 1)
        )

    # Save with external data
    onnx.save(
        model,
        output_path,
        save_as_external_data=True,
        size_threshold=0,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
    )
    print(f"Saved {output_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="INT4 weight-only quantization for ONNX models")
    ap.add_argument("--input", required=True, help="Input ONNX model (FP32 recommended)")
    ap.add_argument("--output", required=True, help="Output INT4 ONNX model")
    args = ap.parse_args()
    quantize_model(args.input, args.output)


if __name__ == "__main__":
    main()
