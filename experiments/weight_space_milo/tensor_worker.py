"""Container-only straight-line Python tensor workspace. Not arbitrary Python."""

import ast
import json
import resource
from pathlib import Path

import torch

# This intentionally small language excludes imports, loops, comprehensions,
# functions, arbitrary attributes, I/O, random generation, solvers and autograd.
# Docker is the security boundary; this validator restricts the scientific protocol.
ATTRS = set(
    "shape dtype T linalg svd vector_norm matrix_norm norm abs max min mean sum std clone detach float double reshape view flatten transpose permute unsqueeze squeeze item tolist topk sort diagonal trace dot outer matmul mm cat stack tensor zeros zeros_like ones_like eye arange float32 float64 int64 clamp sign sqrt square numel size".split()
)
NODES = (
    ast.Module,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.MatMult,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Attribute,
    ast.keyword,
)
BUILTINS = {"print": print, "len": len, "sorted": sorted, "str": str, "int": int, "float": float}


def validate(code):
    if not isinstance(code, str) or len(code.encode()) > 32768:
        raise ValueError("Python code must be a string of at most 32768 bytes")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, NODES):
            raise ValueError(f"Forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (
            node.id.startswith("_")
            or node.id in {"open", "eval", "exec", "compile", "globals", "locals"}
        ):
            raise ValueError("Forbidden name")
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in {"torch", "state", "original", "observations", "scratch", *BUILTINS}
        ):
            raise ValueError("Cannot rebind runtime names")
        if isinstance(node, ast.Attribute) and node.attr not in ATTRS:
            raise ValueError(f"Forbidden attribute: {node.attr}")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Attribute)
            or isinstance(node.func, ast.Name)
            and node.func.id in BUILTINS
        ):
            raise ValueError("Only approved tensor methods and builtins may be called")
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    raise ValueError("Attribute assignment forbidden")
    return tree


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 * 1024,) * 2)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    request = json.loads(Path("/work/request.json").read_text())
    tree = validate(request["code"])
    state = torch.load("/work/current.pt", map_location="cpu", weights_only=True)
    original = torch.load("/source/model.pt", map_location="cpu", weights_only=True)
    if "model_state_dict" in original:
        original = original["model_state_dict"]
    observations = (
        torch.load("/work/observations.pt", map_location="cpu", weights_only=True)
        if Path("/work/observations.pt").exists()
        else {}
    )
    scratch = (
        torch.load("/work/scratch.pt", map_location="cpu", weights_only=True)
        if Path("/work/scratch.pt").exists()
        else {}
    )
    with Path("/work/output.txt").open("w") as output:
        # Bound textual output before materializing large user-visible logs.
        count = 0

        def bounded_print(*args, **kwargs):
            nonlocal count
            value = " ".join(str(x) for x in args) + "\n"
            count += len(value.encode())
            if count > 65536:
                raise ValueError("Output exceeds 65536 bytes")
            output.write(value)

        scope = {
            "__builtins__": {},
            **BUILTINS,
            "print": bounded_print,
            "torch": torch,
            "state": state,
            "original": original,
            "observations": observations,
            "scratch": scratch,
        }
        exec(compile(tree, "<tensor-request>", "exec"), scope, scope)
    torch.save(state, "/work/result.pt")
    if any(
        not isinstance(k, str) or not isinstance(v, torch.Tensor) or v.numel() > 2000000
        for k, v in scratch.items()
    ):
        raise ValueError("scratch requires string keys and tensors with <=2 million entries")
    if (
        len(scratch) > 64
        or sum(v.numel() * v.element_size() for v in scratch.values()) > 32 * 1024 * 1024
    ):
        raise ValueError("Scratch exceeds64 tensors or32MiB")
    torch.save(scratch, "/work/scratch-result.pt")


if __name__ == "__main__":
    main()
