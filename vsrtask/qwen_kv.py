from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _visual_blocks(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    positions = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("No visual tokens found in Qwen processor output")
    positions = positions.tolist()
    blocks: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            blocks.append((start, previous + 1))
            start = position
        previous = position
    blocks.append((start, previous + 1))
    return blocks


def build_image_inputs(processor: Any, image: Image.Image) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this frame."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )


def find_kv_modules(model: torch.nn.Module) -> tuple[list[tuple[str, Any]], list[str]]:
    modules: list[tuple[str, Any]] = []
    prefixes: set[str] = set()
    for name, module in model.named_modules():
        if name.endswith("self_attn.k_proj"):
            modules.append(("k", module))
            prefixes.add(name.rsplit(".self_attn.", 1)[0])
        elif name.endswith("self_attn.v_proj"):
            modules.append(("v", module))
            prefixes.add(name.rsplit(".self_attn.", 1)[0])
    layer_names = sorted(
        prefixes,
        key=lambda name: (
            int(name.rsplit(".", 1)[-1])
            if name.rsplit(".", 1)[-1].isdigit()
            else name
        ),
    )
    if not layer_names:
        raise RuntimeError("Could not find Qwen self_attn.k_proj/v_proj modules")
    return modules, layer_names


def capture_edited_kv(
    model: torch.nn.Module,
    processor: Any,
    image_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K/V as [qwen_layers, visual_tokens, kv_heads*head_dim]."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    return capture_edited_kv_image(model, processor, image, device, dtype)


def capture_edited_kv_image(
    model: torch.nn.Module,
    processor: Any,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K/V for one PIL image as [qwen_layers, visual_tokens, kv_dim]."""
    captured_k: dict[int, torch.Tensor] = {}
    captured_v: dict[int, torch.Tensor] = {}
    modules, layer_names = find_kv_modules(model)
    layer_index = {
        name: index for index, name in enumerate(layer_names)
    }
    hooks = []

    def make_hook(index: int, kind: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            target = captured_k if kind == "k" else captured_v
            target[index] = value.detach().float().cpu()

        return hook

    for kind, module in modules:
        # named_modules is not retained in modules, so derive the layer by
        # matching module identity against the model's named module list.
        module_name = next(
            name
            for name, candidate in model.named_modules()
            if candidate is module
        )
        prefix = module_name.rsplit(".self_attn.", 1)[0]
        hooks.append(
            module.register_forward_hook(
                make_hook(layer_index[prefix], kind)
            )
        )

    image = image.convert("RGB")
    batch = build_image_inputs(processor, image)
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise RuntimeError("Qwen config has no image_token_id")
    blocks = _visual_blocks(batch["input_ids"], int(image_token_id))
    if len(blocks) != 1:
        raise RuntimeError(f"Expected one visual token block, got {blocks}")
    start, end = blocks[0]
    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda" and dtype != torch.float32,
    ):
        model(**model_inputs, use_cache=False, return_dict=True)

    for hook in hooks:
        hook.remove()
    if len(captured_k) != len(layer_names) or len(captured_v) != len(layer_names):
        raise RuntimeError(
            f"Captured {len(captured_k)} K layers and {len(captured_v)} V layers; "
            f"expected {len(layer_names)}"
        )

    def stack(captured: dict[int, torch.Tensor]) -> torch.Tensor:
        values = []
        for index in range(len(layer_names)):
            value = captured[index][:, start:end].squeeze(0)
            values.append(value)
        return torch.stack(values, dim=0)

    return stack(captured_k), stack(captured_v)
