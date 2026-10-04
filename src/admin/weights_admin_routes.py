# SysMeAIEngine - Private AI Model Engine
# Copyright 2026 Jose Rodriguez
#
# Developer: Jose Rodriguez <jrpcone@gmail.com>
# GitHub:    https://github.com/jorodriguezpr
# Site:      https://microrepair.net
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Weight explorer + editor UI."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..core.model_loader import get_loaded, load_model, tensor_stats
from ..core.state import EngineState
from ..api.chat_routes import get_state

router = APIRouter()


@router.get("/ui/weights", response_class=HTMLResponse)
def weights_page(request: Request, state: EngineState = Depends(get_state)):
    # The default model may be unregistered (fresh install) or a backend with no
    # PyTorch tensors (Ollama / llama.cpp handles are plain dicts) -- show why
    # instead of a 500.
    notice = None
    try:
        rows = tensor_stats(state)
    except ValueError as e:
        rows, notice = [], str(e)
    except AttributeError:
        rows = []
        notice = f"The current model '{state.config.default_model}' doesn't expose PyTorch weight tensors."
    return request.app.state.templates.TemplateResponse(
        request,
        name="weights.html",
        context={"title": "Weights", "rows": rows, "rows_json": json.dumps(rows), "active": "weights",
                 "notice": notice},
    )


class WeightEditIn(BaseModel):
    tensor: str
    op: str  # scale | bias | clip | sparsify
    value: Optional[float] = None  # scaling factor / clip threshold / sparsity amount


@router.post("/api/weights/edit")
def edit_weight(body: WeightEditIn, state: EngineState = Depends(get_state)):
    loaded = get_loaded(state)
    if loaded is None:
        load_model(state, state.config.default_model)
        loaded = get_loaded(state)
    assert loaded is not None

    model = loaded.model
    if not hasattr(model, "named_parameters"):
        raise HTTPException(status_code=400, detail="model does not expose named_parameters")

    import torch  # lazy

    target = None
    for name, p in model.named_parameters():
        if name == body.tensor:
            target = p
            break
    if target is None:
        raise HTTPException(status_code=404, detail=f"tensor {body.tensor} not found")

    with torch.no_grad():
        if body.op == "scale":
            if body.value is None:
                raise HTTPException(status_code=400, detail="scale requires value")
            target.mul_(float(body.value))
        elif body.op == "bias":
            if body.value is None:
                raise HTTPException(status_code=400, detail="bias requires value")
            target.add_(float(body.value))
        elif body.op == "clip":
            if body.value is None:
                raise HTTPException(status_code=400, detail="clip requires value (abs max)")
            target.clamp_(min=-abs(body.value), max=abs(body.value))
        elif body.op == "sparsify":
            if body.value is None:
                raise HTTPException(
                    status_code=400,
                    detail="sparsify requires value in [0,1) (fraction to zero)",
                )
            q = float(body.value)
            if q <= 0 or q >= 1:
                raise HTTPException(status_code=400, detail="value must be in [0,1)")
            flat = target.abs().flatten()
            k = max(1, int(flat.numel() * q))
            thresh = torch.kthvalue(flat, k).values
            mask = target.abs() >= thresh
            target.mul_(mask)
        else:
            raise HTTPException(status_code=400, detail=f"unknown op '{body.op}'")

    # persist TinyGPT automatically
    if hasattr(model, "tokenizer") and hasattr(model, "cfg"):
        ckpt = loaded.registry_entry and state.config.weights_dir / loaded.name / "minilm.pt"
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)

    return {"ok": True, "tensor": body.tensor, "op": body.op, "value": body.value}
