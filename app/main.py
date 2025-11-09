# app/main.py

print("🔥 main.py loaded!")
import sys
print("✅ Loaded modules:", list(sys.modules.keys())[:10])
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.llm import call_llm_stage1, call_llm_stage2, validate_ir
from app.render import render_manim_scene
from app.render_cnn import render_cnn_scene




app = FastAPI(title="IR-to-Animation")
print("✅ LOADED main.py with /explain and /ir_from_trace endpoints")

# ---- Request/Response models ----
class ExplainReq(BaseModel):
    text: str

class IRFromTraceReq(BaseModel):
    explain: Dict[str, Any]  # stage1에서 받은 JSON 그대로

class GenerateReq(BaseModel):
    text: str
    out_format: str | None = "gif"
    basename: str | None = "result"

class GenerateResp(BaseModel):
    ir: Dict[str, Any]
    file_path: str

class RenderReq(BaseModel):
    ir: Dict[str, Any]
    out_format: str | None = "gif"
    basename: str | None = "embedding_demo"

# ---- Endpoints ----
@app.post("/explain")
def explain(req: ExplainReq) -> Dict[str, Any]:
    """프롬프트1: 알고리즘 설명+예시+trace 생성"""
    return call_llm_stage1(req.text)

@app.post("/ir_from_trace")
def ir_from_trace(req: IRFromTraceReq) -> Dict[str, Any]:
    ir = call_llm_stage2(req.explain)
    if "events" not in ir: ir["events"] = []
    errs = validate_ir(ir)
    print("🧩 Validation errors:", errs)   # 👈 여기에 추가
    if errs:
        raise HTTPException(status_code=422, detail=errs)
    return ir


@app.post("/generate", response_model=GenerateResp)
def generate(req: GenerateReq):
    """전체 파이프라인: text → explain → IR → render"""
    ex = call_llm_stage1(req.text)
    ir = call_llm_stage2(ex)
    errs = validate_ir(ir)
    if errs:
        raise HTTPException(status_code=422, detail=errs)
    path = render_manim_scene(
        ir, out_basename=req.basename or "result", fmt=req.out_format or "gif"
    )  # manim 호출 경로/옵션은 현재 render.py 규약을 그대로 사용:contentReference[oaicite:0]{index=0}
    return GenerateResp(ir=ir, file_path=path)


@app.post("/render_embedding")
def render_embedding(req: dict):
    domain = req.get("ir", {}).get("metadata", {}).get("domain", "sorting")
    print(f"⚙️ Detected domain: {domain}")

    # CNN 파라미터 기반 애니메이션
    if domain == "cnn_param":
        print("🎞 Using render_cnn_matrix() with param config !!")
        from app.render_cnn_matrix import render_cnn_matrix
        cnn_cfg = req.get("ir", {}).get("params", {})
        path = render_cnn_matrix(cnn_cfg, "cnn_param_demo", fmt="mp4")

    elif domain == "cnn":
        print("🧠 Using render_cnn_scene() !!")
        from app.render_cnn import render_cnn_scene
        path = render_cnn_scene(req["ir"], req.get("basename") or "cnn_demo", fmt=req.get("out_format") or "gif")

    elif domain == "sorting":
        print("🔢 Using render_manim_scene() !!")
        from app.render import render_manim_scene
        path = render_manim_scene(req["ir"], req.get("basename") or "embedding_demo", fmt=req.get("out_format") or "gif")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported domain: {domain}")

    print(f"✅ Render complete -> {path}")
    return {"file_path": path}


@app.post("/render_cnn_matrix")
def render_cnn_matrix_endpoint(req: dict):
    from app.render_cnn_matrix import render_cnn_matrix
    path = render_cnn_matrix(req, "cnn_param_demo", fmt="mp4")
    return {"file_path": path}
