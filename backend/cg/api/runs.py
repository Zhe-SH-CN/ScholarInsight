"""Run and research APIs."""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from cg.auth import require_session_user
from cg.orchestrator.pipeline import ResearchPipeline
from cg.repositories.run import RunRepository
from cg.schemas.research import ResearchRequest, RunDetail, RunStarted, RunStatus
from cg.settings import get_settings
from cg.tools.search import SearchTool

router = APIRouter(prefix="/api", tags=["runs"])


class ChatMessage(BaseModel):
    message: str


class RenameRunRequest(BaseModel):
    project_name: str = Field(min_length=1, max_length=120)



@router.post("/runs", response_model=RunStarted)
async def start_run(
    request: ResearchRequest,
    background: BackgroundTasks,
    username: str = Depends(require_session_user),
) -> RunStarted:
    settings = get_settings()
    if not settings.active_llm_api_key:
        raise HTTPException(
            status_code=422,
            detail=(
                "⚠️ 需要配置 LLM API Key 才能启动智能 Agent 研究。"
                "请在 backend/.env 中填写 ARK_API_KEY（豆包）/ DEEPSEEK_API_KEY / QWEN_API_KEY，"
                "重启后端后重新发起研究。"
            ),
        )
    pipeline = ResearchPipeline(settings)
    status = await pipeline.prepare_run(request, owner=username)
    background.add_task(pipeline.run, request, status.run_id)
    return RunStarted(run_id=status.run_id, status=status.status)


@router.get("/runs", response_model=list[RunStatus])
async def list_runs(username: str = Depends(require_session_user)) -> list[RunStatus]:
    settings = get_settings()
    return await RunRepository(settings.data_dir).mark_stale_running(
        settings.cg_run_stale_after_seconds,
        owner=username,
    )


@router.get("/capabilities", response_model=dict)
async def capabilities(username: str = Depends(require_session_user)) -> dict:
    _ = username
    settings = get_settings()
    search = SearchTool(settings)
    active_search_providers = search.active_provider_names
    return {
        "llm_configured": bool(settings.active_llm_api_key),
        "llm_provider": settings.cg_llm_provider,
        "llm_model": settings.cg_llm_model,
        "search_provider": ",".join(active_search_providers) or "none",
        "search_providers": active_search_providers,
        "configured_search_providers": search.provider_names,
        "zhihu_configured": bool(settings.zhihu_api_key),
        "max_search_rounds": settings.cg_max_search_rounds,
        "max_research_loops": settings.cg_max_research_loops,
    }


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: str, username: str = Depends(require_session_user)) -> RunDetail:
    try:
        settings = get_settings()
        repo = RunRepository(settings.data_dir)
        await repo.mark_stale_running(settings.cg_run_stale_after_seconds, owner=username)
        return await repo.detail(run_id, owner=username)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.patch("/runs/{run_id}", response_model=RunStatus)
async def rename_run(
    run_id: str,
    body: RenameRunRequest,
    username: str = Depends(require_session_user),
) -> RunStatus:
    name = body.project_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Project name cannot be empty.")
    try:
        settings = get_settings()
        repo = RunRepository(settings.data_dir)
        return await repo.rename_run(run_id, name, owner=username)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.delete("/runs/{run_id}", status_code=204)
async def delete_run(run_id: str, username: str = Depends(require_session_user)) -> None:
    try:
        settings = get_settings()
        repo = RunRepository(settings.data_dir)
        await repo.delete_run(run_id, owner=username)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.post("/runs/{run_id}/stop", response_model=RunStatus)
async def stop_run(run_id: str, username: str = Depends(require_session_user)) -> RunStatus:
    try:
        settings = get_settings()
        repo = RunRepository(settings.data_dir)
        await repo.assert_access(run_id, username)
        return await repo.request_stop(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc


@router.post("/runs/{run_id}/chat")
async def chat_about_run(
    run_id: str,
    body: ChatMessage,
    username: str = Depends(require_session_user),
) -> dict:
    settings = get_settings()
    if not settings.active_llm_api_key:
        return {
            "response": (
                "LLM 尚未配置。请在 `backend/.env` 中填入 API Key，重启后端即可使用 AI 对话功能。\n\n"
                "支持豆包（ARK_API_KEY）、DeepSeek、Qwen 等 OpenAI 兼容接口。"
            )
        }

    try:
        repo = RunRepository(settings.data_dir)
        detail = await repo.detail(run_id, owner=username)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}") from exc

    parts: list[str] = [
        f"**研究方向**: {detail.request.target_topic}",
        f"**研究目标**: {detail.request.research_goal}",
    ]
    if detail.executive_summary_markdown:
        parts.append(f"\n## 执行摘要\n{detail.executive_summary_markdown[:3000]}")
    elif detail.report_markdown:
        parts.append(f"\n## 报告摘录\n{detail.report_markdown[:2000]}")
    if detail.claims:
        claims_text = "\n".join(
            f"- {c.final_wording or c.claim} [{c.risk_level} 风险, {round(c.confidence * 100)}% 置信度]"
            for c in detail.claims[:15]
        )
        parts.append(f"\n## 核心结论（共 {len(detail.claims)} 条）\n{claims_text}")

    context = "\n".join(parts)

    from cg.llm.client import LLMClient  # noqa: PLC0415
    client = LLMClient(settings)
    try:
        response = await client.complete(
            system=(
                "你是一位学术论文分析专家。请基于提供的研究上下文简洁地回答问题。"
                "如果上下文不足以回答，请明确说明。用与用户提问相同的语言回复。"
            ),
            user=f"## 研究上下文\n\n{context}\n\n---\n\n问题：{body.message}",
        )
        return {"response": response}
    except Exception as exc:  # noqa: BLE001
        return {"response": f"LLM 调用出错：{str(exc)[:300]}"}
