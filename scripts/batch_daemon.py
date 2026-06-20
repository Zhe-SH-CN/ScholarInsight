"""
ScholarInsight 24/7 批量蒸馏 Daemon。
自动从 topics.json 读取话题，调用 ResearchPipeline.run() 持续消耗 MiMo tokens。

支持断点续传：每个 runner 在 pipeline 各阶段保存 checkpoint，
中断后重启会从 checkpoint 恢复，跳过已完成的步骤。

用法:
  # 前台运行（调试）
  cd backend && uv run python ../scripts/batch_daemon.py

  # 后台运行
  tmux new-session -d -s scholar-batch "cd backend && uv run python ../scripts/batch_daemon.py"

  # Smoke test（只跑一个话题）
  cd backend && uv run python ../scripts/batch_daemon.py --smoke-test
  cd backend && uv run python ../scripts/batch_daemon.py --pilot-count 3
"""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

# 添加 backend 到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from cg.orchestrator.pipeline import ResearchPipeline
from cg.schemas.research import ResearchRequest
from cg.settings import Settings

# 路径
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
TOPICS_PATH = SCRIPT_DIR / "topics.json"
PROGRESS_PATH = DATA_DIR / "batch_progress.json"
ERRORS_PATH = DATA_DIR / "batch_errors.json"
MARKDOWN_PROGRESS_PATH = PROJECT_DIR / "batch_topic_progress.md"
STATS_INTERVAL = 10  # 每 10 个 run 打印统计
MAX_CONCURRENT = 3   # 并发数
PROGRESS_REFRESH_SECONDS = 15


def load_topics() -> list[str]:
    """加载话题列表。"""
    if not TOPICS_PATH.exists():
        print(f"ERROR: topics.json not found at {TOPICS_PATH}")
        sys.exit(1)
    with open(TOPICS_PATH, "r", encoding="utf-8") as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} topics from {TOPICS_PATH}")
    return topics


def load_progress() -> dict:
    """加载断点续传进度。"""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "completed": [],
        "failed": [],
        "started_at": None,
        "total_tokens_estimate": 0,
        "runs": {},
    }


def save_progress(progress: dict) -> None:
    """保存进度。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def log_error(topic: str, error: str) -> None:
    """记录错误。"""
    errors = []
    if ERRORS_PATH.exists():
        with open(ERRORS_PATH, "r", encoding="utf-8") as f:
            errors = json.load(f)
    errors.append({
        "topic": topic,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)


def topic_run_id(index: int, topic: str) -> str:
    """Stable folder name: 001_Topic_Name."""
    slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in topic)
    slug = "_".join(part for part in slug.split("_") if part)
    return f"{index:03d}_{slug[:72].rstrip('_') or 'topic'}"


def markdown_escape(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def topic_meta(topics: list[str]) -> list[dict]:
    return [
        {"index": index, "topic": topic, "run_id": topic_run_id(index, topic)}
        for index, topic in enumerate(topics, start=1)
    ]


def collect_run_metrics(run_dir: Path) -> dict:
    status = read_json(run_dir / "status.json", {}) or {}
    checkpoint = read_json(run_dir / "checkpoint.json", None)
    claims = read_jsonl(run_dir / "claims" / "_index.jsonl")
    verification_counts = Counter(str(claim.get("verification_status") or "unknown") for claim in claims)
    risk_counts: Counter[str] = Counter()
    for claim in claims:
        for note in claim.get("red_team_notes") or []:
            if isinstance(note, dict):
                risk_counts[str(note.get("risk_type") or "other")] += 1

    if status.get("status") == "completed":
        display_status = "Done"
        checkpoint_label = ""
    elif status.get("status") == "failed":
        display_status = "Failed"
        checkpoint_label = ""
    elif checkpoint:
        step = checkpoint.get("step", "unknown")
        loop = checkpoint.get("loop_round", "?")
        max_loops = checkpoint.get("max_loops", "?")
        display_status = f"Checkpoint:{step}"
        checkpoint_label = f"L{loop}/{max_loops}"
    elif status.get("status") in {"queued", "running"}:
        display_status = str(status.get("current_stage") or status.get("status") or "Running")
        checkpoint_label = ""
    else:
        display_status = "Waiting"
        checkpoint_label = ""

    metrics = status.get("metrics") or {}
    return {
        "status": display_status,
        "checkpoint": checkpoint_label,
        "claims": metrics.get("claim_count") or len(claims),
        "verified": metrics.get("verified_claim_count") or verification_counts.get("verified", 0),
        "needs_evidence": verification_counts.get("needs_evidence", 0),
        "challenged": metrics.get("challenged_claim_count") or verification_counts.get("challenged", 0),
        "rejected": verification_counts.get("rejected", 0),
        "single_source": risk_counts.get("single_source", 0),
        "insufficient": risk_counts.get("insufficient_evidence", 0),
        "wording": risk_counts.get("wording_risk", 0) + risk_counts.get("absolute_wording", 0),
        "over_inference": risk_counts.get("over_inference", 0),
        "other_risks": sum(
            count
            for risk, count in risk_counts.items()
            if risk not in {"single_source", "insufficient_evidence", "wording_risk", "absolute_wording", "over_inference"}
        ),
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "error": status.get("error") or "",
    }


def render_markdown_progress(topics: list[str], progress: dict, runs_dir: Path) -> str:
    completed = set(progress.get("completed") or [])
    failed = set(progress.get("failed") or [])
    rows = []
    headers = [
        "No.",
        "Topic",
        "Status",
        "Run folder",
        "Checkpoint",
        "Claims",
        "Verified",
        "Needs evidence",
        "Challenged",
        "Rejected",
        "Single source",
        "Insufficient evidence",
        "Wording risk",
        "Over-inference",
        "Other risks",
        "Updated",
        "Error",
    ]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")

    for meta in topic_meta(topics):
        run_id = meta["run_id"]
        run_dir = runs_dir / run_id
        if run_dir.exists():
            values = collect_run_metrics(run_dir)
        else:
            topic = meta["topic"]
            values = {
                "status": "Done" if topic in completed else "Failed" if topic in failed else "Waiting",
                "checkpoint": "",
                "claims": 0,
                "verified": 0,
                "needs_evidence": 0,
                "challenged": 0,
                "rejected": 0,
                "single_source": 0,
                "insufficient": 0,
                "wording": 0,
                "over_inference": 0,
                "other_risks": 0,
                "updated": "",
                "error": "",
            }
        row = [
            meta["index"],
            meta["topic"],
            values["status"],
            run_id if run_dir.exists() else "",
            values["checkpoint"],
            values["claims"],
            values["verified"],
            values["needs_evidence"],
            values["challenged"],
            values["rejected"],
            values["single_source"],
            values["insufficient"],
            values["wording"],
            values["over_inference"],
            values["other_risks"],
            values["updated"],
            values["error"],
        ]
        rows.append("| " + " | ".join(markdown_escape(item) for item in row) + " |")

    completed_count = len(completed)
    failed_count = len(failed)
    return "\n".join(
        [
            "# ScholarInsight Batch Topic Progress",
            "",
            "This file is maintained by `scripts/batch_daemon.py` during batch runs.",
            "",
            f"- Topics: {len(topics)}",
            f"- Completed: {completed_count}",
            f"- Failed: {failed_count}",
            f"- Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            *rows,
            "",
        ]
    )


def write_markdown_progress(topics: list[str], progress: dict, settings: Settings) -> None:
    MARKDOWN_PROGRESS_PATH.write_text(
        render_markdown_progress(topics, progress, settings.data_dir / "runs"),
        encoding="utf-8",
    )


async def monitor_topic_progress(
    topics: list[str],
    progress: dict,
    settings: Settings,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        write_markdown_progress(topics, progress, settings)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=PROGRESS_REFRESH_SECONDS)
        except asyncio.TimeoutError:
            pass


def print_stats(progress: dict, total: int, start_time: float) -> None:
    """打印统计信息。"""
    done = len(progress["completed"])
    failed = len(progress["failed"])
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1) * 3600  # runs/hour
    remaining = (total - done - failed) / max(rate, 1)
    tokens_est = done * 800_000  # ~800K tokens per run
    print(f"\n{'='*60}")
    print(f"进度: {done}/{total} 完成, {failed} 失败")
    print(f"速率: {rate:.1f} runs/hour")
    print(f"预估剩余: {remaining:.1f} 小时")
    print(f"累计 tokens 估算: {tokens_est/1e6:.0f}M")
    print(f"运行时间: {elapsed/3600:.1f} 小时")
    print(f"{'='*60}\n")


async def run_topic(
    pipeline: ResearchPipeline,
    index: int,
    topic: str,
    semaphore: asyncio.Semaphore,
    topics: list[str],
    progress: dict,
    settings: Settings,
) -> tuple[str, bool]:
    """运行单个话题（支持断点续传）。"""
    async with semaphore:
        run_id = topic_run_id(index, topic)
        request = ResearchRequest(
            project_name=run_id,
            target_topic=topic,
            research_goal=f"分析 {topic} 方向中论文之间的创新关系和推理模式分布",
            max_sources=100,
            max_search_rounds=2,
            max_research_loops=5,
        )
        progress.setdefault("runs", {})[topic] = {"index": index, "run_id": run_id}
        save_progress(progress)
        write_markdown_progress(topics, progress, settings)

        try:
            # 检查是否有未完成的 run（checkpoint 存在）
            runs_dir = Path(pipeline.settings.data_dir) / "runs"
            existing_run_id = run_id if (runs_dir / run_id).exists() else None
            existing_run_dir = runs_dir / run_id
            existing_status = read_json(existing_run_dir / "status.json", {}) if existing_run_id else {}

            if existing_run_id and (existing_run_dir / "checkpoint.json").exists():
                print(f"  RESUME: {topic} → {existing_run_id}")
                await pipeline.resume(request, run_id)
            elif existing_run_id and existing_status.get("status") == "completed":
                print(f"  SKIP DONE: {topic} → {existing_run_id}")
            else:
                status = await pipeline.prepare_run(request, owner="batch_daemon")
                run_id = status.run_id
                topic_input_path = Path(pipeline.runs.run_dir(run_id)) / "topic_input.json"
                topic_input_path.write_text(
                    json.dumps(
                        {
                            "index": index,
                            "topic": topic,
                            "run_id": run_id,
                            "request": request.model_dump(mode="json"),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                await pipeline.run(request, run_id)

            write_markdown_progress(topics, progress, settings)
            return topic, True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  FAILED: {topic} — {error_msg}")
            log_error(topic, error_msg)
            write_markdown_progress(topics, progress, settings)
            return topic, False


async def main() -> None:
    """主循环。"""
    parser = argparse.ArgumentParser(description="ScholarInsight batch daemon")
    parser.add_argument("--smoke-test", action="store_true", help="只跑一个话题进行测试")
    parser.add_argument("--pilot-count", type=int, default=0, help="只跑前 N 个尚未完成的话题")
    args = parser.parse_args()
    if args.pilot_count < 0:
        parser.error("--pilot-count must be >= 0")
    if args.smoke_test and args.pilot_count:
        parser.error("--smoke-test and --pilot-count cannot be used together")

    topics = load_topics()
    progress = load_progress()
    progress.setdefault("runs", {})
    completed_set = set(progress["completed"])
    failed_set = set(progress["failed"])

    # 过滤已完成的话题
    pending = [meta for meta in topic_meta(topics) if meta["topic"] not in completed_set]
    print(f"待处理: {len(pending)}/{len(topics)} 话题")

    if not pending:
        print("所有话题已完成！")
        return

    # Smoke test 模式：只跑第一个待处理话题
    if args.smoke_test:
        pending = pending[:1]
        print(f"Smoke test 模式：只跑 {pending[0]['index']}. {pending[0]['topic']}")
    elif args.pilot_count:
        pending = pending[:args.pilot_count]
        print(f"Pilot 模式：只跑前 {len(pending)} 个未完成话题")

    if not progress["started_at"]:
        progress["started_at"] = datetime.now(timezone.utc).isoformat()

    settings = Settings()
    pipeline = ResearchPipeline(settings)
    concurrency = 1 if args.smoke_test else MAX_CONCURRENT
    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.time()
    batch_count = 0
    save_progress(progress)
    write_markdown_progress(topics, progress, settings)
    stop_monitor = asyncio.Event()
    monitor = asyncio.create_task(monitor_topic_progress(topics, progress, settings, stop_monitor))

    try:
        for i in range(0, len(pending), concurrency):
            batch = pending[i:i + concurrency]
            tasks = [
                run_topic(
                    pipeline,
                    int(meta["index"]),
                    str(meta["topic"]),
                    semaphore,
                    topics,
                    progress,
                    settings,
                )
                for meta in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"  BATCH ERROR: {result}")
                    continue
                topic, success = result
                batch_count += 1
                if success:
                    if topic not in progress["completed"]:
                        progress["completed"].append(topic)
                    if topic in failed_set:
                        failed_set.discard(topic)
                    progress["failed"] = [item for item in progress["failed"] if item != topic]
                else:
                    if topic not in progress["failed"]:
                        progress["failed"].append(topic)
                    failed_set.add(topic)

                # 每完成一个就保存进度
                save_progress(progress)
                write_markdown_progress(topics, progress, settings)

                # 每 10 个打印统计
                if batch_count % STATS_INTERVAL == 0:
                    print_stats(progress, len(topics), start_time)
    finally:
        stop_monitor.set()
        await monitor

    # 最终统计
    write_markdown_progress(topics, progress, settings)
    print_stats(progress, len(topics), start_time)
    print("Daemon 结束。")


if __name__ == "__main__":
    asyncio.run(main())
