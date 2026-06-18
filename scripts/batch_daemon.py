"""
ScholarInsight 24/7 批量蒸馏 Daemon。
自动从 topics.json 读取话题，调用 ResearchPipeline.run() 持续消耗 MiMo tokens。

用法:
  # 前台运行（调试）
  cd backend && uv run python ../scripts/batch_daemon.py

  # 后台运行
  tmux new-session -d -s scholar-batch "cd backend && uv run python ../scripts/batch_daemon.py"
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# 添加 backend 到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from cg.orchestrator.pipeline import ResearchPipeline
from cg.schemas.research import ResearchRequest
from cg.settings import Settings

# 路径
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
DATA_DIR = BACKEND_DIR / "data"
TOPICS_PATH = SCRIPT_DIR / "topics.json"
PROGRESS_PATH = DATA_DIR / "batch_progress.json"
ERRORS_PATH = DATA_DIR / "batch_errors.json"
STATS_INTERVAL = 10  # 每 10 个 run 打印统计
MAX_CONCURRENT = 3   # 并发数


def load_topics() -> list[str]:
    """加载话题列表。"""
    if not TOPICS_PATH.exists():
        print(f"ERROR: topics.json not found at {TOPICS_PATH}")
        sys.exit(1)
    with open(TOPICS_PATH, "r") as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} topics from {TOPICS_PATH}")
    return topics


def load_progress() -> dict:
    """加载断点续传进度。"""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "started_at": None, "total_tokens_estimate": 0}


def save_progress(progress: dict) -> None:
    """保存进度。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def log_error(topic: str, error: str) -> None:
    """记录错误。"""
    errors = []
    if ERRORS_PATH.exists():
        with open(ERRORS_PATH, "r") as f:
            errors = json.load(f)
    errors.append({
        "topic": topic,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with open(ERRORS_PATH, "w") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)


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
    topic: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, bool]:
    """运行单个话题。"""
    async with semaphore:
        request = ResearchRequest(
            project_name=f"论文推理模式分析: {topic}",
            target_product=topic,
            competitors=[],
            research_goal=f"分析 {topic} 方向中论文之间的创新关系和推理模式分布",
            max_sources=50,
            max_search_rounds=2,
            max_research_loops=2,
        )

        run_id = f"batch_{uuid4().hex[:8]}"
        try:
            status = await pipeline.prepare_run(request, owner="batch_daemon")
            run_id = status.run_id
            await pipeline.run(request, run_id)
            return topic, True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  FAILED: {topic} — {error_msg}")
            log_error(topic, error_msg)
            return topic, False


async def main() -> None:
    """主循环。"""
    topics = load_topics()
    progress = load_progress()
    completed_set = set(progress["completed"])
    failed_set = set(progress["failed"])

    # 过滤已完成的话题
    pending = [t for t in topics if t not in completed_set]
    print(f"待处理: {len(pending)}/{len(topics)} 话题")

    if not pending:
        print("所有话题已完成！")
        return

    if not progress["started_at"]:
        progress["started_at"] = datetime.now(timezone.utc).isoformat()

    settings = Settings()
    pipeline = ResearchPipeline(settings)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    start_time = time.time()
    batch_count = 0

    for i in range(0, len(pending), MAX_CONCURRENT):
        batch = pending[i:i + MAX_CONCURRENT]
        tasks = [run_topic(pipeline, topic, semaphore) for topic in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                print(f"  BATCH ERROR: {result}")
                continue
            topic, success = result
            batch_count += 1
            if success:
                progress["completed"].append(topic)
                if topic in failed_set:
                    failed_set.discard(topic)
            else:
                progress["failed"].append(topic)
                failed_set.add(topic)

            # 每完成一个就保存进度
            save_progress(progress)

            # 每 10 个打印统计
            if batch_count % STATS_INTERVAL == 0:
                print_stats(progress, len(topics), start_time)

    # 最终统计
    print_stats(progress, len(topics), start_time)
    print("Daemon 结束。")


if __name__ == "__main__":
    asyncio.run(main())
