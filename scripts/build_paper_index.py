"""
用 PyMuPDF 批量提取 PDF 元数据，生成 paper_index.json。
用法: python scripts/build_paper_index.py /path/to/papers data/paper_index.json

递归扫描所有子目录中的 .pdf 文件。
"""
import sys, json, os, re
import fitz  # PyMuPDF


def extract_metadata(pdf_path: str) -> dict:
    try:
        doc = fitz.open(pdf_path)
        # 前 5 页文本
        pages_text = []
        for i in range(min(5, len(doc))):
            pages_text.append(doc[i].get_text())
        full_text = "\n".join(pages_text)
        doc.close()

        # 启发式提取标题（第一页前几行非空文本）
        lines = [l.strip() for l in pages_text[0].split("\n") if l.strip()]
        title = lines[0] if lines else os.path.basename(pdf_path)

        # 启发式提取摘要
        abstract = ""
        abstract_match = re.search(
            r"(?i)abstract[:\s]*\n?(.*?)(?:\n\s*\n|introduction|1[\.\s])",
            full_text, re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()[:1500]

        return {
            "title": title,
            "abstract": abstract,
            "pdf_path": pdf_path,
            "focused_text": full_text[:8000],  # 前 8K 字符供 Evidence Agent 使用
        }

    except Exception as e:
        return {"title": os.path.basename(pdf_path), "error": str(e), "pdf_path": pdf_path}


if __name__ == "__main__":
    pdf_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/paper_index.json"

    # 递归扫描所有子目录中的 PDF
    pdfs = []
    for root, dirs, files in os.walk(pdf_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    pdfs.sort()

    # 断点续传：加载已有结果，跳过已处理的 PDF
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            old_results = json.load(f)
        for r in old_results:
            existing[r.get("pdf_path", "")] = r
        print(f"Loaded {len(existing)} existing entries for resume")

    # 只处理新增的 PDF
    new_pdfs = [p for p in pdfs if p not in existing]
    print(f"Total PDFs: {len(pdfs)}, already processed: {len(existing)}, new: {len(new_pdfs)}")

    if not new_pdfs:
        print("No new PDFs to process. Done!")
        sys.exit(0)

    # 处理新增 PDF
    new_results = []
    for i, pdf in enumerate(new_pdfs):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(new_pdfs)} new PDFs...")
        new_results.append(extract_metadata(pdf))

    # 合并：旧结果 + 新结果（旧的保留，新的追加）
    all_results = list(existing.values()) + new_results

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in all_results if "error" not in r)
    print(f"Done: {ok}/{len(all_results)} successful, saved to {output_path}")
