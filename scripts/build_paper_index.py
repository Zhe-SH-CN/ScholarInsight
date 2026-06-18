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

    print(f"Processing {len(pdfs)} PDFs from {pdf_dir}...")

    results = []
    for i, pdf in enumerate(pdfs):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pdfs)}...")
        results.append(extract_metadata(pdf))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if "error" not in r)
    print(f"Done: {ok}/{len(results)} successful, saved to {output_path}")
