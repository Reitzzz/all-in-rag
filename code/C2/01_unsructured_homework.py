from unstructured.partition.pdf import partition_pdf

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

from collections import Counter


def parse_pdf(strategy):
    """使用指定策略解析PDF并输出结果。"""
    elements = partition_pdf(
        filename=pdf_path,
        strategy=strategy,
    )

    print(f"\n解析策略: {strategy}")
    print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

    types = Counter(e.category for e in elements)
    print(f"元素类型: {dict(types)}")

    print("\n所有元素:")
    for i, element in enumerate(elements, 1):
        print(f"Element {i} ({element.category}):")
        print(element)
        print("=" * 60)


# 分别使用高分辨率布局分析和OCR解析
parse_pdf("hi_res")
parse_pdf("ocr_only")
