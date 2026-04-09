#!/usr/bin/env python3
"""将 .tex 中残留英文弯引号与 ASCII 直引号统一为中文双引号 “”（U+201C / U+201D）。

行内 $...$ 数学模式内的 ASCII 引号不替换。

用法:
  python3 replace_tex_cn_quotes.py path/to/chap01.tex [更多文件...]
"""

from __future__ import annotations

import sys
from pathlib import Path

# 中文双引号（与 GB/T 15834 常用的弯引号同码位）
CURLY_L = "\u201c"
CURLY_R = "\u201d"


def replace_ascii_quotes_line(line: str) -> str:
    out: list[str] = []
    i = 0
    n = len(line)
    math_depth = 0
    quote_toggle = True

    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n:
            out.append(c)
            out.append(line[i + 1])
            i += 2
            continue
        if c == "$":
            math_depth ^= 1
            out.append(c)
            i += 1
            continue
        if c == '"' and math_depth == 0:
            out.append(CURLY_L if quote_toggle else CURLY_R)
            quote_toggle = not quote_toggle
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def process_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(replace_ascii_quotes_line(line) for line in lines)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("用法: python3 replace_tex_cn_quotes.py <tex 文件> ...", file=sys.stderr)
        sys.exit(1)
    for fp in paths:
        raw = fp.read_text(encoding="utf-8")
        new = process_text(raw)
        if new != raw:
            fp.write_text(new, encoding="utf-8")
            print(f"已更新: {fp}")
        else:
            print(f"无变化: {fp}")


if __name__ == "__main__":
    main()
