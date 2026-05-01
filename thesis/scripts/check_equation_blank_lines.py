#!/usr/bin/env python3
"""
检查 / 修正 equation 环境周围的空行约定：
  - \\end{equation} 之后：若下一非空行不是 \\begin{equation}，则须有空行；文件末尾的 \\end{equation} 后补一空行。
  - \\begin{equation} 之前：不得有空行。
  - 连续两个 equation 之间：不插空行（否则与「公式前无空行」冲突）。

用法:
  python3 check_equation_blank_lines.py --check  chap01.tex ...
  python3 check_equation_blank_lines.py --fix   chap01.tex ...
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BEGIN_EQ = re.compile(r"^\s*\\begin\{equation\}")
END_EQ = re.compile(r"^\s*\\end\{equation\}")


def fix_lines(lines: list[str]) -> tuple[list[str], dict]:
    stats = {
        "after_end_insert_blank": 0,
        "after_end_append_eof": 0,
        "after_end_remove_extra_blanks": 0,
        "before_begin_remove_blanks": 0,
    }
    lines = [ln.rstrip("\r\n") for ln in lines]

    i = 0
    while i < len(lines):
        if END_EQ.match(lines[i]):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1

            if j >= len(lines):
                lines.append("")
                stats["after_end_append_eof"] += 1
                i += 1
                continue

            nxt = lines[j]
            if BEGIN_EQ.match(nxt):
                while j > i + 1:
                    del lines[i + 1]
                    j -= 1
                    stats["after_end_remove_extra_blanks"] += 1
            else:
                if j == i + 1:
                    lines.insert(i + 1, "")
                    stats["after_end_insert_blank"] += 1
                    i += 2
                    continue
                if j > i + 2:
                    for _ in range(j - i - 2):
                        del lines[i + 2]
                        stats["after_end_remove_extra_blanks"] += 1
        i += 1

    i = 0
    while i < len(lines):
        if BEGIN_EQ.match(lines[i]):
            while i > 0 and lines[i - 1].strip() == "":
                del lines[i - 1]
                i -= 1
                stats["before_begin_remove_blanks"] += 1
        i += 1

    return lines, stats


def check_lines(lines: list[str]) -> list[tuple[int, str, str]]:
    lines = [ln.rstrip("\r\n") for ln in lines]
    issues: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        if BEGIN_EQ.match(line):
            if i > 0 and lines[i - 1].strip() == "":
                issues.append((i + 1, r"\begin{equation} 前有不应存在的空行", ""))
        if END_EQ.match(line):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines):
                issues.append((i + 1, r"\end{equation} 在文件末尾后缺少空行", ""))
            elif not BEGIN_EQ.match(lines[j]) and j == i + 1:
                issues.append((i + 1, r"\end{equation} 后缺少空行", lines[j][:60]))
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    exit_code = 0
    for fp in args.files:
        if not fp.is_file():
            print(f"跳过（非文件）: {fp}", file=sys.stderr)
            continue
        raw = fp.read_text(encoding="utf-8")
        ends_nl = raw.endswith("\n")
        content_lines = raw.split("\n")
        if raw.endswith("\n") and content_lines and content_lines[-1] == "":
            content_lines = content_lines[:-1]

        if args.check:
            iss = check_lines(content_lines)
            print(f"=== {fp} ===")
            if not iss:
                print("  无违规")
            else:
                exit_code = 1
                for ln, msg, ctx in iss:
                    extra = f" | 下一行: {ctx!r}" if ctx else ""
                    print(f"  行 {ln}: {msg}{extra}")
        else:
            new_lines, st = fix_lines(content_lines)
            new_raw = "\n".join(new_lines)
            if ends_nl:
                new_raw += "\n"
            if new_raw != raw:
                fp.write_text(new_raw, encoding="utf-8")
                t = (
                    st["after_end_insert_blank"]
                    + st["after_end_append_eof"]
                    + st["after_end_remove_extra_blanks"]
                    + st["before_begin_remove_blanks"]
                )
                print(f"已写入 {fp}（改动项数合计 {t}）")
            else:
                print(f"无变化: {fp}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
