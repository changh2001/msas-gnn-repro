#!/usr/bin/env python3
r"""
Pre-clean expanded LaTeX for higher-fidelity Pandoc DOCX conversion.

This script targets known Pandoc pain points in thesis documents:
- \label{...} commands embedded in math blocks
- \Bigl / \Bigr style delimiters that Pandoc's math reader may reject
- \resizebox{...}{...}{$...$} wrappers around displayed formulas
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MATH_ENVS = [
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "flalign",
    "flalign*",
    "eqnarray",
    "eqnarray*",
]


def _clean_math_block(content: str) -> str:
    # Remove labels inside math blocks (labels are not useful in docx conversion).
    content = re.sub(r"\\label\{[^{}]+\}", "", content)
    # Normalize delimiter commands that often trigger Pandoc parse warnings.
    content = re.sub(r"\\Biggl|\\Biggr|\\biggl|\\biggr", "", content)
    content = re.sub(r"\\Bigl|\\Bigr|\\bigl|\\bigr|\\Big|\\big", "", content)
    return content


def _clean_resizebox_math(text: str) -> str:
    # Convert: \resizebox{...}{...}{$ ... $} -> $$ ... $$
    pattern = re.compile(
        r"\\resizebox\{[^{}]*\}\{[^{}]*\}\{\s*\$\s*(.*?)\s*\$\s*\}",
        flags=re.DOTALL,
    )
    return pattern.sub(lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n", text)


def clean_tex(text: str) -> str:
    text = _clean_resizebox_math(text)

    # Clean named math environments.
    for env in MATH_ENVS:
        pattern = re.compile(
            rf"\\begin\{{{re.escape(env)}\}}(.*?)\\end\{{{re.escape(env)}\}}",
            flags=re.DOTALL,
        )

        def _replace_env(match: re.Match[str]) -> str:
            body = _clean_math_block(match.group(1))
            return f"\\begin{{{env}}}{body}\\end{{{env}}}"

        text = pattern.sub(_replace_env, text)

    # Clean display math blocks: \[ ... \]
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: r"\[" + _clean_math_block(m.group(1)) + r"\]",
        text,
        flags=re.DOTALL,
    )
    # Clean inline math blocks: \( ... \)
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: r"\(" + _clean_math_block(m.group(1)) + r"\)",
        text,
        flags=re.DOTALL,
    )

    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean expanded TeX for Pandoc DOCX.")
    parser.add_argument("--infile", required=True, help="Input expanded .tex file")
    parser.add_argument("--outfile", required=True, help="Output cleaned .tex file")
    args = parser.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)

    raw = infile.read_text(encoding="utf-8")
    cleaned = clean_tex(raw)
    outfile.write_text(cleaned, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
