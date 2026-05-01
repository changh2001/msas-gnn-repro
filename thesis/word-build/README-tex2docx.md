# LaTeX -> Word 高保真转换说明（MSAS-GNN）

本目录用于论文 `main.tex` 的高保真 DOCX 产物管理。

## 一键转换

在项目根目录执行：

```bash
bash scripts/tex2docx.sh
```

输出文件：

- `word-build/main-final.docx`
- `word-build/pandoc-warnings.log`

## 使用 reference.docx（强烈建议）

为了让 Word 样式尽量贴近 PDF，请先准备 `word-build/reference.docx`，然后执行：

```bash
bash scripts/tex2docx.sh --reference-doc word-build/reference.docx
```

## reference.docx 必调样式

请至少统一以下样式（与 PDF 保持一致）：

- 页面：A4、页边距、页眉页脚距离
- 正文：中文字体、英文字体、字号、行距、首行缩进、段前段后
- 标题 1/2/3：字号、加粗、编号间距、段前段后
- 图题/表题：字体、字号、位置（图下表上）、段落样式
- 参考文献：编号样式、悬挂缩进、行距
- 脚注：字号、行距、分隔线

## 严格对齐流程

1. 运行脚本生成 `main-final.docx`
2. 打开 `pandoc-warnings.log`，优先处理公式和引用告警
3. 按 `word-build/PDF-Word-Strict-Checklist.md` 逐项核对
4. 仅在 Word 中做版面微调，不回写 TeX（避免源文件漂移）

## 说明

- `bnuthesis` 模板的封面/授权页/前置页样式不能 100% 自动复刻，建议在 Word 中单独模板化。
- 复杂公式、跨页长表、图表浮动位置需要人工复核。
