import csv
from pathlib import Path


# Fill this block manually, then run:
#   python src/tools/build_llm_stability_table.py
#
# Each row is: (method, prompt_type, tag_count, Recall@20, NDCG@20)
TABLE_ROWS = [
    ("LSDCDR", "Long-form Profile", "-", 0.0730, 0.0663),
    ("", r"\shortstack{Concept-Level\\Interest Tags}", "1--2", 0.0723, 0.0650),
    ("", "", "2--4 (ours)", 0.0763, 0.0700),
    ("", "", "4--6", 0.0622, 0.0584),
]


OUTPUT_TEX = "src/tools/llm_stability_cloth_sports_table.tex"
OUTPUT_CSV = "src/tools/llm_stability_cloth_sports_table.csv"
CAPTION = "Effect of prompt design and tag stability on Amazon Cloth$\\rightarrow$Sport."
LABEL = "tab:prompt_formulation"
BOLD_LSDCDR = True


def _validate_metric(value, name):
    try:
        metric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got: {value}") from exc
    if metric < 0:
        raise ValueError(f"{name} must be non-negative, got: {value}")
    return metric


def validate_rows(rows):
    normalized = []
    for idx, row in enumerate(rows, start=1):
        if len(row) != 5:
            raise ValueError(
                f"TABLE_ROWS entry {idx} must be (method, prompt_type, tag_count, recall20, ndcg20)."
            )
        method, prompt_type, tag_count, recall, ndcg = row
        normalized.append(
            (
                str(method),
                str(prompt_type),
                str(tag_count),
                _validate_metric(recall, f"{prompt_type} {tag_count} Recall@20"),
                _validate_metric(ndcg, f"{prompt_type} {tag_count} NDCG@20"),
            )
        )
    return normalized


def _fmt(value, bold=False):
    text = f"{value:.4f}"
    return rf"\textbf{{{text}}}" if bold else text


def build_latex(rows, caption, label, bold_lsdcdr=True):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{7pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{c|c|c|c|c}",
        r"\hline",
        r"Method & Prompt Type & Tag Count & R@20 & N@20 \\",
        r"\hline",
    ]

    for idx, (method, prompt_type, tag_count, recall, ndcg) in enumerate(rows):
        is_ours = tag_count == "2--4 (ours)"
        method_cell = method
        if idx == 0 and method == "LSDCDR":
            method_cell = r"\multirow{4}{*}{LSDCDR}"
        prompt_cell = prompt_type
        if idx == 1:
            prompt_cell = r"\multirow{3}{*}{" + prompt_type + r"}"
        if is_ours and prompt_type:
            prompt_cell = rf"\textbf{{{prompt_cell}}}"
        tag_cell = rf"\textbf{{{tag_count}}}" if is_ours else tag_count
        lines.append(
            f"{method_cell} & "
            + prompt_cell
            + " & "
            + tag_cell
            + " & "
            + _fmt(recall, bold=bold_lsdcdr and is_ours)
            + " & "
            + _fmt(ndcg, bold=bold_lsdcdr and is_ours)
            + r" \\"
        )
        if idx == 0:
            lines.append(r"\cline{2-5}")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Prompt Type", "Tag Count", "Cloth->Sport:R@20", "Cloth->Sport:N@20"])
        for method, prompt_type, tag_count, recall, ndcg in rows:
            writer.writerow([method, prompt_type, tag_count, f"{recall:.4f}", f"{ndcg:.4f}"])


def main():
    rows = validate_rows(TABLE_ROWS)

    tex = build_latex(
        rows=rows,
        caption=CAPTION,
        label=LABEL,
        bold_lsdcdr=BOLD_LSDCDR,
    )

    output_tex = Path(OUTPUT_TEX)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(tex, encoding="utf-8")

    output_csv = Path(OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, rows)

    print(f"Saved LaTeX: {output_tex}")
    print(f"Saved CSV  : {output_csv}")


if __name__ == "__main__":
    main()
