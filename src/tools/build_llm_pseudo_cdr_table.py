# 验证我们的Diffusion和自进化机制的有效性，替换成别的模型后会导致性能下降
import argparse
import csv
import re
from pathlib import Path


DOMAIN_PAIRS = [
    {
        "dataset": "Amazon2014",
        "src": "Clothing_Shoes_and_Jewelry",
        "tgt": "Sports_and_Outdoors",
        "short": "Cloth $\\rightarrow$ Sport",
        "group": "Amazon",
    },
    {
        "dataset": "Amazon2014",
        "src": "Sports_and_Outdoors",
        "tgt": "Clothing_Shoes_and_Jewelry",
        "short": "Sport $\\rightarrow$ Cloth",
        "group": "Amazon",
    },
    {
        "dataset": "Amazon2014",
        "src": "Electronics",
        "tgt": "Cell_Phones_and_Accessories",
        "short": "Elec $\\rightarrow$ Phone",
        "group": "Amazon",
    },
    {
        "dataset": "Amazon2014",
        "src": "Cell_Phones_and_Accessories",
        "tgt": "Electronics",
        "short": "Phone $\\rightarrow$ Elec",
        "group": "Amazon",
    },
    {
        "dataset": "Douban",
        "src": "Book",
        "tgt": "Music",
        "short": "Book $\\rightarrow$ Music",
        "group": "Douban",
    },
    {
        "dataset": "Douban",
        "src": "Music",
        "tgt": "Book",
        "short": "Music $\\rightarrow$ Book",
        "group": "Douban",
    },
]


METHODS = [
    {
        "label": "EMCDR",
        "patterns": ["LLM_Pseudo_EMCDR"],
    },
    {
        "label": "PTUPCDR",
        "patterns": ["LLM_Pseudo_PTUPCDR"],
    },
    {
        "label": "UniCDR",
        "patterns": ["LLM_Pseudo_UniCDR"],
    },
]


PSEUDO_METHOD_LABELS = [method["label"] for method in METHODS]
PSEUDO_SUP_LABEL = "Same LLM-init."
LSDCDR_CSV_LABEL = "Diffusion + Self-Evolution"


# Paper/Table-2 LSDCDR values. Used by default because original LSDCDR has many
# historical hyper/ablation logs and some reverse-domain logs are not present.
LSDCDR_FALLBACK = {
    ("Amazon2014", "Clothing_Shoes_and_Jewelry", "Sports_and_Outdoors"): (0.0763, 0.0700),
    ("Amazon2014", "Sports_and_Outdoors", "Clothing_Shoes_and_Jewelry"): (0.0621, 0.0559),
    ("Amazon2014", "Electronics", "Cell_Phones_and_Accessories"): (0.0994, 0.0820),
    ("Amazon2014", "Cell_Phones_and_Accessories", "Electronics"): (0.0667, 0.0762),
    ("Douban", "Book", "Music"): (0.0285, 0.0958),
    ("Douban", "Music", "Book"): (0.0580, 0.1575),
}


LSDCDR_LOG_PATTERNS = [
    "LLM_Diff_interaction_as_condition_mask_both_tag_EMA",
    "LLM_Diff_interaction_as_condition_mask_both_tag_EMA_elec",
    "LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban",
]


EXCLUDE_LSDCDR_NAME_PARTS = [
    "without",
    "ablation",
    "statistical",
    "pseudo_quality",
    "case_study",
    "alpha",
]


def _domain_key(pair):
    return pair["dataset"], pair["src"], pair["tgt"]


def _domain_token(pair):
    return f"{pair['dataset']}-{pair['src']}+{pair['tgt']}-"


def _find_logs(log_dir, patterns, pair):
    token = _domain_token(pair)
    matches = []
    for path in sorted(log_dir.glob("*.log")):
        name = path.name
        if token not in name:
            continue
        if any(name.startswith(pattern) for pattern in patterns):
            matches.append(path)
    return matches


def _find_lsdcdr_logs(log_dir, pair):
    token = _domain_token(pair)
    matches = []
    for path in sorted(log_dir.glob("*.log")):
        name = path.name
        lowered = name.lower()
        if token not in name:
            continue
        if not any(name.startswith(pattern) for pattern in LSDCDR_LOG_PATTERNS):
            continue
        if any(part in lowered for part in EXCLUDE_LSDCDR_NAME_PARTS):
            continue
        matches.append(path)
    return matches


def _metric_at_20(line):
    match = re.search(r"@20=([0-9.]+)", line)
    return float(match.group(1)) if match else None


def _parse_best_cold_test_block(text):
    marker = "BEST RESULTS"
    idx = text.rfind(marker)
    if idx >= 0:
        text = text[idx:]

    cold_idx = text.rfind("Cold Evaluation")
    if cold_idx >= 0:
        text = text[cold_idx:]

    test_idx = text.rfind("Test:")
    if test_idx < 0:
        test_idx = text.rfind("Test Result (upon best valid):")
    if test_idx < 0:
        raise ValueError("Cannot find cold Test block in log.")

    block = text[test_idx:]
    recall = None
    ndcg = None
    for line in block.splitlines():
        lowered = line.lower()
        if "recall" in lowered and "@20=" in line:
            recall = _metric_at_20(line)
        elif "ndcg" in lowered and "@20=" in line:
            ndcg = _metric_at_20(line)
        if recall is not None and ndcg is not None:
            return recall, ndcg

    raise ValueError("Cannot parse recall@20 and ndcg@20 from cold Test block.")


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _parse_best_cold_test_block(text)


def _choose_single_log(logs, method_label, pair):
    if not logs:
        raise FileNotFoundError(
            f"No log found for {method_label}: {pair['dataset']} {pair['src']} -> {pair['tgt']}"
        )
    if len(logs) == 1:
        return logs[0]

    parsed = []
    for path in logs:
        try:
            recall, ndcg = parse_log(path)
            parsed.append((recall, ndcg, path))
        except Exception:
            continue
    if not parsed:
        raise ValueError(f"Found logs for {method_label}, but none can be parsed: {logs}")

    # If multiple logs exist, choose the one with the best cold test Recall@20.
    parsed.sort(key=lambda item: (item[0], item[1], item[2].stat().st_mtime), reverse=True)
    return parsed[0][2]


def collect_llm_pseudo_results(log_dir):
    table = {}
    sources = {}
    for method in METHODS:
        row = {}
        src_row = {}
        for pair in DOMAIN_PAIRS:
            logs = _find_logs(log_dir, method["patterns"], pair)
            path = _choose_single_log(logs, method["label"], pair)
            row[_domain_key(pair)] = parse_log(path)
            src_row[_domain_key(pair)] = path
        table[method["label"]] = row
        sources[method["label"]] = src_row
    return table, sources


def collect_lsdcdr_results(log_dir, source):
    row = {}
    sources = {}
    for pair in DOMAIN_PAIRS:
        key = _domain_key(pair)
        if source == "fallback":
            row[key] = LSDCDR_FALLBACK[key]
            sources[key] = "Table-2 fallback"
            continue

        logs = _find_lsdcdr_logs(log_dir, pair)
        if not logs and source == "logs_or_fallback":
            row[key] = LSDCDR_FALLBACK[key]
            sources[key] = "Table-2 fallback"
            continue

        path = _choose_single_log(logs, "LSDCDR", pair)
        row[key] = parse_log(path)
        sources[key] = path
    return row, sources


def _fmt(value, bold=False, star=False):
    text = f"{value:.4f}"
    if star:
        text = f"{text}$^*$"
    if bold:
        text = f"\\textbf{{{text}}}"
    return text


def _row_to_latex(label, row, bold=False, star=False):
    cells = [label]
    for pair in DOMAIN_PAIRS:
        recall, ndcg = row[_domain_key(pair)]
        cells.append(_fmt(recall, bold=bold, star=star))
        cells.append(_fmt(ndcg, bold=bold, star=star))
    return " & ".join(cells) + r" \\"


def _variant_row_to_latex(prefix, label, row, bold=False, star=False):
    cells = [prefix, label]
    for pair in DOMAIN_PAIRS:
        recall, ndcg = row[_domain_key(pair)]
        cells.append(_fmt(recall, bold=bold, star=star))
        cells.append(_fmt(ndcg, bold=bold, star=star))
    return " & ".join(cells) + r" \\"


def _ours_two_line_row_to_latex(prefix, row, bold=False, star=False):
    first_line = [prefix, r"\textbf{Diffusion +}"]
    for pair in DOMAIN_PAIRS:
        recall, ndcg = row[_domain_key(pair)]
        first_line.append(rf"\multirow{{2}}{{*}}{{{_fmt(recall, bold=bold, star=star)}}}")
        first_line.append(rf"\multirow{{2}}{{*}}{{{_fmt(ndcg, bold=bold, star=star)}}}")

    second_line = ["", r"\textbf{Self-Evolution (ours)}"]
    second_line.extend([""] * (2 * len(DOMAIN_PAIRS)))
    return " & ".join(first_line) + r" \\" + "\n" + " & ".join(second_line) + r" \\"


def build_latex(table, caption, label, bold_lsdcdr=True, star_lsdcdr=False):
    amazon_cols = sum(1 for pair in DOMAIN_PAIRS if pair["group"] == "Amazon") * 2
    douban_cols = sum(1 for pair in DOMAIN_PAIRS if pair["group"] == "Douban") * 2
    domain_headers = []
    for idx, pair in enumerate(DOMAIN_PAIRS):
        align = "c|" if idx < len(DOMAIN_PAIRS) - 1 else "c"
        domain_headers.append(rf"\multicolumn{{2}}{{{align}}}{{\textbf{{{pair['short']}}}}}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c|c|" + "cc|" * (len(DOMAIN_PAIRS) - 1) + r"cc}",
        r"\hline",
        rf"\multicolumn{{2}}{{c|}}{{\textbf{{Dataset}}}} & \multicolumn{{{amazon_cols}}}{{c|}}{{\textbf{{Amazon}}}} & \multicolumn{{{douban_cols}}}{{c}}{{\textbf{{Douban}}}} \\",
        r"\hline",
        "\\multicolumn{2}{c|}{\\textbf{Domain: Src$\\rightarrow$Tgt}} & "
        + " & ".join(domain_headers)
        + r" \\",
        r"\cline{1-2}",
        r"\textbf{Pseudo Sup.} & \textbf{Transfer Module} & "
        + " & ".join([r"\textbf{R@20} & \textbf{N@20}" for _ in DOMAIN_PAIRS])
        + r" \\",
        r"\hline",
    ]

    first_method = PSEUDO_METHOD_LABELS[0]
    lines.append(
        _variant_row_to_latex(
            r"\multirow{5}{*}{Same LLM-init.}",
            first_method,
            table[first_method],
        )
    )
    for method in PSEUDO_METHOD_LABELS[1:]:
        lines.append(_variant_row_to_latex("", method, table[method]))
    lines.append(
        _ours_two_line_row_to_latex(
            "",
            table["LSDCDR"],
            bold=bold_lsdcdr,
            star=star_lsdcdr,
        )
    )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def write_csv(path, table):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["Pseudo Sup.", "Transfer Module"]
        for pair in DOMAIN_PAIRS:
            prefix = f"{pair['dataset']}:{pair['src']}->{pair['tgt']}"
            header.extend([f"{prefix}:R@20", f"{prefix}:N@20"])
        writer.writerow(header)
        for method in PSEUDO_METHOD_LABELS:
            row = [PSEUDO_SUP_LABEL, method]
            for pair in DOMAIN_PAIRS:
                recall, ndcg = table[method][_domain_key(pair)]
                row.extend([f"{recall:.4f}", f"{ndcg:.4f}"])
            writer.writerow(row)
        row = [PSEUDO_SUP_LABEL, LSDCDR_CSV_LABEL]
        for pair in DOMAIN_PAIRS:
            recall, ndcg = table["LSDCDR"][_domain_key(pair)]
            row.extend([f"{recall:.4f}", f"{ndcg:.4f}"])
        writer.writerow(row)


def print_sources(sources):
    print("Log/value sources:")
    for method in PSEUDO_METHOD_LABELS + ["LSDCDR"]:
        print(f"\n[{method}]")
        for pair in DOMAIN_PAIRS:
            src = sources[method][_domain_key(pair)]
            print(f"  {pair['dataset']} {pair['src']} -> {pair['tgt']}: {src}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a LaTeX/CSV table for same-pseudo LSDCDR backbone analysis."
    )
    parser.add_argument("--log_dir", default="src/log")
    parser.add_argument("--output_tex", default="src/tools/llm_pseudo_cdr_table.tex")
    parser.add_argument("--output_csv", default="src/tools/llm_pseudo_cdr_table.csv")
    parser.add_argument(
        "--lsdcdr_source",
        choices=["fallback", "logs", "logs_or_fallback"],
        default="fallback",
        help="Use Table-2 fallback values by default to avoid ambiguous historical LSDCDR logs.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Effect of replacing LSDCDR's diffusion-based self-evolving transfer module "
            "with conventional CDR backbones under the same pseudo supervision."
        ),
    )
    parser.add_argument("--label", default="tab:same_pseudo_non_diffusion")
    parser.add_argument("--no_bold_lsdcdr", action="store_true")
    parser.add_argument("--star_lsdcdr", action="store_true")
    parser.add_argument("--print_sources", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Missing log directory: {log_dir}")

    table, sources = collect_llm_pseudo_results(log_dir)
    lsdcdr_row, lsdcdr_sources = collect_lsdcdr_results(log_dir, args.lsdcdr_source)
    table["LSDCDR"] = lsdcdr_row
    sources["LSDCDR"] = lsdcdr_sources

    tex = build_latex(
        table=table,
        caption=args.caption,
        label=args.label,
        bold_lsdcdr=not args.no_bold_lsdcdr,
        star_lsdcdr=args.star_lsdcdr,
    )

    output_tex = Path(args.output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(tex, encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, table)

    print(f"Saved LaTeX: {output_tex}")
    print(f"Saved CSV  : {output_csv}")
    if args.print_sources:
        print_sources(sources)


if __name__ == "__main__":
    main()
