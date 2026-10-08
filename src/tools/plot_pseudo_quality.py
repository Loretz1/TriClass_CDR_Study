# 绘制Epoch-Pseudo Interaction Recall曲线。默认从pseudo_quality_records/<run>/epoch_metrics.csv
# 读取伪交互集合质量指标，并将图片保存回同一个record目录。
# --curve_log 仅作为旧兼容选项：只有未提供 --record_dir 时才会从训练log读取曲线。
# cd /data/guqiwen/CDRec

# /data/guqiwen/anaconda3/envs/1.12.0+cu116/bin/python \
#   src/tools/plot_pseudo_quality.py \
#   --record_dir src/pseudo_quality_records/<record_dir_name> \
#   --metric recall@5 \
#   --prepend_initial_value 0.0009 \
#   --hline "LLM4Rerank on LSDCDR-Sem candidates=0.0102" \
#   --hline "LSDCDR-SemOnly w/ GPT-5.2+TE3L=0.0000" \
#   --hline "LSDCDR-SemOnly w/ Qwen3-Max+TE-v4=0.0000"

import argparse
import csv
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# You can edit this list directly if you do not want to pass --hline each time.
# Format: ("label", value). Values should match the selected metric, e.g. recall@5.
DEFAULT_HLINES = [
    # ("LSDCDR-SemOnly w/ GPT-5.2+TE3L", 0.0000),
    # ("LSDCDR-SemOnly w/ Qwen3-Max+TE-v4", 0.0000),
    # ("LLM4Rerank on LSDCDR-Sem candidates", 0.0000),
]


def _load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _metric_label(metric):
    if "@" not in metric:
        return metric
    name, k = metric.split("@", 1)
    if name.lower() == "recall":
        return "Pseudo Interaction Recall"
    return f"{name.capitalize()}@{k}"


def _safe_metric_name(metric):
    return re.sub(r"[^A-Za-z0-9]+", "", metric.lower())


def _load_pseudo_start_epoch(record_dir):
    path = os.path.join(record_dir, "config_snapshot.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    value = config.get("pseudo_start_epoch")
    if isinstance(value, list) and value:
        return value[0]
    return value


def _parse_hline(value):
    if "=" in value:
        label, raw = value.split("=", 1)
    elif ":" in value:
        label, raw = value.split(":", 1)
    else:
        raw = value
        label = value

    label = label.strip()
    raw = raw.strip()
    if not label:
        label = raw
    return label, float(raw)


def _parse_extra_curve(value):
    if "=" not in value:
        raise ValueError(
            'Extra curves must use the form "Label=/path/to/record_dir".'
        )
    label, record_dir = value.split("=", 1)
    label = label.strip()
    record_dir = record_dir.strip()
    if not label or not record_dir:
        raise ValueError(
            'Extra curves must use the form "Label=/path/to/record_dir".'
        )
    return label, record_dir


def _metric_at_k(line, k):
    match = re.search(rf"@{re.escape(str(k))}=([0-9.]+)", line)
    return float(match.group(1)) if match else None


def _parse_log_epoch_metrics(log_path, metric):
    if "@" not in metric:
        raise ValueError("Log curve metric must use the form metric@K, e.g. recall@20.")
    metric_name, k = metric.lower().split("@", 1)
    log_path = os.path.abspath(log_path)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Missing curve log: {log_path}")

    valid_scores = {}
    test_values = {}
    active_epoch = None
    active_block = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_match = re.search(r"\[Epoch\s+(\d+)\]\s+Validation Summary", line)
            if epoch_match:
                active_epoch = int(epoch_match.group(1))
                active_block = "valid"
                continue

            epoch_match = re.search(r"\[Epoch\s+(\d+)\]\s+Test Summary", line)
            if epoch_match:
                active_epoch = int(epoch_match.group(1))
                active_block = "test"
                continue

            if active_epoch is not None and active_block == "valid":
                score_match = re.search(r"Validation Score:\s*([0-9.]+)", line)
                if score_match:
                    valid_scores[active_epoch] = float(score_match.group(1))

            if active_epoch is not None and active_block == "test" and re.search(
                r"\[Epoch\s+\d+\]\s+(Validation|Test) Summary",
                line,
            ):
                active_epoch = None
                active_block = None
                continue

            if active_epoch is None or active_block != "test":
                continue

            stripped = line.strip().lower()
            if stripped.startswith(f"{metric_name} ") or stripped.startswith(f"{metric_name}:"):
                value = _metric_at_k(line, k)
                if value is not None:
                    test_values[active_epoch] = value
                    active_epoch = None
                    active_block = None

    if not test_values:
        raise ValueError(f"No {metric} Test Summary points found in {log_path}")
    return valid_scores, test_values


def _load_log_test_curve(log_path, metric, curve_mode):
    valid_scores, test_values = _parse_log_epoch_metrics(log_path, metric)

    if curve_mode == "raw_test":
        return sorted(test_values.items(), key=lambda item: item[0])

    if curve_mode not in {"best_valid_test", "best_valid_updates"}:
        raise ValueError(f"Unsupported curve_mode={curve_mode}")

    best_valid = None
    selected_test = None
    points = []
    for epoch in sorted(test_values):
        valid = valid_scores.get(epoch)
        if valid is not None and (best_valid is None or valid > best_valid + 1e-12):
            best_valid = valid
            selected_test = test_values[epoch]
            points.append((epoch, selected_test))
        elif curve_mode == "best_valid_test" and selected_test is not None:
            points.append((epoch, selected_test))

    if not points:
        raise ValueError(f"No best-validation-selected {metric} points found in {log_path}")
    return points


def _load_record_curve(record_dir, metric):
    record_dir = os.path.abspath(record_dir)
    csv_path = os.path.join(record_dir, "epoch_metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing epoch_metrics.csv: {csv_path}")

    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    if metric not in rows[0]:
        available = [k for k in rows[0].keys() if "@" in k]
        raise KeyError(f"Metric {metric} not found. Available metrics: {available}")

    points = []
    for row in rows:
        epoch = int(row["epoch"])
        value = _safe_float(row.get(metric))
        if value is not None:
            points.append((epoch, value))
    if not points:
        raise ValueError(f"Metric {metric} has no values in {csv_path}")
    return sorted(points, key=lambda x: x[0])


def _merge_hlines(cli_hlines, use_default_hlines):
    hlines = []
    if use_default_hlines:
        hlines.extend(DEFAULT_HLINES)
    for item in cli_hlines or []:
        hlines.append(_parse_hline(item))
    return hlines


def _auto_ylim(values, hlines):
    all_values = list(values) + [value for _, value in hlines]
    if not all_values:
        return None
    ymin = min(all_values)
    ymax = max(all_values)
    if ymin == ymax:
        pad = max(abs(ymax) * 0.1, 0.005)
    else:
        pad = (ymax - ymin) * 0.12
    return max(0.0, ymin - pad), ymax + pad


def _sqrt_forward(values):
    values = np.asarray(values)
    return np.sqrt(np.clip(values, 0.0, None))


def _sqrt_inverse(values):
    values = np.asarray(values)
    return np.square(values)


def _fourthroot_forward(values):
    values = np.asarray(values)
    return np.power(np.clip(values, 0.0, None), 0.25)


def _fourthroot_inverse(values):
    values = np.asarray(values)
    return np.power(np.clip(values, 0.0, None), 4.0)


def _power065_forward(values):
    values = np.asarray(values)
    return np.power(np.clip(values, 0.0, None), 0.65)


def _power065_inverse(values):
    values = np.asarray(values)
    return np.power(np.clip(values, 0.0, None), 1.0 / 0.65)


def _power058_forward(values):
    values = np.asarray(values)
    return np.power(np.clip(values, 0.0, None), 0.58)


def _power058_inverse(values):
    values = np.asarray(values)
    return np.power(np.clip(values, 0.0, None), 1.0 / 0.58)


def _power035_forward(values):
    values = np.asarray(values)
    return np.power(np.clip(values, 0.0, None), 0.35)


def _power035_inverse(values):
    values = np.asarray(values)
    return np.power(np.clip(values, 0.0, None), 1.0 / 0.35)


def _format_y_tick(value):
    if value == 0:
        return "0"
    if value < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _apply_y_scale(ax, y_scale, ymin, ymax):
    if y_scale == "linear":
        return
    if y_scale not in {"sqrt", "fourthroot", "power065", "power058", "power035"}:
        raise ValueError(f"Unsupported y_scale={y_scale}")

    if y_scale == "sqrt":
        ax.set_yscale("function", functions=(_sqrt_forward, _sqrt_inverse))
    elif y_scale == "fourthroot":
        ax.set_yscale("function", functions=(_fourthroot_forward, _fourthroot_inverse))
    elif y_scale == "power065":
        ax.set_yscale("function", functions=(_power065_forward, _power065_inverse))
    elif y_scale == "power035":
        ax.set_yscale("function", functions=(_power035_forward, _power035_inverse))
    else:
        ax.set_yscale("function", functions=(_power058_forward, _power058_inverse))

    if y_scale in {"power065", "power058", "power035"}:
        candidate_ticks = [0.0, 0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    else:
        candidate_ticks = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    ticks = [tick for tick in candidate_ticks if ymin - 1e-12 <= tick <= ymax + 1e-12]
    existing_labels = {_format_y_tick(tick) for tick in ticks}
    if ticks and ymax not in ticks and ymax > ticks[-1] and _format_y_tick(ymax) not in existing_labels:
        ticks.append(ymax)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_format_y_tick(tick) for tick in ticks])


def _apply_x_scale(ax, x_scale, xmin, xmax):
    if x_scale == "linear":
        return
    if x_scale not in {"sqrt", "power08"}:
        raise ValueError(f"Unsupported x_scale={x_scale}")

    if x_scale == "sqrt":
        ax.set_xscale("function", functions=(_sqrt_forward, _sqrt_inverse))
        candidate_ticks = [0, 1, 5, 10, 20, 50, 100, 150, 200]
    else:
        ax.set_xscale(
            "function",
            functions=(
                lambda values: np.power(np.clip(np.asarray(values), 0.0, None), 0.8),
                lambda values: np.power(np.clip(np.asarray(values), 0.0, None), 1.0 / 0.8),
            ),
        )
        candidate_ticks = [0, 20, 50, 100, 150, 200]

    if xmax > 200 and x_scale != "power08":
        candidate_ticks.append(int(xmax))
    ticks = [tick for tick in candidate_ticks if xmin - 1e-12 <= tick <= xmax + 1e-12]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])


def _near_start_hlines(start_value, hline_specs):
    if not hline_specs:
        return []
    threshold = max(start_value * 1.8, start_value + 0.003)
    return [spec for spec in hline_specs if spec["value"] <= threshold]


def _build_hline_specs(hlines):
    line_styles = ["-", "--", "-.", ":"]
    colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]
    specs = []
    for idx, (label, value) in enumerate(hlines):
        specs.append({
            "label": label,
            "value": value,
            "color": colors[idx % len(colors)],
            "linestyle": line_styles[(idx // len(colors)) % len(line_styles)],
        })
    return specs


def _add_initial_inset(
    ax,
    xs,
    ys,
    hline_specs,
    pseudo_start_epoch,
    show_pseudo_start,
    inset_epochs,
    curve_linewidth,
    marker_size,
    hline_width,
):
    if not xs:
        return

    start_x = xs[0]
    start_y = ys[0]
    nearby_hlines = _near_start_hlines(start_y, hline_specs)
    if not nearby_hlines:
        return

    x_max = start_x + max(1, int(inset_epochs))
    inset_points = [(x, y) for x, y in zip(xs, ys) if start_x <= x <= x_max]
    low_values = [start_y] + [spec["value"] for spec in nearby_hlines]
    ymin = min(low_values)
    ymax = max(low_values)
    pad = max((ymax - ymin) * 0.45, 0.00032)

    iax = ax.inset_axes([0.075, 0.58, 0.30, 0.30])
    iax.plot(
        [p[0] for p in inset_points],
        [p[1] for p in inset_points],
        color="#1f77b4",
        linewidth=max(curve_linewidth * 0.9, 0.8),
        marker="o",
        markersize=max(marker_size, 2.2),
        zorder=3,
    )

    for spec in nearby_hlines:
        iax.axhline(
            spec["value"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=max(hline_width, 1.0),
            alpha=0.95,
            zorder=2,
        )

    if show_pseudo_start and pseudo_start_epoch is not None:
        ps = int(pseudo_start_epoch)
        if start_x <= ps <= x_max:
            iax.axvline(ps, color="#d62728", linestyle="--", linewidth=0.9, zorder=1)

    iax.scatter(
        [start_x],
        [start_y],
        color="#1f77b4",
        edgecolors="white",
        linewidths=0.8,
        s=34,
        zorder=5,
    )
    iax.annotate(
        f"epoch {start_x}: {start_y:.4f}",
        xy=(start_x, start_y),
        xytext=(start_x + 0.18, start_y - pad * 0.22),
        fontsize=7.0,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.6,
            "color": "#555555",
        },
    )

    iax.set_xlim(start_x - 0.15, x_max + 0.15)
    iax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    iax.set_title("Initial zoom", fontsize=7.5)
    iax.tick_params(axis="both", labelsize=7, length=2)
    iax.grid(True, linestyle="--", linewidth=0.45, alpha=0.28)


def plot_curve(
    metric,
    record_dir=None,
    curve_log=None,
    curve_mode="best_valid_test",
    output_prefix=None,
    hlines=None,
    use_default_hlines=True,
    show_pseudo_start=False,
    title=None,
    ylimit=None,
    curve_linewidth=1.35,
    marker_size=1.6,
    hline_width=1.1,
    show_initial_inset=False,
    inset_epochs=1,
    curve_label="LSDCDR",
    hide_legend=False,
    y_scale="linear",
    x_scale="linear",
    no_markers=False,
    prepend_initial_value=None,
    label_line_ends=False,
    numbers_only=False,
    save_svg=False,
    extra_curves=None,
    extra_curve_styles=None,
    warmup_end=None,
    warmup_label=None,
    max_epoch=None,
):
    if record_dir:
        record_dir = os.path.abspath(record_dir)
        points = _load_record_curve(record_dir, metric)
        pseudo_start_epoch = _load_pseudo_start_epoch(record_dir)
    else:
        if not curve_log:
            raise ValueError("Either --record_dir or --curve_log is required.")
        points = _load_log_test_curve(curve_log, metric, curve_mode)
        pseudo_start_epoch = None

    if max_epoch is not None:
        points = [point for point in points if point[0] <= max_epoch]
        if not points:
            raise ValueError(f"No main-curve points remain at or before epoch {max_epoch}.")

    if prepend_initial_value is not None:
        points = [(int(p[0]) + 1, p[1]) for p in points]
        points.insert(0, (0, float(prepend_initial_value)))
        if pseudo_start_epoch is not None:
            pseudo_start_epoch = int(pseudo_start_epoch) + 1

    extra_curve_data = []
    style_map = {
        "solid": "-",
        "dashed": "--",
        "dashdot": "-.",
        "dotted": ":",
    }
    for index, (label, extra_record_dir) in enumerate(extra_curves or []):
        extra_points = _load_record_curve(extra_record_dir, metric)
        if max_epoch is not None:
            extra_points = [point for point in extra_points if point[0] <= max_epoch]
            if not extra_points:
                raise ValueError(
                    f"No points for extra curve {label!r} remain at or before "
                    f"epoch {max_epoch}."
                )
        if prepend_initial_value is not None:
            extra_points = [(int(p[0]) + 1, p[1]) for p in extra_points]
            extra_points.insert(0, (0, float(prepend_initial_value)))
        requested_style = None
        if extra_curve_styles and index < len(extra_curve_styles):
            requested_style = extra_curve_styles[index]
        extra_curve_data.append(
            {
                "label": label,
                "points": extra_points,
                "color": ["#d62728", "#17becf", "#8c564b", "#e377c2"][index % 4],
                "linestyle": style_map.get(
                    requested_style,
                    ["--", "-.", ":", "--"][index % 4],
                ),
            }
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    hlines = _merge_hlines(hlines, use_default_hlines)
    hline_specs = _build_hline_specs(hlines)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=180)
    if warmup_end is not None:
        ax.axvspan(0, warmup_end, color="#d62728", alpha=0.035, zorder=0)
        ax.axvline(
            warmup_end,
            color="#7f7f7f",
            linestyle=":",
            linewidth=1.0,
            zorder=1,
        )
    ax.plot(
        xs,
        ys,
        color="#1f77b4",
        linewidth=curve_linewidth,
        marker=None if no_markers else "o",
        markersize=0 if no_markers else marker_size,
        label=curve_label,
    )
    for extra in extra_curve_data:
        extra_xs = [point[0] for point in extra["points"]]
        extra_ys = [point[1] for point in extra["points"]]
        ax.plot(
            extra_xs,
            extra_ys,
            color=extra["color"],
            linestyle=extra["linestyle"],
            linewidth=curve_linewidth,
            marker=None if no_markers else "o",
            markersize=0 if no_markers else marker_size,
            label=extra["label"],
        )
    ax.set_xlabel("" if numbers_only else "Epoch")
    ax.set_ylabel("" if numbers_only else _metric_label(metric))
    if title and not numbers_only:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    if show_pseudo_start and pseudo_start_epoch is not None:
        ax.axvline(
            int(pseudo_start_epoch),
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label=f"pseudo_start_epoch={pseudo_start_epoch}",
        )

    for spec in hline_specs:
        ax.axhline(
            spec["value"],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=hline_width,
            alpha=0.95,
            label=f"{spec['label']} ({spec['value']:.4f})",
        )

    if show_initial_inset and not numbers_only:
        _add_initial_inset(
            ax=ax,
            xs=xs,
            ys=ys,
            hline_specs=hline_specs,
            pseudo_start_epoch=pseudo_start_epoch,
            show_pseudo_start=show_pseudo_start,
            inset_epochs=inset_epochs,
            curve_linewidth=curve_linewidth,
            marker_size=marker_size,
            hline_width=hline_width,
        )

    if ylimit:
        ymin, ymax = ylimit
        ax.set_ylim(ymin, ymax)
    else:
        all_curve_values = list(ys)
        for extra in extra_curve_data:
            all_curve_values.extend(point[1] for point in extra["points"])
        auto_ylim = _auto_ylim(all_curve_values, hlines)
        if auto_ylim is not None:
            ymin, ymax = auto_ylim
            ax.set_ylim(ymin, ymax)
        else:
            ymin, ymax = ax.get_ylim()

    ymin, ymax = ax.get_ylim()
    _apply_y_scale(ax, y_scale, ymin, ymax)

    all_xs = list(xs)
    for extra in extra_curve_data:
        all_xs.extend(point[0] for point in extra["points"])
    x_max = max(all_xs) if all_xs else 0
    x_pad = 0.0
    if xs:
        ax.set_xlim(0, x_max + x_pad)
        _apply_x_scale(ax, x_scale, 0, x_max + x_pad)

    if warmup_end is not None and not numbers_only:
        annotation = warmup_label or f"Pseudo refinement starts (epoch {warmup_end:g})"
        ax.annotate(
            annotation,
            xy=(warmup_end, 0.975),
            xycoords=ax.get_xaxis_transform(),
            xytext=(5, 0),
            textcoords="offset points",
            color="#666666",
            fontsize=7.8,
            rotation=0,
            va="top",
            ha="left",
            clip_on=True,
        )

    if label_line_ends and xs and not numbers_only:
        label_x = x_max - max(x_max * 0.012, 2.5)
        label_box = {
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.65,
            "pad": 0.4,
        }
        ax.annotate(
            curve_label,
            xy=(label_x, ys[-1]),
            xytext=(0, -9),
            textcoords="offset points",
            color="#1f77b4",
            fontsize=8.5,
            va="top",
            ha="right",
            clip_on=True,
        )
        for index, extra in enumerate(extra_curve_data):
            if not extra["points"]:
                continue
            extra_last_x = extra["points"][-1][0]
            extra_last_y = extra["points"][-1][1]
            extra_label_x = extra_last_x - max(extra_last_x * 0.012, 2.5)
            ax.annotate(
                extra["label"],
                xy=(extra_label_x, extra_last_y),
                xytext=(0, 7 if index % 2 == 0 else -7),
                textcoords="offset points",
                color=extra["color"],
                fontsize=8.0,
                va="center",
                ha="right",
                bbox=label_box,
                clip_on=True,
            )
        hline_x = label_x
        for spec in hline_specs:
            offset_y = 0
            if spec["color"] in {"#ff7f0e", "#2ca02c"}:
                offset_y = 5
            elif spec["color"] == "#9467bd":
                offset_y = -5
            ax.annotate(
                spec["label"],
                xy=(hline_x, spec["value"]),
                xytext=(0, offset_y),
                textcoords="offset points",
                color=spec["color"],
                fontsize=8.0,
                va="center",
                ha="right",
                bbox=label_box,
                clip_on=True,
            )

    if not hide_legend and not numbers_only:
        ax.legend(frameon=False, fontsize=8.2)

    fig.tight_layout()

    if output_prefix is None:
        if record_dir:
            output_prefix = os.path.join(record_dir, f"pseudo_{_safe_metric_name(metric)}_curve")
        else:
            output_prefix = f"pseudo_{_safe_metric_name(metric)}_curve"
    png_path = f"{output_prefix}.png"
    pdf_path = f"{output_prefix}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    svg_path = None
    if save_svg:
        svg_path = f"{output_prefix}.svg"
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, svg_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_dir", default=None)
    parser.add_argument(
        "--extra_curve",
        action="append",
        default=[],
        help=(
            'Additional dynamic curve, repeatable. Format: '
            '"Label=/path/to/pseudo_quality_record_dir".'
        ),
    )
    parser.add_argument(
        "--extra_curve_style",
        action="append",
        choices=["solid", "dashed", "dashdot", "dotted"],
        default=[],
        help=(
            "Line style for each --extra_curve in the same order. "
            "Repeat this option when plotting multiple extra curves."
        ),
    )
    parser.add_argument(
        "--warmup_end",
        type=float,
        default=None,
        help="Shade the warm-up interval and mark the epoch at which refinement starts.",
    )
    parser.add_argument(
        "--warmup_label",
        default=None,
        help="Optional label shown beside --warmup_end.",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=None,
        help="Ignore main and extra curve points after this source-data epoch.",
    )
    parser.add_argument(
        "--curve_log",
        default=None,
        help="Legacy option: parse [Epoch X] Test Summary curve from a training log. Ignored when --record_dir is set.",
    )
    parser.add_argument(
        "--curve_mode",
        choices=["best_valid_test", "best_valid_updates", "raw_test"],
        default="best_valid_test",
        help=(
            "For --curve_log: best_valid_test plots the Test metric selected by validation at each epoch; "
            "raw_test plots every epoch's raw Test Summary."
        ),
    )
    parser.add_argument("--metric", default="recall@5")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument(
        "--hline",
        action="append",
        default=[],
        help='Horizontal baseline line, repeatable. Format: "Label=0.0523" or "Label:0.0523".',
    )
    parser.add_argument(
        "--no_default_hlines",
        action="store_true",
        help="Ignore DEFAULT_HLINES defined near the top of this script.",
    )
    parser.add_argument(
        "--no_pseudo_start",
        action="store_true",
        help="Do not draw pseudo_start_epoch vertical line. Default is already no vertical line.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits, e.g. --ylim 0 0.09.",
    )
    parser.add_argument("--curve_linewidth", type=float, default=1.35)
    parser.add_argument("--marker_size", type=float, default=1.6)
    parser.add_argument("--no_markers", action="store_true", help="Draw the main curve without point markers.")
    parser.add_argument("--hline_width", type=float, default=1.1)
    parser.add_argument(
        "--hide_legend",
        action="store_true",
        help="Hide the legend. Useful when labels are explained in the caption.",
    )
    parser.add_argument(
        "--label_line_ends",
        action="store_true",
        help="Label the main curve and horizontal baselines near their right ends.",
    )
    parser.add_argument(
        "--numbers_only",
        action="store_true",
        help="Remove all non-numeric text from the figure while keeping numeric tick labels.",
    )
    parser.add_argument("--save_svg", action="store_true", help="Also save the figure as SVG.")
    parser.add_argument(
        "--y_scale",
        choices=["linear", "power065", "power058", "power035", "sqrt", "fourthroot"],
        default="linear",
        help=(
            "Use a nonlinear y-axis scale. power065 is mild, power058 is stronger, "
            "power035 lifts very small baselines more, sqrt is stronger, and fourthroot is strongest."
        ),
    )
    parser.add_argument(
        "--x_scale",
        choices=["linear", "power08", "sqrt"],
        default="linear",
        help=(
            "Use a nonlinear x-axis scale. power08 mildly expands early epochs; "
            "sqrt expands them more strongly."
        ),
    )
    parser.add_argument("--show_initial_inset", action="store_true")
    parser.add_argument(
        "--inset_epochs",
        type=int,
        default=1,
        help="Number of early epochs shown in the initial zoom inset.",
    )
    parser.add_argument("--curve_label", default="LSDCDR")
    parser.add_argument(
        "--prepend_initial_value",
        type=float,
        default=None,
        help=(
            "Prepend a manually supplied initial point at epoch 0 and shift all loaded "
            "epoch points by +1. Use this for the pre-training semantic retrieval baseline."
        ),
    )
    args = parser.parse_args()

    png_path, pdf_path, svg_path = plot_curve(
        record_dir=args.record_dir,
        metric=args.metric,
        curve_log=args.curve_log,
        curve_mode=args.curve_mode,
        output_prefix=args.output_prefix,
        hlines=args.hline,
        use_default_hlines=not args.no_default_hlines,
        show_pseudo_start=False if args.no_pseudo_start else False,
        title=args.title,
        ylimit=args.ylim,
        curve_linewidth=args.curve_linewidth,
        marker_size=args.marker_size,
        hline_width=args.hline_width,
        show_initial_inset=args.show_initial_inset,
        inset_epochs=args.inset_epochs,
        curve_label=args.curve_label,
        hide_legend=args.hide_legend,
        y_scale=args.y_scale,
        x_scale=args.x_scale,
        no_markers=args.no_markers,
        prepend_initial_value=args.prepend_initial_value,
        label_line_ends=args.label_line_ends,
        numbers_only=args.numbers_only,
        save_svg=args.save_svg,
        extra_curves=[_parse_extra_curve(value) for value in args.extra_curve],
        extra_curve_styles=args.extra_curve_style,
        warmup_end=args.warmup_end,
        warmup_label=args.warmup_label,
        max_epoch=args.max_epoch,
    )
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    if svg_path:
        print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
