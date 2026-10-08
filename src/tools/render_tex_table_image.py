import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def _find_latex_engine():
    for engine in ("pdflatex", "xelatex", "lualatex"):
        if shutil.which(engine):
            return engine
    return None


def _strip_table_float(tex):
    lines = []
    for line in tex.splitlines():
        stripped = line.strip()
        if re.match(r"\\begin\{table\*?\}", stripped):
            continue
        if re.match(r"\\end\{table\*?\}", stripped):
            continue
        if stripped == r"\centering":
            continue
        if stripped.startswith(r"\caption"):
            continue
        if stripped.startswith(r"\label"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _build_document(tabular_tex):
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{multirow}",
            r"\pagestyle{empty}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\hoffset}{-1in}",
            r"\setlength{\voffset}{-1in}",
            r"\setlength{\oddsidemargin}{6pt}",
            r"\setlength{\topmargin}{6pt}",
            r"\setlength{\headheight}{0pt}",
            r"\setlength{\headsep}{0pt}",
            r"\setlength{\textwidth}{24in}",
            r"\setlength{\textheight}{8in}",
            r"\ifdefined\pdfpagewidth",
            r"  \pdfpagewidth=24in",
            r"  \pdfpageheight=8in",
            r"\fi",
            r"\begin{document}",
            r"\noindent",
            tabular_tex,
            r"\end{document}",
            "",
        ]
    )


def _crop_png_whitespace(path, padding=12):
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return

    image = Image.open(path).convert("RGB")
    white = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, white)
    bbox = diff.point(lambda pixel: 255 if pixel > 10 else 0).getbbox()
    if bbox is None:
        return

    left = max(0, bbox[0] - padding)
    upper = max(0, bbox[1] - padding)
    right = min(image.size[0], bbox[2] + padding)
    lower = min(image.size[1], bbox[3] + padding)
    image.crop((left, upper, right, lower)).save(path)


def _run(cmd, cwd):
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or ""
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nLaTeX/tool output:\n"
            + output[-8000:]
        ) from exc


def render_tex_to_image(input_tex, output_png, dpi):
    engine = _find_latex_engine()
    if engine is None:
        raise RuntimeError(
            "No LaTeX engine found. Please install pdflatex/xelatex/lualatex "
            "or run this script on a machine with TeX Live/MiKTeX."
        )
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("Missing pdftoppm. Please install poppler-utils.")

    input_tex = Path(input_tex)
    output_png = Path(output_png)
    tex = input_tex.read_text(encoding="utf-8")
    doc = _build_document(_strip_table_float(tex))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tex_table_") as tmp:
        tmp_path = Path(tmp)
        wrapped_tex = tmp_path / "table_image.tex"
        wrapped_tex.write_text(doc, encoding="utf-8")

        _run(
            [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                wrapped_tex.name,
            ],
            cwd=tmp_path,
        )

        pdf_path = tmp_path / "table_image.pdf"
        output_pdf = output_png.with_suffix(".pdf")
        shutil.copyfile(pdf_path, output_pdf)

        png_prefix = output_png.with_suffix("")
        _run(
            [
                "pdftoppm",
                "-png",
                "-singlefile",
                "-r",
                str(dpi),
                str(output_pdf),
                str(png_prefix),
            ],
            cwd=Path.cwd(),
        )
        _crop_png_whitespace(output_png)

    return output_png, output_pdf


def parse_args():
    parser = argparse.ArgumentParser(description="Render a LaTeX table snippet to PNG/PDF.")
    parser.add_argument("input_tex", help="Path to a LaTeX table snippet.")
    parser.add_argument(
        "--output_png",
        help="Output PNG path. Defaults to input filename with .png suffix.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    input_tex = Path(args.input_tex)
    output_png = Path(args.output_png) if args.output_png else input_tex.with_suffix(".png")
    png_path, pdf_path = render_tex_to_image(input_tex, output_png, args.dpi)
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
