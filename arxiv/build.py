"""Assemble main.tex from the section file and the live benchmark artifacts.

Keeping the prose in sections.tex and splicing it here means the paper cannot
drift from the numbers: the appendices are read straight out of the code and the
case files, so if a rubric or a prompt changes, the paper changes with it.

Usage:  python arxiv/build.py   ->  writes arxiv/ivf_bench.tex
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARXIV = ROOT / "arxiv"
sys.path.insert(0, str(ROOT / "src"))

ABSTRACT = r"""
Embryo selection still rests on a subjective morphology grade, and the machine
learning built to improve it scores images alone. The largest randomised trial of
that approach did not beat trained embryologists. The information that drives the
decision, the patient's history and lifestyle, sits with the physician and
reaches neither the embryologist nor the model, and no public dataset carries it
alongside embryo images and outcomes. We propose IVF-Bench, a standard for
measuring how well vision-language models reason about a transfer decision: 753
real day-5 cases, with open-ended answers graded against five clinical rubrics
and against the recorded outcome. Across eight systems, post-training a 9B open
model on 550 preference pairs lifts every rubric and takes it past Claude Opus
4.6 at a twentieth of the cost, while outcome prediction stays at a floor we
argue is set by the missing data rather than by the models. Benchmark, model, and
all 5,194 responses are released.
"""

CITATION = r"""@article{correa2026ivfbench,
  author  = {Correa, Andrew G. A. and Yoon, Brittany},
  title   = {{IVF-Bench}: Vision-Language Models Explain Embryo Cases Well and
             Predict Outcomes Poorly},
  journal = {arXiv preprint},
  year    = {2026}
}"""


def section(name: str, text: str) -> str:
    """Pull one '% ===== NAME =====' block out of sections.tex."""
    body = text.split(f"% ===== {name} =====")[1]
    return body.split("% =====")[0].rstrip() + "\n"


def prompt_appendix() -> str:
    case = json.loads((ROOT / "data/cases/IVF-BENCH-0001.json").read_text())
    return (
        "Every model receives this prompt together with the embryo image. The "
        "example is case IVF-BENCH-0001; laboratory and patient values change "
        "per case, the structure does not.\n\n"
        "\\begin{lstlisting}\n" + case["prompt"].replace("\t", "  ") + "\n\\end{lstlisting}\n"
    )


def rubric_appendix() -> str:
    from ivf_bench.eval.judge import RUBRIC_DEFINITIONS

    out = [
        "The judge returns a score and a one-line justification for each rubric, "
        "plus the numeric probability it extracted from the response. Rubric text "
        "is reproduced verbatim from \\texttt{src/ivf\\_bench/eval/judge.py}.\n"
    ]
    for key, r in RUBRIC_DEFINITIONS.items():
        out.append("\\paragraph{%s}" % r["name"].replace("&", "\\&"))
        out.append(re.sub(r"\s+", " ", r["description"]).strip().replace("&", "\\&"))
        out.append("\\begin{itemize}")
        for score in sorted(r["scale"]):
            anchor = str(r["scale"][score]).replace("&", "\\&")
            out.append(f"\\item \\textbf{{{score}}}: {anchor}")
        out.append("\\end{itemize}\n")
    return "\n".join(out)


def artifact_appendix() -> str:
    rows = [
        ("Code and benchmark", "\\url{https://github.com/The-Fertility-Plan/ivf-bench}"),
        ("Trained model",
         "\\url{https://huggingface.co/thefertilityplan/ivf-bench-qwen9b-vlm-orpo}"),
        ("Preference dataset",
         "\\url{https://huggingface.co/datasets/thefertilityplan/ivf-bench-orpo-qwen9b-clipped}"),
        ("Source embryo data (CC BY 4.0)",
         "\\url{https://doi.org/10.6084/m9.figshare.20123153.v3}"),
        ("Model responses and judge transcripts", "\\texttt{data/runs/} in the repository"),
        ("Statistics in this paper", "\\texttt{scripts/paper\\_analysis.py}"),
    ]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    return (
        "\\begin{table}[H]\n\\centering\\small\n"
        "\\begin{tabular}{ll}\n\\toprule\n"
        "\\textbf{Artifact} & \\textbf{Location} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Everything released with this paper.}\n\\end{table}\n\n"
        "To cite this work:\n\n\\begin{lstlisting}\n" + CITATION + "\n\\end{lstlisting}\n"
    )


def main() -> None:
    tpl = (ARXIV / "main.tex").read_text()
    sec = (ARXIV / "sections.tex").read_text()

    for placeholder, value in [
        ("ABSTRACT_PLACEHOLDER", ABSTRACT.strip()),
        ("RESULTS_SETUP_PLACEHOLDER", section("RESULTS_SETUP", sec)),
        ("RESULTS_PLACEHOLDER", section("RESULTS", sec)),
        ("ORPO_PLACEHOLDER", section("ORPO", sec)),
        ("ROBUSTNESS_PLACEHOLDER", section("ROBUSTNESS", sec)),
        ("LIMITATIONS_PLACEHOLDER", section("LIMITATIONS", sec)),
        ("RELEASE_PLACEHOLDER", section("RELEASE", sec)),
        ("PROMPT_APPENDIX_PLACEHOLDER", prompt_appendix()),
        ("RUBRIC_APPENDIX_PLACEHOLDER", rubric_appendix()),
        ("ARTIFACT_APPENDIX_PLACEHOLDER", artifact_appendix()),
    ]:
        if placeholder not in tpl:
            raise SystemExit(f"placeholder {placeholder} missing from main.tex")
        tpl = tpl.replace(placeholder, value)

    out = ARXIV / "ivf_bench.tex"
    out.write_text(tpl)
    print(f"wrote {out} ({len(tpl.splitlines())} lines)")
    left = re.findall(r"[A-Z_]+_PLACEHOLDER", tpl)
    print("unfilled placeholders:", left or "none")


if __name__ == "__main__":
    main()
