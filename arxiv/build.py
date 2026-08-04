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
Embryo selection in {IVF} still rests on morphology, and the systems built to
standardise it rank blastocysts from images alone, without the patient context
clinicians weigh. We present IVF-Bench: 753 blastocyst cases from the public
Kromp dataset, each pairing a real embryo image and its Gardner annotation with
that patient's cycle data and recorded outcome, plus patient-history fields
sampled from published distributions and labelled as generated. Models write an
open-ended assessment, scored on five clinical rubrics by a judge shown the same
embryo and on the implantation probability they state. Across eight systems,
clinical integration is uniformly strong and outcome discrimination is not:
{AUROC} never exceeds 0.562, which a logistic regression on the measured fields
alone matches, placing the ceiling in the data rather than the models.
Post-training Qwen 3.5-9B with {ORPO} on 500 benchmark-derived pairs raises it
24.6\% over its own base, a gain a second judge puts at 15.5\%; the second place
it buys, ahead of Claude Opus 4.6, holds under one judge and not the other,
because judges favour their own lineage. We release the benchmark, the model, and
all 5,194 responses with transcripts from both judges.
"""

CITATION = r"""@article{correa2026ivfbench,
  author  = {Correa, Andrew G. A. and Yoon, Brittany},
  title   = {{IVF-Bench}: Evaluating Vision-Language Models on IVF Reasoning},
  journal = {arXiv preprint},
  year    = {2026}
}"""


def figure(name: str) -> str:
    """Pull one '% ===== NAME =====' block out of figures.tex."""
    text = (ARXIV / "figures.tex").read_text()
    return text.split(f"% ===== {name} =====")[1].split("% =====")[0].rstrip() + "\n"


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
        "plus the numeric probability it extracted from the response. The text "
        "below is what the judge sees, verbatim.\n"
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



def distribution_appendix() -> str:
    """Every generated field, its distribution, and where the parameters come from."""
    rows = [
        ("BMI (kg/m$^2$)", "Truncated normal, mean 25.0, SD 5.0, range [17, 45]",
         "\\citet{eshre2025}"),
        ("Basal FSH (IU/L)", "Truncated normal, mean 6.0 at age 30, $+0.15$ per year",
         "\\citet{eshre2025}"),
        ("Diagnosis", "Categorical: tubal 0.25, male 0.25, unexplained 0.15, "
         "PCOS 0.12, endometriosis 0.10, DOR 0.05, uterine 0.03, other 0.05",
         "\\citet{eshre2025}"),
        ("Stimulation protocol", "Categorical: antagonist 0.70, long agonist 0.20, "
         "short agonist 0.05, natural 0.03, other 0.02",
         "\\citet{yoo2026}"),
        ("Previous cycles", "Categorical: 0 at 0.40, then 0.25, 0.15, 0.10, 0.05, 0.05",
         "\\citet{vanderborght2025}"),
        ("Previous outcomes", "Categorical per cycle: no pregnancy 0.45, biochemical "
         "0.20, miscarriage 0.15, live birth 0.15, ectopic 0.05",
         "\\citet{vanderborght2025}"),
        ("Partner age (years)", "Female age $+$ Normal(2.4, 4.3), clipped to [20, 65]",
         "\\citet{jelinkova2026}"),
        ("Sperm concentration (M/mL)", "Truncated normal, mean 50.0, SD 25.0, range [5, 200]",
         "\\citet{who2021semen}"),
        ("Sperm motility (\\%)", "Truncated normal, mean 55.0, SD 15.0, range [10, 95]",
         "\\citet{who2021semen}"),
        ("Smoking", "Categorical: never 0.65, former 0.25, current 0.10",
         "\\citet{dodge2015}"),
        ("Alcohol", "Categorical: none 0.40, light 0.35, moderate 0.20, heavy 0.05",
         "\\citet{dodge2015}"),
        ("Physical activity", "Categorical: sedentary 0.30, light 0.25, moderate 0.25, "
         "active 0.15, very active 0.05", "\\citet{sherwin2022}"),
        ("Diet", "Categorical: standard 0.50, health-conscious 0.25, Mediterranean 0.15, "
         "vegetarian 0.05, restricted 0.05", "\\citet{gaskins2019}"),
    ]
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
    return (
        "These are the fields that no public dataset carries alongside embryo "
        "images and outcomes, and that we therefore generate. Each is drawn from "
        "the distribution shown, seeded deterministically from the case "
        "identifier; basal FSH and partner age are conditioned on the patient's "
        "real age and the rest are drawn independently, as Section~\\ref{sec:synthetic} sets out. Sources are the nearest published population "
        "estimate we could find for an IVF-treated cohort; where registries "
        "disagree by geography we chose a Western-representative central "
        "value.\n\n"
        "\\begin{table}[H]\n\\centering\\footnotesize\n"
        "\\begin{tabular}{p{3.1cm}p{7.4cm}p{2.6cm}}\n\\toprule\n"
        "\\textbf{Field} & \\textbf{Distribution} & \\textbf{Source} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\caption{Generated patient-context fields. None is sampled with "
        "reference to the recorded outcome. Basal FSH and partner age are "
        "conditioned on the patient's real age and therefore act as weak age "
        "proxies; the remainder are drawn independently of one another, so "
        "genuine inter-field clinical correlations are absent.}\n"
        "\\end{table}\n"
    )


def artifact_appendix() -> str:
    return (
        "The benchmark, the evaluation code, every response and judge transcript, and "
        "the analysis behind each interval are at "
        "\\url{https://github.com/The-Fertility-Plan/ivf-bench}. "
        "The post-trained model is at "
        "\\url{https://huggingface.co/thefertilityplan/ivf-bench-qwen9b-vlm-orpo} "
        "and the preference dataset at "
        "\\url{https://huggingface.co/datasets/thefertilityplan/ivf-bench-orpo-qwen9b-clipped}. "
        "Code is released under Apache 2.0.\n\n"
        "\\begin{sloppypar}\n"
        "Judge transcripts are grouped by the condition the judge was in, so any "
        "comparison here can be recomputed and any claim about a confound checked. "
        "\\texttt{scores\\_sighted} holds the scores reported throughout, produced "
        "by a judge shown the embryo; \\texttt{scores} holds the earlier blind "
        "judge, which Section~\\ref{sec:robustness} compares against; "
        "\\texttt{scores\\_sonnet\\_sighted} holds the cross-judge pass; and the "
        "directories named for a confound hold the superseded runs described in "
        "Appendix~\\ref{app:defects}. "
        "\\texttt{provider\\_manifest.json} records which third-party host served "
        "each call, so the provider mixture noted in Section~\\ref{sec:setup} can "
        "be recomputed. We do not ship the full raw API envelopes, which run to "
        "125\\,MB and add nothing the response and manifest files do not "
        "already carry.\n\\end{sloppypar}\n\n"
        "We do not redistribute the embryo images. They belong to the Kromp "
        "blastocyst dataset \\citep{kromp2023}, are available under CC BY 4.0 at "
        "\\url{https://doi.org/10.6084/m9.figshare.20123153.v3}, and our release "
        "rebuilds every case from that source deterministically.\n\n"
        "To cite:\n\\vspace{-0.4em}\n\\begin{lstlisting}[frame=none, backgroundcolor=\\color{white}]\n" + CITATION + "\n\\end{lstlisting}\n"
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
        ("JUDGELESSONS_PLACEHOLDER", section("JUDGELESSONS", sec)),
        ("LIMITATIONS_PLACEHOLDER", section("LIMITATIONS", sec)),
        ("RELEASE_PLACEHOLDER", section("RELEASE", sec)),
        ("PROMPT_APPENDIX_PLACEHOLDER", prompt_appendix()),
        ("RUBRIC_APPENDIX_PLACEHOLDER", rubric_appendix()),
        ("DEFECTS_APPENDIX_PLACEHOLDER", (ARXIV / "defects.tex").read_text()),
        ("DISTRIBUTION_APPENDIX_PLACEHOLDER", distribution_appendix()),
        ("ARTIFACT_APPENDIX_PLACEHOLDER", artifact_appendix()),
        ("FIG_PIPELINE_PLACEHOLDER", figure("FIG_PIPELINE")),
        ("FIG_RUBRICS_PLACEHOLDER", figure("FIG_RUBRICS")),
        ("FIG_BASERATE_PLACEHOLDER", figure("FIG_BASERATE")),
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
