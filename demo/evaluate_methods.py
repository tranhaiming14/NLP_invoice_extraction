"""
evaluate_methods.py — Evaluate Method 1 (OCR + Rules) and Method 2 (OCR + LLM)
on the SROIE2019 test set.

Fields evaluated: company, date, address, total

Metrics per field:
  - Exact Match (EM): after lowercasing + whitespace normalisation
  - Token F1       : overlap of whitespace-split tokens (same as SROIE / SQuAD)

Usage examples:
    # Evaluate both methods on the first 50 images
    python evaluate_methods.py --samples 50

    # Only rule-based, all images, save detailed results
    python evaluate_methods.py --methods rule --output results_rule.json

    # Only LLM, 30 images, 2-second delay between requests (avoids rate-limit)
    python evaluate_methods.py --methods llm --samples 30 --llm-delay 2.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── path setup so demo/ modules are importable ──────────────────────────────
WORKSPACE = Path(__file__).parent.parent  # Go up from demo/ to root
sys.path.insert(0, str(WORKSPACE / "demo"))

IMG_DIR      = WORKSPACE / "train" / "SROIE2019" / "test" / "img"
ENTITIES_DIR = WORKSPACE / "train" / "SROIE2019" / "test" / "entities"
FIELDS       = ["company", "date", "address", "total"]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lower-case, strip, collapse internal whitespace."""
    return " ".join(str(text).lower().split())


def exact_match(pred: str, truth: str) -> bool:
    return _norm(pred) == _norm(truth)


def token_f1(pred: str, truth: str) -> float:
    """Token-level F1 (SROIE / SQuAD style)."""
    p_toks = _norm(pred).split()
    t_toks = _norm(truth).split()
    if not p_toks and not t_toks:
        return 1.0
    if not p_toks or not t_toks:
        return 0.0
    common = {}
    for tok in set(p_toks) & set(t_toks):
        common[tok] = min(p_toks.count(tok), t_toks.count(tok))
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(p_toks)
    recall    = num_common / len(t_toks)
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth() -> dict[str, dict]:
    """Return {file_stem: {company, date, address, total}} from SROIE entities."""
    gt: dict[str, dict] = {}
    for f in sorted(ENTITIES_DIR.glob("*.txt")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            gt[f.stem] = {k: str(v).strip() for k, v in data.items()}
        except Exception as e:
            print(f"[WARN] Could not parse ground truth {f.name}: {e}")
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation core
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    extract_fn,
    method_name: str,
    gt: dict,
    samples: int | None,
    llm_delay: float,
) -> tuple[dict, list[dict]]:
    """
    Run *extract_fn* on images and compare to ground truth.

    Returns
    -------
    summary : dict  — per-field and overall EM / F1 scores
    rows    : list  — per-image detail rows for JSON output
    """
    stems = sorted(gt.keys())
    if samples:
        stems = stems[:samples]

    per_field: dict[str, dict] = {
        f: {"em": [], "f1": []} for f in FIELDS
    }
    rows: list[dict] = []
    errors = 0

    print(f"\n{'─'*60}")
    print(f"  Evaluating: {method_name}  ({len(stems)} images)")
    print(f"{'─'*60}")

    for i, stem in enumerate(stems):
        img_path = IMG_DIR / f"{stem}.jpg"
        if not img_path.exists():
            print(f"  [SKIP] {stem} — image not found")
            continue

        try:
            pred = extract_fn(str(img_path))
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")
            errors += 1
            continue

        truth = gt[stem]
        row = {"file": stem, "method": method_name}
        for field in FIELDS:
            p = str(pred.get(field, "") or "")
            t = str(truth.get(field, "") or "")
            em = exact_match(p, t)
            f1 = token_f1(p, t)
            per_field[field]["em"].append(em)
            per_field[field]["f1"].append(f1)
            row[f"{field}_pred"]  = p
            row[f"{field}_truth"] = t
            row[f"{field}_em"]    = em
            row[f"{field}_f1"]    = round(f1, 4)
        rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == len(stems):
            print(f"  [{i+1:>4}/{len(stems)}] done  (errors so far: {errors})")

        if llm_delay and i < len(stems) - 1:
            time.sleep(llm_delay)

    # ── aggregate ──
    summary: dict = {"method": method_name, "n_evaluated": len(rows), "n_errors": errors}
    overall_em, overall_f1 = [], []
    for field in FIELDS:
        ems = per_field[field]["em"]
        f1s = per_field[field]["f1"]
        field_em = sum(ems) / len(ems) if ems else 0.0
        field_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        summary[field] = {
            "exact_match": round(field_em * 100, 2),
            "token_f1":    round(field_f1 * 100, 2),
        }
        overall_em.append(field_em)
        overall_f1.append(field_f1)

    summary["overall"] = {
        "exact_match": round(sum(overall_em) / len(FIELDS) * 100, 2),
        "token_f1":    round(sum(overall_f1) / len(FIELDS) * 100, 2),
    }
    return summary, rows


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(summaries: list[dict]) -> None:
    col_w = 14
    header_fields = ["Method"] + [f.capitalize() for f in FIELDS] + ["Overall"]
    sep = "+" + "+".join(["-" * (col_w + 2)] * len(header_fields)) + "+"

    def row(*cells):
        return "| " + " | ".join(str(c).ljust(col_w) for c in cells) + " |"

    print("\n" + "=" * 80)
    print("  RESULTS — Token F1 (%) / Exact Match (%)")
    print("=" * 80)
    print(sep)
    print(row(*header_fields))
    print(sep)
    for s in summaries:
        f1_row   = [s["method"][:col_w]]
        em_row   = ["(exact match)"]
        for field in FIELDS:
            f1_row.append(f"{s[field]['token_f1']:.1f}%")
            em_row.append(f"{s[field]['exact_match']:.1f}%")
        f1_row.append(f"{s['overall']['token_f1']:.1f}%")
        em_row.append(f"{s['overall']['exact_match']:.1f}%")
        print(row(*f1_row))
        print(row(*em_row))
        print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR+Rule and OCR+LLM methods on SROIE2019 test set."
    )
    parser.add_argument(
        "--methods",
        default="rule,llm",
        help="Comma-separated list of methods to run: rule, llm  (default: rule,llm)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Max number of test images to evaluate (default: all ~347).",
    )
    parser.add_argument(
        "--llm-delay",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Seconds to wait between LLM API calls to avoid rate-limiting (default: 1.0).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save detailed per-image results as JSON.",
    )
    args = parser.parse_args()

    methods_to_run = [m.strip().lower() for m in args.methods.split(",")]
    valid_methods = {"rule", "llm"}
    invalid = set(methods_to_run) - valid_methods
    if invalid:
        sys.exit(f"Unknown method(s): {invalid}. Choose from: rule, llm")

    # ── load ground truth ──
    print("Loading ground truth …", end=" ", flush=True)
    gt = load_ground_truth()
    print(f"{len(gt)} files found.")

    if not gt:
        sys.exit(f"No ground truth files found in {ENTITIES_DIR}. Check your dataset path.")

    summaries: list[dict] = []
    all_rows:  list[dict] = []

    # ── Method 1: OCR + Rules ──
    if "rule" in methods_to_run:
        from method1_rule import extract_with_rules
        summary, rows = evaluate(
            extract_fn=extract_with_rules,
            method_name="OCR + Rule-based",
            gt=gt,
            samples=args.samples,
            llm_delay=0.0,
        )
        summaries.append(summary)
        all_rows.extend(rows)

    # ── Method 2: OCR + LLM ──
    if "llm" in methods_to_run:
        from method2_llm import extract_with_llm
        summary, rows = evaluate(
            extract_fn=extract_with_llm,
            method_name="OCR + LLM (Gemini)",
            gt=gt,
            samples=args.samples,
            llm_delay=args.llm_delay,
        )
        summaries.append(summary)
        all_rows.extend(rows)

    # ── print table ──
    print_summary_table(summaries)

    # ── save JSON ──
    output_path = args.output
    if output_path is None:
        n_tag = f"_{args.samples}" if args.samples else "_all"
        output_path = str(WORKSPACE / f"eval_results{n_tag}.json")

    out = {"summaries": summaries, "details": all_rows}
    Path(output_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
