#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import cohen_kappa_score

FIELDS = [
    "catalyst", "metal_node", "linker", "phase", "solvent", "sacrificial_donor",
    "photosensitizer", "additive", "reaction_time_h", "reaction_temp_c",
    "catalyst_mass_mg", "co2_pressure_bar", "light_type", "product",
    "performance_value", "performance_unit", "source_anchor"
]
NUMERIC_FIELDS = {
    "reaction_time_h", "reaction_temp_c", "catalyst_mass_mg",
    "co2_pressure_bar", "performance_value"
}
DEFAULT_GROUPS = {
    "catalyst_identity": ["catalyst", "metal_node", "linker"],
    "reaction_conditions": [
        "phase", "solvent", "sacrificial_donor", "photosensitizer", "additive",
        "reaction_time_h", "reaction_temp_c", "catalyst_mass_mg",
        "co2_pressure_bar", "light_type"
    ],
    "products": ["product"],
    "performance": ["performance_value", "performance_unit"],
}


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s in {"", "none", "null", "nan", "n/a", "na", "not reported", "missing"}:
        return ""
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([,;/+])\s*", r"\1", s)
    return s


def to_float(x: Any) -> Optional[float]:
    s = norm_text(x)
    if not s:
        return None
    s = s.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def numeric_equal(a: Any, b: Any, rtol: float, atol: float) -> bool:
    x, y = to_float(a), to_float(b)
    if x is None or y is None:
        return norm_text(a) == norm_text(b)
    return math.isclose(x, y, rel_tol=rtol, abs_tol=atol)


def field_equal(field: str, a: Any, b: Any, cfg: Dict[str, Any]) -> bool:
    if field in NUMERIC_FIELDS:
        return numeric_equal(a, b, cfg.get("numeric_relative_tolerance", 0.01), cfg.get("numeric_absolute_tolerance", 1e-8))
    return norm_text(a) == norm_text(b)


def similarity(a: Any, b: Any) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def numeric_similarity(a: Any, b: Any) -> float:
    x, y = to_float(a), to_float(b)
    if x is None and y is None:
        return 1.0
    if x is None or y is None:
        return 0.0
    denom = max(abs(x), abs(y), 1e-12)
    return max(0.0, 1.0 - min(abs(x-y)/denom, 1.0))


def record_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    cat = similarity(a.get("catalyst"), b.get("catalyst"))
    prod = similarity(a.get("product"), b.get("product"))
    phase = similarity(a.get("phase"), b.get("phase"))
    solvent = similarity(a.get("solvent"), b.get("solvent"))
    donor = similarity(a.get("sacrificial_donor"), b.get("sacrificial_donor"))
    ps = similarity(a.get("photosensitizer"), b.get("photosensitizer"))
    cond_num = np.mean([
        numeric_similarity(a.get("reaction_time_h"), b.get("reaction_time_h")),
        numeric_similarity(a.get("reaction_temp_c"), b.get("reaction_temp_c")),
        numeric_similarity(a.get("catalyst_mass_mg"), b.get("catalyst_mass_mg")),
        numeric_similarity(a.get("co2_pressure_bar"), b.get("co2_pressure_bar")),
    ])
    return float(0.45*cat + 0.20*prod + 0.05*phase + 0.10*solvent + 0.05*donor + 0.05*ps + 0.10*cond_num)


def align_records(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]], threshold: float) -> Tuple[List[Tuple[int,int,float]], List[int], List[int]]:
    if not a_rows or not b_rows:
        return [], list(range(len(a_rows))), list(range(len(b_rows)))
    score = np.zeros((len(a_rows), len(b_rows)), dtype=float)
    for i, a in enumerate(a_rows):
        for j, b in enumerate(b_rows):
            score[i, j] = record_similarity(a, b)
    r, c = linear_sum_assignment(1.0 - score)
    pairs, used_a, used_b = [], set(), set()
    for i, j in zip(r, c):
        s = float(score[i, j])
        if s >= threshold:
            pairs.append((i, j, s)); used_a.add(i); used_b.add(j)
    return pairs, [i for i in range(len(a_rows)) if i not in used_a], [j for j in range(len(b_rows)) if j not in used_b]


def group_by_paper(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = defaultdict(list)
    for r in rows:
        out[str(r.get("paper_id", "")).strip()].append(r)
    return dict(out)


def cmd_sample(args: argparse.Namespace) -> None:
    rows = read_csv(args.master)
    if not rows:
        raise SystemExit("Master CSV is empty.")
    if args.paper_col not in rows[0]:
        raise SystemExit(f"Missing paper column: {args.paper_col}")
    excluded = set()
    if args.exclude_papers and Path(args.exclude_papers).exists():
        excluded = {x.strip() for x in Path(args.exclude_papers).read_text(encoding="utf-8").splitlines() if x.strip()}

    paper_rows = defaultdict(list)
    for r in rows:
        p = str(r.get(args.paper_col, "")).strip()
        if p and p not in excluded:
            paper_rows[p].append(r)

    strata = defaultdict(list)
    for p, rs in paper_rows.items():
        phase = norm_text(next((r.get(args.phase_col, "") for r in rs if norm_text(r.get(args.phase_col, ""))), "")) if args.phase_col else ""
        products = sorted({norm_text(r.get(args.product_col, "")) for r in rs if norm_text(r.get(args.product_col, ""))}) if args.product_col else []
        product_key = "+".join(products[:3]) or "unknown"
        strata[(phase or "unknown", product_key)].append(p)

    rng = random.Random(args.seed)
    for s in strata.values(): rng.shuffle(s)
    selected = []
    keys = list(strata.keys()); rng.shuffle(keys)
    while len(selected) < min(args.n_papers, len(paper_rows)):
        progressed = False
        for k in keys:
            if strata[k]:
                selected.append(strata[k].pop()); progressed = True
                if len(selected) >= args.n_papers: break
        if not progressed: break

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    Path(outdir/"selected_papers.txt").write_text("\n".join(selected)+"\n", encoding="utf-8")
    sampled = []
    for p in selected:
        sampled.extend(paper_rows[p])
    fns = list(rows[0].keys())
    write_csv(outdir/"sampled_master_records.csv", sampled, fns)
    print(f"Selected {len(selected)} papers / {len(sampled)} existing records. Excluded {len(excluded)} papers.")


def cmd_make_templates(args: argparse.Namespace) -> None:
    papers = [x.strip() for x in Path(args.selected_papers).read_text(encoding="utf-8").splitlines() if x.strip()]
    headers = ["paper_id", "record_local_id"] + FIELDS + ["annotation_notes"]
    rows = []
    for p in papers:
        rows.append({"paper_id": p, "record_local_id": "1"})
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir/"annotator1.csv", rows, headers)
    write_csv(outdir/"annotator2.csv", rows, headers)
    instructions = (
        "BLIND DOUBLE-ANNOTATION INSTRUCTIONS\n\n"
        "1. Annotators must read the source article/Supporting Information directly and must NOT see the LLM output or the previously curated database.\n"
        "2. Add one row per catalyst-condition-product record. If a paper contains several records, add rows and number record_local_id by first appearance in the article.\n"
        "3. If multiple products are reported under the same conditions, use separate rows with identical condition fields and different product/performance fields.\n"
        "4. Leave unreported information blank. Never infer or write zero for missing values.\n"
        "5. Preserve the reported performance unit; do not normalize during annotation.\n"
        "6. source_anchor should identify the table/figure/paragraph or a short local phrase that supports the record.\n"
        "7. Annotators work independently. Disagreements are adjudicated only after both files are frozen.\n"
    )
    Path(outdir/"ANNOTATION_INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
    print(f"Created blind annotation templates for {len(papers)} papers.")


def load_annotation(path: str | Path) -> List[Dict[str, Any]]:
    rows = read_csv(path)
    out = []
    for r in rows:
        # Drop completely blank placeholder rows except paper_id.
        if any(norm_text(r.get(f, "")) for f in FIELDS if f != "source_anchor"):
            out.append(r)
    return out


def cmd_agreement(args: argparse.Namespace) -> None:
    cfg = read_json(args.config)
    a1, a2 = load_annotation(args.annotator1), load_annotation(args.annotator2)
    g1, g2 = group_by_paper(a1), group_by_paper(a2)
    papers = sorted(set(g1)|set(g2))
    threshold = cfg.get("record_match_threshold", 0.55)
    low_thr = cfg.get("low_confidence_alignment_threshold", 0.75)

    alignment_rows, disagreements = [], []
    field_pairs = defaultdict(lambda: [[], []])
    group_stats = defaultdict(lambda: {"agree":0, "total":0})
    unmatched_a1 = unmatched_a2 = 0

    groups = cfg.get("field_groups", DEFAULT_GROUPS)
    field_to_group = {f:g for g, fs in groups.items() for f in fs}

    for p in papers:
        r1, r2 = g1.get(p, []), g2.get(p, [])
        pairs, ua, ub = align_records(r1, r2, threshold)
        unmatched_a1 += len(ua); unmatched_a2 += len(ub)
        for i,j,s in pairs:
            id1 = r1[i].get("record_local_id", str(i+1)); id2 = r2[j].get("record_local_id", str(j+1))
            alignment_rows.append({
                "paper_id":p,"annotator1_record_id":id1,"annotator2_record_id":id2,
                "similarity":f"{s:.4f}","review_alignment":"YES" if s < low_thr else ""
            })
            for f in [x for x in FIELDS if x != "source_anchor"]:
                v1, v2 = r1[i].get(f,""), r2[j].get(f,"")
                eq = field_equal(f,v1,v2,cfg)
                g = field_to_group.get(f, "other")
                # agreement includes missing vs missing as agreement; this is reported separately from extraction metrics.
                group_stats[g]["total"] += 1
                group_stats[g]["agree"] += int(eq)
                if f not in NUMERIC_FIELDS:
                    field_pairs[f][0].append(norm_text(v1) or "__MISSING__")
                    field_pairs[f][1].append(norm_text(v2) or "__MISSING__")
                if not eq:
                    disagreements.append({
                        "paper_id":p,"annotator1_record_id":id1,"annotator2_record_id":id2,
                        "field":f,"annotator1_value":v1,"annotator2_value":v2,
                        "adjudicated_value":"","resolution_note":""
                    })
        for i in ua:
            alignment_rows.append({"paper_id":p,"annotator1_record_id":r1[i].get("record_local_id",str(i+1)),"annotator2_record_id":"","similarity":"","review_alignment":"UNMATCHED_A1"})
        for j in ub:
            alignment_rows.append({"paper_id":p,"annotator1_record_id":"","annotator2_record_id":r2[j].get("record_local_id",str(j+1)),"similarity":"","review_alignment":"UNMATCHED_A2"})

    agreement_rows = []
    for g, d in group_stats.items():
        agreement_rows.append({"level":"group","field_or_group":g,"percent_agreement":d["agree"]/d["total"] if d["total"] else "","cohen_kappa":"","n":d["total"]})
    for f, (x,y) in field_pairs.items():
        try:
            k = cohen_kappa_score(x,y) if len(set(x+y)) > 1 else float("nan")
        except Exception:
            k = float("nan")
        agree = sum(a==b for a,b in zip(x,y))/len(x) if x else float("nan")
        agreement_rows.append({"level":"field","field_or_group":f,"percent_agreement":agree,"cohen_kappa":"" if math.isnan(k) else k,"n":len(x)})

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir/"record_alignment.csv", alignment_rows, ["paper_id","annotator1_record_id","annotator2_record_id","similarity","review_alignment"])
    write_csv(outdir/"interannotator_agreement.csv", agreement_rows, ["level","field_or_group","percent_agreement","cohen_kappa","n"])
    write_csv(outdir/"adjudication_template.csv", disagreements, ["paper_id","annotator1_record_id","annotator2_record_id","field","annotator1_value","annotator2_value","adjudicated_value","resolution_note"])
    write_json(outdir/"agreement_summary.json", {"matched_record_pairs":len([x for x in alignment_rows if x["similarity"]]),"unmatched_annotator1_records":unmatched_a1,"unmatched_annotator2_records":unmatched_a2,"field_disagreements":len(disagreements)})
    print(f"Agreement complete. {len(disagreements)} field disagreements. Review record_alignment.csv before adjudication.")


def cmd_finalize_gold(args: argparse.Namespace) -> None:
    cfg = read_json(args.config)
    a1, a2 = load_annotation(args.annotator1), load_annotation(args.annotator2)
    g1, g2 = group_by_paper(a1), group_by_paper(a2)
    adjud = read_csv(args.adjudication)
    amap = {}
    for r in adjud:
        key=(r["paper_id"],r["annotator1_record_id"],r["annotator2_record_id"],r["field"])
        if not norm_text(r.get("adjudicated_value","")) and norm_text(r.get("annotator1_value","")) != norm_text(r.get("annotator2_value","")):
            raise SystemExit(f"Unresolved adjudication: {key}")
        amap[key]=r.get("adjudicated_value","")

    threshold = cfg.get("record_match_threshold", 0.55)
    gold=[]
    for p in sorted(set(g1)|set(g2)):
        r1,r2=g1.get(p,[]),g2.get(p,[])
        pairs,ua,ub=align_records(r1,r2,threshold)
        if ua or ub:
            raise SystemExit(f"Unmatched annotation records remain in paper {p}. Resolve record segmentation/alignment manually before finalizing gold.")
        for n,(i,j,s) in enumerate(pairs,1):
            out={"paper_id":p,"record_id":f"{p}_G{n:03d}"}
            id1=r1[i].get("record_local_id",str(i+1)); id2=r2[j].get("record_local_id",str(j+1))
            for f in FIELDS:
                v1,v2=r1[i].get(f,""),r2[j].get(f,"")
                if field_equal(f,v1,v2,cfg): out[f]=v1
                else:
                    out[f]=amap.get((p,id1,id2,f),"")
            gold.append(out)
    write_csv(args.output, gold, ["paper_id","record_id"]+FIELDS)
    print(f"Gold standard written: {len(gold)} records.")


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def chunk_text(text: str, max_chars: int, overlap_paragraphs: int=1) -> List[str]:
    paras=[p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks=[]; cur=[]; n=0
    for p in paras:
        if cur and n+len(p)+2>max_chars:
            chunks.append("\n\n".join(cur))
            cur=cur[-overlap_paragraphs:] if overlap_paragraphs else []
            n=sum(len(x)+2 for x in cur)
        cur.append(p); n+=len(p)+2
    if cur: chunks.append("\n\n".join(cur))
    return chunks


def find_paper_file(corpus: Path, paper_id: str) -> Path:
    """Find the merged Main+SI markdown for one paper.

    Supported layout (preferred):
        corpus/1/*_merged.md
        corpus/2/*_merged.md
        ...
    Legacy flat-corpus lookup is retained as a fallback.
    """
    paper_dir = corpus / str(paper_id)
    if paper_dir.is_dir():
        merged = sorted(paper_dir.glob("*_merged.md"))
        if len(merged) == 1:
            return merged[0]
        if len(merged) > 1:
            raise RuntimeError(
                f"Multiple *_merged.md files found for paper {paper_id}: "
                + "; ".join(x.name for x in merged)
            )
        raise FileNotFoundError(
            f"No *_merged.md found for paper {paper_id} in {paper_dir}"
        )

    # Backward-compatible flat corpus.
    for ext in (".md", ".txt"):
        p = corpus / (str(paper_id) + ext)
        if p.exists():
            return p
    matches = list(corpus.glob(str(paper_id) + ".*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No source found for {paper_id} in {corpus}")


def discover_papers(corpus: Path) -> List[str]:
    """Auto-discover paper IDs from subfolders containing exactly one *_merged.md."""
    found = []
    for d in corpus.iterdir():
        if not d.is_dir():
            continue
        if len(list(d.glob("*_merged.md"))) == 1:
            found.append(d.name)

    def key(x: str):
        return (0, int(x)) if x.isdigit() else (1, x.lower())

    return sorted(found, key=key)


def parse_json_response(text: str) -> Dict[str,Any]:
    t=text.strip()
    t=re.sub(r"^```(?:json)?\s*","",t,flags=re.I); t=re.sub(r"\s*```$","",t)
    return json.loads(t)


def cmd_extract(args: argparse.Namespace) -> None:
    from openai import OpenAI
    from rank_bm25 import BM25Okapi

    cfg=read_json(args.config)
    schema=Path(args.schema).read_text(encoding="utf-8")
    template=Path(args.prompt).read_text(encoding="utf-8")
    examples=[]
    with open(args.examples,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): examples.append(json.loads(line))
    tokenized=[re.findall(r"\w+",str(e.get("text","" )).lower()) for e in examples]
    bm25=BM25Okapi(tokenized)
    corpus=Path(args.corpus_dir)
    if args.selected_papers:
        papers=[x.strip() for x in Path(args.selected_papers).read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        papers=discover_papers(corpus)
        if not papers:
            raise SystemExit(f"No subfolders containing exactly one *_merged.md were found in {corpus}")
        print(f"Auto-discovered {len(papers)} merged paper packages: {', '.join(papers)}")
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    rawdir=outdir/"raw_responses"; rawdir.mkdir(exist_ok=True)
    logdir=outdir/"retrieval_logs"; logdir.mkdir(exist_ok=True)

    api_key=os.environ.get(cfg.get("api_key_env","LLM_API_KEY"))
    if not api_key: raise SystemExit(f"Set environment variable {cfg.get('api_key_env','LLM_API_KEY')} with your API key.")
    client=OpenAI(api_key=api_key, base_url=cfg.get("base_url") or None)
    k=int(cfg.get("bm25_k",5))
    all_records=[]
    for pidx,paper_id in enumerate(papers,1):
        text=find_paper_file(corpus,paper_id).read_text(encoding="utf-8",errors="ignore")
        chunks=chunk_text(text,int(cfg.get("chunk_max_chars",7000)),int(cfg.get("chunk_overlap_paragraphs",1)))
        paper_recs=[]
        retrieval_log=[]
        for ci,ch in enumerate(chunks,1):
            q=re.findall(r"\w+",ch.lower())
            scores=bm25.get_scores(q)
            idxs=np.argsort(scores)[::-1][:k]
            rex=[examples[int(i)] for i in idxs]
            retrieval_log.append({"chunk":ci,"example_ids":[e.get("example_id") for e in rex],"scores":[float(scores[int(i)]) for i in idxs]})
            ex_text="\n\n".join(json.dumps({"text":e.get("text"),"output":e.get("output")},ensure_ascii=False) for e in rex)
            extraction_guard = r"""

MANDATORY PAPER-PACKAGE RULES:
- The source may combine the MAIN ARTICLE and SUPPORTING INFORMATION of ONE paper.
- Extract ONLY catalysts, conditions, products, and performance experimentally reported by the authors of THIS paper.
- DO NOT extract records from Introduction/background discussion, literature reviews, References, or literature/state-of-the-art/comparison tables.
- In Supporting Information comparison tables, extract a row only when it is explicitly the current paper / 'this work'.
- Use Main and SI as complementary evidence; do not create duplicate records merely because the same experiment is stated in both.
- Do not infer missing values. Leave unreported fields null/blank according to the supplied schema.
- IMPORTANT: Create a record ONLY when performance_value is explicitly reported. Do NOT output records with a missing/null/blank performance_value.
- source_anchor should preserve provenance when possible (MAIN/SI + section/table/figure/local phrase).
"""
            user_prompt=(
                template.replace("{SCHEMA}",schema)
                        .replace("{EXAMPLES}",ex_text)
                        .replace("{TEXT}",ch)
                + extraction_guard
            )
            kwargs={
                "model":cfg["model"],
                "messages":[{"role":"user","content":user_prompt}],
                "temperature":cfg.get("temperature",0.0),
                "top_p":cfg.get("top_p",1.0),
                "max_tokens":cfg.get("max_tokens",4000),
            }
            if cfg.get("seed") is not None: kwargs["seed"]=cfg["seed"]
            resp=client.chat.completions.create(**kwargs)
            content=resp.choices[0].message.content or ""
            (rawdir/f"{paper_id}_chunk{ci:03d}.txt").write_text(content,encoding="utf-8")
            try:
                obj=parse_json_response(content)
                for r in obj.get("records",[]):
                    rr={"paper_id":paper_id,"source_chunk":ci,**{f:r.get(f) for f in FIELDS}}
                    paper_recs.append(rr)
            except Exception as e:
                with open(outdir/"parse_errors.log","a",encoding="utf-8") as ef:
                    ef.write(f"{paper_id}\tchunk={ci}\t{type(e).__name__}: {e}\n")
        write_json(logdir/f"{paper_id}.json",retrieval_log)
        # Remove records without an explicit performance value BEFORE deduplication.
        # Keep numeric zero (0 / 0.0) as a valid reported value.
        def has_performance_value(record):
            value = record.get("performance_value")
            if value is None:
                return False
            if isinstance(value, str):
                value = value.strip()
                if not value or value.lower() in {"null", "none", "n/a", "na", "nan", "not reported", "not available"}:
                    return False
            return True

        before_filter = len(paper_recs)
        paper_recs = [r for r in paper_recs if has_performance_value(r)]
        removed_empty_performance = before_filter - len(paper_recs)

        # Conservative exact-signature deduplication only. Do not merge conflicting records automatically.
        seen=set()
        for r in paper_recs:
            sig=tuple(norm_text(r.get(f)) for f in FIELDS if f!="source_anchor")
            if sig not in seen:
                seen.add(sig); all_records.append(r)
        print(
            f"[{pidx}/{len(papers)}] {paper_id}: {len(chunks)} chunks, "
            f"removed {removed_empty_performance} records with empty performance_value, "
            f"{len(seen)} deduplicated records"
        )

    write_csv(outdir/"llm_raw_extracted_records.csv",all_records,["paper_id","source_chunk"]+FIELDS)
    print(f"Raw extraction complete: {len(all_records)} records. Do NOT manually correct this file before evaluation.")


def count_field_items(record: Dict[str,Any], fields: Sequence[str]) -> int:
    return sum(1 for f in fields if norm_text(record.get(f,"")))


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg=read_json(args.config)
    gold=read_csv(args.gold); pred=read_csv(args.pred)
    gg,pg=group_by_paper(gold),group_by_paper(pred)
    groups=cfg.get("field_groups",DEFAULT_GROUPS)
    threshold=cfg.get("record_match_threshold",0.55)
    counts={g:{"tp":0,"fp":0,"fn":0} for g in groups}
    error_counts=Counter()
    error_rows=[]
    total_generated=0; generated_needing_correction=0
    total_gold=0; missing_additions=0
    alignment_rows=[]

    field_to_group={f:g for g,fs in groups.items() for f in fs}
    eval_fields=[f for f in field_to_group if f!="source_anchor"]

    for p in sorted(set(gg)|set(pg)):
        gr,pr=gg.get(p,[]),pg.get(p,[])
        pairs,ug,up=align_records(gr,pr,threshold)
        for i,j,s in pairs:
            alignment_rows.append({"paper_id":p,"gold_record_id":gr[i].get("record_id",str(i+1)),"pred_record_index":j+1,"similarity":s})
            for f in eval_fields:
                g=field_to_group[f]; gv,pv=gr[i].get(f,""),pr[j].get(f,"")
                gh,ph=bool(norm_text(gv)),bool(norm_text(pv))
                total_gold += int(gh); total_generated += int(ph)
                if not gh and not ph: continue
                if gh and ph and field_equal(f,gv,pv,cfg):
                    counts[g]["tp"]+=1
                elif gh and ph:
                    counts[g]["fp"]+=1; counts[g]["fn"]+=1; generated_needing_correction+=1
                    mode="numeric_mismatch" if f in NUMERIC_FIELDS else ("unit_mismatch" if f=="performance_unit" else "wrong_text_or_category")
                    error_counts[mode]+=1
                    error_rows.append({"paper_id":p,"gold_record_id":gr[i].get("record_id",str(i+1)),"field":f,"gold_value":gv,"pred_value":pv,"error_mode":mode})
                elif gh and not ph:
                    counts[g]["fn"]+=1; missing_additions+=1; error_counts["missing_field"]+=1
                    error_rows.append({"paper_id":p,"gold_record_id":gr[i].get("record_id",str(i+1)),"field":f,"gold_value":gv,"pred_value":"","error_mode":"missing_field"})
                elif ph and not gh:
                    counts[g]["fp"]+=1; generated_needing_correction+=1; error_counts["hallucinated_field"]+=1
                    error_rows.append({"paper_id":p,"gold_record_id":gr[i].get("record_id",str(i+1)),"field":f,"gold_value":"","pred_value":pv,"error_mode":"hallucinated_field"})
        for i in ug:
            error_counts["missed_record"]+=1
            for f in eval_fields:
                if norm_text(gr[i].get(f,"")):
                    counts[field_to_group[f]]["fn"]+=1; total_gold+=1; missing_additions+=1
            error_rows.append({"paper_id":p,"gold_record_id":gr[i].get("record_id",str(i+1)),"field":"__RECORD__","gold_value":"present","pred_value":"missing","error_mode":"missed_record"})
        for j in up:
            error_counts["spurious_record"]+=1
            for f in eval_fields:
                if norm_text(pr[j].get(f,"")):
                    counts[field_to_group[f]]["fp"]+=1; total_generated+=1; generated_needing_correction+=1
            error_rows.append({"paper_id":p,"gold_record_id":"","field":"__RECORD__","gold_value":"missing","pred_value":"present","error_mode":"spurious_record"})

    metric_rows=[]
    for g,c in counts.items():
        tp,fp,fn=c["tp"],c["fp"],c["fn"]
        precision=tp/(tp+fp) if tp+fp else float("nan")
        recall=tp/(tp+fn) if tp+fn else float("nan")
        f1=2*precision*recall/(precision+recall) if precision+recall and not (math.isnan(precision) or math.isnan(recall)) else float("nan")
        metric_rows.append({"field_group":g,"TP":tp,"FP":fp,"FN":fn,"precision":precision,"recall":recall,"F1":f1})
    # micro overall
    tp=sum(c["tp"] for c in counts.values()); fp=sum(c["fp"] for c in counts.values()); fn=sum(c["fn"] for c in counts.values())
    prc=tp/(tp+fp) if tp+fp else float("nan"); rec=tp/(tp+fn) if tp+fn else float("nan"); f1=2*prc*rec/(prc+rec) if prc+rec else float("nan")
    metric_rows.append({"field_group":"OVERALL_MICRO","TP":tp,"FP":fp,"FN":fn,"precision":prc,"recall":rec,"F1":f1})

    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    write_csv(outdir/"Table_SXX_field_level_extraction_metrics.csv",metric_rows,["field_group","TP","FP","FN","precision","recall","F1"])
    write_csv(outdir/"Table_SXX_error_modes.csv",[{"error_mode":k,"count":v} for k,v in error_counts.most_common()],["error_mode","count"])
    write_csv(outdir/"error_details.csv",error_rows,["paper_id","gold_record_id","field","gold_value","pred_value","error_mode"])
    write_csv(outdir/"gold_vs_llm_record_alignment.csv",alignment_rows,["paper_id","gold_record_id","pred_record_index","similarity"])
    summary={
        "gold_records":len(gold),"predicted_records":len(pred),
        "total_gold_nonempty_fields":total_gold,"total_generated_nonempty_fields":total_generated,
        "generated_fields_requiring_correction_or_deletion":generated_needing_correction,
        "generated_field_correction_fraction":generated_needing_correction/total_generated if total_generated else None,
        "gold_fields_missing_from_llm":missing_additions,
        "missing_field_addition_fraction":missing_additions/total_gold if total_gold else None,
        "record_match_threshold":threshold,
        "numeric_relative_tolerance":cfg.get("numeric_relative_tolerance",0.01),
        "error_modes":dict(error_counts)
    }
    write_json(outdir/"validation_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def cmd_manifest(args: argparse.Namespace) -> None:
    cfg=read_json(args.config)
    examples_n=0
    with open(args.examples,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): examples_n+=1
    manifest={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "provider":cfg.get("provider"),"base_url":cfg.get("base_url"),
        "model":cfg.get("model"),"access_date":cfg.get("access_date"),
        "temperature":cfg.get("temperature"),"top_p":cfg.get("top_p"),
        "max_tokens":cfg.get("max_tokens"),"seed":cfg.get("seed"),
        "bm25_k":cfg.get("bm25_k"),"annotated_example_database_size":examples_n,
        "prompt_sha256":sha256_file(args.prompt),"schema_sha256":sha256_file(args.schema),
        "examples_sha256":sha256_file(args.examples),"config_sha256":sha256_file(args.config),
        "notes":"Archive this manifest together with prompt, schema, few-shot examples, retrieval logs, raw model outputs, and evaluation tables."
    }
    write_json(args.output,manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="LLM extraction reproducibility + gold-set validation pipeline")
    sp=p.add_subparsers(dest="cmd",required=True)

    q=sp.add_parser("sample")
    q.add_argument("--master",required=True,help="CSV with one or more records per paper")
    q.add_argument("--paper-col",default="paper_id")
    q.add_argument("--phase-col",default="phase")
    q.add_argument("--product-col",default="product")
    q.add_argument("--exclude-papers",help="Text file of papers used in few-shot DB/prompt development")
    q.add_argument("--n-papers",type=int,default=40)
    q.add_argument("--seed",type=int,default=20260911)
    q.add_argument("--outdir",default="validation_work/sample")
    q.set_defaults(func=cmd_sample)

    q=sp.add_parser("make-templates")
    q.add_argument("--selected-papers",required=True)
    q.add_argument("--outdir",default="validation_work/annotation")
    q.set_defaults(func=cmd_make_templates)

    q=sp.add_parser("extract")
    q.add_argument("--config",required=True); q.add_argument("--schema",required=True); q.add_argument("--prompt",required=True)
    q.add_argument("--examples",required=True,help="JSONL: example_id,paper_id,text,output")
    q.add_argument("--selected-papers",help="Optional text file of paper IDs. If omitted, auto-discover subfolders containing exactly one *_merged.md."); q.add_argument("--corpus-dir",required=True); q.add_argument("--outdir",default="validation_work/llm_run")
    q.set_defaults(func=cmd_extract)

    q=sp.add_parser("agreement")
    q.add_argument("--config",required=True); q.add_argument("--annotator1",required=True); q.add_argument("--annotator2",required=True); q.add_argument("--outdir",default="validation_work/agreement")
    q.set_defaults(func=cmd_agreement)

    q=sp.add_parser("finalize-gold")
    q.add_argument("--config",required=True); q.add_argument("--annotator1",required=True); q.add_argument("--annotator2",required=True); q.add_argument("--adjudication",required=True); q.add_argument("--output",default="validation_work/gold_standard.csv")
    q.set_defaults(func=cmd_finalize_gold)

    q=sp.add_parser("evaluate")
    q.add_argument("--config",required=True); q.add_argument("--gold",required=True); q.add_argument("--pred",required=True); q.add_argument("--outdir",default="validation_work/evaluation")
    q.set_defaults(func=cmd_evaluate)

    q=sp.add_parser("manifest")
    q.add_argument("--config",required=True); q.add_argument("--prompt",required=True); q.add_argument("--schema",required=True); q.add_argument("--examples",required=True); q.add_argument("--output",default="validation_work/run_manifest.json")
    q.set_defaults(func=cmd_manifest)
    return p


def main() -> None:
    p=build_parser(); args=p.parse_args(); args.func(args)

if __name__=="__main__": main()
