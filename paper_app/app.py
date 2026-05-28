from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse


ROOT = Path(__file__).resolve().parent
PAPERS_JSON = ROOT / "papers.json"
BUFFER_MD = ROOT / "buffer.md"

ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"
GITHUB_API = "https://api.github.com/repos"
OPENALEX_API = "https://api.openalex.org/works"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
CURRENT_YEAR = datetime.now().year

DIRECTIONS: dict[str, dict[str, Any]] = {
    "LLM Disaggregation": {
        "seed_ids": [
            "2311.18677",
            "2401.09670",
            "2406.17565",
            "2407.00079",
            "2408.08147",
            "2504.03775",
            "2504.02263",
            "2504.19867",
            "2505.11916",
            "2507.06608",
            "2508.15881",
            "2508.19559",
            "2510.08544",
            "2510.13668",
            "2511.21862",
            "2601.08833",
            "2602.12029",
        ],
        "queries": [
            'all:"disaggregated LLM serving"',
            'all:"prefill decode" AND all:"LLM serving"',
            'all:"prefill-decode" AND all:"LLM inference"',
            'all:"KV cache" AND all:"disaggregated" AND all:"serving"',
            'all:"stateful inference" AND all:"LLM"',
            'all:"long-context" AND all:"LLM serving" AND all:"KV cache"',
            'all:"adaptive scheduling" AND all:"disaggregated" AND all:"LLM"',
            'all:"agent" AND all:"LLM serving"',
        ],
        "terms": [
            "disaggregated",
            "prefill",
            "decode",
            "kv cache",
            "serving",
            "scheduler",
            "long-context",
            "long context",
            "stateful",
        ],
    },
    "Speculative Decoding in Test Time": {
        "seed_ids": [
            "2211.17192",
            "2302.01318",
            "2402.01528",
            "2404.11912",
            "2408.11049",
            "2408.11850",
            "2506.04708",
            "2605.09329",
            "2502.01662",
            "2504.05598",
            "2512.02337",
            "2604.14612",
        ],
        "queries": [
            'all:"speculative decoding" AND all:"LLM"',
            'all:"self-speculative decoding"',
            'all:"speculative sampling" AND all:"LLM"',
            'all:"adaptive draft length"',
            'all:"draft model" AND all:"verifier"',
            'all:"partial verification" AND all:"speculative decoding"',
            'all:"test-time scaling" AND all:"speculative"',
            'all:"reasoning" AND all:"speculative decoding"',
            'all:"long-context" AND all:"speculative decoding"',
        ],
        "terms": [
            "speculative",
            "draft",
            "acceptance",
            "verifier",
            "test-time",
            "test time",
            "adaptive",
            "reasoning",
            "long-context",
        ],
    },
}

KNOWN_METADATA: dict[str, dict[str, str]] = {
    "2407.00079": {
        "conference": "FAST '25, Best Paper",
        "acceptance": "Accepted",
        "org_company": "Moonshot AI, Tsinghua University",
        "code_url": "https://github.com/kvcache-ai/Mooncake",
    },
    "2504.02263": {
        "conference": "SIGCOMM '25",
        "acceptance": "Accepted",
        "org_company": "ByteDance, Peking University",
    },
    "2605.09329": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "org_company": "University of Texas at Austin",
    },
    "2401.09670": {
        "conference": "OSDI '24",
        "acceptance": "Accepted",
        "org_company": "Peking University",
        "code_url": "https://github.com/LLMServe/DistServe",
    },
    "2311.18677": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "org_company": "Microsoft",
    },
    "2406.17565": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "org_company": "Huawei Cloud",
    },
    "2211.17192": {
        "conference": "ICML '23",
        "acceptance": "Accepted",
        "org_company": "Google Research",
    },
    "2302.01318": {
        "conference": "ICML '23",
        "acceptance": "Accepted",
        "org_company": "DeepMind",
    },
    "2402.01528": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "org_company": "University of Wisconsin-Madison",
        "code_url": "https://github.com/uw-mad-dash/decoding-speculative-decoding",
    },
    "2404.11912": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "org_company": "CMU, Meta AI",
        "code_url": "https://github.com/Infini-AI-Lab/TriForce",
    },
    "2408.11049": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "org_company": "CMU",
        "code_url": "https://github.com/Infini-AI-Lab/MagicDec",
    },
    "2408.11850": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "org_company": "USTC",
        "code_url": "https://github.com/smart-lty/ParallelSpeculativeDecoding",
    },
    "2506.04708": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
    },
    "2504.05598": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "org_company": "University of Southern California",
        "code_url": "https://github.com/hoenza/DEL",
    },
    "2502.01662": {
        "conference": "arXiv preprint",
        "acceptance": "Preprint",
        "code_url": "https://github.com/Kamichanw/Speculative-Ensemble",
    },
}

app = FastAPI(title="Awesome Agentic Inference Paper App")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def arxiv_id_from_url(url: str) -> str:
    raw = url.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", raw)


def month_from_timestamp(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y.%m")


def openreview_search_url(title: str) -> str:
    return "https://openreview.net/search?term=" + urllib.parse.quote(title)


def fetch_arxiv(direction: str, query: str, limit: int) -> list[dict[str, Any]]:
    return fetch_arxiv_params(
        direction,
        {
            "search_query": query,
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
        is_seed=False,
    )


def fetch_arxiv_ids(direction: str, arxiv_ids: list[str]) -> list[dict[str, Any]]:
    if not arxiv_ids:
        return []
    return fetch_arxiv_params(
        direction,
        {
            "id_list": ",".join(arxiv_ids),
            "max_results": len(arxiv_ids),
        },
        is_seed=True,
    )


def fetch_arxiv_params(direction: str, params: dict[str, Any], is_seed: bool) -> list[dict[str, Any]]:
    response = requests.get(
        ARXIV_API,
        params=params,
        headers={"User-Agent": "awesome-agentic-inference-paper-app/0.1"},
        timeout=30,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall("atom:entry", ATOM):
        title = normalize_text(entry.findtext("atom:title", "", ATOM))
        abstract = normalize_text(entry.findtext("atom:summary", "", ATOM))
        published = entry.findtext("atom:published", "", ATOM)
        arxiv_url = entry.findtext("atom:id", "", ATOM)
        arxiv_id = arxiv_id_from_url(arxiv_url)
        authors = [
            normalize_text(author.findtext("atom:name", "", ATOM))
            for author in entry.findall("atom:author", ATOM)
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        known = KNOWN_METADATA.get(arxiv_id, {})
        paper = {
            "title": title,
            "authors": authors,
            "date": month_from_timestamp(published),
            "direction": direction,
            "arxiv_id": arxiv_id,
            "arxiv_url": arxiv_url,
            "pdf_url": pdf_url,
            "openreview_url": known.get("openreview_url", openreview_search_url(title)),
            "abstract": abstract,
            "conference": known.get("conference", "arXiv preprint"),
            "acceptance": known.get("acceptance", "Preprint"),
            "org_company": known.get("org_company", "Unknown"),
            "code_url": known.get("code_url", ""),
            "status": "unread",
            "is_seed": is_seed,
            "citation_count": 0,
            "influential_citation_count": 0,
            "github_stars": 0,
            "metrics_status": "local",
            "score_breakdown": {},
            "score": 0,
        }
        paper["score"] = score_paper(paper)
        papers.append(paper)
    return papers


def relevance_matches(paper: dict[str, Any]) -> list[str]:
    text = f"{paper['title']} {paper['abstract']}".lower()
    terms = DIRECTIONS[paper["direction"]]["terms"]
    return [term for term in terms if term in text]


def conference_score(conference: str) -> int:
    value = conference.lower()
    if not value or "preprint" in value or value == "unknown":
        return 3
    top_venues = [
        "osdi",
        "sosp",
        "nsdi",
        "sigcomm",
        "fast",
        "mlsys",
        "icml",
        "neurips",
        "iclr",
        "asplos",
        "isca",
    ]
    score = 15 if any(venue in value for venue in top_venues) else 10
    if "best paper" in value:
        score = min(15, score + 2)
    return score


def recency_score(date: str) -> int:
    try:
        year = int(date.split(".")[0])
    except (ValueError, IndexError):
        return 0
    age = max(0, CURRENT_YEAR - year)
    if age == 0:
        return 10
    if age == 1:
        return 8
    if age == 2:
        return 6
    if age == 3:
        return 4
    return 2


def citation_score(citations: int) -> int:
    if citations <= 0:
        return 0
    if citations >= 1000:
        return 25
    if citations >= 300:
        return 22
    if citations >= 100:
        return 18
    if citations >= 50:
        return 14
    if citations >= 20:
        return 10
    if citations >= 5:
        return 6
    return 3


def github_score(stars: int, has_code: bool) -> int:
    if stars >= 10_000:
        return 15
    if stars >= 3_000:
        return 13
    if stars >= 1_000:
        return 11
    if stars >= 300:
        return 8
    if stars >= 100:
        return 6
    if stars >= 20:
        return 4
    if has_code:
        return 2
    return 0


def score_paper(paper: dict[str, Any]) -> int:
    matches = relevance_matches(paper)
    relevance = min(30, len(matches) * 5)
    citations = int(paper.get("citation_count", 0) or 0)
    stars = int(paper.get("github_stars", 0) or 0)
    breakdown = {
        "relevance": relevance,
        "relevance_matches": matches,
        "citations": citation_score(citations),
        "citation_count": citations,
        "github": github_score(stars, bool(paper.get("code_url"))),
        "github_stars": stars,
        "conference": conference_score(str(paper.get("conference", ""))),
        "recency": recency_score(str(paper.get("date", ""))),
    }
    paper["score_breakdown"] = breakdown
    return (
        breakdown["relevance"]
        + breakdown["citations"]
        + breakdown["github"]
        + breakdown["conference"]
        + breakdown["recency"]
    )


def enrich_metrics(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(enrich_one_paper, paper) for paper in papers]
        for future in as_completed(futures):
            future.result()
    return papers


def enrich_one_paper(paper: dict[str, Any]) -> None:
    if os.environ.get("S2_API_KEY"):
        enrich_semantic_scholar(paper)
    if int(paper.get("citation_count", 0) or 0) == 0:
        enrich_openalex(paper)
    enrich_github(paper)
    paper["score"] = score_paper(paper)


def enrich_semantic_scholar(paper: dict[str, Any]) -> None:
    arxiv_id = str(paper.get("arxiv_id", "")).split("v")[0]
    if not arxiv_id:
        return
    try:
        headers = {"User-Agent": "awesome-agentic-inference-paper-app/0.1"}
        if os.environ.get("S2_API_KEY"):
            headers["x-api-key"] = os.environ["S2_API_KEY"]
        response = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/ARXIV:{arxiv_id}",
            params={"fields": "citationCount,influentialCitationCount,venue,year,url"},
            headers=headers,
            timeout=8,
        )
        if response.status_code == 404:
            paper["metrics_status"] = "semantic-scholar-missing"
            return
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        paper["metrics_status"] = "semantic-scholar-error"
        return

    paper["citation_count"] = int(data.get("citationCount") or 0)
    paper["influential_citation_count"] = int(data.get("influentialCitationCount") or 0)
    venue = str(data.get("venue") or "").strip()
    if venue.lower() in {"arxiv", "arxiv.org"}:
        venue = ""
    if venue and paper.get("conference") == "arXiv preprint":
        paper["conference"] = venue
        paper["acceptance"] = "Unknown"
    paper["semantic_scholar_url"] = data.get("url", "")
    paper["metrics_status"] = "semantic-scholar"


def enrich_openalex(paper: dict[str, Any]) -> None:
    try:
        response = requests.get(
            OPENALEX_API,
            params={
                "search": paper.get("title", ""),
                "per-page": 1,
                "select": "display_name,cited_by_count,publication_year",
            },
            headers={"User-Agent": "awesome-agentic-inference-paper-app/0.1"},
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except requests.RequestException:
        if paper.get("metrics_status") != "semantic-scholar":
            paper["metrics_status"] = "metrics-error"
        return

    if not results:
        return
    result = results[0]
    result_title = normalize_text(str(result.get("display_name", ""))).lower()
    paper_title = normalize_text(str(paper.get("title", ""))).lower()
    if result_title and (result_title == paper_title or result_title[:48] == paper_title[:48]):
        paper["citation_count"] = int(result.get("cited_by_count") or 0)
        paper["metrics_status"] = "openalex"


def enrich_github(paper: dict[str, Any]) -> None:
    repo = github_repo_from_url(str(paper.get("code_url", "")))
    if not repo:
        return
    try:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "awesome-agentic-inference-paper-app/0.1",
        }
        if os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        response = requests.get(
            f"{GITHUB_API}/{repo}",
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return
    paper["github_stars"] = int(data.get("stargazers_count") or 0)


def github_repo_from_url(url: str) -> str:
    match = re.search(r"github\.com/([^/\s]+)/([^/#?\s]+)", url)
    if not match:
        return ""
    owner, repo = match.groups()
    return f"{owner}/{repo.removesuffix('.git')}"


def dedupe(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for paper in papers:
        key = paper.get("arxiv_id") or paper["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return sorted(unique, key=lambda p: (-int(p["score"]), p["direction"], p["date"], p["title"]))


def collect_papers(limit_per_direction: int = 12) -> list[dict[str, Any]]:
    existing = {paper.get("arxiv_id"): paper for paper in load_papers()}
    papers: list[dict[str, Any]] = []
    for direction, config in DIRECTIONS.items():
        papers.extend(fetch_arxiv_ids(direction, config.get("seed_ids", [])))
        for query in config.get("queries", []):
            papers.extend(fetch_arxiv(direction, query, limit_per_direction))
    papers = dedupe(papers)
    papers = filter_relevant(papers)
    papers = enrich_metrics(papers)
    merge_existing_metrics(papers, existing)
    papers = dedupe(papers)
    save_papers(papers)
    return papers


def filter_relevant(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relevant = []
    for paper in papers:
        matches = relevance_matches(paper)
        if paper.get("is_seed") or len(matches) >= 2:
            relevant.append(paper)
    return relevant


def merge_existing_metrics(papers: list[dict[str, Any]], existing: dict[str, dict[str, Any]]) -> None:
    for paper in papers:
        old = existing.get(paper.get("arxiv_id"))
        if not old:
            continue
        for key in ["citation_count", "influential_citation_count", "github_stars"]:
            paper[key] = max(int(paper.get(key, 0) or 0), int(old.get(key, 0) or 0))
        for key in ["semantic_scholar_url", "code_url"]:
            if not paper.get(key) and old.get(key):
                paper[key] = old[key]
        paper["score"] = score_paper(paper)


def load_papers() -> list[dict[str, Any]]:
    if not PAPERS_JSON.exists():
        return []
    return json.loads(PAPERS_JSON.read_text(encoding="utf-8"))


def save_papers(papers: list[dict[str, Any]]) -> None:
    mark_seed_papers(papers)
    PAPERS_JSON.write_text(json.dumps(papers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_buffer(papers)


def mark_seed_papers(papers: list[dict[str, Any]]) -> None:
    seed_ids = {
        arxiv_id
        for config in DIRECTIONS.values()
        for arxiv_id in config.get("seed_ids", [])
    }
    for paper in papers:
        if paper.get("arxiv_id") in seed_ids:
            paper["is_seed"] = True


def write_buffer(papers: list[dict[str, Any]]) -> None:
    lines = [
        "# Paper Buffer",
        "",
        "Candidate papers collected by the paper reading app. Promote to `README.md` only after reading.",
        "",
    ]
    for direction in DIRECTIONS:
        lines.extend(
            [
                f"## {direction}",
                "",
                "| Date | Seed | Score | Citations | Stars | Title | arXiv | OpenReview | Acceptance | Conference | Status | Notes |",
                "|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|",
            ]
        )
        direction_papers = [paper for paper in papers if paper["direction"] == direction]
        for paper in sorted(direction_papers, key=lambda p: (-int(p["score"]), p["date"], p["title"])):
            lines.append(
                f"| {paper['date']} | {seed_label(paper)} | {paper['score']} | {paper.get('citation_count', 0)} | "
                f"{paper.get('github_stars', 0)} | {paper['title']} | "
                f"[[arXiv]]({paper['arxiv_url']}) | [[OpenReview]]({paper['openreview_url']}) | "
                f"{paper['acceptance']} | {paper['conference']} | {paper['status']} |  |"
            )
        lines.append("")
    BUFFER_MD.write_text("\n".join(lines), encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return render_html(load_papers())


@app.get("/api/papers")
def api_papers() -> JSONResponse:
    return JSONResponse(load_papers())


@app.post("/api/refresh")
def api_refresh(limit: int = Query(default=12, ge=1, le=50)) -> JSONResponse:
    papers = collect_papers(limit)
    return JSONResponse({"count": len(papers), "papers": papers})


@app.get("/buffer.md", response_class=PlainTextResponse)
def buffer_markdown() -> str:
    if not BUFFER_MD.exists():
        return ""
    return BUFFER_MD.read_text(encoding="utf-8")


def render_html(papers: list[dict[str, Any]]) -> str:
    cards = "\n".join(render_card(index + 1, paper) for index, paper in enumerate(papers))
    direction_buttons = "\n".join(
        f'<button class="chip" data-direction="{html.escape(direction)}">{html.escape(direction)}</button>'
        for direction in DIRECTIONS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Reading Buffer</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #20242a;
      --muted: #687076;
      --line: #d8dad5;
      --accent: #0f766e;
      --accent-soft: #e1f4ef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 247, 244, .97);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto auto;
      gap: 10px;
      align-items: center;
    }}
    input, select, button {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      padding: 8px 10px;
    }}
    button {{ cursor: pointer; }}
    main {{
      display: grid;
      grid-template-columns: 250px minmax(0, 1fr);
      gap: 18px;
      padding: 18px 24px 32px;
    }}
    aside {{
      position: sticky;
      top: 93px;
      align-self: start;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
    }}
    .chip {{
      width: 100%;
      margin-top: 8px;
      text-align: left;
    }}
    .chip.active {{
      background: var(--accent-soft);
      border-color: var(--accent);
    }}
    .paper {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 12px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .badge {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #fafafa;
    }}
    .score {{
      color: var(--accent);
      font-weight: 700;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 10px 0;
    }}
    a {{
      color: #0b5cad;
      text-decoration: none;
      font-weight: 650;
    }}
    a:hover {{ text-decoration: underline; }}
    details {{
      border-top: 1px solid var(--line);
      margin-top: 12px;
      padding-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 14px;
    }}
    .empty {{
      padding: 40px;
      color: var(--muted);
      text-align: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    @media (max-width: 860px) {{
      .controls {{ grid-template-columns: 1fr; }}
      main {{ grid-template-columns: 1fr; padding: 14px; }}
      aside {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Paper Reading Buffer</h1>
    <div class="controls">
      <input id="search" type="search" placeholder="Search title, abstract, org, conference">
      <select id="sort">
        <option value="score">Sort by score</option>
        <option value="seed">Sort by seed + score</option>
        <option value="citations">Sort by citations</option>
        <option value="stars">Sort by GitHub stars</option>
        <option value="date">Sort by date</option>
        <option value="title">Sort by title</option>
      </select>
      <button id="refresh">Fetch Papers</button>
      <button id="reset">Reset</button>
    </div>
  </header>
  <main>
    <aside>
      <strong>Directions</strong>
      <button class="chip active" data-direction="all">All papers</button>
      {direction_buttons}
      <p class="subtle">Abstracts are folded by default. README promotion happens only after reading.</p>
      <p class="subtle"><a href="/buffer.md" target="_blank">Open buffer.md</a></p>
    </aside>
    <section id="papers">
      {cards or '<div class="empty">No papers yet. Click Fetch Papers.</div>'}
    </section>
  </main>
  <script>
    const search = document.querySelector('#search');
    const sort = document.querySelector('#sort');
    const reset = document.querySelector('#reset');
    const refresh = document.querySelector('#refresh');
    const chips = Array.from(document.querySelectorAll('.chip'));
    const list = document.querySelector('#papers');
    let activeDirection = 'all';

    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      const cards = Array.from(document.querySelectorAll('.paper'));
      cards.forEach(card => {{
        const directionOk = activeDirection === 'all' || card.dataset.direction === activeDirection;
        const searchOk = !q || card.textContent.toLowerCase().includes(q);
        card.style.display = directionOk && searchOk ? '' : 'none';
      }});
      const sorted = cards.sort((a, b) => {{
        if (sort.value === 'seed') {{
          const seedDiff = Number(b.dataset.seed) - Number(a.dataset.seed);
          if (seedDiff !== 0) return seedDiff;
          return Number(b.dataset.score) - Number(a.dataset.score);
        }}
        if (sort.value === 'citations') return Number(b.dataset.citations) - Number(a.dataset.citations);
        if (sort.value === 'stars') return Number(b.dataset.stars) - Number(a.dataset.stars);
        if (sort.value === 'date') return b.dataset.date.localeCompare(a.dataset.date);
        if (sort.value === 'title') return a.dataset.title.localeCompare(b.dataset.title);
        return Number(b.dataset.score) - Number(a.dataset.score);
      }});
      sorted.forEach(card => list.appendChild(card));
    }}

    chips.forEach(chip => {{
      chip.addEventListener('click', () => {{
        chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        activeDirection = chip.dataset.direction;
        applyFilters();
      }});
    }});
    search.addEventListener('input', applyFilters);
    sort.addEventListener('change', applyFilters);
    reset.addEventListener('click', () => {{
      search.value = '';
      sort.value = 'score';
      activeDirection = 'all';
      chips.forEach(c => c.classList.toggle('active', c.dataset.direction === 'all'));
      applyFilters();
    }});
    refresh.addEventListener('click', async () => {{
      refresh.disabled = true;
      refresh.textContent = 'Fetching...';
      await fetch('/api/refresh?limit=12', {{ method: 'POST' }});
      window.location.reload();
    }});
    applyFilters();
  </script>
</body>
</html>
"""


def render_card(index: int, paper: dict[str, Any]) -> str:
    title = html.escape(str(paper["title"]))
    authors_list = paper.get("authors", [])
    authors = html.escape(", ".join(authors_list[:6]))
    if len(authors_list) > 6:
        authors += ", et al."
    abstract = html.escape(str(paper.get("abstract", "")))
    direction = html.escape(str(paper.get("direction", "")))
    conference = html.escape(str(paper.get("conference", "")))
    acceptance = html.escape(str(paper.get("acceptance", "")))
    org = html.escape(str(paper.get("org_company", "Unknown")))
    score = int(paper.get("score", 0))
    citations = int(paper.get("citation_count", 0) or 0)
    influential = int(paper.get("influential_citation_count", 0) or 0)
    stars = int(paper.get("github_stars", 0) or 0)
    date = html.escape(str(paper.get("date", "")))
    breakdown = paper.get("score_breakdown", {})
    relevance_terms = ", ".join(breakdown.get("relevance_matches", [])) or "none"
    score_summary = (
        f"relevance {breakdown.get('relevance', 0)}; "
        f"citations {breakdown.get('citations', 0)}; "
        f"github {breakdown.get('github', 0)}; "
        f"conference {breakdown.get('conference', 0)}; "
        f"recency {breakdown.get('recency', 0)}"
    )

    links = []
    links.append(link("arXiv", paper.get("arxiv_url")))
    links.append(link("PDF", paper.get("pdf_url")))
    links.append(link("OpenReview", paper.get("openreview_url")))
    if paper.get("code_url"):
        links.append(link("Code", paper.get("code_url")))
    else:
        links.append('<span class="badge">Code: unknown</span>')

    return f"""
<article class="paper" data-direction="{direction}" data-score="{score}" data-seed="{1 if paper.get('is_seed') else 0}" data-citations="{citations}" data-stars="{stars}" data-date="{date}" data-title="{title.lower()}">
  <div class="meta">
    <span class="badge">#{index}</span>
    {seed_badge(paper)}
    <span class="badge">{date}</span>
    <span class="badge">{direction}</span>
    <span class="badge">{conference}</span>
    <span class="badge">{acceptance}</span>
    <span class="badge score">Score {score}</span>
    <span class="badge">Citations {citations}</span>
    <span class="badge">Influential {influential}</span>
    <span class="badge">Stars {stars}</span>
  </div>
  <h2>{title}</h2>
  <div class="subtle">{authors}</div>
  <div class="subtle">Org / Company: {org}</div>
  <div class="subtle">Score: {html.escape(score_summary)}</div>
  <div class="subtle">Relevance terms: {html.escape(relevance_terms)}</div>
  <div class="links">{' '.join(links)}</div>
  <details>
    <summary>Abstract</summary>
    <p>{abstract}</p>
  </details>
</article>
"""


def link(label: str, url: str | None) -> str:
    if not url:
        return f'<span class="badge">{html.escape(label)}: none</span>'
    safe_url = html.escape(url)
    return f'<a href="{safe_url}" target="_blank" rel="noreferrer">{html.escape(label)}</a>'


def seed_label(paper: dict[str, Any]) -> str:
    return "Seed" if paper.get("is_seed") else ""


def seed_badge(paper: dict[str, Any]) -> str:
    if not paper.get("is_seed"):
        return ""
    return '<span class="badge score">Seed</span>'
