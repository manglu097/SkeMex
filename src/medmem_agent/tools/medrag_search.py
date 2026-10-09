"""MedRAG-based medical knowledge retrieval tool.

This tool wraps the MedRAG toolkit (https://github.com/gzxiong/MedRAG) to
provide retrieval-augmented generation over four curated medical corpora:

  - PubMed      : 23.9 M biomedical abstracts
  - Textbooks   : 18 standard medical textbooks (Harrison's, etc.)
  - StatPearls  : 9.3 k clinical decision-support articles

MedRAG requires an LLM to synthesise the retrieved snippets into a final
answer.  The LLM is specified via ``llm_name`` (e.g. ``"OpenAI/gpt-4o"``),
and the API credentials are injected through ``api_key`` / ``api_base``.
MedRAG reads these credentials from the ``openai`` module's global attributes
at import time, so this tool sets them before constructing the MedRAG object.

**PubMed Direct Path (corpus="PubMed")**

When ``corpus`` is set to ``"PubMed"``, the tool bypasses MedRAG entirely and
uses the PubMed E-utilities API (esearch + efetch) to fetch real abstracts
directly from NCBI.  The retrieved abstracts are then passed to the same LLM
(configured via ``llm_name`` / ``api_key`` / ``api_base``) for synthesis,
producing a grounded answer with inline citations.  This avoids the need for
a locally downloaded PubMed corpus and always retrieves the most up-to-date
literature from NCBI.

When MedRAG is not installed or the corpus has not been downloaded, the tool
falls back to the Semantic Scholar public API so that the agent can still
retrieve medical literature without any local setup.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ToolParam, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic Scholar / PubMed fallback helpers
# ---------------------------------------------------------------------------
_SEMANTIC_SCHOLAR_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
_PUBMED_ESEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
_PUBMED_EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)


def _semantic_scholar_search(query: str, limit: int = 5, api_key: str = "") -> List[Dict[str, str]]:
    """Query Semantic Scholar for medical papers. Returns a list of snippets."""
    params = urllib.parse.urlencode({
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,externalIds",
    })
    url = f"{_SEMANTIC_SCHOLAR_URL}?{params}"
    headers = {"Accept": "application/json"}
    # Priority: explicit arg > env var
    ss_api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if ss_api_key:
        headers["x-api-key"] = ss_api_key

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Semantic Scholar API error: {e}")
        return []

    results = []
    for paper in data.get("data", []):
        title = paper.get("title", "")
        abstract = paper.get("abstract") or ""
        year = paper.get("year", "")
        authors = ", ".join(
            a.get("name", "") for a in (paper.get("authors") or [])[:3]
        )
        pmid = (paper.get("externalIds") or {}).get("PubMed", "")
        source = f"PubMed:{pmid}" if pmid else "Semantic Scholar"
        snippet = abstract[:2000] if abstract else "(No abstract available)"
        results.append({
            "title": title,
            "authors": authors,
            "year": str(year),
            "source": source,
            "content": snippet,
        })
    return results


def _pubmed_search(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Query PubMed E-utilities for biomedical abstracts."""
    import xml.etree.ElementTree as ET
    ncbi_api_key = os.environ.get("NCBI_API_KEY", "")

    search_params: Dict[str, Any] = {
        "db": "pubmed",
        "term": query,
        "retmax": limit,
        "retmode": "json",
        "sort": "relevance",
    }
    if ncbi_api_key:
        search_params["api_key"] = ncbi_api_key

    search_url = f"{_PUBMED_ESEARCH_URL}?{urllib.parse.urlencode(search_params)}"
    try:
        with urllib.request.urlopen(search_url, timeout=20) as resp:
            search_data = json.loads(resp.read().decode("utf-8"))
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning(f"PubMed esearch error: {e}")
        return []

    if not pmids:
        return []

    fetch_params: Dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if ncbi_api_key:
        fetch_params["api_key"] = ncbi_api_key

    fetch_url = f"{_PUBMED_EFETCH_URL}?{urllib.parse.urlencode(fetch_params)}"
    try:
        with urllib.request.urlopen(fetch_url, timeout=20) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
    except Exception as e:
        logger.warning(f"PubMed efetch error: {e}")
        return []

    results = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = article.findtext(".//ArticleTitle") or ""
        
        abstract_texts = article.findall(".//AbstractText")
        abstract = " ".join([a.text for a in abstract_texts if a.text])
        if not abstract:
            abstract = "(No abstract available)"
            
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.findtext("LastName")
            initials = author.findtext("Initials")
            if last_name and initials:
                authors.append(f"{last_name} {initials}")
            elif last_name:
                authors.append(last_name)
        authors_str = ", ".join(authors[:3])
        
        year = article.findtext(".//PubDate/Year")
        if not year:
            year = article.findtext(".//MedlineDate") or ""
            
        results.append({
            "title": title,
            "authors": authors_str,
            "year": year,
            "source": f"PubMed:{pmid}",
            "content": abstract[:2000],
        })
    return results


# ---------------------------------------------------------------------------
# MedRAGSearchTool
# ---------------------------------------------------------------------------

@dataclass
class MedRAGSearchTool:
    """Medical knowledge retrieval + synthesis tool backed by MedRAG.

    MedRAG performs dense retrieval over curated medical corpora and then
    uses an LLM to synthesise the retrieved snippets into a final answer.
    This matches the official ``MedRAG with corpus cached in memory`` usage::

        medrag = MedRAG(
            llm_name="OpenAI/gpt-4o",
            rag=True,
            retriever_name="MedCPT",
            corpus_name="Textbooks",
            corpus_cache=True,
        )
        answer, snippets, scores = medrag.answer(question=question, options=options, k=32)

    LLM credentials are injected via ``api_key`` and ``api_base``, which are
    written into the ``openai`` module's global attributes before MedRAG is
    initialised.  ``llm_name`` must follow MedRAG's ``"Provider/model"``
    convention, e.g. ``"OpenAI/gpt-4o"`` or ``"OpenAI/gpt-4.1-mini"``.

    **Special case — corpus="PubMed":**

    When ``corpus`` is set to ``"PubMed"``, the tool **bypasses MedRAG entirely**
    and uses the PubMed E-utilities API (esearch + efetch XML) to retrieve real
    abstracts directly from NCBI.  The abstracts are then passed to the same LLM
    (``llm_name`` / ``api_key`` / ``api_base``) for synthesis.  This path:

    - Does **not** require a locally downloaded PubMed corpus.
    - Always returns the most up-to-date literature from NCBI.
    - Supports the same ``options`` MCQ format as the MedRAG path.
    - Falls back gracefully if the LLM call fails.

    When MedRAG is not installed or the corpus is unavailable, the tool falls
    back to the Semantic Scholar public API.

    Args:
        llm_name: MedRAG LLM identifier in ``"Provider/model"`` format.
            Must start with ``"OpenAI/"`` for OpenAI-compatible endpoints.
            Defaults to ``"OpenAI/gpt-4o"``.
        api_key: API key for the LLM.  Falls back to the ``OPENAI_API_KEY``
            environment variable when empty.
        api_base: Base URL of the OpenAI-compatible API endpoint.  When set,
            overrides the default OpenAI endpoint, allowing locally deployed
            models (e.g. vLLM) to be used.  Falls back to the
            ``OPENAI_API_BASE`` environment variable when empty.
        corpus: MedRAG corpus to search. One of ``"PubMed"``, ``"Textbooks"``,
            ``"StatPearls"``, ``"Wikipedia"``, or ``"MedCorp"`` (all combined).
            Setting ``"PubMed"`` activates the direct NCBI API path (no local
            corpus required).
        retriever: MedRAG retriever. One of ``"MedCPT"``, ``"BM25"``,
            ``"Contriever"``, or ``"SPECTER"``.
        db_dir: Local directory where MedRAG corpus files are stored.
        top_k: Number of snippets to retrieve per query (passed as ``k`` to
            ``medrag.answer()``).
        corpus_cache: Load the entire corpus into memory for faster repeated
            retrieval.  Corresponds to ``corpus_cache=True`` in MedRAG.
        HNSW: Build/use an HNSW faiss index for further acceleration of dense
            retrieval.  Corresponds to ``HNSW=True`` in MedRAG.
        timeout_seconds: Timeout for HTTP requests (both PubMed API and fallback).
    """

    name: str = "medrag_search"
    description: str = (
    "Retrieve medical knowledge from curated corpora: "
    "PubMed (recent research and clinical studies), "
    "Textbooks (foundational knowledge and standard clinical practice), "
    "StatPearls (concise clinical summaries and guideline-style content). "
    "Use when additional medical evidence is needed for diagnosis, management, "
    "or drug-related reasoning."
)

    # LLM configuration -------------------------------------------------------
    llm_name: str = "OpenAI/gpt-4o"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.0
    top_p: float = 1.0

    # Retrieval configuration -------------------------------------------------
    default_corpus: str = "MedCorp"   # used when caller does not pass corpus in payload
    retriever: str = "MedCPT"
    db_dir: str = "./corpus"
    top_k: int = 5
    corpus_cache: bool = False
    HNSW: bool = False
    timeout_seconds: int = 30

    # Semantic Scholar API key (optional; raises rate limit from 1 req/s to 10 req/s)
    semantic_scholar_api_key: str = ""

    # Internal state — not part of the public API ----------------------------
    _medrag_instances: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _medrag_available: Optional[bool] = field(default=None, init=False, repr=False)
    _medrag_last_error: str = field(default="", init=False, repr=False)
    _medrag_last_traceback: str = field(default="", init=False, repr=False)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _resolve_credentials(self) -> tuple[str, str]:
        """Return (api_key, api_base), falling back to environment variables."""
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        base = self.api_base or os.environ.get("OPENAI_API_BASE", "")
        return key, base

    def _try_init_medrag(self, corpus: str) -> bool:
        """Attempt to initialise the MedRAG instance for the given corpus.

        Uses a cache to avoid re-initialising the same corpus multiple times.
        Each corpus gets its own MedRAG instance stored in _medrag_instances.

        Injects LLM credentials into the ``openai`` module's global attributes
        **before** importing MedRAG, because MedRAG reads ``openai.api_key``
        and ``openai.api_base`` at module level when it is first imported.

        Returns True on success, False otherwise.
        """
        # Check if this corpus is already cached
        if corpus in self._medrag_instances:
            logger.debug(f"Reusing cached MedRAG instance for corpus: {corpus}")
            return True

        # Check if MedRAG is available at all (only check once)
        if self._medrag_available is False:
            logger.warning(
                "Skipping MedRAG re-initialisation because a previous attempt already failed. "
                f"corpus={corpus}, last_error={self._medrag_last_error or '(unknown)'}"
            )
            if self._medrag_last_traceback:
                logger.warning(
                    "Previous MedRAG initialisation traceback for corpus=%s:\n%s",
                    corpus,
                    self._medrag_last_traceback,
                )
            return False

        api_key, api_base = self._resolve_credentials()
        db_dir_exists = os.path.isdir(self.db_dir)
        if not db_dir_exists:
            logger.warning(
                "Configured MedRAG db_dir does not exist or is not a directory: %s",
                self.db_dir,
            )
        if "/" not in self.llm_name:
            logger.warning(
                "Configured llm_name does not follow MedRAG's expected 'Provider/model' format: %s",
                self.llm_name,
            )

        logger.info(
            "Attempting MedRAG initialisation: corpus=%s, llm_name=%s, retriever=%s, "
            "db_dir=%s, db_dir_exists=%s, corpus_cache=%s, HNSW=%s, api_base=%s",
            corpus,
            self.llm_name,
            self.retriever,
            self.db_dir,
            db_dir_exists,
            self.corpus_cache,
            self.HNSW,
            api_base or "(default)",
        )

        try:
            import openai as _openai_mod
            if api_key:
                _openai_mod.api_key = api_key
            if api_base:
                _openai_mod.api_base = api_base
                try:
                    _openai_mod.base_url = api_base  # type: ignore[attr-defined]
                except AttributeError:
                    pass

            from medrag import MedRAG  # type: ignore

            init_kwargs: Dict[str, Any] = dict(
                llm_name=self.llm_name,
                rag=True,
                retriever_name=self.retriever,
                corpus_name=corpus,
                db_dir=self.db_dir,
                corpus_cache=self.corpus_cache,
            )
            if self.HNSW:
                init_kwargs["HNSW"] = True

            self._medrag_instances[corpus] = MedRAG(**init_kwargs)
            self._medrag_available = True
            self._medrag_last_error = ""
            self._medrag_last_traceback = ""
            logger.info(
                f"MedRAG initialised: llm={self.llm_name}, corpus={corpus}, "
                f"retriever={self.retriever}, corpus_cache={self.corpus_cache}, "
                f"HNSW={self.HNSW}"
            )
        except Exception as e:
            self._medrag_last_error = f"{type(e).__name__}: {e}"
            self._medrag_last_traceback = traceback.format_exc()
            logger.error(
                "MedRAG initialisation failed: corpus=%s, llm_name=%s, retriever=%s, db_dir=%s, "
                "db_dir_exists=%s, corpus_cache=%s, HNSW=%s, error=%s\n%s",
                corpus,
                self.llm_name,
                self.retriever,
                self.db_dir,
                db_dir_exists,
                self.corpus_cache,
                self.HNSW,
                self._medrag_last_error,
                self._medrag_last_traceback,
            )
            self._medrag_available = False

        return self._medrag_available  # type: ignore[return-value]

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Retrieve and synthesise medical knowledge for a query.

        Execution paths (in priority order):

        **Path A — PubMed Direct** (``corpus="PubMed"``):
            Bypasses MedRAG.  Queries NCBI via esearch + efetch XML to fetch
            real abstracts, then calls the configured LLM for synthesis.
            No local corpus required; always returns up-to-date literature.

        **Path B — MedRAG** (all other corpora, when MedRAG is available):
            Uses the locally installed MedRAG library with the configured
            corpus and retriever for dense retrieval + LLM synthesis.

        **Path C — Fallback** (MedRAG unavailable):
            Queries Semantic Scholar (then PubMed as secondary fallback)
            without LLM synthesis.

        Args:
            payload: Must contain ``"query"`` (str). Optionally:
                - ``"top_k"`` (int, 1–32): number of snippets to retrieve.
                - ``"options"`` (dict): answer choices for MCQ questions,
                  e.g. ``{"A": "...", "B": "...", "C": "...", "D": "..."}``.
                  When provided, the LLM returns a structured JSON answer with
                  ``step_by_step_thinking`` and ``answer_choice``.

        Returns:
            ToolResult with keys:
                - ``query``: the original query string.
                - ``source``: corpus/API used (e.g. ``"PubMed API (direct)"``,
                  ``"MedRAG/Textbooks"``, or ``"Semantic Scholar / PubMed (fallback)"``).
                - ``answer``: synthesised answer from the LLM.
                - ``snippets``: list of dicts with ``title``, ``source``, and
                  ``content`` (top retrieved evidence snippets).

        Raises:
            ValueError: If ``query`` is missing or empty.
        """
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("'query' field is required and must not be empty.")

        corpus = str(payload.get("corpus", self.default_corpus)).strip()
        top_k = int(payload.get("top_k", self.top_k))
        top_k = max(1, min(top_k, 32))
        options = payload.get("options", None)

        # ------------------------------------------------------------------
        # Path A: PubMed + Semantic Scholar dual search + LLM synthesis
        # ------------------------------------------------------------------
        if corpus == "PubMed":
            logger.info(
                "Corpus is PubMed. Querying both PubMed E-utilities and "
                "Semantic Scholar in parallel, then synthesising with LLM."
            )
            # Fetch from both sources; each returns up to top_k results.
            # We then merge and de-duplicate by title before LLM synthesis.
            pubmed_snippets = _pubmed_search(query, limit=top_k)
            ss_snippets = _semantic_scholar_search(
                query, limit=top_k, api_key=self.semantic_scholar_api_key
            )

            # Merge: PubMed first (more authoritative), then Semantic Scholar
            # entries whose titles are not already covered by PubMed.
            seen_titles: set = set()
            snippets = []
            for s in pubmed_snippets:
                key = s["title"].lower().strip()
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    snippets.append(s)
            for s in ss_snippets:
                key = s["title"].lower().strip()
                if key and key not in seen_titles:
                    seen_titles.add(key)
                    snippets.append(s)

            source_label = "PubMed API + Semantic Scholar"
            if not pubmed_snippets and not ss_snippets:
                logger.warning("Both PubMed and Semantic Scholar returned no results.")
                snippets = [{
                    "title": "No results found",
                    "source": source_label,
                    "content": f"No medical literature found for query: '{query}'.",
                }]
                answer_text = (
                    "No relevant literature found in PubMed or Semantic Scholar "
                    "to answer the query."
                )
            else:
                if not pubmed_snippets:
                    logger.warning("PubMed returned no results; using Semantic Scholar only.")
                    source_label = "Semantic Scholar"
                elif not ss_snippets:
                    logger.warning("Semantic Scholar returned no results; using PubMed only.")
                    source_label = "PubMed API (direct)"

                api_key, api_base = self._resolve_credentials()
                context_parts = []
                for i, s in enumerate(snippets):
                    authors = s.get("authors", "")
                    year = s.get("year", "")
                    meta = ""
                    if authors:
                        meta += f"Authors: {authors}  "
                    if year:
                        meta += f"Year: {year}"
                    context_parts.append(
                        f"Document [{i+1}]\nTitle: {s['title']}\n"
                        + (meta.strip() + "\n" if meta.strip() else "")
                        + f"Abstract: {s['content']}"
                    )
                context_str = "\n\n".join(context_parts)

                system_prompt = """You are a helpful medical assistant. Your task is to summarize the provided relevant documents based on the user's query. Please provide a comprehensive and concise summary of the key information found in the documents. Do not attempt to answer any multiple-choice questions. Just provide the summary in plain text."""

                if options:
                    system_prompt += (
                        f"\n\nThe user has provided multiple choice options: "
                        f"{json.dumps(options)}. "
                        "You must select the best option based on the evidence. "
                        "Output your response in valid JSON format with two keys: "
                        "'step_by_step_thinking' (your reasoning) and "
                        "'answer_choice' (the exact key of the chosen option, "
                        "e.g., 'A')."
                    )

                try:
                    from .reasoning_tools import _call_llm
                    model = self.llm_name
                    answer_raw = _call_llm(
                        api_key=api_key,
                        api_base=api_base,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=f"Here are the relevant documents:\n{context_str}\n\nHere is the user's query/topic: {query}\n\nPlease summarize the relevant documents based on the query above:",
                        max_tokens=1024,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        timeout=self.timeout_seconds,
                    )

                    answer_text = answer_raw
                    if options:
                        try:
                            clean_raw = answer_raw.strip()
                            if clean_raw.startswith("```json"):
                                clean_raw = clean_raw[7:]
                            if clean_raw.endswith("```"):
                                clean_raw = clean_raw[:-3]
                            parsed = json.loads(clean_raw.strip())
                            if isinstance(parsed, dict):
                                thinking = parsed.get("step_by_step_thinking", "")
                                choice = parsed.get("answer_choice", "")
                                if thinking or choice:
                                    answer_text = (
                                        f"{thinking}\n\nAnswer: {choice}".strip()
                                        if thinking
                                        else choice
                                    )
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(
                        "LLM synthesis failed for PubMed+SS path: query=%s, llm_name=%s, error=%s\n%s",
                        query,
                        self.llm_name,
                        f"{type(e).__name__}: {e}",
                        traceback.format_exc(),
                    )
                    answer_text = f"LLM synthesis failed: {e}"

            return ToolResult(
                name=self.name,
                payload={
                    "query": query,
                    "source": source_label,
                    "answer": answer_text,
                    "snippets": snippets,
                },
                render_fields=["answer"],
            )

        # ------------------------------------------------------------------
        # Path B: MedRAG — retrieval + LLM synthesis
        # ------------------------------------------------------------------
        if self._try_init_medrag(corpus) and corpus in self._medrag_instances:
            try:
                answer_raw, snippets_raw, scores = self._medrag_instances[corpus].answer(
                    question=query,
                    options=options,
                    k=top_k,
                )
                # Parse answer: MedRAG returns a JSON string for OpenAI models
                answer_text = answer_raw
                if isinstance(answer_raw, str):
                    try:
                        parsed = json.loads(answer_raw)
                        if isinstance(parsed, dict):
                            # Prefer step_by_step_thinking + answer_choice if present
                            thinking = parsed.get("step_by_step_thinking", "")
                            choice = parsed.get("answer_choice", "")
                            if thinking or choice:
                                answer_text = (
                                    f"{thinking}\n\nAnswer: {choice}".strip()
                                    if thinking
                                    else choice
                                )
                            else:
                                answer_text = json.dumps(parsed, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass  # keep answer_raw as-is

                # Normalise snippets
                snippets = []
                for s in (snippets_raw or []):
                    snippets.append({
                        "title": s.get("title", ""),
                        "source": s.get("id", corpus),
                        "content": s.get("content", s.get("contents", ""))[:600],
                    })

                return ToolResult(
                    name=self.name,
                    payload={
                        "query": query,
                        "source": f"MedRAG/{corpus}",
                        "answer": answer_text,
                        "snippets": snippets,
                    },
                    render_fields=["answer"],
                )
            except Exception as e:
                self._medrag_last_error = f"{type(e).__name__}: {e}"
                self._medrag_last_traceback = traceback.format_exc()
                logger.error(
                    "MedRAG answer() failed: corpus=%s, query=%s, top_k=%s, llm_name=%s, retriever=%s, error=%s\n%s",
                    corpus,
                    query,
                    top_k,
                    self.llm_name,
                    self.retriever,
                    self._medrag_last_error,
                    self._medrag_last_traceback,
                )
                raise

        # ------------------------------------------------------------------
        # Path B: Fallback — Semantic Scholar + PubMed (no LLM synthesis)
        # ------------------------------------------------------------------
        fallback_reason = self._medrag_last_error or "unknown initialisation failure"
        logger.warning(
            "MedRAG not available; falling back to Semantic Scholar / PubMed API. "
            "No LLM synthesis will be performed. corpus=%s, reason=%s",
            corpus,
            fallback_reason,
        )
        if self._medrag_last_traceback:
            logger.warning(
                "Latest MedRAG traceback before fallback for corpus=%s:\n%s",
                corpus,
                self._medrag_last_traceback,
            )
        snippets = _semantic_scholar_search(
            query, limit=top_k, api_key=self.semantic_scholar_api_key
        )
        if not snippets:
            snippets = _pubmed_search(query, limit=top_k)

        if not snippets:
            snippets = [{
                "title": "No results found",
                "source": "API",
                "content": (
                    f"No medical literature found for query: '{query}'. "
                    "Try rephrasing with more specific medical terminology."
                ),
            }]

        return ToolResult(
            name=self.name,
            payload={
                "query": query,
                "source": "Semantic Scholar / PubMed (fallback)",
                "answer": (
                    "MedRAG is not available. The following snippets were retrieved "
                    "from Semantic Scholar / PubMed without LLM synthesis. "
                    f"Reason: {fallback_reason}"
                ),
                "snippets": snippets,
                "debug": {
                    "corpus": corpus,
                    "llm_name": self.llm_name,
                    "retriever": self.retriever,
                    "db_dir": self.db_dir,
                    "last_error": self._medrag_last_error,
                    "last_traceback": self._medrag_last_traceback,
                },
            },
            render_fields=["answer"],
        )

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="query",
                type="string",
                required=True,
                description=(
                    "Medical question or keyword to search for. Use precise clinical "
                    "terminology (e.g., drug names, disease names, ICD terms) for best results."
                ),
            ),
            ToolParam(
                name="corpus",
                type="string",
                required=True,
                description=(
                    "Corpus to search. One of: \"PubMed\" (live NCBI API, no local corpus required), "
                    "\"Textbooks\", \"StatPearls\". Any other value falls back to PubMed."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"query": "first-line treatment for type 2 diabetes guidelines", "corpus": "PubMed"}'
        )
