"""Medical knowledge base tools.

This module provides three tools for querying structured medical knowledge:

  1. DrugInfoLookupTool       — Look up comprehensive drug information
                                (mechanism, indications, side effects, etc.)
  2. DrugInteractionCheckTool — Check for clinically significant interactions
                                between two or more drugs.
  3. UMLSConceptLookupTool    — Retrieve standardised medical concepts,
                                definitions, and semantic relations from UMLS.

Data sources and fallback strategy
------------------------------------
DrugBank (local .pkl file, as used by ReflecTool)
  → RxNorm API (NIH, free, no key required)
  → OpenFDA Drug API (free, no key required)

UMLS REST API (requires free NIH account and API key via $UMLS_API_KEY)
  → NLM MedlinePlus Connect API (free, no key required)
"""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import ToolParam, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API endpoints (no key required)
# ---------------------------------------------------------------------------
_RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
_OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
_UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
_MEDLINEPLUS_BASE = "https://connect.medlineplus.gov/service"


# ===========================================================================
# Helper: simple HTTP GET returning parsed JSON
# ===========================================================================
def _get_json(url: str, timeout: int = 20) -> Optional[Dict]:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.debug(f"HTTP GET failed for {url}: {e}")
        return None


# ===========================================================================
# DrugInfoLookupTool
# ===========================================================================
@dataclass
class DrugInfoLookupTool:
    """Look up comprehensive information about a drug.

    Retrieves mechanism of action, indications, contraindications, common
    adverse effects, dosage forms, and drug class from RxNorm and OpenFDA.

    Args:
        name: Tool name used in agent prompts and the tool registry.
        description: Human-readable description injected into the system prompt.
        drugbank_path: Optional path to a local DrugBank ``drugbank.pkl`` file
            (as used by ReflecTool). When provided and the file exists, it is
            used as the primary data source.
        timeout_seconds: HTTP request timeout for API calls.
    """

    name: str = "drug_info_lookup"
    description: str = (
        "Look up a drug's mechanism of action, indications, contraindications, adverse effects, "
        "and dosage forms. One drug per call."
    )
    drugbank_path: str = "./drugbank/drugbank.pkl"
    timeout_seconds: int = 20
    _drugbank_df: Any = field(default=None, init=False, repr=False)
    _drugbank_loaded: Optional[bool] = field(default=None, init=False, repr=False)
    _drugbank_cache: Dict = field(default_factory=dict, init=False, repr=False)

    def _try_load_drugbank(self) -> bool:
        if self._drugbank_loaded is not None:
            return self._drugbank_loaded
        try:
            import pandas as pd  # type: ignore
            self._drugbank_df = pd.read_pickle(self.drugbank_path)
            self._drugbank_loaded = True
            logger.info(f"DrugBank loaded from {self.drugbank_path}")
        except Exception as e:
            logger.info(f"DrugBank not available ({e}). Using RxNorm/OpenFDA APIs.")
            self._drugbank_loaded = False
        return self._drugbank_loaded  # type: ignore[return-value]

    def _query_drugbank(self, drug_name: str) -> Optional[Dict]:
        """Fuzzy-match drug name in local DrugBank DataFrame."""
        if drug_name in self._drugbank_cache:
            return self._drugbank_cache[drug_name]
        if not self._try_load_drugbank():
            self._drugbank_cache[drug_name] = None
            return None
        try:
            import difflib
            df = self._drugbank_df
            df = df.copy()
            df["_sim"] = df["synonyms"].apply(
                lambda syns: max(
                    (difflib.SequenceMatcher(None, drug_name.lower(), s.lower()).ratio()
                     for s in syns),
                    default=0.0,
                )
            )
            best = df.loc[df["_sim"].idxmax()]
            if best["_sim"] < 0.6:
                self._drugbank_cache[drug_name] = None
                return None
            row = best.drop("_sim").to_dict()
            row.pop("similarity", None)
            self._drugbank_cache[drug_name] = row
            return row
        except Exception as e:
            logger.debug(f"DrugBank query error: {e}")
            self._drugbank_cache[drug_name] = None
            return None

    def _query_rxnorm(self, drug_name: str) -> Dict[str, Any]:
        """Query RxNorm for drug concept info."""
        result: Dict[str, Any] = {}

        # Get RxCUI
        cui_url = (
            f"{_RXNORM_BASE}/rxcui.json?"
            f"name={urllib.parse.quote(drug_name)}&search=2"
        )
        cui_data = _get_json(cui_url, self.timeout_seconds)
        if not cui_data:
            return result

        rxcui = (
            cui_data.get("idGroup", {}).get("rxnormId", [None])[0]
        )
        if not rxcui:
            return result

        result["rxcui"] = rxcui

        # Get drug properties
        props_url = f"{_RXNORM_BASE}/rxcui/{rxcui}/allProperties.json?prop=all"
        props_data = _get_json(props_url, self.timeout_seconds)
        if props_data:
            props = props_data.get("propConceptGroup", {}).get("propConcept", [])
            for p in props:
                pname = p.get("propName", "")
                pval = p.get("propValue", "")
                if pname == "RxNorm Name":
                    result["name"] = pval
                elif pname == "TTY":
                    result["term_type"] = pval

        # Get drug classes
        class_url = f"{_RXNORM_BASE}/rxcui/{rxcui}/classes.json"
        class_data = _get_json(class_url, self.timeout_seconds)
        if class_data:
            classes = class_data.get("rxclassDrugInfoList", {}).get(
                "rxclassDrugInfo", []
            )
            result["drug_classes"] = list({
                c.get("rxclassMinConceptItem", {}).get("className", "")
                for c in classes
                if c.get("rxclassMinConceptItem", {}).get("className")
            })[:5]

        return result

    def _query_openfda(self, drug_name: str) -> Dict[str, Any]:
        """Query OpenFDA drug label API for clinical information."""
        url = (
            f"{_OPENFDA_BASE}?search=openfda.brand_name:{urllib.parse.quote(drug_name)}"
            f"+openfda.generic_name:{urllib.parse.quote(drug_name)}&limit=1"
        )
        data = _get_json(url, self.timeout_seconds)
        if not data or not data.get("results"):
            # Try generic name search
            url2 = (
                f"{_OPENFDA_BASE}?search=openfda.generic_name:"
                f"{urllib.parse.quote(drug_name)}&limit=1"
            )
            data = _get_json(url2, self.timeout_seconds)

        if not data or not data.get("results"):
            return {}

        label = data["results"][0]
        result: Dict[str, Any] = {}

        def _first(key: str) -> str:
            val = label.get(key, [])
            return val[0][:800] if val else ""

        indications = _first("indications_and_usage")
        if indications:
            result["indications"] = indications

        contraindications = _first("contraindications")
        if contraindications:
            result["contraindications"] = contraindications[:600]

        warnings = _first("warnings")
        if warnings:
            result["warnings"] = warnings[:600]

        adverse = _first("adverse_reactions")
        if adverse:
            result["adverse_reactions"] = adverse[:600]

        mechanism = _first("mechanism_of_action")
        if mechanism:
            result["mechanism_of_action"] = mechanism[:600]

        dosage = _first("dosage_and_administration")
        if dosage:
            result["dosage_and_administration"] = dosage[:600]

        openfda = label.get("openfda", {})
        brand = openfda.get("brand_name", [])
        generic = openfda.get("generic_name", [])
        if brand:
            result["brand_names"] = brand[:3]
        if generic:
            result["generic_name"] = generic[0]

        return result

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Look up drug information.

        Args:
            payload: Must contain ``"drug_name"`` (str). Optionally
                ``"fields"`` (list[str]) to restrict which fields to return.

        Returns:
            ToolResult with comprehensive drug information.

        Raises:
            ValueError: If ``drug_name`` is missing or empty.
        """
        drug_name = str(payload.get("drug_name", "")).strip()
        if not drug_name:
            raise ValueError("'drug_name' field is required and must not be empty.")

        info: Dict[str, Any] = {"drug_name": drug_name}

        # Try DrugBank first (local)
        db_result = self._query_drugbank(drug_name)
        if db_result:
            info["source"] = "DrugBank (local)"
            info.update({k: v for k, v in db_result.items() if v})
        else:
            # RxNorm for drug class / CUI
            rxnorm = self._query_rxnorm(drug_name)
            if rxnorm:
                info["rxcui"] = rxnorm.get("rxcui", "")
                info["drug_classes"] = rxnorm.get("drug_classes", [])

            # OpenFDA for clinical label information
            fda = self._query_openfda(drug_name)
            info.update(fda)

            if len(info) <= 1:
                info["note"] = (
                    f"No detailed information found for '{drug_name}'. "
                    "Verify the drug name spelling or try the generic name."
                )
                info["source"] = "RxNorm/OpenFDA (no results)"
            else:
                info["source"] = "RxNorm + OpenFDA"

        return ToolResult(name=self.name, payload=info, render_fields=[
            "drug_name", "rxcui", "drug_classes", "indications", "contraindications",
            "adverse_reactions", "mechanism_of_action", "dosage_and_administration",
            "warnings", "brand_names", "generic_name", "source",
        ])

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="drug_name",
                type="string",
                required=True,
                description=(
                    "The name of a single drug to look up. Accepts generic names (e.g., "
                    "\"metformin\"), brand names (e.g., \"Glucophage\"), or drug class "
                    "names. Use the most common generic name for best results. "
                    "Only one drug name is accepted per call — do not pass a "
                    "comma-separated list of multiple drugs."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return '{"drug_name": "metformin"}'


# ===========================================================================
# DrugInteractionCheckTool
# ===========================================================================
@dataclass
class DrugInteractionCheckTool:
    """Check for clinically significant drug-drug interactions.

    Primary data source: local DrugBank file — either a ``.pkl`` pre-parsed
    DataFrame or the raw DrugBank XML full-database download (``.xml``).
    Each row in the DataFrame is expected to have a ``drug_interactions``
    column containing a list of dicts representing ``<drug-interaction>``
    elements from the DrugBank XML schema.  According to the DrugBank XML
    specification (https://docs.drugbank.com/xml/#drug-interactions), each
    ``<drug-interaction>`` element contains:

    - ``<drugbank-id>`` — DrugBank ID of the interacting drug (hyphenated key)
    - ``<name>``        — Name of the interacting drug
    - ``<description>`` — Textual description of the interaction

    The tool therefore accepts both ``"drugbank-id"`` (hyphenated, as produced
    directly by XML parsers) and ``"drugbank_id"`` (underscored, as used in
    some pre-processed DataFrames) when reading interaction entries.

    Fallback: when the local DrugBank file is unavailable, the tool falls back
    to the RxNorm Interaction API (free, no key required).  Note that the
    RxNorm endpoint may be unreliable; the DrugBank path is strongly preferred.

    Query logic (DrugBank path):
    1. Fuzzy-match each input drug name against the ``synonyms`` column to
       find the corresponding DrugBank row (same algorithm as
       :class:`DrugInfoLookupTool`).
    2. For every resolved drug, retrieve its ``drug_interactions`` list.
    3. Cross-filter: keep only entries where the interacting partner's
       DrugBank ID (``drugbank-id`` or ``drugbank_id``) or canonical name
       matches one of the *other* queried drugs.  Name matching also checks
       the ``synonyms`` list of the resolved row to handle cases where the
       interaction entry uses a trade name or synonym rather than the
       canonical DrugBank name.
    4. De-duplicate symmetric pairs (A→B and B→A produce one record).

    Args:
        name: Tool name used in agent prompts and the tool registry.
        description: Human-readable description injected into the system prompt.
        drugbank_path: Path to the local DrugBank ``.pkl`` or ``.xml`` file.
        timeout_seconds: HTTP request timeout (used only for the RxNorm fallback).
    """

    name: str = "drug_interaction_check"
    description: str = (
        "Check for known drug-drug interactions between two or more medications."
    )
    drugbank_path: str = "./drugbank/drugbank.pkl"
    timeout_seconds: int = 20
    _drugbank_df: Any = field(default=None, init=False, repr=False)
    _drugbank_loaded: Optional[bool] = field(default=None, init=False, repr=False)
    _drugbank_cache: Dict = field(default_factory=dict, init=False, repr=False)
    _rxcui_cache: Dict = field(default_factory=dict, init=False, repr=False)
    # Inverted index built once after loading: synonym_lower → row dict.
    # Enables O(1) exact-match lookups instead of O(N) full-table scans.
    _synonym_index: Dict = field(default_factory=dict, init=False, repr=False)
    # Flat sorted list of all (synonym_lower, row_dict) pairs for fuzzy search.
    _synonym_keys: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # DrugBank helpers (shared logic with DrugInfoLookupTool)
    # ------------------------------------------------------------------

    def _try_load_drugbank(self) -> bool:
        """Load the DrugBank DataFrame once and build lookup indexes.

        Supports two file formats:
        - ``.pkl``: a pre-parsed pandas DataFrame (fastest).
        - ``.xml``: the raw DrugBank full-database XML download.  The XML is
          parsed on first load and an auto-generated ``.pkl`` cache is written
          alongside the XML file so that subsequent loads skip XML parsing
          entirely (typically 10-100× faster).

        After loading, two in-memory indexes are built:
        - ``_synonym_index``: ``{synonym_lower: row_dict}`` for O(1) exact
          lookups.  When multiple drugs share a synonym the one with the
          higher synonym-list position wins (canonical name first).
        - ``_synonym_keys``: sorted list of all synonym strings used as the
          candidate pool for ``difflib.get_close_matches`` fuzzy search.
        """
        if self._drugbank_loaded is not None:
            return self._drugbank_loaded
        try:
            import pandas as pd  # type: ignore
            path = self.drugbank_path
            if path.lower().endswith(".xml"):
                # Derive a sibling .pkl cache path next to the XML file
                pkl_cache = path[:-4] + "_interaction_cache.pkl"
                if os.path.exists(pkl_cache):
                    logger.info(f"Loading DrugBank from pkl cache: {pkl_cache}")
                    self._drugbank_df = pd.read_pickle(pkl_cache)
                else:
                    logger.info(f"Parsing DrugBank XML (first run, may take a while): {path}")
                    self._drugbank_df = self._parse_drugbank_xml(path)
                    try:
                        self._drugbank_df.to_pickle(pkl_cache)
                        logger.info(f"DrugBank pkl cache written to {pkl_cache}")
                    except Exception as cache_err:
                        logger.warning(f"Could not write pkl cache: {cache_err}")
            else:
                self._drugbank_df = pd.read_pickle(path)
            self._build_synonym_index()
            self._drugbank_loaded = True
            logger.info(
                f"DrugBank ready: {len(self._drugbank_df)} drugs, "
                f"{len(self._synonym_index)} synonym entries indexed."
            )
        except Exception as e:
            logger.info(
                f"DrugBank not available ({e}). "
                "Falling back to RxNorm Interaction API."
            )
            self._drugbank_loaded = False
        return self._drugbank_loaded  # type: ignore[return-value]

    def _build_synonym_index(self) -> None:
        """Build the inverted synonym index and flat synonym key list.

        Called once after the DataFrame is loaded.  Iterates over every row
        and every synonym, populating:

        - ``_synonym_index``: ``{synonym_lower: row_dict}``
          The canonical name (first element of the synonyms list) takes
          priority; later synonyms only fill gaps.
        - ``_synonym_keys``: sorted ``list[str]`` of all synonym strings,
          used as the candidate pool for ``difflib.get_close_matches``.
        """
        index: Dict[str, Dict] = {}
        for row in self._drugbank_df.itertuples(index=False):
            row_dict = row._asdict()
            syns = row_dict.get("synonyms") or []
            for syn in syns:
                key = syn.lower()
                if key not in index:  # canonical name wins on collision
                    index[key] = row_dict
        self._synonym_index = index
        self._synonym_keys = sorted(index.keys())

    @staticmethod
    def _parse_drugbank_xml(xml_path: str):
        """Parse a DrugBank full-database XML file into a pandas DataFrame.

        Extracts the fields required by :class:`DrugInteractionCheckTool`:
        ``drugbank_id``, ``name``, ``synonyms``, and ``drug_interactions``.

        The ``drug_interactions`` column contains lists of dicts with keys
        ``drugbank-id``, ``name``, and ``description`` — exactly as they
        appear in the DrugBank XML schema (hyphenated key names preserved).

        Args:
            xml_path: Absolute or relative path to the DrugBank XML file.

        Returns:
            A :class:`pandas.DataFrame` with one row per drug.
        """
        import xml.etree.ElementTree as ET  # stdlib, no extra deps
        import pandas as pd  # type: ignore

        NS = "http://www.drugbank.ca"

        def _tag(local: str) -> str:
            return f"{{{NS}}}{local}"

        def _text(elem, local: str) -> str:
            child = elem.find(_tag(local))
            return (child.text or "").strip() if child is not None else ""

        records = []
        context = ET.iterparse(xml_path, events=("end",))
        for _event, elem in context:
            if elem.tag != _tag("drug"):
                continue

            # Primary DrugBank ID (attribute primary="true")
            primary_id = ""
            for db_id_elem in elem.findall(_tag("drugbank-id")):
                if db_id_elem.get("primary") == "true":
                    primary_id = (db_id_elem.text or "").strip()
                    break
            if not primary_id:
                elem.clear()
                continue

            name = _text(elem, "name")

            # Synonyms: include the canonical name itself for completeness
            synonyms: List[str] = [name] if name else []
            syns_elem = elem.find(_tag("synonyms"))
            if syns_elem is not None:
                for syn in syns_elem.findall(_tag("synonym")):
                    val = (syn.text or "").strip()
                    if val and val not in synonyms:
                        synonyms.append(val)

            # Drug interactions
            drug_interactions: List[Dict[str, str]] = []
            di_container = elem.find(_tag("drug-interactions"))
            if di_container is not None:
                for di in di_container.findall(_tag("drug-interaction")):
                    partner_id = _text(di, "drugbank-id")
                    partner_name = _text(di, "name")
                    description = _text(di, "description")
                    drug_interactions.append({
                        "drugbank-id": partner_id,  # preserve hyphenated key
                        "name": partner_name,
                        "description": description,
                    })

            records.append({
                "drugbank_id": primary_id,
                "name": name,
                "synonyms": synonyms,
                "drug_interactions": drug_interactions,
            })
            elem.clear()  # free memory

        return pd.DataFrame(records)

    def _query_drugbank_row(self, drug_name: str) -> Optional[Dict]:
        """Look up *drug_name* in the DrugBank index.

        Strategy (fastest-first):

        1. **Cache hit** — return immediately if already resolved.
        2. **Exact match** — O(1) lookup in ``_synonym_index`` using the
           lower-cased query string.
        3. **Fuzzy match** — ``difflib.get_close_matches`` over the flat
           ``_synonym_keys`` list (all ~140 k synonym strings).  This is
           O(N) but avoids DataFrame copies and per-row Python loops,
           making it roughly 5-10× faster than the previous approach.

        Returns the best-matching row as a dict, or ``None`` if no match
        exceeds the similarity threshold (0.6).
        """
        if drug_name in self._drugbank_cache:
            return self._drugbank_cache[drug_name]
        if not self._try_load_drugbank():
            self._drugbank_cache[drug_name] = None
            return None
        try:
            key = drug_name.lower()

            # --- Step 1: exact match (O(1)) ---
            row = self._synonym_index.get(key)
            if row is not None:
                self._drugbank_cache[drug_name] = row
                return row

            # --- Step 2: fuzzy match via get_close_matches ---
            import difflib
            matches = difflib.get_close_matches(
                key,
                self._synonym_keys,
                n=1,
                cutoff=0.6,
            )
            if not matches:
                self._drugbank_cache[drug_name] = None
                return None
            row = self._synonym_index[matches[0]]
            self._drugbank_cache[drug_name] = row
            return row
        except Exception as e:
            logger.debug(f"DrugBank query error for '{drug_name}': {e}")
            self._drugbank_cache[drug_name] = None
            return None

    def _check_drugbank_interactions(
        self,
        drugs: List[str],
    ) -> tuple[List[Dict[str, str]], Dict[str, str]]:
        """Look up interactions between *drugs* using the local DrugBank data.

        Returns:
            A 2-tuple of:
            - interactions: list of interaction dicts
              (keys: ``drugs``, ``description``, ``source``).
            - drugbank_ids: mapping of input drug name → resolved DrugBank ID
              (or ``"not found"`` when the drug could not be matched).
        """
        # Step 1: resolve each drug to its DrugBank row
        rows: Dict[str, Optional[Dict]] = {}
        drugbank_ids: Dict[str, str] = {}
        for drug in drugs:
            row = self._query_drugbank_row(drug)
            rows[drug] = row
            if row:
                drugbank_ids[drug] = row.get("drugbank_id", "unknown")
            else:
                drugbank_ids[drug] = "not found"

        # Step 2: build lookup structures for cross-filtering.
        # resolved_ids:    query_name → DrugBank ID (upper-cased)
        # resolved_names:  query_name → canonical name (lower-cased)
        # resolved_synonyms: query_name → set of all synonyms (lower-cased)
        resolved_ids: Dict[str, str] = {
            drug: row.get("drugbank_id", "").upper()
            for drug, row in rows.items()
            if row and row.get("drugbank_id")
        }
        resolved_names: Dict[str, str] = {
            drug: (row.get("name") or drug).lower()
            for drug, row in rows.items()
            if row
        }
        # Build a synonym set per drug for broader name matching.
        # DrugBank interaction entries may use a trade name or synonym rather
        # than the canonical name stored in row["name"].
        resolved_synonyms: Dict[str, set] = {}
        for drug, row in rows.items():
            if row is None:
                continue
            syns = row.get("synonyms") or []
            resolved_synonyms[drug] = {
                s.lower() for s in syns if isinstance(s, str)
            }
            # Always include the canonical name
            resolved_synonyms[drug].add(resolved_names.get(drug, drug.lower()))

        # Step 3: cross-filter interactions
        seen_pairs: set = set()  # de-duplicate symmetric pairs
        interactions: List[Dict[str, str]] = []

        for drug_a, row_a in rows.items():
            if row_a is None:
                continue
            raw_interactions = row_a.get("drug_interactions", []) or []
            for entry in raw_interactions:
                # Accept both "drugbank-id" (hyphenated, as in DrugBank XML)
                # and "drugbank_id" (underscored, used in some pre-processed
                # DataFrames).  See https://docs.drugbank.com/xml/#drug-interactions
                partner_id = (
                    str(entry.get("drugbank-id") or entry.get("drugbank_id") or "")
                ).upper()
                partner_name = str(entry.get("name", "")).lower()
                description = str(entry.get("description", ""))

                # Check whether the partner is one of the other queried drugs.
                # Match by DrugBank ID first (most reliable), then by name
                # against both the canonical name and all known synonyms.
                matched_drug_b: Optional[str] = None
                for drug_b, db_id in resolved_ids.items():
                    if drug_b == drug_a:
                        continue
                    id_match = bool(partner_id and partner_id == db_id)
                    name_match = bool(
                        partner_name
                        and partner_name in resolved_synonyms.get(drug_b, set())
                    )
                    if id_match or name_match:
                        matched_drug_b = drug_b
                        break

                if matched_drug_b is None:
                    continue

                # De-duplicate: (A, B) and (B, A) are the same pair
                pair_key = tuple(sorted([drug_a, matched_drug_b]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                interactions.append({
                    "drugs": f"{drug_a} + {matched_drug_b}",
                    "description": description[:600],
                    "source": "DrugBank (local)",
                })

        return interactions, drugbank_ids

    # ------------------------------------------------------------------
    # RxNorm fallback helpers
    # ------------------------------------------------------------------

    def _get_rxcui(self, drug_name: str) -> Optional[str]:
        """Resolve drug name to RxCUI (used only for the RxNorm fallback)."""
        if drug_name in self._rxcui_cache:
            return self._rxcui_cache[drug_name]
        url = (
            f"{_RXNORM_BASE}/rxcui.json?"
            f"name={urllib.parse.quote(drug_name)}&search=2"
        )
        data = _get_json(url, self.timeout_seconds)
        if not data:
            self._rxcui_cache[drug_name] = None
            return None
        ids = data.get("idGroup", {}).get("rxnormId", [])
        result = ids[0] if ids else None
        self._rxcui_cache[drug_name] = result
        return result

    def _check_rxnorm_interactions(
        self, rxcuis: List[str]
    ) -> List[Dict[str, str]]:
        """Query RxNorm interaction API for a list of RxCUIs (fallback)."""
        if len(rxcuis) < 2:
            return []
        cuis_param = urllib.parse.quote("+".join(rxcuis))
        url = (
            f"{_RXNORM_BASE}/interaction/list.json?"
            f"rxcuis={cuis_param}"
        )
        data = _get_json(url, self.timeout_seconds)
        if not data:
            return []
        interactions = []
        full_list = data.get("fullInteractionTypeGroup", [])
        for group in full_list:
            for itype in group.get("fullInteractionType", []):
                for pair in itype.get("interactionPair", []):
                    severity = pair.get("severity", "unknown")
                    description = pair.get("description", "")
                    drugs_involved = [
                        c.get("minConceptItem", {}).get("name", "")
                        for c in pair.get("interactionConcept", [])
                    ]
                    interactions.append({
                        "drugs": " + ".join(d for d in drugs_involved if d),
                        "severity": severity,
                        "description": description[:600],
                        "source": group.get("sourceDisclaimer", "RxNorm"),
                    })
        return interactions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Check drug interactions.

        Args:
            payload: Must contain either:
                - ``"drugs"`` (list[str]): list of drug names to check.
                - ``"drug1"`` and ``"drug2"`` (str): two drug names to check.
                  ``"drug3"`` and ``"drug4"`` are also accepted.

        Returns:
            ToolResult with keys:
                - ``drugs_queried``: list of drug names checked.
                - ``drugbank_ids``: resolved DrugBank IDs (or ``"not found"``).
                - ``interactions``: list of interaction dicts (may be empty).
                  Each dict has ``drugs``, ``description``, and ``source``
                  keys; the DrugBank path also includes ``drugbank_id`` of
                  the interacting partner.
                - ``summary``: plain-text summary.
                - ``source``: data source used (``"DrugBank (local)"`` or
                  ``"RxNorm Interaction API (fallback)"``).

        Raises:
            ValueError: If fewer than two drug names are provided.
        """
        try:
            # ---- Parse input ------------------------------------------------
            drugs: List[str] = []
            if "drugs" in payload:
                raw = payload["drugs"]
                if isinstance(raw, str):
                    drugs = [d.strip() for d in raw.split(",") if d.strip()]
                else:
                    drugs = [str(d).strip() for d in raw if str(d).strip()]
            else:
                for key in ("drug1", "drug2", "drug3", "drug4"):
                    val = payload.get(key, "").strip()
                    if val:
                        drugs.append(val)

            if len(drugs) < 2:
                raise ValueError(
                    "At least two drug names are required. "
                    "Provide 'drugs' as a list, or 'drug1' and 'drug2' as separate fields."
                )

            # ---- Primary path: DrugBank local file --------------------------
            if self._try_load_drugbank():
                interactions, drugbank_ids = self._check_drugbank_interactions(drugs)
                data_source = "DrugBank (local)"
            else:
                # ---- Fallback: RxNorm API -----------------------------------
                rxcuis: List[str] = []
                drugbank_ids: Dict[str, str] = {}
                for drug in drugs:
                    cui = self._get_rxcui(drug)
                    if cui:
                        rxcuis.append(cui)
                        drugbank_ids[drug] = cui  # store RxCUI under same key
                    else:
                        drugbank_ids[drug] = "not found"
                interactions = self._check_rxnorm_interactions(rxcuis)
                data_source = "RxNorm Interaction API (fallback)"

            # ---- Build summary ----------------------------------------------
            if interactions:
                summary = (
                    f"{len(interactions)} interaction(s) found among "
                    f"{', '.join(drugs)}."
                )
            else:
                summary = (
                    f"No known interactions found between {', '.join(drugs)} "
                    f"in the {data_source} database. "
                    "This does not guarantee safety — always consult clinical "
                    "resources for comprehensive interaction checking."
                )

            return ToolResult(
                name=self.name,
                payload={
                    "drugs_queried": drugs,
                    "drugbank_ids": drugbank_ids,
                    "interactions": interactions[:10],  # cap at 10
                    "summary": summary,
                    "source": data_source,
                },
                render_fields=["interactions"],
            )
        except Exception:
            traceback.print_exc()
            logger.exception("DrugInteractionCheckTool failed")
            raise

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="drugs",
                type="list[string]",
                required=False,
                description=(
                    "List of drug names to check for interactions. Provide 2 or more "
                    "drug names, e.g. [\"warfarin\", \"aspirin\", \"ibuprofen\"]. "
                    "Use this OR the drug1/drug2 fields, not both."
                ),
            ),
            ToolParam(
                name="drug1",
                type="string",
                required=False,
                description=(
                    "First drug name. Use together with 'drug2' (and optionally "
                    "'drug3', 'drug4'). Ignored if 'drugs' list is provided."
                ),
            ),
            ToolParam(
                name="drug2",
                type="string",
                required=False,
                description=(
                    "Second drug name. Use together with 'drug1'. "
                    "Ignored if 'drugs' list is provided."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"drugs": ["warfarin", "aspirin"]} '
            'or {"drug1": "metformin", "drug2": "ibuprofen"}'
        )


# ===========================================================================
# UMLSConceptLookupTool
# ===========================================================================
@dataclass
class UMLSConceptLookupTool:
    """Look up standardised medical concepts in UMLS / MedlinePlus.

    When a UMLS API key is available (``$UMLS_API_KEY``), queries the full
    UMLS REST API for concept definitions, semantic types, and relations.

    Falls back to the NLM MedlinePlus Connect API (no key required) for
    consumer-friendly definitions and related information.

    Args:
        name: Tool name used in agent prompts and the tool registry.
        description: Human-readable description injected into the system prompt.
        umls_api_key: UMLS API key. If not provided, reads ``$UMLS_API_KEY``.
        timeout_seconds: HTTP request timeout.
    """

    name: str = "umls_concept_lookup"
    description: str = (
        "Look up a medical term in UMLS to get its official definition, semantic type, and related concepts."
    )
    umls_api_key: str = ""
    timeout_seconds: int = 20
    max_results: int = 3

    def __post_init__(self) -> None:
        if not self.umls_api_key:
            self.umls_api_key = os.environ.get("UMLS_API_KEY", "")

    def _get_umls_ticket(self) -> Optional[str]:
        """Obtain a single-use UMLS service ticket."""
        # Step 1: TGT (Ticket-Granting Ticket)
        tgt_url = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        data = urllib.parse.urlencode({"apikey": self.umls_api_key}).encode()
        try:
            req = urllib.request.Request(tgt_url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
            # Extract TGT URL from response
            match = re.search(r'action="([^"]+)"', body)
            if not match:
                return None
            tgt_endpoint = match.group(1)
        except Exception as e:
            logger.debug(f"UMLS TGT error: {e}")
            return None

        # Step 2: Service Ticket
        st_data = urllib.parse.urlencode(
            {"service": "http://umlsks.nlm.nih.gov"}
        ).encode()
        try:
            req2 = urllib.request.Request(tgt_endpoint, data=st_data, method="POST")
            with urllib.request.urlopen(req2, timeout=self.timeout_seconds) as resp2:
                return resp2.read().decode("utf-8").strip()
        except Exception as e:
            logger.debug(f"UMLS ST error: {e}")
            return None

    def _search_umls(self, term: str) -> Dict[str, Any]:
        """Search UMLS for a term and return concept info."""
        ticket = self._get_umls_ticket()
        if not ticket:
            return {}

        search_url = (
            f"{_UMLS_BASE}/search/current?"
            f"string={urllib.parse.quote(term)}&ticket={ticket}&pageSize=5"
        )
        data = _get_json(search_url, self.timeout_seconds)
        if not data:
            return {}

        results = data.get("result", {}).get("results", [])
        if not results:
            return {}

        best = results[0]
        cui = best.get("ui", "")
        name = best.get("name", "")
        root_source = best.get("rootSource", "")

        result: Dict[str, Any] = {
            "cui": cui,
            "preferred_name": name,
            "source": root_source,
        }

        # Get definitions
        if cui:
            ticket2 = self._get_umls_ticket()
            if ticket2:
                def_url = (
                    f"{_UMLS_BASE}/content/current/CUI/{cui}/definitions?"
                    f"ticket={ticket2}&pageSize={self.max_results}"
                )
                def_data = _get_json(def_url, self.timeout_seconds)
                if def_data:
                    defs = def_data.get("result", [])
                    result["definitions"] = [
                        {"source": d.get("rootSource", ""), "value": d.get("value", "")[:400]}
                        for d in defs[:self.max_results]
                    ]

        return result

    def _search_medlineplus(self, term: str) -> Dict[str, Any]:
        """Query NLM MedlinePlus Connect (no key required)."""
        url = (
            f"{_MEDLINEPLUS_BASE}?"
            f"mainSearchCriteria.v.cs=2.16.840.1.113883.6.177"
            f"&mainSearchCriteria.v.dn={urllib.parse.quote(term)}"
            f"&knowledgeResponseType=application/json"
        )
        data = _get_json(url, self.timeout_seconds)
        if not data:
            return {}

        feed = data.get("feed", {})
        entries = feed.get("entry", [])
        if not entries:
            return {}

        result: Dict[str, Any] = {
            "preferred_name": term,
            "source": "MedlinePlus",
            "related_topics": [],
        }
        for entry in entries[:5]:
            title = entry.get("title", {})
            title_text = title.get("_value", "") if isinstance(title, dict) else str(title)
            summary = entry.get("summary", {})
            summary_text = (
                summary.get("_value", "")[:400]
                if isinstance(summary, dict)
                else str(summary)[:400]
            )
            result["related_topics"].append({
                "title": title_text,
                "summary": summary_text,
            })

        return result

    def _search_rxnorm_concept(self, term: str) -> Dict[str, Any]:
        """Lightweight fallback: use RxNorm to get concept info for drug terms."""
        url = (
            f"{_RXNORM_BASE}/rxcui.json?"
            f"name={urllib.parse.quote(term)}&search=2"
        )
        data = _get_json(url, self.timeout_seconds)
        if not data:
            return {}
        ids = data.get("idGroup", {}).get("rxnormId", [])
        if not ids:
            return {}
        return {
            "preferred_name": term,
            "rxcui": ids[0],
            "source": "RxNorm",
            "note": "Drug concept found in RxNorm. For full UMLS info, provide UMLS_API_KEY.",
        }

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Look up a medical concept in UMLS / MedlinePlus.

        Args:
            payload: Must contain ``"term"`` (str): the medical term to look up.
                Optionally ``"include_relations"`` (bool, default False).

        Returns:
            ToolResult with keys:
                - ``term``: the queried term.
                - ``cui``: UMLS Concept Unique Identifier (if available).
                - ``preferred_name``: standardised preferred name.
                - ``definitions``: list of definitions from different sources.
                - ``source``: data source used.

        Raises:
            ValueError: If ``term`` is missing or empty.
        """
        term = str(payload.get("term", "")).strip()
        if not term:
            raise ValueError("'term' field is required and must not be empty.")

        result: Dict[str, Any] = {"term": term}

        # Try UMLS first (requires API key)
        _umls_render = ["term", "cui", "preferred_name", "source", "definitions"]

        if self.umls_api_key:
            umls_result = self._search_umls(term)
            if umls_result:
                result.update(umls_result)
                return ToolResult(name=self.name, payload=result, render_fields=_umls_render)

        # Fallback: MedlinePlus
        ml_result = self._search_medlineplus(term)
        if ml_result:
            result.update(ml_result)
            return ToolResult(name=self.name, payload=result, render_fields=_umls_render)

        # Fallback: RxNorm (for drug terms)
        rx_result = self._search_rxnorm_concept(term)
        if rx_result:
            result.update(rx_result)
            return ToolResult(name=self.name, payload=result, render_fields=_umls_render)

        result["note"] = (
            f"No concept found for '{term}'. "
            "Set UMLS_API_KEY for full UMLS access, or try a more specific term."
        )
        result["source"] = "none"
        return ToolResult(name=self.name, payload=result, render_fields=_umls_render)

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="term",
                type="string",
                required=True,
                description=(
                    "Medical term, disease name, symptom, procedure, or drug to look up. "
                    "Examples: \"hyponatremia\", \"myocardial infarction\", \"SGLT2 inhibitor\"."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return '{"term": "hyponatremia"}'
