from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig, OmegaConf
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm
import re


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


def cfg_select(config: DictConfig, key: str, default=None):
    """Safely read nested config values."""
    try:
        value = OmegaConf.select(config, key, default=default)
        return value
    except Exception:
        return default


def keyword_in_text(text: str, keyword: str) -> bool:
    """
    Match keyword in title/abstract.

    For short abbreviations such as CT, MRI, DWI, ADC, SAM, use word-boundary matching
    to avoid false positives such as 'object' containing 'ct' or 'sample' containing 'sam'.
    For longer phrases, use substring matching.
    """
    keyword = str(keyword).strip().lower()
    if not keyword:
        return False

    # Short pure alphanumeric terms should be matched as independent tokens.
    # Examples: CT, MRI, DWI, ADC, SAM.
    if re.fullmatch(r"[a-z0-9]+", keyword) and len(keyword) <= 4:
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        return re.search(pattern, text) is not None

    return keyword in text


def get_paper_text(paper) -> str:
    """Collect searchable text from a paper object."""
    fields = []

    for name in ["title", "abstract", "summary", "description"]:
        if isinstance(paper, dict):
            value = paper.get(name, "")
        else:
            value = getattr(paper, name, "")
        if value:
            fields.append(str(value))

    return " ".join(fields).lower()


def keyword_filter_papers(papers, config: DictConfig):
    """
    Hard filter papers for medical imaging / stroke / segmentation topics.

    It reads:
      filters.include_keywords
      filters.exclude_keywords

    from CUSTOM_CONFIG.
    """
    filters = cfg_select(config, "filters", None)
    if filters is None:
        logger.info("No filters section found in config. Keyword filtering skipped.")
        return papers

    include_keywords = cfg_select(config, "filters.include_keywords", []) or []
    exclude_keywords = cfg_select(config, "filters.exclude_keywords", []) or []

    include_keywords = [str(k).strip().lower() for k in include_keywords if str(k).strip()]
    exclude_keywords = [str(k).strip().lower() for k in exclude_keywords if str(k).strip()]

    if not include_keywords and not exclude_keywords:
        logger.info("No include/exclude keywords configured. Keyword filtering skipped.")
        return papers

    filtered = []
    dropped_by_include = 0
    dropped_by_exclude = 0

    for p in papers:
        text = get_paper_text(p)

        include_hit = True
        if include_keywords:
            include_hit = any(keyword_in_text(text, keyword) for keyword in include_keywords)

        if not include_hit:
            dropped_by_include += 1
            continue

        exclude_hit = False
        if exclude_keywords:
            exclude_hit = any(keyword_in_text(text, keyword) for keyword in exclude_keywords)

        if exclude_hit:
            dropped_by_exclude += 1
            continue

        filtered.append(p)

    logger.info(
        f"Keyword filter kept {len(filtered)} / {len(papers)} papers. "
        f"Dropped by include filter: {dropped_by_include}. "
        f"Dropped by exclude filter: {dropped_by_exclude}."
    )

    if len(filtered) > 0:
        sample_titles = []
        for p in filtered[:5]:
            if isinstance(p, dict):
                sample_titles.append(str(p.get("title", "")))
            else:
                sample_titles.append(str(getattr(p, "title", "")))
        logger.info("Keyword filter sample kept papers:\n" + "\n".join(sample_titles))

    return filtered


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config

        include_path = cfg_select(config, "zotero.include_path", None)
        ignore_path = cfg_select(config, "zotero.ignore_path", None)

        self.include_path_patterns = normalize_path_patterns(include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(ignore_path, "ignore_path")

        executor_sources = cfg_select(config, "executor.source", None)
        if executor_sources is None:
            raise ValueError(
                "Missing config.executor.source. "
                'Example: executor: { source: ["arxiv"] }'
            )

        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in executor_sources
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")

        zot = zotero.Zotero(self.config.zotero.user_id, "user", self.config.zotero.api_key)

        collections = zot.everything(zot.collections())
        collections = {c["key"]: c for c in collections}

        # Only these Zotero item types are used as seed papers.
        # Thesis / book / webpage items are ignored by the original project design.
        corpus = zot.everything(zot.items(itemType="conferencePaper || journalArticle || preprint"))

        # Keep only items with abstracts.
        corpus = [
            c for c in corpus
            if c.get("data", {}).get("abstractNote", "") not in ("", None)
        ]

        def get_collection_path(col_key: str) -> str:
            if col_key not in collections:
                return ""

            parent = collections[col_key]["data"].get("parentCollection")
            current_name = collections[col_key]["data"].get("name", "")

            if parent:
                parent_path = get_collection_path(parent)
                if parent_path:
                    return parent_path + "/" + current_name
                return current_name

            return current_name

        for c in corpus:
            collection_keys = c.get("data", {}).get("collections", []) or []
            paths = [get_collection_path(col) for col in collection_keys]
            paths = [p for p in paths if p]
            c["paths"] = paths

        logger.info(f"Fetched {len(corpus)} zotero papers")

        return [
            CorpusPaper(
                title=c["data"].get("title", ""),
                abstract=c["data"].get("abstractNote", ""),
                added_date=datetime.strptime(c["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"),
                paths=c.get("paths", []),
            )
            for c in corpus
        ]

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]

        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]

        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus))) if len(corpus) > 0 else []
            samples = "\n".join([c.title + " - " + "\n".join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")

        return corpus

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)

        send_empty = bool(cfg_select(self.config, "executor.send_empty", False))

        if len(corpus) == 0:
            logger.error(
                f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}"
            )

            if send_empty:
                logger.info("send_empty=True. Sending an empty diagnostic email.")
                email_content = render_email([])
                send_email(self.config, email_content)
                logger.info("Empty diagnostic email sent successfully.")

            return

        all_papers = []

        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()

            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue

            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)

        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")

        # Hard keyword filtering before reranking.
        # This removes non-medical CV papers such as remote sensing, style transfer,
        # industrial defect detection, robotics, and generic agent papers.
        all_papers = keyword_filter_papers(all_papers, self.config)
        logger.info(f"Total {len(all_papers)} papers after keyword filtering")

        reranked_papers = []

        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[: self.config.executor.max_paper_num]

            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)

        elif not send_empty:
            logger.info("No papers left after keyword filtering. No email will be sent.")
            return

        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
