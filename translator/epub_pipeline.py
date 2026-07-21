"""Walks an extracted EPUB directory and translates every (X)HTML page in place."""

import asyncio
import time
from datetime import timedelta
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from colorama import Fore

from .llm_client import translate_paragraph

# Inline tags stripped before translation so the model only ever sees plain
# text (styling/links are meaningless to the translation and confuse the
# structured-output regex).
INLINE_TAGS_TO_STRIP = ["em", "strong", "b", "i", "span", "a", "sup"]
TRANSLATABLE_SELECTOR = "p, h1, h2, h3, h4, h5, h6, li"
HTML_SUFFIXES = (".html", ".xhtml", ".htm")


def _parser_for(path: Path) -> str:
    # EPUB content documents are XHTML; parsing/serializing them as XML
    # keeps the file well-formed (self-closing tags, etc).
    return "lxml-xml" if path.suffix == ".xhtml" else "lxml"


def _strip_inline_tags(element) -> None:
    for tag in reversed(list(element.find_all(INLINE_TAGS_TO_STRIP))):
        if tag.parent is not None:
            tag.unwrap()


def _element_text(element) -> str:
    # get_text(strip=True) strips each text node individually with no
    # separator, so adjacent inline tags glue words together, e.g.
    # "Hello <b>world</b>" -> "Helloworld". Join with a space and collapse
    # whitespace instead.
    return " ".join(element.get_text(separator=" ").split())


async def _translate_element(client, semaphore, element):
    async with semaphore:
        result = await translate_paragraph(client, _element_text(element))
        translation = result["choices"][0]["message"]["content"]
        tokens = result["usage"]["total_tokens"]
        return translation, tokens


async def translate_file(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, path: Path) -> tuple[int, int]:
    """Translate every paragraph-like element of one file in place.

    Returns (elements_translated, tokens_used).
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), _parser_for(path))
    elements = soup.select(TRANSLATABLE_SELECTOR)
    for element in elements:
        _strip_inline_tags(element)

    # All paragraphs of a file are dispatched at once; `semaphore` is what
    # actually bounds how many are in flight against vLLM at any time.
    results = await asyncio.gather(
        *(_translate_element(client, semaphore, element) for element in elements)
    )
    for element, (translation, _tokens) in zip(elements, results):
        element.string = translation

    path.write_text(str(soup), encoding="utf-8")
    return len(elements), sum(tokens for _, tokens in results)


def _count_translatable_elements(path: Path) -> int:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), _parser_for(path))
    return len(soup.select(TRANSLATABLE_SELECTOR))


async def translate_directory(root: Path, concurrency: int = 8) -> None:
    """Recursively translate every HTML/XHTML page found under `root`."""
    html_files = sorted(p for p in root.rglob("*") if p.suffix in HTML_SUFFIXES)
    if not html_files:
        raise FileNotFoundError(f"No .html/.xhtml files found under {root}")

    total_elements = sum(_count_translatable_elements(p) for p in html_files)
    semaphore = asyncio.Semaphore(concurrency)

    done = 0
    total_tokens = 0
    start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        for path in html_files:
            count, tokens = await translate_file(client, semaphore, path)
            done += count
            total_tokens += tokens

            elapsed = time.perf_counter() - start
            eta = (elapsed / done) * total_elements if done else 0
            tokens_per_min = total_tokens / (elapsed / 60) if elapsed else 0
            print(
                f"\r{Fore.GREEN}{(done / total_elements) * 100:6.2f}%  "
                f"elapsed: {timedelta(seconds=int(elapsed))}  "
                f"ETA: {timedelta(seconds=int(eta))}  "
                f"tokens: {total_tokens:_}  "
                f"{Fore.RED}tokens/min: {round(tokens_per_min):_}",
                end="",
                flush=True,
            )
    print()
