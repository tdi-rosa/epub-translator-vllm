#!/usr/bin/env python3
"""End-to-end EPUB translator: extract -> translate every page -> repack.

Requires a running OpenAI-compatible chat-completions endpoint (see
serve_vllm.sh). Point it elsewhere with TRANSLATOR_API_BASE_URL /
TRANSLATOR_MODEL_NAME if not using the defaults.
"""

import argparse
import asyncio
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from colorama import init as colorama_init

from translator.epub_pipeline import translate_directory

colorama_init(autoreset=True)


def extract_epub(epub_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(epub_path) as zf:
        zf.extractall(dest)


def repack_epub(source_dir: Path, epub_path: Path) -> None:
    """Zip `source_dir` back into a valid EPUB.

    The EPUB spec requires `mimetype` to be the first entry and stored
    uncompressed; everything else is regular deflate.
    """
    mimetype = source_dir / "mimetype"

    with zipfile.ZipFile(epub_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if mimetype.exists():
            zf.write(mimetype, "mimetype", compress_type=zipfile.ZIP_STORED)

        for path in sorted(source_dir.rglob("*")):
            if path.is_dir() or path == mimetype:
                continue
            zf.write(path, path.relative_to(source_dir))


async def run(input_epub: Path, output_epub: Path, concurrency: int) -> None:
    with TemporaryDirectory(prefix="epub-translate-") as tmp:
        workdir = Path(tmp)

        print(f"Extracting {input_epub} ...")
        extract_epub(input_epub, workdir)

        print(f"Translating (concurrency={concurrency}) ...")
        await translate_directory(workdir, concurrency=concurrency)

        print(f"Repacking into {output_epub} ...")
        repack_epub(workdir, output_epub)

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_epub", type=Path, help="Source .epub file (English)")
    parser.add_argument("output_epub", type=Path, help="Destination .epub file (translated)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of paragraphs translated concurrently against vLLM (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_epub.exists():
        raise SystemExit(f"Input file not found: {args.input_epub}")
    if not args.input_epub.suffix == ".epub":
        raise SystemExit(f"Input file must be a .epub: {args.input_epub}")

    asyncio.run(run(args.input_epub, args.output_epub, args.concurrency))


if __name__ == "__main__":
    main()
