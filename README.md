# EPUB Translator — vLLM + Qwen3.5

Self-hosted EN→FR book translation with grammar-constrained decoding: one
command extracts an EPUB, translates every paragraph concurrently, and
repacks a valid EPUB — no cloud API, no data leaving the machine.

Built to translate books for my own reading list, and to work through the
practical side of running a local LLM as a real pipeline rather than a demo:
constraining generation with a regex grammar so the model can't corrupt the
markup, disabling Qwen3.5's `<think>` mode to keep per-paragraph latency
sane, and bounding async concurrency with a semaphore instead of firing
everything at vLLM at once.

## How it works

```
book.epub ──extract──▶ workdir/*.xhtml ──translate──▶ workdir/*.xhtml ──repack──▶ book_fr.epub
                              │
                              ▼
                    BeautifulSoup selects every
                    p / h1-h6 / li element, sends
                    each to vLLM concurrently
                    (bounded by an asyncio.Semaphore)
```

- **`serve_vllm.sh`** — starts vLLM in Docker with an OpenAI-compatible API,
  Qwen3.5's XML tool-call/reasoning parsers, and a `guidance` structured-output
  backend.
- **`translator/llm_client.py`** — builds the chat-completion request: a
  translation system prompt, `temperature=0.2`, `enable_thinking=False`
  (Qwen3.5 emits a `<think>` block by default; skipping it matters a lot on
  short paragraphs), and a **regex-constrained structured output** so the
  model can only emit Latin-script text + standard punctuation — this is
  what stops occasional stray Unicode/control characters from corrupting the
  XHTML on repack.
- **`translator/epub_pipeline.py`** — walks the extracted EPUB, strips inline
  styling tags (`em`, `strong`, `span`, `a`, ...) so the model only sees plain
  text, fires all paragraphs of a file concurrently via `asyncio.gather` (rate
  limited by a semaphore against vLLM), and writes the translated XHTML back
  in place. Prints live progress: %, ETA, tokens used, tokens/min.
- **`translate_epub.py`** — the CLI entry point: extracts the EPUB to a temp
  directory, runs the pipeline, repacks it (`mimetype` stored first and
  uncompressed, per the EPUB spec), cleans up.

## Requirements

- Docker + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (a CUDA GPU with enough VRAM for a 9B AWQ model, ~7-8 GB)
- Python 3.11+

## Usage

```bash
pip install -r requirements.txt

# 1. Start the model server (leave running in its own terminal)
./serve_vllm.sh

# 2. Translate a book
python translate_epub.py book_en.epub book_fr.epub

# Optional: tune how many paragraphs are in flight against vLLM at once
python translate_epub.py book_en.epub book_fr.epub --concurrency 16
```

`serve_vllm.sh` and `translator/llm_client.py` read a few environment
variables if you want to point at a different cache dir, model, port, or
endpoint:

| Variable                    | Default                                 |
| ---------------------------- | ---------------------------------------- |
| `HF_CACHE_DIR`               | `$HOME/.cache/huggingface`               |
| `MODEL`                      | `QuantTrio/Qwen3.5-9B-AWQ`                |
| `SERVED_MODEL_NAME`          | `qwen3.5`                                |
| `PORT`                       | `8000`                                   |
| `MAX_MODEL_LEN`              | `32768`                                  |
| `GPU_MEM_UTIL`               | `0.90`                                   |
| `TRANSLATOR_API_BASE_URL`    | `http://localhost:8000/v1`               |
| `TRANSLATOR_MODEL_NAME`      | `qwen3.5`                                |

## Notes / limitations

- The translation direction and system prompt (EN→FR) are hardcoded — this
  was built for a specific personal use case, not as a general translation
  service. Making the target language a CLI flag would be the natural next
  step.
- EPUB structural files (`content.opf`, `nav.xhtml`, `toc.ncx`, CSS, images,
  ...) are left untouched — only content pages with translatable text nodes
  are rewritten.
- No retry/backoff around the vLLM calls: if a request fails, the run fails.
  Fine for a controlled local pipeline processing one book at a time.

## License

MIT — see [LICENSE](LICENSE).
