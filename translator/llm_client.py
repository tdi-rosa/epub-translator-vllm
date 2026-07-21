"""Thin async client for an OpenAI-compatible chat-completions endpoint (vLLM)."""

import os

import httpx

API_BASE_URL = os.environ.get("TRANSLATOR_API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("TRANSLATOR_MODEL_NAME", "qwen3.5")

SYSTEM_PROMPT = """Tu es un traducteur expert de l'anglais vers le français. Ton objectif est de fournir une traduction naturelle, fluide et parfaitement idiomatique.

### Directives de traduction :
- Ne traduis pas littéralement : interprète le contexte pour retranscrire le sens de la manière la plus fidèle et naturelle en français.
- Adapte le ton en fonction du texte source (titre, paragraphe, dialogue, etc.).
- N'hésite pas à reformuler, rédiger, pour mieux faire passer le sens général du paragraphe.

### Contraintes strictes de format :
- Retourne uniquement le texte traduit.
- Ne formule aucun commentaire, aucune salutation, ni aucune explication avant ou après le texte.
- Si l'utilisateur n'envoie aucun texte, ne renvoie absolument rien (laisse la réponse vide).
- Si le texte reçu est une ligne simple, une référence (ex: "Ch. 4", "p. 120"), une mention d'annexe (ex: "Appendix A"), ou une simple note bibliographique, ne cherche pas à la traduire ou à la modifier : retourne-la exactement telle qu'elle t'a été envoyée.
"""

# Restricts generation to a fixed character set (Latin script + common punctuation).
# Structured output at the character-class level avoids the raw model wandering into
# stray unicode/control characters that break the EPUB's XHTML on repack.
TRANSLATION_CHARSET_REGEX = (
    r"^[a-zA-Z0-9À-ÿœŒæÆ\s.,'’\"?!;:()\[\]{}«»<>/\\|@#*+=_~$%&`.…\-–—]*$"
)


def build_request(paragraph: str) -> dict:
    return {
        # Pin the model name explicitly so vLLM's routing can't silently
        # fall back to a different model behind the same endpoint.
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": paragraph},
        ],
        "temperature": 0.2,
        "structured_outputs": {"regex": TRANSLATION_CHARSET_REGEX},
        # Qwen3.5 emits a <think>...</think> block by default; disabling it
        # keeps latency reasonable on short paragraphs.
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def translate_paragraph(client: httpx.AsyncClient, paragraph: str) -> dict:
    """Send one paragraph to the model and return the parsed JSON response."""
    response = await client.post(
        f"{API_BASE_URL}/chat/completions",
        json=build_request(paragraph),
        timeout=120,
    )
    response.raise_for_status()
    return response.json()
