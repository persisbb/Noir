
import logging
import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

MAX_ATTEMPTS = 3


class LLMUnavailableError(Exception):
    """Raised when Gemini cannot complete a request."""


def generate_text(prompt: str) -> str:
    """Generate text with bounded retries and error handling."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise LLMUnavailableError(
            "GEMINI_API_KEY is not configured."
        )

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30000),
        )

        config = types.GenerateContentConfig(
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            )
        )

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=config,
                )

                answer = response.text

                if not answer or not answer.strip():
                    raise LLMUnavailableError(
                        "Gemini returned an empty response."
                    )

                return answer.strip()

            except (errors.ServerError, errors.ClientError) as exc:
                status = getattr(exc, "code", None)

                # Retry temporary server failures and rate limits.
                if status not in (429, 500, 502, 503, 504):
                    raise LLMUnavailableError(
                        "Gemini request failed."
                    ) from exc

                if attempt == MAX_ATTEMPTS - 1:
                    break

                delay = min(
                    2 ** (attempt + 1) + random.uniform(0, 1),
                    10,
                )

                logger.warning(
                    "Gemini request failed (HTTP %s). "
                    "Retrying in %.1f seconds.",
                    status,
                    delay,
                )

                time.sleep(delay)

        raise LLMUnavailableError(
            "Gemini is temporarily unavailable."
        )

    except LLMUnavailableError:
        raise

    except Exception as exc:
        logger.exception("Unexpected Gemini service failure")

        raise LLMUnavailableError(
            "Unable to generate an answer right now."
        ) from exc

    finally:
        if "client" in locals():
            client.close()
