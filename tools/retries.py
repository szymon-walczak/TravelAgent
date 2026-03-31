import asyncio
import time
from typing import Awaitable, Callable, TypeVar

from config import MAX_MODEL_RETRIES, MODEL_RETRY_DELAY_SECONDS

T = TypeVar("T")


def is_transient_model_error(error: Exception) -> bool:
    message = str(error).lower()
    transient_markers = (
        "503",
        "unavailable",
        "high demand",
        "service is currently unavailable",
    )
    return any(marker in message for marker in transient_markers)


def invoke_with_retry(operation: Callable[[], T], label: str) -> T:
    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            return operation()
        except Exception as error:
            if attempt < MAX_MODEL_RETRIES and is_transient_model_error(error):
                print(
                    f"Retrying {label} after transient model error "
                    f"({attempt}/{MAX_MODEL_RETRIES}): {error}"
                )
                time.sleep(MODEL_RETRY_DELAY_SECONDS)
                continue
            raise


async def invoke_with_retry_async(operation: Callable[[], Awaitable[T]], label: str) -> T:
    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            return await operation()
        except Exception as error:
            if attempt < MAX_MODEL_RETRIES and is_transient_model_error(error):
                print(
                    f"Retrying {label} after transient model error "
                    f"({attempt}/{MAX_MODEL_RETRIES}): {error}"
                )
                await asyncio.sleep(MODEL_RETRY_DELAY_SECONDS)
                continue
            raise
