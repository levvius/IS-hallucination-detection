"""
Response caching system using TTL cache.

Caches classification results to avoid duplicate processing of the same text.
Uses MD5 hashing for cache keys and TTL (time-to-live) for automatic expiration.
"""
import hashlib
import logging
from typing import Optional, Dict, Any
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Cache configuration: 100 entries with 5-minute TTL
response_cache = TTLCache(maxsize=100, ttl=300)


def get_cache_key(text: str) -> str:
    """
    Generate a cache key from input text using MD5 hash.

    Args:
        text: Input text to hash

    Returns:
        MD5 hash of the text as hex string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_cached_result(text: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached classification result for the given text.

    Args:
        text: Input text to look up

    Returns:
        Cached result dictionary if found, None otherwise
    """
    key = get_cache_key(text)
    result = response_cache.get(key)

    if result is not None:
        logger.info(f"Cache hit for key: {key[:8]}...")

    return result


def cache_result(text: str, result: Dict[str, Any]) -> None:
    """
    Cache a classification result for the given text.

    Args:
        text: Input text that was classified
        result: Classification result to cache
    """
    key = get_cache_key(text)
    response_cache[key] = result
    logger.debug(f"Cached result for key: {key[:8]}...")


def clear_cache() -> None:
    """Clear all cached results."""
    response_cache.clear()
    logger.info("Cache cleared")


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache state.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "size": len(response_cache),
        "maxsize": response_cache.maxsize,
        "ttl": response_cache.ttl,
        "currsize": response_cache.currsize
    }
