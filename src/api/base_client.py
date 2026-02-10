"""Base API client with rate limiting for Polymarket APIs"""
import time
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import api_config

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_10s: int
    delay_seconds: float


class RateLimiter:
    """Simple rate limiter with configurable delay"""

    def __init__(self, delay_seconds: float = 0.2):
        self.delay_seconds = delay_seconds
        self.last_request_time: float = 0

    def wait(self) -> None:
        """Wait if necessary to respect rate limit"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.delay_seconds:
            sleep_time = self.delay_seconds - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()


class BaseClient:
    """Base HTTP client with rate limiting and retry logic"""

    def __init__(
        self,
        base_url: str,
        delay_seconds: float = api_config.DEFAULT_DELAY,
        timeout: int = api_config.REQUEST_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limiter = RateLimiter(delay_seconds)
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a session with retry logic"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make an HTTP request with rate limiting"""
        self.rate_limiter.wait()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            if response.status_code == 429:
                logger.warning("Rate limited, backing off...")
                time.sleep(5)
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            return None
        except ValueError as e:
            logger.error(f"JSON decode error for {url}: {e}")
            return None

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make a GET request"""
        return self._make_request("GET", endpoint, params=params, headers=headers)

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make a POST request"""
        return self._make_request(
            "POST", endpoint, params=params, data=data, headers=headers
        )

    def get_paginated(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        page_size: int = api_config.DEFAULT_PAGE_SIZE,
        max_pages: Optional[int] = None,
        offset_key: str = "offset",
        limit_key: str = "limit",
        results_key: Optional[str] = None,
    ) -> list:
        """
        Fetch all pages of a paginated endpoint.

        Args:
            endpoint: API endpoint
            params: Query parameters
            page_size: Number of items per page
            max_pages: Maximum pages to fetch (None = all)
            offset_key: Parameter name for offset
            limit_key: Parameter name for limit
            results_key: Key in response containing results (None = response is list)

        Returns:
            List of all items across pages
        """
        all_results = []
        params = params or {}
        params[limit_key] = page_size
        offset = 0
        page_count = 0

        while True:
            params[offset_key] = offset
            response = self.get(endpoint, params)

            if response is None:
                break

            # Extract results from response
            if results_key:
                items = response.get(results_key, [])
            elif isinstance(response, list):
                items = response
            else:
                items = []

            if not items:
                break

            all_results.extend(items)
            page_count += 1

            # Check if we've reached max pages
            if max_pages and page_count >= max_pages:
                break

            # Check if we got fewer items than requested (last page)
            if len(items) < page_size:
                break

            offset += page_size

        return all_results

    def close(self) -> None:
        """Close the session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
