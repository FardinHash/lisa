import time
from collections import defaultdict
from typing import Callable, Dict, List

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.enabled = settings.rate_limit_enabled
        self.calls = settings.rate_limit_calls
        self.period = settings.rate_limit_period
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def _get_client_id(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _clean_old_requests(self, client_id: str, current_time: float) -> None:
        cutoff_time = current_time - self.period
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] if req_time > cutoff_time
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        client_id = self._get_client_id(request)
        current_time = time.time()

        self._clean_old_requests(client_id, current_time)

        if len(self.requests[client_id]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.calls} requests per {self.period} seconds.",
                headers={"Retry-After": str(self.period)},
            )

        self.requests[client_id].append(current_time)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(
            self.calls - len(self.requests[client_id])
        )
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.period))

        return response
