import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MonitoringService:
    def __init__(self):
        self.metrics: Dict[str, Any] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "errors": 0}
        )
        self.start_time = time.time()

    def record_request(
        self, endpoint: str, duration: float, success: bool = True
    ) -> None:
        metric = self.metrics[endpoint]
        metric["count"] += 1
        metric["total_time"] += duration
        if not success:
            metric["errors"] += 1

        logger.info(
            f"Request to {endpoint} completed in {duration:.3f}s | Success: {success}"
        )

    def record_llm_call(
        self, model: str, tokens: int, duration: float, cost: Optional[float] = None
    ) -> None:
        key = f"llm_{model}"
        metric = self.metrics[key]
        metric["count"] += 1
        metric["total_time"] += duration
        metric["total_tokens"] = metric.get("total_tokens", 0) + tokens

        if cost:
            metric["total_cost"] = metric.get("total_cost", 0.0) + cost

        logger.info(
            f"LLM call to {model} | Tokens: {tokens} | Duration: {duration:.3f}s"
        )

    def record_rag_search(
        self, query_length: int, results_count: int, duration: float
    ) -> None:
        metric = self.metrics["rag_search"]
        metric["count"] += 1
        metric["total_time"] += duration
        metric["total_results"] = metric.get("total_results", 0) + results_count

        logger.debug(
            f"RAG search | Query length: {query_length} | Results: {results_count} | Duration: {duration:.3f}s"
        )

    def record_error(self, error_type: str, error_message: str) -> None:
        metric = self.metrics["errors"]
        metric["count"] += 1
        error_key = f"error_{error_type}"
        self.metrics[error_key]["count"] = self.metrics[error_key].get("count", 0) + 1

        logger.error(f"Error recorded: {error_type} - {error_message}")

    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        summary = {
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
        }

        for key, value in self.metrics.items():
            if value["count"] > 0:
                avg_time = value["total_time"] / value["count"]
                summary["metrics"][key] = {
                    "count": value["count"],
                    "total_time": round(value["total_time"], 3),
                    "avg_time": round(avg_time, 3),
                    "errors": value.get("errors", 0),
                }

                if "total_tokens" in value:
                    summary["metrics"][key]["total_tokens"] = value["total_tokens"]

                if "total_cost" in value:
                    summary["metrics"][key]["total_cost"] = round(
                        value["total_cost"], 4
                    )

                if "total_results" in value:
                    summary["metrics"][key]["total_results"] = value["total_results"]

        return summary

    def _format_uptime(self, seconds: float) -> str:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def reset_metrics(self) -> None:
        self.metrics.clear()
        logger.info("Metrics reset")


monitoring_service = MonitoringService()
