"""
Structured metrics collection for observability.
Tracks inference latency, token usage, fallback events, RAG performance.
"""
import json
import time
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class Metrics:
    """Singleton metrics collector."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.metrics_dir = Path("processed_files") / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / "events.jsonl"
        self._initialized = True
    
    def _write_event(self, event: dict):
        """Write an event to the metrics log."""
        event["timestamp"] = datetime.utcnow().isoformat()
        try:
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[METRICS] Warning: could not write event: {e}")
    
    def record_inference(self, task: str, duration_sec: float, 
                        tokens_generated: int, success: bool, 
                        model_name: str = "", temperature: float = 0.0):
        """Record inference latency and token usage."""
        event = {
            "type": "inference",
            "task": task,
            "duration_sec": round(duration_sec, 3),
            "tokens_generated": tokens_generated,
            "success": success,
            "model": model_name,
            "temperature": temperature,
        }
        self._write_event(event)
    
    def record_rag_retrieval(self, query_len: int, results_count: int, 
                            latency_sec: float, db_name: str = ""):
        """Record RAG retrieval performance."""
        event = {
            "type": "rag_retrieval",
            "query_len": query_len,
            "results_count": results_count,
            "latency_sec": round(latency_sec, 3),
            "db": db_name,
        }
        self._write_event(event)
    
    def record_fallback_event(self, tier: int, reason: str, 
                             context_trimmed_pct: float = 0.0):
        """Record when fallback was triggered."""
        event = {
            "type": "fallback",
            "tier": tier,
            "reason": reason,
            "context_trimmed_pct": round(context_trimmed_pct, 1),
        }
        self._write_event(event)
    
    def record_file_upload(self, filename: str, size_bytes: int, 
                          file_type: str, page_count: int = 0):
        """Record file upload for audit trail."""
        event = {
            "type": "file_upload",
            "filename": filename,
            "size_bytes": size_bytes,
            "file_type": file_type,
            "page_count": page_count,
        }
        self._write_event(event)
    
    def record_validation_error(self, error_code: str, details: str):
        """Record input validation failures."""
        event = {
            "type": "validation_error",
            "error_code": error_code,
            "details": details,
        }
        self._write_event(event)
    
    def record_ollama_health(self, healthy: bool, response_time_ms: float):
        """Record Ollama server health check."""
        event = {
            "type": "ollama_health",
            "healthy": healthy,
            "response_time_ms": round(response_time_ms, 1),
        }
        self._write_event(event)
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics from recent metrics."""
        try:
            events = []
            if self.metrics_file.exists():
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                events.append(json.loads(line))
                            except:
                                pass
            
            # Last 1000 events
            events = events[-1000:]
            
            stats = {
                "total_events": len(events),
                "inference_count": 0,
                "fallback_count": 0,
                "errors": 0,
                "avg_inference_duration_sec": 0.0,
                "avg_tokens_per_inference": 0,
            }
            
            inference_durations = []
            inference_tokens = []
            
            for evt in events:
                if evt.get("type") == "inference":
                    stats["inference_count"] += 1
                    if evt.get("success"):
                        inference_durations.append(evt.get("duration_sec", 0))
                        inference_tokens.append(evt.get("tokens_generated", 0))
                elif evt.get("type") == "fallback":
                    stats["fallback_count"] += 1
                elif evt.get("type") == "validation_error":
                    stats["errors"] += 1
            
            if inference_durations:
                stats["avg_inference_duration_sec"] = round(
                    sum(inference_durations) / len(inference_durations), 2)
            if inference_tokens:
                stats["avg_tokens_per_inference"] = int(
                    sum(inference_tokens) / len(inference_tokens))
            
            return stats
        except Exception as e:
            print(f"[METRICS] Error getting stats: {e}")
            return {}


# Global singleton
_metrics = Metrics()

def get_metrics() -> Metrics:
    """Get the metrics singleton."""
    return _metrics
