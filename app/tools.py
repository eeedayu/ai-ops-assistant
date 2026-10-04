from typing import Any
from app.rag.service import RAGService


def get_device_status() -> dict[str, Any]:
    """
    查询服务器实时状态。
    """

    return {
        "device_id": "server-001",
        "cpu_usage": 92,
        "cpu_temperature": 87,
        "memory_usage": 73,
        "disk_usage": 68,
        "network_latency_ms": 21,
    }


def get_recent_logs(
    limit: int = 5,
) -> list[str]:
    """
    查询最近系统日志。
    """

    logs = [
        "10:01 CPU usage > 90%",
        "10:03 nginx response time increased",
        "10:05 worker process restarted",
        "10:06 upstream connection timeout",
        "10:08 request latency increased",
        "10:10 worker restart completed",
    ]

    return logs[-limit:]


def query_monitor_metrics(
    metric: str,
) -> dict[str, Any]:
    """
    查询监控指标。
    """

    metrics = {
        "cpu": {
            "avg_5m": 91.2,
            "avg_15m": 87.6,
            "load_1m": 8.42,
            "load_5m": 7.91,
        },
        "memory": {
            "avg_5m": 71.3,
            "avg_15m": 69.8,
        },
        "network": {
            "avg_latency_ms": 23.4,
            "packet_loss": 0.1,
        },
    }

    if metric not in metrics:
        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    return metrics[metric]



def restart_service(
    service_name: str,
) -> dict[str, Any]:

    return {
        "service": service_name,
        "action": "restart",
        "message": (
            f"模拟重启服务: {service_name}"
        ),
    }

