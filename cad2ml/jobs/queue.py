"""Job transport on Valkey (Redis protocol) using the reliable-queue pattern.

    enqueue:  LPUSH cad2ml:queue <job_id>
    reserve:  BLMOVE cad2ml:queue cad2ml:processing:<worker> RIGHT LEFT <timeout>
    ack:      LREM  cad2ml:processing:<worker> 0 <job_id>

The queue is *transport only*. Correctness never depends on it:
  * a job is claimed by a compare-and-set on its PostgreSQL state (queued -> parsing),
    so duplicate deliveries are harmless;
  * leases/heartbeats live in PostgreSQL, and the reaper re-queues expired leases;
  * the reconciler re-enqueues every ``queued`` job found in PostgreSQL, so an emptied
    or restarted Valkey loses no work.
"""

from __future__ import annotations

from typing import Any

import redis

QUEUE = "cad2ml:queue"
PROCESSING_PREFIX = "cad2ml:processing:"


def make_redis(url: str, backend: str = "valkey") -> Any:
    if backend == "fake":
        import fakeredis

        return fakeredis.FakeRedis(decode_responses=True)
    return redis.Redis.from_url(
        url, decode_responses=True, socket_timeout=30, socket_connect_timeout=5, health_check_interval=15
    )


class JobQueue:
    def __init__(self, client: Any) -> None:
        self.r = client

    def enqueue(self, job_id: str) -> None:
        self.r.lpush(QUEUE, job_id)

    def reserve(self, worker_id: str, timeout_s: int = 2) -> str | None:
        return self.r.blmove(QUEUE, PROCESSING_PREFIX + worker_id, timeout_s, "RIGHT", "LEFT")  # type: ignore[no-any-return]

    def ack(self, worker_id: str, job_id: str) -> None:
        self.r.lrem(PROCESSING_PREFIX + worker_id, 0, job_id)

    def depth(self) -> int:
        return int(self.r.llen(QUEUE))

    def queued_ids(self) -> set[str]:
        return set(self.r.lrange(QUEUE, 0, -1))

    def release_worker(self, worker_id: str) -> int:
        """Move anything a (restarted) worker still holds back to the queue."""
        n = 0
        while self.r.lmove(PROCESSING_PREFIX + worker_id, QUEUE, "RIGHT", "LEFT") is not None:
            n += 1
        return n

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False
