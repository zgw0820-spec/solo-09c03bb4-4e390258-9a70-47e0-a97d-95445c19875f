#!/usr/bin/env python3
"""End-to-end self-check against a running service.

    python3 scripts/selfcheck.py [base-url]

Verifies, over real HTTP:
1. health + two distinct models
2. switching the active model
3. version pinning: requests accepted before a switch keep the old version,
   requests accepted after it use the new one, and no batch mixes versions
4. batching: 16 emails -> batches of at most 4
5. deadlines: a slow email times out, its batch-mates still succeed, and the
   late result never flips the timeout back to success
6. early dispatch when the earliest deadline is near
7. interleaved demo
8. human review: observation windows, duplicate/conflict handling, automatic
   rollback at the missed-spam threshold, and late reviews from an old window
   never rolling back the current model
"""

from __future__ import annotations

import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SPAM = "Congratulations! You are a winner. Claim your free prize now."
HAM = "Hi team, attached is the quarterly report for our meeting tomorrow."
HAM2 = "Can we schedule the sprint review? Thanks, regards."
HAM3 = "Please review the attached invoice from our vendor and approve the budget."
HAM4 = "Lunch is booked at noon in the usual place, see you there."
CRYPTO = "Limited offer! Invest in crypto today and double your money fast."

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def wait_job(client: httpx.Client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    state = {}
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("completed", "failed"):
            return state
        time.sleep(0.02)
    return state


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=30.0)

    print("1. health & models")
    health = client.get("/healthz").json()
    check(health["status"] == "ok", "service healthy")
    check({"v1", "v2"} <= set(health["models"]), "both v1 and v2 shipped")
    models = client.get("/api/models").json()["models"]
    sizes = {m["version"]: m["vocabulary_size"] for m in models}
    check(sizes["v1"] != sizes["v2"], f"models differ (v1={sizes['v1']}, v2={sizes['v2']} terms)")

    print("2. model switch")
    switched = client.post("/api/model", json={"version": "v2"}).json()
    check(switched["active"] == "v2", "active model switched to v2")
    check(client.get("/healthz").json()["active_model"] == "v2", "health reflects the switch")
    check(client.post("/api/model", json={"version": "nope"}).status_code == 404, "unknown version rejected")

    print("3. version pinning across a live switch")
    client.post("/api/model", json={"version": "v1"})
    job = client.post(
        "/api/jobs",
        json={
            "label": "pin",
            "emails": [
                {"text": SPAM, "deadline_ms": 900},
                {"text": SPAM, "deadline_ms": 900, "submit_after_ms": 260},
                {"text": SPAM, "deadline_ms": 900, "submit_after_ms": 420},
            ],
        },
    ).json()
    time.sleep(0.12)
    switch = client.post("/api/model", json={"version": "v2"}).json()
    check(switch["event"]["from"] == "v1" and switch["event"]["to"] == "v2", "switched v1 -> v2 mid-queue")
    pinned = wait_job(client, job["id"])
    versions = [e["model_version"] for e in pinned["entries"]]
    check(versions[0] == "v1", "email accepted before the switch kept v1")
    check(all(v == "v2" for v in versions[1:]), "emails accepted after the switch used v2")
    batches = client.get("/api/batches?limit=50").json()["batches"]
    for entry in pinned["entries"]:
        batch = next((b for b in batches if b["batch_id"] == entry["batch_id"]), None)
        if batch:
            expected = entry["model_version"]
            check(batch["model_version"] == expected, f"batch {batch['batch_id']} is pure {expected}")

    print("4. batching limits (16 emails)")
    client.post("/api/model", json={"version": "v1"})
    bulk = client.post(
        "/api/jobs",
        json={
            "label": "bulk16",
            "emails": [
                {"text": f"{SPAM} ref {i}", "deadline_ms": 1000}
                for i in range(16)
            ],
        },
    ).json()
    bulk_state = wait_job(client, bulk["id"])
    check(bulk_state["ok"] == 16, f"all 16 classified (ok={bulk_state['ok']})")
    bulk_ids = {e["batch_id"] for e in bulk_state["entries"]}
    check(len(bulk_ids) == 4, f"16 emails formed 4 batches (got {len(bulk_ids)})")
    sizes = [b["size"] for b in client.get("/api/batches?limit=50").json()["batches"] if b["batch_id"] in bulk_ids]
    check(max(sizes) <= 4 and sorted(sizes) == [4, 4, 4, 4], f"batch sizes {sorted(sizes)}")

    print("5. deadline timeout + late result cannot flip it")
    client.post("/api/model", json={"version": "v2"})
    timeout_job = client.post(
        "/api/classify",
        json={
            "label": "batch timeout",
            "emails": [
                {"text": SPAM, "deadline_ms": 100, "simulate_latency_ms": 600},
                {"text": CRYPTO, "deadline_ms": 1000},
            ],
        },
    ).json()
    late = next(e for e in timeout_job["entries"] if e["deadline_ms"] == 100)
    fast = next(e for e in timeout_job["entries"] if e["deadline_ms"] == 1000)
    check(late["status"] == "timeout", "slow email timed out")
    check(late["score"] is None, "timed-out email has no score")
    check(late["late_result_discarded"] is True, "late result explicitly discarded")
    check(fast["status"] == "ok", "batch-mate still succeeded")
    check(late["batch_id"] == fast["batch_id"], "they were in the same batch")
    time.sleep(0.7)
    reread = client.get(f"/api/jobs/{timeout_job['id']}").json()
    again = next(e for e in reread["entries"] if e["deadline_ms"] == 100)
    check(again["status"] == "timeout", "timeout still a timeout after the late thread finished")

    print("6. early dispatch when the earliest deadline is near")
    near = client.post(
        "/api/jobs",
        json={"label": "near", "emails": [{"text": SPAM, "deadline_ms": 150}]},
    ).json()
    near_state = wait_job(client, near["id"])
    entry = near_state["entries"][0]
    check(entry["status"] == "ok", "near-deadline request was dispatched early and scored")
    check(entry["latency_ms"] < 200, f"latency {entry['latency_ms']}ms < 200ms window")

    print("7. interleaved demo")
    demo = client.post("/api/demo/interleaved").json()
    demo_state = wait_job(client, demo["id"], timeout=10)
    old = [e for e in demo_state["entries"] if e["index"] <= 2]
    new = [e for e in demo_state["entries"] if e["index"] > 2]
    check({e["model_version"] for e in old} == {"v1"}, "demo old requests pinned to v1")
    check({e["model_version"] for e in new} == {"v2"}, "demo new requests used v2")
    slow = next(e for e in new if e["simulate_latency_ms"] > 0)
    check(slow["status"] == "timeout", "demo slow email timed out")
    check(client.get("/healthz").json()["active_model"] == "v2", "demo left v2 active")

    print("8. human review: observation window + automatic rollback")
    # The demo above left v2 active; go v1 -> v2 so the window's rollback
    # target is unambiguously v1.
    client.post("/api/model", json={"version": "v1"})
    client.post("/api/model", json={"version": "v2"})
    obs = client.get("/api/observation").json()
    current = obs["current"]
    check(
        current is not None and current["model_version"] == "v2" and current["previous_version"] == "v1",
        "manual switch opened an observation window on v2 (rollback target v1)",
    )
    window_id = current["id"]
    review_job = client.post(
        "/api/classify",
        json={
            "label": "review",
            "emails": [
                {"text": HAM, "deadline_ms": 900},
                {"text": HAM2, "deadline_ms": 900},
                {"text": HAM3, "deadline_ms": 900},
                {"text": HAM4, "deadline_ms": 900},
            ],
        },
    ).json()
    check(
        all(e["status"] == "ok" and e["label"] == "ham" for e in review_job["entries"]),
        "four review emails classified ok as ham by v2",
    )
    check(
        all(e["window_id"] == window_id for e in review_job["entries"]),
        "accepted emails pinned to the current observation window",
    )

    def post_review(index: int, truth: str) -> httpx.Response:
        return client.post(
            "/api/reviews",
            json={"job_id": review_job["id"], "entry_index": index, "true_label": truth},
        )

    first = post_review(1, "spam").json()
    check(first["duplicate"] is False and first["review"]["missed_spam"] is True,
          "first review recorded (spam judged ham = missed)")
    retry = post_review(1, "spam").json()
    check(
        retry["duplicate"] is True and retry["window"]["review_count"] == 1,
        "same-truth retry not double counted",
    )
    conflict = post_review(1, "ham")
    check(conflict.status_code == 409, "conflicting truth rejected with 409")

    post_review(2, "spam")
    third = post_review(3, "ham").json()
    check(
        third["rollback"] is None and third["window"]["review_count"] == 3,
        "no rollback below 4 distinct reviews",
    )
    fourth = post_review(4, "ham").json()
    rollback = fourth["rollback"]
    check(rollback is not None, "4th review with 2 missed spam triggered auto-rollback")
    if rollback:
        check(
            rollback["from"] == "v2" and rollback["to"] == "v1" and rollback["window_id"] == window_id,
            f"rolled back v2 -> v1 from window #{window_id}",
        )
        check(
            len(rollback["trigger_emails"]) == 2
            and all(t["true_label"] == "spam" and t["predicted_label"] == "ham" for t in rollback["trigger_emails"]),
            "rollback record lists the two missed-spam trigger emails",
        )
    check(client.get("/healthz").json()["active_model"] == "v1", "active model is now v1")
    obs = client.get("/api/observation").json()
    check(obs["current"] is None, "rollback did not open a new observation window")
    check(
        obs["windows"] and obs["windows"][0]["status"] == "triggered",
        "triggered window recorded in history",
    )
    check(
        obs["rollbacks"] and obs["rollbacks"][0]["window_id"] == window_id,
        "rollback listed in rollback records",
    )

    print("9. late reviews from an old window cannot roll back the current model")
    client.post("/api/model", json={"version": "v2"})
    obs = client.get("/api/observation").json()
    second_window = obs["current"]["id"]
    check(second_window != window_id, "re-switching to v2 opened a fresh window")
    late_job = client.post(
        "/api/classify",
        json={
            "label": "late-reviews",
            "emails": [
                {"text": HAM, "deadline_ms": 900},
                {"text": HAM2, "deadline_ms": 900},
                {"text": HAM3, "deadline_ms": 900},
                {"text": HAM4, "deadline_ms": 900},
            ],
        },
    ).json()
    # Newer manual decision supersedes the window before its reviews arrive.
    client.post("/api/model", json={"version": "v1"})
    rollbacks_before = len(client.get("/api/observation").json()["rollbacks"])
    for index in range(1, 5):
        late = client.post(
            "/api/reviews",
            json={"job_id": late_job["id"], "entry_index": index, "true_label": "spam"},
        ).json()
    check(late["review"]["late"] is True, "late review flagged as late")
    check(late["rollback"] is None, "late review did not trigger a rollback")
    obs = client.get("/api/observation").json()
    check(
        len(obs["rollbacks"]) == rollbacks_before,
        "no extra rollback recorded for the superseded window",
    )
    check(
        client.get("/healthz").json()["active_model"] == "v1",
        "current model left untouched by late reviews",
    )
    windows = {w["id"]: w for w in obs["windows"]}
    check(
        windows.get(second_window, {}).get("status") == "superseded"
        and windows[second_window]["review_count"] == 4,
        "superseded window kept its tally but stayed inert",
    )

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
