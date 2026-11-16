"""Human-review observation windows and automatic rollback tests.

Contract under test
-------------------
* Every manual switch opens a fresh observation window counting only distinct
  emails accepted while it is current.
* At >=4 distinct reviews with >=2 spam-judged-as-ham the gateway rolls back
  to the version active before that window's switch.
* Same-truth retries do not double-count; a different truth is a conflict.
* Late reviews for closed windows never roll back the current model, and an
  interleaved manual switch is never overridden by an automatic rollback.
* Emails accepted before a rollback keep their pinned version; new requests
  use the rolled-back version.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import main as main_module  # noqa: E402

# Distinct ham-looking texts: both models classify them as ham, so a human
# "spam" verdict is always a missed-spam (false negative) review.
HAM_TEXTS = [
    "Hi team, attached is the quarterly report for our meeting tomorrow.",
    "Can we schedule the sprint review? Thanks, regards.",
    "Please review the attached invoice from our vendor and approve the budget.",
    "Lunch is booked at noon in the usual place, see you there.",
    "Reminder: dentist appointment on Friday morning, reply to confirm.",
    "The library book you reserved is ready for pickup this week.",
]
SPAM_TEXT = "Congratulations! You are a winner. Claim your free prize now."


@pytest.fixture()
def client():
    main_module.jobs.clear()
    main_module.switches.clear()
    main_module.active_version = main_module.registry.default_version
    main_module.windows.clear()
    main_module.windows_by_id.clear()
    main_module.current_window_id = None
    main_module.rollbacks.clear()
    main_module.reviews_by_email.clear()
    main_module.email_registry.clear()
    main_module.worker.metrics.update({key: 0 for key in main_module.worker.metrics})
    with TestClient(main_module.app) as test_client:
        yield test_client


def classify(client: TestClient, texts: list[str], label: str = "review") -> dict:
    response = client.post(
        "/api/classify",
        json={
            "label": label,
            "emails": [{"text": text, "deadline_ms": 900} for text in texts],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def review(client: TestClient, job_id: str, index: int, true_label: str):
    return client.post(
        "/api/reviews",
        json={"job_id": job_id, "entry_index": index, "true_label": true_label},
    )


def observation(client: TestClient) -> dict:
    return client.get("/api/observation").json()


def test_manual_switch_opens_observation_window(client):
    assert observation(client)["current"] is None  # nothing before a manual switch
    client.post("/api/model", json={"version": "v2"})
    obs = observation(client)
    current = obs["current"]
    assert current is not None
    assert current["status"] == "open"
    assert current["model_version"] == "v2"
    assert current["previous_version"] == "v1"
    assert current["review_count"] == 0
    assert obs["thresholds"] == {"min_reviews": 4, "min_missed_spam": 2}


def test_review_records_truth_and_counts_distinct_emails(client):
    client.post("/api/model", json={"version": "v2"})
    job = classify(client, HAM_TEXTS[:2])
    first = review(client, job["id"], 1, "spam")
    assert first.status_code == 200
    body = first.json()
    assert body["duplicate"] is False
    assert body["rollback"] is None
    assert body["review"]["true_label"] == "spam"
    assert body["review"]["missed_spam"] is True  # model said ham
    assert body["review"]["late"] is False
    window = body["window"]
    assert window["review_count"] == 1
    assert window["missed_spam_count"] == 1

    # Same truth again: idempotent, no double counting.
    again = review(client, job["id"], 1, "spam")
    assert again.status_code == 200
    assert again.json()["duplicate"] is True
    assert again.json()["window"]["review_count"] == 1

    # A different email adds a second distinct review.
    second = review(client, job["id"], 2, "ham")
    assert second.json()["window"]["review_count"] == 2
    assert second.json()["window"]["missed_spam_count"] == 1

    # The entry on the results page carries its review.
    entry = client.get(f"/api/jobs/{job['id']}").json()["entries"][0]
    assert entry["review"]["true_label"] == "spam"
    assert entry["window_id"] == window["id"]


def test_conflicting_truth_is_rejected(client):
    client.post("/api/model", json={"version": "v2"})
    job = classify(client, [HAM_TEXTS[0]])
    assert review(client, job["id"], 1, "spam").status_code == 200
    conflict = review(client, job["id"], 1, "ham")
    assert conflict.status_code == 409
    assert "冲突" in conflict.json()["detail"]
    # The original truth still stands and is counted only once.
    assert observation(client)["current"]["review_count"] == 1


def test_review_requires_successful_classification(client):
    client.post("/api/model", json={"version": "v2"})
    timed_out = client.post(
        "/api/classify",
        json={
            "label": "late",
            "emails": [
                {"text": SPAM_TEXT, "deadline_ms": 100, "simulate_latency_ms": 600}
            ],
        },
    ).json()
    assert timed_out["entries"][0]["status"] == "timeout"
    response = review(client, timed_out["id"], 1, "spam")
    assert response.status_code == 409
    assert "成功分类" in response.json()["detail"]

    missing = review(client, "no-such-job", 1, "spam")
    assert missing.status_code == 404
    bad_label = client.post(
        "/api/reviews",
        json={"job_id": "x", "entry_index": 1, "true_label": "unknown"},
    )
    assert bad_label.status_code == 422


def test_rollback_triggers_at_threshold(client):
    client.post("/api/model", json={"version": "v2"})
    window_id = observation(client)["current"]["id"]
    job = classify(client, HAM_TEXTS[:4])

    # 2 missed spam + 1 correct: thresholds not both met yet.
    assert review(client, job["id"], 1, "spam").json()["rollback"] is None
    assert review(client, job["id"], 2, "spam").json()["rollback"] is None
    assert review(client, job["id"], 3, "ham").json()["rollback"] is None
    assert client.get("/healthz").json()["active_model"] == "v2"

    # 4th distinct review: 4 reviews, 2 missed spam -> automatic rollback.
    final = review(client, job["id"], 4, "ham").json()
    rollback = final["rollback"]
    assert rollback is not None
    assert rollback["from"] == "v2"
    assert rollback["to"] == "v1"
    assert rollback["window_id"] == window_id
    assert rollback["review_count"] == 4
    assert rollback["missed_spam_count"] == 2
    assert len(rollback["trigger_emails"]) == 2
    assert all(t["true_label"] == "spam" for t in rollback["trigger_emails"])
    assert all(t["predicted_label"] == "ham" for t in rollback["trigger_emails"])
    assert {t["email_id"] for t in rollback["trigger_emails"]} == {
        f"{job['id']}:1",
        f"{job['id']}:2",
    }
    assert final["active_model"] == "v1"
    assert client.get("/healthz").json()["active_model"] == "v1"

    obs = observation(client)
    assert obs["current"] is None  # rollback opens no new window
    assert obs["windows"][0]["status"] == "triggered"
    assert obs["rollbacks"][0]["window_id"] == window_id

    # The switch history records the automatic rollback as its own event.
    sources = {event["source"] for event in client.get("/api/state").json()["switches"]}
    assert "auto-rollback" in sources


def test_rollback_requires_both_thresholds(client):
    client.post("/api/model", json={"version": "v2"})
    job = classify(client, HAM_TEXTS[:4])
    # 4 distinct reviews but only 1 missed spam -> no rollback.
    review(client, job["id"], 1, "spam")
    review(client, job["id"], 2, "ham")
    review(client, job["id"], 3, "ham")
    result = review(client, job["id"], 4, "ham").json()
    assert result["rollback"] is None
    assert client.get("/healthz").json()["active_model"] == "v2"


def test_late_review_from_old_window_cannot_rollback(client):
    client.post("/api/model", json={"version": "v2"})
    old_window_id = observation(client)["current"]["id"]
    job = classify(client, HAM_TEXTS[:4])

    # A newer manual decision supersedes the window before reviews arrive.
    client.post("/api/model", json={"version": "v1"})
    assert observation(client)["current"]["id"] != old_window_id

    for index, truth in enumerate(["spam", "spam", "spam", "spam"], start=1):
        result = review(client, job["id"], index, truth).json()
        assert result["rollback"] is None
        assert result["review"]["late"] is True

    # Old window accumulated the reviews but cannot roll anything back.
    windows = {w["id"]: w for w in observation(client)["windows"]}
    old = windows[old_window_id]
    assert old["status"] == "superseded"
    assert old["review_count"] == 4
    assert old["missed_spam_count"] == 4
    assert observation(client)["rollbacks"] == []
    assert client.get("/healthz").json()["active_model"] == "v1"


def test_rollback_never_overrides_newer_manual_switch(client):
    """Reviews trickle in; a manual switch lands before the 4th review."""
    client.post("/api/model", json={"version": "v2"})
    job = classify(client, HAM_TEXTS[:4])
    review(client, job["id"], 1, "spam")
    review(client, job["id"], 2, "spam")
    review(client, job["id"], 3, "ham")

    # The human switches away (newer decision) before the threshold is met.
    client.post("/api/model", json={"version": "v1"})

    # The 4th review would have triggered the rollback had the window lived.
    result = review(client, job["id"], 4, "spam").json()
    assert result["rollback"] is None
    assert result["review"]["late"] is True
    assert observation(client)["rollbacks"] == []
    assert client.get("/healthz").json()["active_model"] == "v1"


def test_reswitch_to_same_version_requires_fresh_observation(client):
    client.post("/api/model", json={"version": "v2"})
    first_window = observation(client)["current"]["id"]
    stale_job = classify(client, HAM_TEXTS[:4])
    review(client, stale_job["id"], 1, "spam")
    review(client, stale_job["id"], 2, "spam")

    client.post("/api/model", json={"version": "v1"})
    client.post("/api/model", json={"version": "v2"})
    obs = observation(client)
    second_window = obs["current"]["id"]
    assert second_window != first_window
    assert obs["current"]["review_count"] == 0  # tally restarted from zero

    # Reviews of the first window's emails never leak into the new window.
    review(client, stale_job["id"], 3, "spam")
    review(client, stale_job["id"], 4, "spam")
    assert observation(client)["current"]["review_count"] == 0
    assert observation(client)["rollbacks"] == []

    # The new window observes independently and can trigger its own rollback.
    fresh = classify(client, HAM_TEXTS[:4])
    review(client, fresh["id"], 1, "spam")
    review(client, fresh["id"], 2, "spam")
    review(client, fresh["id"], 3, "ham")
    result = review(client, fresh["id"], 4, "ham").json()
    assert result["rollback"] is not None
    assert result["rollback"]["window_id"] == second_window
    assert result["rollback"]["to"] == "v1"


def test_rollback_keeps_accepted_emails_on_old_version(client):
    """Accepted-before-rollback emails keep v2; new intake uses v1."""
    client.post("/api/model", json={"version": "v2"})
    reviewed = classify(client, HAM_TEXTS[:4])

    # Second job: one immediate email, one staggered past the rollback.
    job = client.post(
        "/api/jobs",
        json={
            "label": "inflight",
            "emails": [
                {"text": SPAM_TEXT, "deadline_ms": 1000},
                {"text": HAM_TEXTS[4], "deadline_ms": 1000, "submit_after_ms": 500},
            ],
        },
    ).json()
    for _ in range(200):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["entries"][0]["model_version"]:
            break
        time.sleep(0.01)
    assert state["entries"][0]["model_version"] == "v2"

    # Trigger the rollback while the second email is still waiting to be
    # accepted.
    review(client, reviewed["id"], 1, "spam")
    review(client, reviewed["id"], 2, "spam")
    review(client, reviewed["id"], 3, "ham")
    assert review(client, reviewed["id"], 4, "ham").json()["rollback"] is not None

    for _ in range(400):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] == "completed":
            break
        time.sleep(0.01)
    assert state["status"] == "completed"
    first, second = state["entries"]
    assert first["model_version"] == "v2"  # accepted before the rollback
    assert second["model_version"] == "v1"  # accepted after the rollback


def test_reviews_without_window_never_rollback(client):
    """Before any manual switch (and after a rollback) nothing is observed."""
    job = classify(client, HAM_TEXTS[:4])
    for index in range(1, 5):
        result = review(client, job["id"], index, "spam").json()
        assert result["window"] is None
        assert result["rollback"] is None
    assert observation(client)["current"] is None
    assert observation(client)["rollbacks"] == []


def test_post_rollback_emails_are_unobserved_until_next_switch(client):
    client.post("/api/model", json={"version": "v2"})
    job = classify(client, HAM_TEXTS[:4])
    review(client, job["id"], 1, "spam")
    review(client, job["id"], 2, "spam")
    review(client, job["id"], 3, "ham")
    review(client, job["id"], 4, "ham")
    assert client.get("/healthz").json()["active_model"] == "v1"

    # After the rollback there is no current window: new emails are accepted
    # with window_id None and their reviews cannot trigger anything.
    after = classify(client, HAM_TEXTS[2:6])
    assert all(e["window_id"] is None for e in after["entries"])
    for index in range(1, 5):
        result = review(client, after["id"], index, "spam").json()
        assert result["rollback"] is None
    assert len(observation(client)["rollbacks"]) == 1

    # A manual switch re-arms observation.
    client.post("/api/model", json={"version": "v2"})
    assert observation(client)["current"] is not None


def test_interleaved_manual_switches_and_rollback(client):
    """Full interleaving: switch, review, switch away, switch back, rollback."""
    client.post("/api/model", json={"version": "v2"})  # window A (target v1)
    job_a = classify(client, HAM_TEXTS[:2])
    review(client, job_a["id"], 1, "spam")
    review(client, job_a["id"], 2, "spam")

    client.post("/api/model", json={"version": "v1"})  # closes A, opens B
    client.post("/api/model", json={"version": "v2"})  # closes B, opens C
    window_c = observation(client)["current"]["id"]

    job_c = classify(client, HAM_TEXTS[2:6])
    review(client, job_c["id"], 1, "spam")
    review(client, job_c["id"], 2, "spam")
    review(client, job_c["id"], 3, "ham")
    result = review(client, job_c["id"], 4, "ham").json()
    assert result["rollback"] is not None
    rollback = result["rollback"]
    assert rollback["window_id"] == window_c
    assert rollback["from"] == "v2"
    assert rollback["to"] == "v1"
    # Only window C's own missed-spam emails are listed as triggers.
    assert {t["email_id"] for t in rollback["trigger_emails"]} == {
        f"{job_c['id']}:1",
        f"{job_c['id']}:2",
    }
    statuses = {w["id"]: w["status"] for w in observation(client)["windows"]}
    assert statuses[window_c] == "triggered"
    assert set(statuses.values()) == {"triggered", "superseded"}


def test_observation_listing_endpoints(client):
    client.post("/api/model", json={"version": "v2"})
    job = classify(client, [HAM_TEXTS[0]])
    review(client, job["id"], 1, "spam")

    reviews = client.get("/api/reviews").json()["reviews"]
    assert len(reviews) == 1
    assert reviews[0]["email_id"] == f"{job['id']}:1"

    state = client.get("/api/state").json()
    assert state["observation"]["current"]["review_count"] == 1
    assert state["observation"]["thresholds"]["min_reviews"] == 4
