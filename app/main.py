"""FastAPI gateway for the spam moderation review console.

Intake pins the *current* active model version onto each accepted email, then
hands it to the single in-process worker.  A model switch therefore affects
only requests accepted from that moment on: emails already queued keep the
version they were accepted with.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .batching import Worker
from .model import ModelRegistry

logger = logging.getLogger("moderation")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
STATIC_DIR = Path(os.environ.get("STATIC_DIR", BASE_DIR / "static"))

MAX_EMAILS_PER_SUBMISSION = int(os.environ.get("MAX_EMAILS_PER_SUBMISSION", "16"))
DEADLINE_MIN_MS = 100
DEADLINE_MAX_MS = 1000
MAX_JOBS_KEPT = int(os.environ.get("MAX_JOBS_KEPT", "30"))

# Human-review observation windows.  Every *manual* model switch opens a fresh
# window; when it has accumulated at least ROLLBACK_MIN_REVIEWS distinct
# reviewed emails, of which at least ROLLBACK_MIN_MISSED_SPAM are spam the
# model judged as ham, the gateway automatically rolls back to the version
# that was active before that window's switch.
ROLLBACK_MIN_REVIEWS = int(os.environ.get("ROLLBACK_MIN_REVIEWS", "4"))
ROLLBACK_MIN_MISSED_SPAM = int(os.environ.get("ROLLBACK_MIN_MISSED_SPAM", "2"))
MAX_WINDOWS_KEPT = int(os.environ.get("MAX_WINDOWS_KEPT", "50"))
MAX_ROLLBACKS_KEPT = int(os.environ.get("MAX_ROLLBACKS_KEPT", "50"))
MAX_EMAILS_KEPT = int(os.environ.get("MAX_EMAILS_KEPT", "1000"))
MAX_REVIEWS_KEPT = int(os.environ.get("MAX_REVIEWS_KEPT", "1000"))

registry = ModelRegistry.load_dir(MODELS_DIR, os.environ.get("DEFAULT_MODEL_VERSION"))
worker = Worker(
    registry,
    max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "4")),
    batch_wait_ms=float(os.environ.get("BATCH_WAIT_MS", "200")),
    dispatch_guard_ms=float(os.environ.get("DISPATCH_GUARD_MS", "25")),
)

active_version: str = registry.default_version
switches: list[dict[str, Any]] = []
jobs: dict[str, dict[str, Any]] = {}

# ---- human-review / observation-window state ------------------------------
windows: list[ObservationWindow] = []
windows_by_id: dict[int, ObservationWindow] = {}
current_window_id: int | None = None
_window_seq = itertools.count(1)
rollbacks: list[dict[str, Any]] = []
reviews_by_email: dict[str, Review] = {}
email_registry: dict[str, EmailRecord] = {}

_background_tasks: set[asyncio.Task] = set()


def _spawn(coro, *, name: str) -> asyncio.Task:
    """Create a background task while keeping a strong reference to it."""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _set_active(version: str, *, source: str, note: str | None = None) -> dict[str, Any]:
    global active_version
    if version not in registry:
        raise KeyError(version)
    previous = active_version
    event = {
        "at": _now_iso(),
        "from": previous,
        "to": version,
        "source": source,
        "changed": previous != version,
        "note": note,
    }
    active_version = version
    switches.append(event)
    if len(switches) > 100:
        del switches[: len(switches) - 100]
    logger.info("active model %s -> %s (%s)", previous, version, source)
    return event


# ---------------------------------------------------------------------------
# human review: observation windows, reviews, automatic rollback
# ---------------------------------------------------------------------------
@dataclass
class EmailRecord:
    """Intake snapshot of one accepted email.

    Kept independently of ``jobs`` so a human review can still arrive after
    the originating job has been pruned from the job list.
    """

    email_id: str
    job_id: str
    entry_index: int
    text: str
    model_version: str
    window_id: int | None
    accepted_at: str
    status: str = "accepted"
    predicted_label: str | None = None
    score: float | None = None


@dataclass
class Review:
    """One human ground-truth decision for a successfully classified email."""

    email_id: str
    job_id: str
    entry_index: int
    text: str
    model_version: str
    predicted_label: str
    score: float | None
    true_label: str
    window_id: int | None
    reviewer: str | None
    note: str | None
    reviewed_at: str
    # True when the email's observation window was no longer the current open
    # one at review time: recorded for transparency, but it can never roll
    # back the current model.
    late: bool

    @property
    def is_missed_spam(self) -> bool:
        return self.true_label == "spam" and self.predicted_label == "ham"

    def public(self) -> dict[str, Any]:
        return {
            "email_id": self.email_id,
            "job_id": self.job_id,
            "entry_index": self.entry_index,
            "text": self.text,
            "model_version": self.model_version,
            "predicted_label": self.predicted_label,
            "score": self.score,
            "true_label": self.true_label,
            "window_id": self.window_id,
            "reviewer": self.reviewer,
            "note": self.note,
            "reviewed_at": self.reviewed_at,
            "late": self.late,
            "missed_spam": self.is_missed_spam,
        }


@dataclass
class ObservationWindow:
    """Reviews accumulated while one manually selected model version is active.

    A window opens on every manual (console/API) model switch — including
    switching back to a version that was observed before — and counts only
    distinct emails *accepted* while it is the current window.  It closes
    either when a newer switch supersedes it or when its own tally triggers
    an automatic rollback to ``previous_version``.
    """

    id: int
    opened_at: str
    model_version: str
    previous_version: str | None
    source: str
    status: str = "open"  # open | superseded | triggered
    closed_at: str | None = None
    reviews: dict[str, Review] = field(default_factory=dict)

    @property
    def review_count(self) -> int:
        return len(self.reviews)

    @property
    def missed_spam_count(self) -> int:
        return sum(1 for review in self.reviews.values() if review.is_missed_spam)

    def public(self, *, include_reviews: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "status": self.status,
            "source": self.source,
            "model_version": self.model_version,
            "previous_version": self.previous_version,
            "review_count": self.review_count,
            "missed_spam_count": self.missed_spam_count,
            "progress": {
                "reviews": self.review_count,
                "reviews_needed": ROLLBACK_MIN_REVIEWS,
                "missed_spam": self.missed_spam_count,
                "missed_spam_needed": ROLLBACK_MIN_MISSED_SPAM,
            },
        }
        if include_reviews:
            recent = list(self.reviews.values())[-20:][::-1]
            data["reviews"] = [review.public() for review in recent]
        return data


def _close_current_window(*, status: str) -> None:
    global current_window_id
    if current_window_id is None:
        return
    window = windows_by_id.get(current_window_id)
    if window is not None and window.status == "open":
        window.status = status
        window.closed_at = _now_iso()
    current_window_id = None


def _prune_windows() -> None:
    while len(windows) > MAX_WINDOWS_KEPT:
        evicted = windows.pop(0)
        windows_by_id.pop(evicted.id, None)


def _prune_reviews() -> None:
    # Window tallies keep their own references, so pruning the global index
    # only affects the "recent reviews" listing, never rollback accounting.
    while len(reviews_by_email) > MAX_REVIEWS_KEPT:
        reviews_by_email.pop(next(iter(reviews_by_email)), None)


def _apply_switch(version: str, *, source: str, note: str | None = None) -> dict[str, Any]:
    """Change the active model and maintain observation windows.

    Any newer switch closes the current window, so a stale window's late
    reviews can never override a newer (manual or automatic) decision.  Only
    human-initiated switches (``source="console"``) open a fresh observation
    window — switching back to a previously observed version therefore starts
    a brand-new tally instead of resurrecting the old one.
    """
    global current_window_id
    event = _set_active(version, source=source, note=note)
    _close_current_window(status="superseded")
    if source == "console":
        window = ObservationWindow(
            id=next(_window_seq),
            opened_at=_now_iso(),
            model_version=version,
            previous_version=event["from"],
            source="manual",
        )
        windows.append(window)
        windows_by_id[window.id] = window
        current_window_id = window.id
        _prune_windows()
        event["window_id"] = window.id
    return event


def _trigger_rollback(window: ObservationWindow) -> dict[str, Any]:
    """Roll the active model back to the version before ``window``'s switch.

    Only ever called for the *current* open window, so an automatic rollback
    can never override a newer manual decision.  The rollback itself does not
    open a new observation window: observation resumes with the next manual
    switch.  Emails already accepted keep their pinned version; only newly
    accepted requests use the rolled-back version (intake pinning unchanged).
    """
    global current_window_id
    assert window.previous_version is not None
    window.status = "triggered"
    window.closed_at = _now_iso()
    missed = [review for review in window.reviews.values() if review.is_missed_spam]
    event = _set_active(
        window.previous_version,
        source="auto-rollback",
        note=(
            f"观察期 #{window.id}：{window.review_count} 封复核中 "
            f"{window.missed_spam_count} 封垃圾邮件被判为 ham，自动回退"
        ),
    )
    current_window_id = None
    rollback = {
        "at": _now_iso(),
        "window_id": window.id,
        "from": event["from"],
        "to": event["to"],
        "review_count": window.review_count,
        "missed_spam_count": window.missed_spam_count,
        "trigger_emails": [
            {
                "email_id": review.email_id,
                "text": review.text,
                "model_version": review.model_version,
                "predicted_label": review.predicted_label,
                "true_label": review.true_label,
                "score": review.score,
            }
            for review in missed
        ],
    }
    rollbacks.append(rollback)
    if len(rollbacks) > MAX_ROLLBACKS_KEPT:
        del rollbacks[: len(rollbacks) - MAX_ROLLBACKS_KEPT]
    logger.warning(
        "auto-rollback: observation #%d on %s -> %s (%d reviews, %d missed spam)",
        window.id,
        window.model_version,
        window.previous_version,
        window.review_count,
        window.missed_spam_count,
    )
    return rollback


def _register_email(job: "Job", entry: "JobEntry") -> None:
    assert entry.email_id is not None
    email_registry[entry.email_id] = EmailRecord(
        email_id=entry.email_id,
        job_id=job.id,
        entry_index=entry.index,
        text=entry.text,
        model_version=entry.model_version or "",
        window_id=entry.window_id,
        accepted_at=entry.accepted_at or _now_iso(),
    )
    while len(email_registry) > MAX_EMAILS_KEPT:
        # insertion-ordered dict: evict the oldest record first
        email_registry.pop(next(iter(email_registry)), None)


def _update_email_record(entry: "JobEntry") -> None:
    if entry.email_id is None or entry.result is None:
        return
    record = email_registry.get(entry.email_id)
    if record is None:
        return
    record.status = entry.result.get("status", record.status)
    record.predicted_label = entry.result.get("label")
    record.score = entry.result.get("score")


def _observation_state() -> dict[str, Any]:
    current = windows_by_id.get(current_window_id) if current_window_id is not None else None
    return {
        "current_window_id": current_window_id,
        "current": current.public() if current is not None else None,
        "thresholds": {
            "min_reviews": ROLLBACK_MIN_REVIEWS,
            "min_missed_spam": ROLLBACK_MIN_MISSED_SPAM,
        },
        "windows": [w.public(include_reviews=False) for w in reversed(windows[-10:])],
        "rollbacks": list(reversed(rollbacks[-10:])),
        "recent_reviews": [r.public() for r in list(reviews_by_email.values())[-20:][::-1]],
    }


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class EmailIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    deadline_ms: int = Field(ge=DEADLINE_MIN_MS, le=DEADLINE_MAX_MS)
    # Demo / self-test affordances.  Both default to zero and only affect the
    # timing of this individual message, never the batching contract.
    submit_after_ms: int = Field(default=0, ge=0, le=5000)
    simulate_latency_ms: int = Field(default=0, ge=0, le=5000)


class SubmitIn(BaseModel):
    label: str = Field(default="", max_length=80)
    emails: list[EmailIn] = Field(min_length=1, max_length=MAX_EMAILS_PER_SUBMISSION)


class SwitchIn(BaseModel):
    version: str
    note: str | None = Field(default=None, max_length=200)


class ReviewIn(BaseModel):
    job_id: str = Field(min_length=1, max_length=80)
    entry_index: int = Field(ge=1, le=1000)
    true_label: Literal["spam", "ham"]
    reviewer: str | None = Field(default=None, max_length=80)
    note: str | None = Field(default=None, max_length=200)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
@dataclass
class JobEntry:
    index: int
    text: str
    deadline_ms: int
    submit_after_ms: int
    simulate_latency_ms: int
    email_id: str | None = None
    model_version: str | None = None
    window_id: int | None = None
    accepted_at: str | None = None
    result: dict | None = None

    def public(self) -> dict:
        result = self.result or {}
        review = reviews_by_email.get(self.email_id or "")
        return {
            "index": self.index,
            "text": self.text,
            "deadline_ms": self.deadline_ms,
            "submit_after_ms": self.submit_after_ms,
            "simulate_latency_ms": self.simulate_latency_ms,
            "email_id": self.email_id,
            "model_version": self.model_version,
            "window_id": self.window_id,
            "accepted_at": self.accepted_at,
            "status": result.get("status", "accepted"),
            "score": result.get("score"),
            "label": result.get("label"),
            "threshold": result.get("threshold"),
            "batch_id": result.get("batch_id"),
            "request_id": result.get("request_id"),
            "latency_ms": result.get("latency_ms"),
            "reason": result.get("reason"),
            "late_result_discarded": result.get("late_result_discarded", False),
            "submitted_at": result.get("submitted_at"),
            "completed_at": result.get("completed_at"),
            "review": review.public() if review is not None else None,
        }


@dataclass
class Job:
    id: str
    label: str
    created_at: str
    entries: list[JobEntry]
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def public(self, *, include_entries: bool = True) -> dict:
        versions = sorted({e.model_version for e in self.entries if e.model_version})
        data = {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "events": list(self.events),
            "size": len(self.entries),
            "settled": sum(1 for e in self.entries if e.result is not None),
            "ok": sum(1 for e in self.entries if (e.result or {}).get("status") == "ok"),
            "timeouts": sum(
                1 for e in self.entries if (e.result or {}).get("status") == "timeout"
            ),
            "errors": sum(
                1 for e in self.entries if (e.result or {}).get("status") == "error"
            ),
            "versions": versions,
        }
        if include_entries:
            data["entries"] = [e.public() for e in self.entries]
        return data


def _prune_jobs() -> None:
    if len(jobs) <= MAX_JOBS_KEPT:
        return
    completed = [jid for jid, job in jobs.items() if job.status == "completed"]
    for jid in completed[: max(0, len(jobs) - MAX_JOBS_KEPT)]:
        jobs.pop(jid, None)


async def _run_job(job: Job) -> None:
    job.status = "running"
    job.started_at = _now_iso()
    start = time.monotonic()
    tasks: list[asyncio.Task] = []
    try:
        for entry in job.entries:
            delay = start + entry.submit_after_ms / 1000.0 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            # ---- intake: pin the version that is active right now ----
            version = active_version
            entry.model_version = version
            entry.accepted_at = _now_iso()
            entry.window_id = current_window_id
            _register_email(job, entry)
            job.events.append(
                {
                    "at": entry.accepted_at,
                    "type": "accepted",
                    "index": entry.index,
                    "model_version": version,
                    "window_id": entry.window_id,
                }
            )
            coro = worker.submit(
                text=entry.text,
                model_version=version,
                deadline_ms=entry.deadline_ms,
                request_id=f"{job.id}-{entry.index}",
                request_batch_id=job.id,
                latency_sim_ms=entry.simulate_latency_ms,
            )
            tasks.append(asyncio.create_task(coro))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for entry, result in zip(job.entries, results):
            if isinstance(result, BaseException):
                entry.result = {
                    "status": "error",
                    "reason": "gateway_error",
                    "detail": str(result),
                }
            else:
                entry.result = result
            _update_email_record(entry)
        job.status = "completed"
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.error = "shutdown"
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("job %s failed", job.id)
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.finished_at = _now_iso()
        _prune_jobs()


def _new_job(payload: SubmitIn, *, prefix: str = "job") -> Job:
    job = Job(
        id=f"{prefix}-{uuid.uuid4().hex[:10]}",
        label=payload.label or "未命名提交",
        created_at=_now_iso(),
        entries=[
            JobEntry(
                index=i + 1,
                text=email.text,
                deadline_ms=email.deadline_ms,
                submit_after_ms=email.submit_after_ms,
                simulate_latency_ms=email.simulate_latency_ms,
            )
            for i, email in enumerate(payload.emails)
        ],
    )
    for entry in job.entries:
        entry.email_id = f"{job.id}:{entry.index}"
    jobs[job.id] = job
    return job


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await worker.start()
    logger.info(
        "models loaded: %s (active=%s, model_dir=%s)",
        ", ".join(registry.versions),
        active_version,
        MODELS_DIR,
    )
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(
    title="Spam Moderation Review Console",
    version="1.0.0",
    description="Version-pinned, deadline-aware batching spam classifier",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "active_model": active_version, "models": registry.versions}


@app.get("/api/models")
async def list_models() -> dict:
    models = []
    for version in registry.versions:
        model = registry.get(version)
        order = model.weights.argsort()
        models.append(
            {
                "version": version,
                "description": model.description,
                "threshold": model.threshold,
                "vocabulary_size": len(model.vocabulary),
                "top_spam_terms": [
                    {"term": model.vocabulary[i], "weight": round(float(model.weights[i]), 4)}
                    for i in order[::-1][:8]
                ],
                "top_ham_terms": [
                    {"term": model.vocabulary[i], "weight": round(float(model.weights[i]), 4)}
                    for i in order[:8]
                ],
            }
        )
    return {"active": active_version, "models": models}


@app.post("/api/model")
async def switch_model(payload: SwitchIn) -> dict:
    if payload.version not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"未知模型版本 {payload.version!r}，可用: {registry.versions}",
        )
    # A manual switch closes the previous observation window (superseded) and
    # opens a fresh one for the newly selected version.
    event = _apply_switch(payload.version, source="console", note=payload.note)
    window = windows_by_id.get(current_window_id) if current_window_id is not None else None
    return {
        "active": active_version,
        "event": event,
        "models": registry.versions,
        "window": window.public(include_reviews=False) if window is not None else None,
    }


# ---------------------------------------------------------------------------
# human review endpoints
# ---------------------------------------------------------------------------
@app.post("/api/reviews")
async def submit_review(payload: ReviewIn) -> dict:
    """Record the human ground truth for one successfully classified email.

    * The review counts toward the observation window the email was *accepted*
      in — never toward a newer window.
    * Re-submitting the same truth for the same email is an idempotent no-op
      (``duplicate: true``); a different truth is rejected as a conflict.
    * When the current window reaches the rollback thresholds the active model
      is rolled back automatically.  Reviews arriving for emails of an
      already-closed window are marked ``late`` and can never roll back the
      current model.
    """
    email_id = f"{payload.job_id}:{payload.entry_index}"
    record = email_registry.get(email_id)
    if record is None:
        raise HTTPException(status_code=404, detail="邮件不存在或已过期，无法复核")
    if record.status != "ok":
        raise HTTPException(
            status_code=409,
            detail=f"仅成功分类的邮件可提交人工复核（当前状态：{record.status}）",
        )
    existing = reviews_by_email.get(email_id)
    if existing is not None:
        if existing.true_label == payload.true_label:
            window = (
                windows_by_id.get(existing.window_id)
                if existing.window_id is not None
                else None
            )
            return {
                "duplicate": True,
                "detail": "相同真值已记录，未重复计数",
                "review": existing.public(),
                "window": window.public() if window is not None else None,
                "rollback": None,
                "active_model": active_version,
            }
        raise HTTPException(
            status_code=409,
            detail=(
                f"真值冲突：该邮件已复核为 {existing.true_label}，"
                f"本次提交为 {payload.true_label}"
            ),
        )

    window = (
        windows_by_id.get(record.window_id) if record.window_id is not None else None
    )
    is_current_open = (
        window is not None
        and window.status == "open"
        and window.id == current_window_id
    )
    review = Review(
        email_id=email_id,
        job_id=record.job_id,
        entry_index=record.entry_index,
        text=record.text[:160],
        model_version=record.model_version,
        predicted_label=record.predicted_label or "",
        score=record.score,
        true_label=payload.true_label,
        window_id=record.window_id,
        reviewer=payload.reviewer,
        note=payload.note,
        reviewed_at=_now_iso(),
        late=window is not None and not is_current_open,
    )
    reviews_by_email[email_id] = review
    _prune_reviews()

    rollback_event = None
    if window is not None:
        window.reviews[email_id] = review
        if (
            is_current_open
            and window.previous_version is not None
            and window.review_count >= ROLLBACK_MIN_REVIEWS
            and window.missed_spam_count >= ROLLBACK_MIN_MISSED_SPAM
        ):
            rollback_event = _trigger_rollback(window)

    return {
        "duplicate": False,
        "review": review.public(),
        "window": window.public() if window is not None else None,
        "rollback": rollback_event,
        "active_model": active_version,
    }


@app.get("/api/reviews")
async def list_reviews(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    reviews = list(reviews_by_email.values())[-limit:][::-1]
    return {"reviews": [review.public() for review in reviews]}


@app.get("/api/observation")
async def observation() -> dict:
    return _observation_state()


@app.get("/api/state")
async def state() -> dict:
    return {
        "active_model": active_version,
        "models": registry.versions,
        "switches": list(reversed(switches[-20:])),
        "worker": worker.snapshot(),
        "observation": _observation_state(),
        "config": {
            "max_batch_size": worker.max_batch_size,
            "batch_wait_ms": worker.batch_wait_s * 1000,
            "dispatch_guard_ms": worker.dispatch_guard_s * 1000,
            "max_emails_per_submission": MAX_EMAILS_PER_SUBMISSION,
            "deadline_min_ms": DEADLINE_MIN_MS,
            "deadline_max_ms": DEADLINE_MAX_MS,
        },
        "jobs": [job.public(include_entries=False) for job in
                 sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)[:12]],
        "server_time": _now_iso(),
        "pending": worker.pending_count(),
    }


@app.post("/api/jobs", status_code=202)
async def create_job(payload: SubmitIn) -> dict:
    job = _new_job(payload)
    _spawn(_run_job(job), name=f"job-runner-{job.id}")
    return job.public()


@app.post("/api/classify")
async def classify(payload: SubmitIn) -> dict:
    """Synchronous convenience endpoint: accept all emails and await results."""
    job = _new_job(payload, prefix="sync")
    await _run_job(job)
    return job.public()


@app.get("/api/jobs")
async def list_jobs(limit: int = Query(default=12, ge=1, le=50)) -> dict:
    ordered = sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
    return {"jobs": [job.public() for job in ordered]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return job.public()


@app.get("/api/batches")
async def batches(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    return {"batches": worker.recent_batches(limit)}


@app.post("/api/demo/interleaved")
async def demo_interleaved() -> dict:
    """Schedule old-model work, switch models mid-queue, then add new work.

    Demonstrates: queued requests keep the version they were accepted with,
    later requests use the new version immediately, and batches never mix
    versions.  One new-version email is given artificial inference latency to
    show that a timeout is not overturned by a late result.
    """
    if "v1" not in registry or "v2" not in registry:
        raise HTTPException(status_code=409, detail="demo 需要 v1 与 v2 两个模型")
    _apply_switch("v1", source="demo", note="演示开始：先使用 v1")

    old_emails = [
        EmailIn(
            text="Congratulations! You are a winner. Claim your free prize now, click here.",
            deadline_ms=900,
        ),
        EmailIn(
            text="URGENT: your account will be suspended. Verify your account immediately.",
            deadline_ms=800,
        ),
    ]
    new_emails = [
        EmailIn(
            text="Limited offer! Invest in crypto today and double your money fast.",
            deadline_ms=500,
        ),
        EmailIn(
            text="Your parcel could not be delivered. Pay the small fee to reschedule.",
            deadline_ms=400,
        ),
        EmailIn(
            text="Final notice: send your password to unlock the refund.",
            deadline_ms=150,
            simulate_latency_ms=600,
        ),
    ]
    combined = SubmitIn(label="切换交错演示", emails=old_emails + new_emails)
    job = _new_job(combined, prefix="demo")

    async def orchestrate() -> None:
        job.status = "running"
        job.started_at = _now_iso()
        start = time.monotonic()
        tasks: list[asyncio.Task] = []
        try:
            for entry in job.entries[: len(old_emails)]:
                version = active_version
                entry.model_version = version
                entry.accepted_at = _now_iso()
                entry.window_id = current_window_id
                _register_email(job, entry)
                job.events.append(
                    {
                        "at": entry.accepted_at,
                        "type": "accepted",
                        "index": entry.index,
                        "model_version": version,
                        "window_id": entry.window_id,
                    }
                )
                tasks.append(
                    asyncio.create_task(
                        worker.submit(
                            text=entry.text,
                            model_version=version,
                            deadline_ms=entry.deadline_ms,
                            request_id=f"{job.id}-{entry.index}",
                            request_batch_id=job.id,
                            latency_sim_ms=entry.simulate_latency_ms,
                        )
                    )
                )
            await asyncio.sleep(max(0.0, start + 0.06 - time.monotonic()))
            event = _apply_switch(
                "v2", source="demo", note="演示：切换到 v2（旧请求仍在排队）"
            )
            job.events.append(
                {
                    "at": event["at"],
                    "type": "model_switched",
                    "from": event["from"],
                    "to": event["to"],
                    "pending_old": worker.pending_count("v1"),
                }
            )
            await asyncio.sleep(max(0.0, start + 0.10 - time.monotonic()))
            for entry in job.entries[len(old_emails) :]:
                version = active_version
                entry.model_version = version
                entry.accepted_at = _now_iso()
                entry.window_id = current_window_id
                _register_email(job, entry)
                job.events.append(
                    {
                        "at": entry.accepted_at,
                        "type": "accepted",
                        "index": entry.index,
                        "model_version": version,
                        "window_id": entry.window_id,
                    }
                )
                tasks.append(
                    asyncio.create_task(
                        worker.submit(
                            text=entry.text,
                            model_version=version,
                            deadline_ms=entry.deadline_ms,
                            request_id=f"{job.id}-{entry.index}",
                            request_batch_id=job.id,
                            latency_sim_ms=entry.simulate_latency_ms,
                        )
                    )
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for entry, result in zip(job.entries, results):
                if isinstance(result, BaseException):
                    entry.result = {
                        "status": "error",
                        "reason": "gateway_error",
                        "detail": str(result),
                    }
                else:
                    entry.result = result
                _update_email_record(entry)
            job.status = "completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "shutdown"
            raise
        finally:
            job.finished_at = _now_iso()
            _prune_jobs()

    _spawn(orchestrate(), name=f"demo-runner-{job.id}")
    return job.public()


@app.exception_handler(ValueError)
async def value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
