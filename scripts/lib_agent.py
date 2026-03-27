"""
OpenClaw agent execution helpers for PinchBench.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from lib_tasks import Task


logger = logging.getLogger(__name__)
MAX_OPENCLAW_MESSAGE_CHARS = int(os.environ.get("PINCHBENCH_MAX_MSG_CHARS", "4000"))
CONTEXT_HASH_PREFIX_CHARS = int(os.environ.get("PINCHBENCH_CONTEXT_HASH_PREFIX_CHARS", "1024"))
CONTEXT_RECENT_MESSAGES = int(os.environ.get("PINCHBENCH_CONTEXT_RECENT_MESSAGES", "4"))
PINCHBENCH_PROVIDER_TAP_PATH = Path(
    os.environ.get(
        "PINCHBENCH_PROVIDER_TAP_PATH",
        str(Path.home() / ".openclaw" / "ecoclaw-plugin-state" / "ecoclaw" / "provider-traffic.jsonl"),
    )
)
PINCHBENCH_LLM_HOOK_TAP_PATH = Path(
    os.environ.get(
        "PINCHBENCH_LLM_HOOK_TAP_PATH",
        str(PINCHBENCH_PROVIDER_TAP_PATH).replace(".jsonl", ".llm-hooks.jsonl"),
    )
)
PINCHBENCH_RECONCILE_PROVIDER_USAGE = (
    os.environ.get("PINCHBENCH_RECONCILE_PROVIDER_USAGE", "1").strip().lower() not in {"0", "false", "off", "no"}
)
PINCHBENCH_RECONCILE_TIMEOUT_SECONDS = float(
    os.environ.get("PINCHBENCH_RECONCILE_TIMEOUT_SECONDS", "60")
)


def slugify_model(model_id: str) -> str:
    return model_id.replace("/", "-").replace(".", "-").lower()


def _ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                # Keep this generic: capture common payload fields without
                # assuming one provider schema.
                item_type = item.get("type")
                if item_type:
                    parts.append(f"[{item_type}]")
                for key in ("text", "content", "input", "output", "result", "value"):
                    if key in item:
                        parts.append(_message_content_to_text(item.get(key)))
            else:
                parts.append(_message_content_to_text(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(content)
    return str(content)


def _normalize_cache_signature_text(text: str) -> str:
    normalized = text
    normalized = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", "<UUID>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"/tmp/pinchbench/[^\s\"']+", "/tmp/pinchbench/<PATH>", normalized)
    normalized = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ][0-9:\.\+\-Z]{6,}\b", "<TIMESTAMP>", normalized)
    normalized = re.sub(r"\b\d{10,}\b", "<LONGNUM>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _build_call_context_detail(
    transcript: List[Dict[str, Any]],
    assistant_entry_index: int,
) -> Dict[str, Any]:
    message_items: List[Dict[str, Any]] = []
    message_indices: List[int] = []
    for idx, entry in enumerate(transcript[:assistant_entry_index]):
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {}) if isinstance(entry.get("message"), dict) else {}
        role = str(msg.get("role") or "unknown")
        content_text = _message_content_to_text(msg.get("content"))
        message_items.append(
            {
                "transcript_index": idx,
                "role": role,
                "content": content_text,
            }
        )
        message_indices.append(idx)

    signature_payload = json.dumps(message_items, ensure_ascii=False, sort_keys=True)
    normalized_payload = _normalize_cache_signature_text(signature_payload)
    prefix_chars = max(128, CONTEXT_HASH_PREFIX_CHARS)
    recent_count = max(1, CONTEXT_RECENT_MESSAGES)
    recent_messages = message_items[-recent_count:]

    return {
        "assistant_transcript_index": assistant_entry_index,
        "context_message_count": len(message_items),
        "context_message_indices": message_indices,
        "context_chars": len(signature_payload),
        "context_signature_sha256": _sha256_text(signature_payload),
        "context_signature_normalized_sha256": _sha256_text(normalized_payload),
        "prefix_chars": prefix_chars,
        "prefix_signature_sha256": _sha256_text(signature_payload[:prefix_chars]),
        "prefix_signature_normalized_sha256": _sha256_text(normalized_payload[:prefix_chars]),
        "recent_messages": recent_messages,
    }


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_ts(value: Any) -> Optional[float]:
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _normalize_prompt_for_match(prompt: str) -> str:
    text = _ensure_text(prompt)
    text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
    return text


def _extract_request_body_prompt_text(request_body: str) -> str:
    try:
        payload = json.loads(request_body)
    except json.JSONDecodeError:
        return ""
    input_items = payload.get("input")
    if not isinstance(input_items, list):
        return ""
    user_texts: List[str] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "") != "user":
            continue
        user_texts.append(_message_content_to_text(item.get("content")))
    return "\n".join([text for text in user_texts if text]).strip()


def _is_strict_first_turn_request(request_body: str) -> bool:
    try:
        payload = json.loads(request_body)
    except json.JSONDecodeError:
        return False
    input_items = payload.get("input")
    if not isinstance(input_items, list) or len(input_items) != 2:
        return False
    first = input_items[0] if isinstance(input_items[0], dict) else {}
    second = input_items[1] if isinstance(input_items[1], dict) else {}
    return str(first.get("role") or "") == "developer" and str(second.get("role") or "") == "user"


def _load_provider_configs() -> List[Dict[str, Any]]:
    cfg_path = Path.home() / ".openclaw" / "openclaw.json"
    try:
        parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    providers = parsed.get("models", {}).get("providers", {})
    if not isinstance(providers, dict):
        return []
    out: List[Dict[str, Any]] = []
    for provider_id, provider_cfg in providers.items():
        if not isinstance(provider_cfg, dict):
            continue
        base_url = str(provider_cfg.get("baseUrl") or "").rstrip("/")
        api_key = str(provider_cfg.get("apiKey") or "")
        if not base_url or not api_key:
            continue
        out.append(
            {
                "id": str(provider_id),
                "base_url": base_url,
                "api_key": api_key,
                "auth_header": bool(provider_cfg.get("authHeader", True)),
            }
        )
    return out


def _find_provider_config_for_url(url: str) -> Optional[Dict[str, Any]]:
    url = str(url or "")
    best: Optional[Dict[str, Any]] = None
    for provider_cfg in _load_provider_configs():
        base_url = str(provider_cfg.get("base_url") or "")
        if not base_url:
            continue
        if url == base_url or url.startswith(base_url + "/"):
            if best is None or len(base_url) > len(str(best.get("base_url") or "")):
                best = provider_cfg
    return best


def _replay_request_for_usage(url: str, request_body: str) -> Optional[Dict[str, int]]:
    provider_cfg = _find_provider_config_for_url(url)
    if provider_cfg is None:
        return None
    try:
        payload = json.loads(request_body)
    except json.JSONDecodeError:
        return None
    payload["stream"] = False
    payload["max_output_tokens"] = 1
    data = json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if provider_cfg.get("auth_header", True):
        headers["authorization"] = f"Bearer {provider_cfg['api_key']}"
    req = urllib_request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=PINCHBENCH_RECONCILE_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, OSError) as exc:
        logger.debug("Provider usage replay failed for %s: %s", url, exc)
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    usage = parsed.get("usage", {})
    if not isinstance(usage, dict):
        usage = parsed.get("response", {}).get("usage", {})
    if not isinstance(usage, dict):
        return None
    raw_input_tokens = _to_int(usage.get("input_tokens"), _to_int(usage.get("prompt_tokens"), 0))
    raw_cached_tokens = _to_int(
        (usage.get("input_tokens_details") or {}).get("cached_tokens"),
        _to_int((usage.get("prompt_tokens_details") or {}).get("cached_tokens"), 0),
    )
    return {
        "raw_input_tokens": raw_input_tokens,
        "cached_tokens": raw_cached_tokens,
    }


def _read_provider_tap_response_pairs() -> List[Dict[str, Any]]:
    records = _read_jsonl_records(PINCHBENCH_PROVIDER_TAP_PATH)
    pairs: List[Dict[str, Any]] = []
    for rec in records:
        if str(rec.get("method") or "").upper() != "POST":
            continue
        url = str(rec.get("url") or "")
        if not url.endswith("/responses"):
            continue
        request_body = rec.get("requestBody")
        if not isinstance(request_body, str) or not request_body.strip():
            continue
        pairs.append(
            {
                "at": rec.get("at"),
                "at_ts": _parse_iso_ts(rec.get("at")),
                "url": url,
                "request_body": request_body,
                "request_prompt_text": _normalize_prompt_for_match(_extract_request_body_prompt_text(request_body)),
                "is_first_turn": _is_strict_first_turn_request(request_body),
            }
        )
    return pairs


def _read_llm_input_events_for_session(session_id: str) -> List[Dict[str, Any]]:
    events = _read_jsonl_records(PINCHBENCH_LLM_HOOK_TAP_PATH)
    out: List[Dict[str, Any]] = []
    for rec in events:
        if rec.get("hook") != "llm_input":
            continue
        event = rec.get("event")
        if not isinstance(event, dict):
            continue
        if str(event.get("sessionId") or "") != session_id:
            continue
        out.append(
            {
                "at": rec.get("at"),
                "at_ts": _parse_iso_ts(rec.get("at")),
                "prompt": str(event.get("prompt") or ""),
                "provider": str(event.get("provider") or ""),
                "model": str(event.get("model") or ""),
            }
        )
    return out


def _match_provider_requests_to_session_calls(session_id: str) -> List[Optional[Dict[str, Any]]]:
    llm_inputs = _read_llm_input_events_for_session(session_id)
    if not llm_inputs:
        return []
    request_pairs = _read_provider_tap_response_pairs()
    used_indices: set[int] = set()
    matched: List[Optional[Dict[str, Any]]] = []
    for input_event in llm_inputs:
        prompt_match = _normalize_prompt_for_match(input_event.get("prompt", ""))
        prompt_key = prompt_match[:120]
        llm_at = input_event.get("at_ts")
        best_idx: Optional[int] = None
        best_score: Optional[Tuple[int, float]] = None
        for idx, pair in enumerate(request_pairs):
            if idx in used_indices:
                continue
            pair_at = pair.get("at_ts")
            if llm_at is not None and pair_at is not None:
                delta = pair_at - llm_at
                if delta < -2.0 or delta > 10.0:
                    continue
                distance = abs(delta)
            else:
                distance = float("inf")
            prompt_text = str(pair.get("request_prompt_text") or "")
            prompt_hit = int(bool(prompt_key) and prompt_key in prompt_text)
            score = (-prompt_hit, distance)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            matched.append(None)
            continue
        used_indices.add(best_idx)
        matched.append(request_pairs[best_idx])
    return matched


def _summarize_llm_calls(llm_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_hit_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": len(llm_calls),
        "usage_available_count": 0,
        "usage_missing_count": 0,
    }
    for call in llm_calls:
        input_tokens = _to_int(call.get("input_tokens"), 0)
        output_tokens = _to_int(call.get("output_tokens"), 0)
        cache_read_tokens = _to_int(call.get("cache_read_tokens"), _to_int(call.get("cached_tokens"), 0))
        cache_write_tokens = _to_int(
            call.get("cache_write_tokens"),
            _to_int(call.get("cache_creation_input_tokens"), 0),
        )
        total_tokens = _to_int(
            call.get("total_tokens"),
            input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        )
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["cache_read_tokens"] += cache_read_tokens
        totals["cache_write_tokens"] += cache_write_tokens
        totals["cache_hit_tokens"] += cache_read_tokens
        totals["total_tokens"] += total_tokens
        totals["cost_usd"] += _to_float(call.get("cost_usd"), 0.0)
        if input_tokens > 0 or output_tokens > 0 or total_tokens > 0:
            totals["usage_available_count"] += 1
        else:
            totals["usage_missing_count"] += 1
    return totals


def _reconcile_llm_calls_with_provider_tap(
    llm_calls: List[Dict[str, Any]],
    *,
    transcript_session_id: Optional[str],
) -> List[Dict[str, Any]]:
    if not PINCHBENCH_RECONCILE_PROVIDER_USAGE:
        return llm_calls
    if not transcript_session_id:
        return llm_calls
    if not PINCHBENCH_PROVIDER_TAP_PATH.exists() or not PINCHBENCH_LLM_HOOK_TAP_PATH.exists():
        return llm_calls
    matched_requests = _match_provider_requests_to_session_calls(transcript_session_id)
    if not matched_requests:
        return llm_calls
    reconciled_calls: List[Dict[str, Any]] = []
    for idx, call in enumerate(llm_calls):
        next_call = dict(call)
        next_call.setdefault("usage_source", "transcript")
        request_match = matched_requests[idx] if idx < len(matched_requests) else None
        if (
            request_match
            and str(next_call.get("api") or "") == "openai-responses"
            and _to_int(next_call.get("cache_read_tokens"), 0) == 0
            and bool(request_match.get("is_first_turn"))
        ):
            replay_usage = _replay_request_for_usage(
                str(request_match.get("url") or ""),
                str(request_match.get("request_body") or ""),
            )
            if replay_usage and replay_usage.get("cached_tokens", 0) > 0:
                raw_input_tokens = _to_int(replay_usage.get("raw_input_tokens"), 0)
                cached_tokens = _to_int(replay_usage.get("cached_tokens"), 0)
                uncached_input_tokens = max(0, raw_input_tokens - cached_tokens)
                output_tokens = _to_int(next_call.get("output_tokens"), 0)
                cache_write_tokens = _to_int(next_call.get("cache_write_tokens"), 0)
                next_call["input_tokens_raw"] = raw_input_tokens
                next_call["cached_tokens_raw"] = cached_tokens
                next_call["input_tokens"] = uncached_input_tokens
                next_call["cache_read_tokens"] = cached_tokens
                next_call["total_tokens"] = (
                    uncached_input_tokens + output_tokens + cached_tokens + cache_write_tokens
                )
                next_call["usage_source"] = "provider_tap_replay"
                next_call["usage_reconciled"] = True
                next_call["provider_request_at"] = request_match.get("at")
        reconciled_calls.append(next_call)
    return reconciled_calls



def _get_agent_workspace(agent_id: str) -> Path | None:
    """Get the workspace path for an agent from OpenClaw config."""
    try:
        list_result = subprocess.run(
            ["openclaw", "agents", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if list_result.returncode != 0:
            return None

        # Parse the agent list output to find workspace
        # OpenClaw normalizes colons to dashes in agent names, so check both.
        normalized_id = agent_id.replace(":", "-")
        lines = list_result.stdout.split("\n")
        found_agent = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f"- {agent_id}") or stripped.startswith(f"- {normalized_id}"):
                found_agent = True
            elif found_agent and "Workspace:" in line:
                workspace_str = line.split("Workspace:")[1].strip()
                # Expand ~ if present
                if workspace_str.startswith("~/"):
                    workspace_str = str(Path.home() / workspace_str[2:])
                return Path(workspace_str)
            elif found_agent and line.strip().startswith("-"):
                # Found next agent, stop looking
                break
        return None
    except Exception as exc:
        logger.warning("Failed to get agent workspace: %s", exc)
        return None


def ensure_agent_exists(agent_id: str, model_id: str, workspace_dir: Path) -> bool:
    """Ensure the OpenClaw agent exists with the correct workspace.

    If the agent already exists but points to a different workspace, it is
    deleted and recreated so that the new workspace takes effect.
    Returns True if the agent was (re)created.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    try:
        list_result = subprocess.run(
            ["openclaw", "agents", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.error("openclaw CLI not found while listing agents")
        return False

    if list_result.returncode == 0:
        # Check for exact agent ID match — avoid substring false positives
        # (e.g. "bench-foo-4" matching "bench-foo-4-5" in the output).
        # Output format is "- <agent_id>" or "- <agent_id> (default)" per line.
        # OpenClaw normalizes colons to dashes in directory/display names, so
        # also check the normalized form.
        existing_agents = set()
        for line in list_result.stdout.splitlines():
            line = line.strip()
            if line.startswith("- "):
                # Extract agent name: "- bench-foo-4-5" or "- main (default)"
                name_part = line[2:].split()[0] if line[2:].strip() else ""
                if name_part:
                    existing_agents.add(name_part)
        normalized_id = agent_id.replace(":", "-")
        if agent_id in existing_agents or normalized_id in existing_agents:
            # Agent exists — check if workspace matches
            current_workspace = _get_agent_workspace(agent_id)
            if (
                current_workspace is not None
                and current_workspace.resolve() == workspace_dir.resolve()
            ):
                logger.info("Agent %s already exists with correct workspace", agent_id)
                return False
            # Workspace is stale or unknown — delete and recreate
            delete_name = normalized_id if normalized_id in existing_agents else agent_id
            logger.info(
                "Agent %s exists with stale workspace (%s != %s), recreating",
                agent_id,
                current_workspace,
                workspace_dir,
            )
            subprocess.run(
                ["openclaw", "agents", "delete", delete_name, "--force"],
                capture_output=True,
                text=True,
                check=False,
            )

    logger.info("Creating OpenClaw agent %s", agent_id)
    try:
        create_result = subprocess.run(
            [
                "openclaw",
                "agents",
                "add",
                agent_id,
                "--model",
                model_id,
                "--workspace",
                str(workspace_dir),
                "--non-interactive",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.error("openclaw CLI not found while creating agent")
        return False

    if create_result.returncode != 0:
        logger.warning(
            "Agent creation returned %s: %s", create_result.returncode, create_result.stderr
        )
    return True


def cleanup_agent_sessions(agent_id: str) -> None:
    """Remove stored session transcripts for an agent to avoid unbounded growth."""
    agent_dir = _get_agent_store_dir(agent_id)
    sessions_dir = agent_dir / "sessions"
    if not sessions_dir.exists():
        return
    removed = 0
    for pattern in ("*.jsonl", "*.jsonl.lock"):
        for path in sessions_dir.glob(pattern):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Failed to remove session file %s: %s", path, exc)
    sessions_store = sessions_dir / "sessions.json"
    if sessions_store.exists():
        try:
            sessions_store.unlink()
        except OSError as exc:
            logger.warning("Failed to remove session store %s: %s", sessions_store, exc)
    if removed:
        logger.info("Removed %s old OpenClaw session transcripts for %s", removed, agent_id)


def prepare_task_workspace(
    skill_dir: Path,
    run_id: str,
    task: Task,
    agent_id: str,
    workspace_override: Path | None = None,
) -> Path:
    """
    Prepare workspace for a task by copying fixtures.
    Uses the agent's configured workspace to ensure files are in the right place.
    """
    import shutil

    # Prefer explicit workspace from caller (parallel-safe).
    workspace = workspace_override
    if workspace is None:
        # Get agent's workspace from agent config
        workspace = _get_agent_workspace(agent_id)
    if workspace is None:
        # Fallback to task-specific workspace if agent workspace not found
        logger.warning("Could not find agent workspace, using fallback")
        workspace = Path(f"/tmp/pinchbench/{run_id}/{task.task_id}")

    # Clear workspace before each task to prevent stale files from prior tasks
    # from contaminating the agent's context.
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    for file_spec in task.workspace_files:
        if "content" in file_spec:
            dest = workspace / file_spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(file_spec["content"])
            continue

        source = skill_dir / "assets" / file_spec["source"]
        dest = workspace / file_spec["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(source.read_bytes())
        except FileNotFoundError:
            logger.error("Workspace file not found: %s", source)
            raise

    return workspace


def _get_agent_store_dir(agent_id: str) -> Path:
    base_dir = Path.home() / ".openclaw" / "agents"
    direct_dir = base_dir / agent_id
    if direct_dir.exists():
        return direct_dir
    normalized_dir = base_dir / agent_id.replace(":", "-")
    if normalized_dir.exists():
        return normalized_dir
    return direct_dir


def _resolve_session_id_from_store(agent_id: str) -> str | None:
    agent_dir = _get_agent_store_dir(agent_id)
    sessions_store = agent_dir / "sessions" / "sessions.json"
    if not sessions_store.exists():
        return None
    try:
        sessions_payload = json.loads(sessions_store.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse sessions store: %s", exc)
        return None
    if not isinstance(sessions_payload, dict):
        return None

    normalized_id = agent_id.replace(":", "-")
    preferred_keys = [
        f"agent:{agent_id}:main",
        f"agent:{agent_id}:default",
        f"agent:{normalized_id}:main",
        f"agent:{normalized_id}:default",
    ]
    for key in preferred_keys:
        entry = sessions_payload.get(key)
        if isinstance(entry, dict) and entry.get("sessionId"):
            return entry["sessionId"]

    newest_entry = None
    newest_timestamp = -1
    for entry in sessions_payload.values():
        if not isinstance(entry, dict):
            continue
        if "sessionId" not in entry:
            continue
        updated_at = entry.get("updatedAt")
        if isinstance(updated_at, (int, float)) and updated_at > newest_timestamp:
            newest_timestamp = updated_at
            newest_entry = entry
    if newest_entry:
        return newest_entry.get("sessionId")
    return None


def _find_recent_session_path(agent_dir: Path, started_at: float) -> Path | None:
    sessions_dir = agent_dir / "sessions"
    if not sessions_dir.exists():
        return None
    candidates = list(sessions_dir.glob("*.jsonl"))
    if not candidates:
        return None
    tolerance_seconds = 5.0
    recent_candidates = [
        path for path in candidates if path.stat().st_mtime >= (started_at - tolerance_seconds)
    ]
    pool = recent_candidates or candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def _wait_for_transcript_unlock(transcript_path: Path, max_wait_seconds: float = 12.0) -> None:
    """
    Wait until the transcript lock file disappears.

    OpenClaw writes transcript as <uuid>.jsonl while holding <uuid>.jsonl.lock.
    Reading too early can race with writer and yield false negatives.
    """
    lock_path = transcript_path.with_name(f"{transcript_path.name}.lock")
    deadline = time.time() + max_wait_seconds
    while lock_path.exists() and time.time() < deadline:
        time.sleep(0.2)


def _wait_for_transcript_appearance(
    agent_dir: Path,
    session_id: str,
    started_at: float,
    max_wait_seconds: float = 30.0,
) -> Path | None:
    """
    Wait for transcript file to appear when session lock exists.

    In some runs OpenClaw may keep <uuid>.jsonl.lock for several seconds before
    the corresponding .jsonl becomes visible, so a short single sleep is brittle.
    """
    deadline = time.time() + max_wait_seconds
    sessions_dir = agent_dir / "sessions"
    while time.time() < deadline:
        resolved_session_id = _resolve_session_id_from_store(agent_dir.name)
        if resolved_session_id:
            candidate = sessions_dir / f"{resolved_session_id}.jsonl"
            if candidate.exists():
                return candidate
        recent_path = _find_recent_session_path(agent_dir, started_at)
        if recent_path is not None:
            return recent_path
        direct_path = sessions_dir / f"{session_id}.jsonl"
        if direct_path.exists():
            return direct_path
        time.sleep(0.5)
    return None


def _read_transcript_file(transcript_path: Path) -> List[Dict[str, Any]]:
    _wait_for_transcript_unlock(transcript_path)

    transcript: List[Dict[str, Any]] = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            transcript.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse transcript line: %s", exc)
            transcript.append({"raw": line, "parse_error": str(exc)})
    return transcript


def _load_transcript_bundle(agent_id: str, session_id: str, started_at: float) -> Dict[str, Any]:
    agent_dir = _get_agent_store_dir(agent_id)
    transcript_path = None
    resolved_session_id: Optional[str] = None

    # OpenClaw ignores the --session-id we pass and generates its own UUID-based
    # session ID internally.  We need to discover the actual transcript path.
    #
    # Strategy (with retries to handle write-delay):
    #   1. Resolve the real session ID from sessions.json
    #   2. Glob for any .jsonl in the sessions dir (most-recently-modified)
    #   3. Try our passed-in session ID as a last resort
    # Two-phase wait:
    # - discovery retries (existing behavior)
    # - if lock files are present but jsonl is not yet visible, allow extra wait
    max_attempts = 10
    for attempt in range(max_attempts):
        # 1. Try sessions.json first — OpenClaw writes the real UUID here
        candidate_session_id = _resolve_session_id_from_store(agent_id)
        if candidate_session_id:
            resolved_session_id = candidate_session_id
            candidate = agent_dir / "sessions" / f"{candidate_session_id}.jsonl"
            if candidate.exists():
                transcript_path = candidate
                logger.info(
                    "Found transcript via sessions.json: %s (attempt %s)",
                    candidate.name,
                    attempt + 1,
                )
                break

        # 2. Glob fallback — pick the most recently modified .jsonl
        recent_path = _find_recent_session_path(agent_dir, started_at)
        if recent_path is not None:
            transcript_path = recent_path
            logger.info(
                "Found transcript via glob fallback: %s (attempt %s)",
                recent_path.name,
                attempt + 1,
            )
            break

        # 3. Try our passed-in session ID (unlikely to work, but check anyway)
        direct_path = agent_dir / "sessions" / f"{session_id}.jsonl"
        if direct_path.exists():
            transcript_path = direct_path
            logger.info(
                "Found transcript via passed session ID: %s (attempt %s)",
                direct_path.name,
                attempt + 1,
            )
            break

        if attempt < (max_attempts - 1):
            time.sleep(1.0)

    if transcript_path is None:
        sessions_dir = agent_dir / "sessions"
        if sessions_dir.exists():
            all_files = list(sessions_dir.iterdir())
            lock_files = [f for f in all_files if f.name.endswith(".jsonl.lock")]
            if lock_files:
                logger.info(
                    "Transcript still locked for agent %s (%s lock files); waiting for transcript file.",
                    agent_id,
                    len(lock_files),
                )
                waited_path = _wait_for_transcript_appearance(
                    agent_dir,
                    session_id,
                    started_at,
                    max_wait_seconds=30.0,
                )
                if waited_path is not None:
                    transcript_path = waited_path
                    resolved_session_id = waited_path.stem
                    logger.info("Found transcript after lock wait: %s", waited_path.name)
            all_files = list(sessions_dir.iterdir())
        if transcript_path is None and sessions_dir.exists():
            logger.warning(
                "Transcript not found for agent %s. Sessions dir contents: %s",
                agent_id,
                [f.name for f in all_files],
            )
        else:
            logger.warning(
                "Transcript not found — sessions dir does not exist: %s",
                sessions_dir,
            )
        return {
            "transcript": [],
            "transcript_path": None,
            "transcript_session_id": resolved_session_id,
        }

    return {
        "transcript": _read_transcript_file(transcript_path),
        "transcript_path": str(transcript_path),
        "transcript_session_id": transcript_path.stem or resolved_session_id or session_id,
    }


def _load_transcript(agent_id: str, session_id: str, started_at: float) -> List[Dict[str, Any]]:
    bundle = _load_transcript_bundle(agent_id, session_id, started_at)
    return bundle.get("transcript", [])


def _extract_usage_from_transcript(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum token usage and cost from all assistant messages in transcript."""
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_hit_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
        "usage_available_count": 0,
        "usage_missing_count": 0,
    }

    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue
        totals["request_count"] += 1
        usage = msg.get("usage", {}) if isinstance(msg.get("usage"), dict) else {}
        provider_raw = usage.get("providerRaw", {})
        if not isinstance(provider_raw, dict):
            provider_raw = {}

        input_tokens = _to_int(usage.get("input"), _to_int(usage.get("input_tokens"), _to_int(usage.get("prompt_tokens"), 0)))
        output_tokens = _to_int(usage.get("output"), _to_int(usage.get("output_tokens"), _to_int(usage.get("completion_tokens"), 0)))
        if input_tokens == 0:
            input_tokens = _to_int(provider_raw.get("input_tokens"), _to_int(provider_raw.get("prompt_tokens"), 0))
        if output_tokens == 0:
            output_tokens = _to_int(provider_raw.get("output_tokens"), _to_int(provider_raw.get("completion_tokens"), 0))

        # Cross-provider cache fields:
        # - OpenClaw transcript style: cacheRead/cacheWrite
        # - Anthropic style: cache_read_input_tokens/cache_creation_input_tokens
        # - OpenAI style: input_tokens_details.cached_tokens / prompt_tokens_details.cached_tokens
        provider_raw_cached_tokens = _to_int(
            (provider_raw.get("input_tokens_details") or {}).get("cached_tokens"),
            _to_int((provider_raw.get("prompt_tokens_details") or {}).get("cached_tokens"), 0),
        )
        cached_tokens = _to_int(
            usage.get("cachedTokens"),
            _to_int(
                usage.get("cached_tokens"),
                _to_int(
                    (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                    _to_int((usage.get("prompt_tokens_details") or {}).get("cached_tokens"), provider_raw_cached_tokens),
                ),
            ),
        )
        cache_read_tokens = _to_int(
            usage.get("cacheRead"),
            _to_int(
                usage.get("cache_read_tokens"),
                _to_int(usage.get("cache_read_input_tokens"), cached_tokens),
            ),
        )
        cache_write_tokens = _to_int(
            usage.get("cacheWrite"),
            _to_int(usage.get("cache_write_tokens"), _to_int(usage.get("cache_creation_input_tokens"), 0)),
        )
        total_tokens = _to_int(
            usage.get("totalTokens"),
            _to_int(usage.get("total_tokens"), input_tokens + output_tokens),
        )
        if total_tokens == 0:
            total_tokens = _to_int(provider_raw.get("total_tokens"), input_tokens + output_tokens)

        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        totals["cache_read_tokens"] += cache_read_tokens
        totals["cache_write_tokens"] += cache_write_tokens
        totals["cache_hit_tokens"] += cache_read_tokens
        totals["total_tokens"] += total_tokens
        cost = usage.get("cost", {})
        totals["cost_usd"] += _to_float(cost.get("total"), _to_float(usage.get("cost_usd"), 0.0))
        if input_tokens > 0 or output_tokens > 0 or total_tokens > 0:
            totals["usage_available_count"] += 1
        else:
            totals["usage_missing_count"] += 1

    return totals


def _extract_llm_calls_from_transcript(transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract per-assistant-message LLM call metadata for debugging and audit."""
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    calls: List[Dict[str, Any]] = []
    for idx, entry in enumerate(transcript):
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue

        usage = msg.get("usage", {}) if isinstance(msg.get("usage"), dict) else {}
        provider_raw = usage.get("providerRaw", {}) if isinstance(usage.get("providerRaw"), dict) else {}
        provider_raw_cached_tokens = _to_int(
            (provider_raw.get("input_tokens_details") or {}).get("cached_tokens"),
            _to_int((provider_raw.get("prompt_tokens_details") or {}).get("cached_tokens"), 0),
        )
        cached_tokens = _to_int(
            usage.get("cachedTokens"),
            _to_int(
                usage.get("cached_tokens"),
                _to_int(
                    (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                    _to_int((usage.get("prompt_tokens_details") or {}).get("cached_tokens"), provider_raw_cached_tokens),
                ),
            ),
        )
        cost_obj = usage.get("cost", {}) if isinstance(usage.get("cost"), dict) else {}
        context_detail = _build_call_context_detail(transcript, idx)
        calls.append({
            "index": idx,
            "timestamp": msg.get("timestamp") or entry.get("timestamp"),
            "provider": msg.get("provider"),
            "model": msg.get("model"),
            "api": msg.get("api"),
            "stop_reason": msg.get("stopReason"),
            "input_tokens": _to_int(usage.get("input"), _to_int(usage.get("input_tokens"), _to_int(usage.get("prompt_tokens"), 0))),
            "output_tokens": _to_int(usage.get("output"), _to_int(usage.get("output_tokens"), _to_int(usage.get("completion_tokens"), 0))),
            "cache_read_tokens": _to_int(
                usage.get("cacheRead"),
                _to_int(usage.get("cache_read_tokens"), _to_int(usage.get("cache_read_input_tokens"), cached_tokens)),
            ),
            "cache_write_tokens": _to_int(usage.get("cacheWrite"), _to_int(usage.get("cache_write_tokens"), _to_int(usage.get("cache_creation_input_tokens"), 0))),
            "total_tokens": _to_int(usage.get("totalTokens"), _to_int(usage.get("total_tokens"), 0)),
            "cost_usd": _to_float(cost_obj.get("total"), _to_float(usage.get("cost_usd"), 0.0)),
            "context_detail": context_detail,
        })
    return calls


def execute_openclaw_task(
    *,
    task: Task,
    agent_id: str,
    model_id: str,
    run_id: str,
    timeout_multiplier: float,
    skill_dir: Path,
    agent_workspace: Path | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    logger.info("🤖 Agent [%s] starting task: %s", agent_id, task.task_id)
    logger.info("   Task: %s", task.name)
    logger.info("   Category: %s", task.category)
    if verbose:
        logger.info(
            "   Prompt: %s", task.prompt[:500] + "..." if len(task.prompt) > 500 else task.prompt
        )

    # Clean up previous session transcripts so we can reliably find this task's
    # transcript (OpenClaw uses its own UUID-based naming, not our session ID).
    cleanup_agent_sessions(agent_id)

    start_time = time.time()
    workspace = prepare_task_workspace(
        skill_dir=skill_dir,
        run_id=run_id,
        task=task,
        agent_id=agent_id,
        workspace_override=agent_workspace,
    )
    session_id = f"{task.task_id}_{int(time.time() * 1000)}"
    timeout_seconds = task.timeout_seconds * timeout_multiplier
    stdout = ""
    stderr = ""
    exit_code = -1
    timed_out = False

    def _run_once(current_session_id: str, current_timeout_seconds: float) -> tuple[str, str, int, bool]:
        run_stdout = ""
        run_stderr = ""
        run_exit_code = -1
        run_timed_out = False
        try:
            result = subprocess.run(
                [
                    "openclaw",
                    "agent",
                    "--agent",
                    agent_id,
                    "--session-id",
                    current_session_id,
                    "--message",
                    task.prompt,
                ],
                capture_output=True,
                text=True,
                cwd=str(workspace),
                timeout=current_timeout_seconds,
                check=False,
            )
            run_stdout = result.stdout
            run_stderr = result.stderr
            run_exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            run_timed_out = True
            run_stdout = _ensure_text(exc.stdout)
            run_stderr = _ensure_text(exc.stderr)
        except FileNotFoundError as exc:
            run_stderr = f"openclaw command not found: {exc}"
        return run_stdout, run_stderr, run_exit_code, run_timed_out

    stdout, stderr, exit_code, timed_out = _run_once(session_id, timeout_seconds)

    transcript_bundle = _load_transcript_bundle(agent_id, session_id, start_time)
    transcript = transcript_bundle.get("transcript", [])

    # Parallel runs occasionally race with transcript persistence. Retry once
    # to reduce false negatives when execution succeeded but transcript is empty.
    if (
        not transcript
        and not timed_out
        and exit_code in (0, -1)
        and "openclaw command not found" not in str(stderr)
    ):
        logger.warning(
            "Empty transcript for %s; retrying task execution once (session sync fallback).",
            task.task_id,
        )
        cleanup_agent_sessions(agent_id)
        retry_session_id = f"{session_id}_retry"
        retry_started_at = time.time()
        retry_stdout, retry_stderr, retry_exit_code, retry_timed_out = _run_once(
            retry_session_id, timeout_seconds
        )
        stdout = f"{stdout}\n{retry_stdout}".strip() if stdout else retry_stdout
        stderr = f"{stderr}\n{retry_stderr}".strip() if stderr else retry_stderr
        exit_code = retry_exit_code
        timed_out = retry_timed_out
        transcript_bundle = _load_transcript_bundle(agent_id, retry_session_id, retry_started_at)
        transcript = transcript_bundle.get("transcript", [])

    llm_calls = _extract_llm_calls_from_transcript(transcript)
    transcript_session_id = transcript_bundle.get("transcript_session_id")
    llm_calls = _reconcile_llm_calls_with_provider_tap(
        llm_calls,
        transcript_session_id=str(transcript_session_id) if transcript_session_id else None,
    )
    usage = _summarize_llm_calls(llm_calls) if llm_calls else _extract_usage_from_transcript(transcript)
    usage["usage_source"] = (
        "llm_calls_reconciled"
        if any(bool(call.get("usage_reconciled")) for call in llm_calls)
        else ("llm_calls" if llm_calls else "transcript")
    )
    usage["usage_reconciled_count"] = sum(1 for call in llm_calls if call.get("usage_reconciled"))
    execution_time = time.time() - start_time

    status = "success"
    if timed_out:
        status = "timeout"
    if not transcript:
        status = "error"
    if exit_code not in (0, -1) and not timed_out:
        status = "error"
    if stderr and "openclaw command not found" in str(stderr):
        status = "error"

    # Verbose logging for debugging
    if verbose:
        logger.info("   [VERBOSE] Exit code: %s", exit_code)
        logger.info("   [VERBOSE] Execution time: %.2fs", execution_time)
        logger.info("   [VERBOSE] Workspace: %s", workspace)
        if stdout:
            logger.info("   [VERBOSE] Stdout (first 1000 chars):\n%s", stdout[:1000])
        if stderr:
            logger.info("   [VERBOSE] Stderr:\n%s", stderr[:1000])
        logger.info("   [VERBOSE] Transcript entries: %d", len(transcript))

        # Show agent responses from transcript
        for entry in transcript:
            if entry.get("type") == "message":
                msg = entry.get("message", {})
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "assistant":
                    # Truncate long responses
                    preview = content[:500] + "..." if len(content) > 500 else content
                    logger.info("   [VERBOSE] Agent response: %s", preview)
                elif role == "user":
                    preview = content[:200] + "..." if len(content) > 200 else content
                    logger.info("   [VERBOSE] User message: %s", preview)

        # Show workspace files after task
        if workspace.exists():
            logger.info("   [VERBOSE] Workspace files after task:")
            for f in sorted(workspace.rglob("*")):
                if f.is_file():
                    try:
                        size = f.stat().st_size
                        logger.info("      %s (%d bytes)", f.relative_to(workspace), size)
                    except OSError:
                        logger.info("      %s", f.relative_to(workspace))

    return {
        "agent_id": agent_id,
        "task_id": task.task_id,
        "status": status,
        "transcript": transcript,
        "transcript_session_id": transcript_session_id,
        "transcript_path": transcript_bundle.get("transcript_path"),
        "llm_calls": llm_calls,
        "llm_models": sorted({str(call.get("model")) for call in llm_calls if call.get("model")}),
        "usage": usage,
        "workspace": str(workspace),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_time": execution_time,
        "stdout": stdout,
        "stderr": stderr,
    }


def run_openclaw_prompt(
    *,
    agent_id: str,
    prompt: str,
    workspace: Path,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Run a single OpenClaw prompt for helper agents like the judge."""
    # Clean up previous session transcripts so we can reliably find this
    # prompt's transcript (OpenClaw uses its own UUID-based naming).
    cleanup_agent_sessions(agent_id)

    start_time = time.time()
    workspace.mkdir(parents=True, exist_ok=True)
    session_id = f"judge_{int(time.time() * 1000)}"
    stdout = ""
    stderr = ""
    exit_code = -1
    timed_out = False

    chunks = [
        prompt[i : i + MAX_OPENCLAW_MESSAGE_CHARS]
        for i in range(0, max(1, len(prompt)), MAX_OPENCLAW_MESSAGE_CHARS)
    ]
    if len(chunks) > 1:
        total_chunks = len(chunks)
        chunks = [
            (
                f"You are receiving a long prompt in {total_chunks} parts.\n"
                f"Ignore and do not respond until the final part.\n\n"
                f"Part 1/{total_chunks}:\n{chunks[0]}"
            )
        ] + [
            (
                f"Part {i + 2}/{total_chunks}:\n{chunks[i + 1]}"
                if i + 2 < total_chunks
                else (
                    f"Part {i + 2}/{total_chunks} (final):\n{chunks[i + 1]}\n"
                    "All parts received. Proceed with final judgment now."
                )
            )
            for i in range(0, total_chunks - 1)
        ]
    for chunk in chunks:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            timed_out = True
            break
        try:
            result = subprocess.run(
                [
                    "openclaw",
                    "agent",
                    "--agent",
                    agent_id,
                    "--session-id",
                    session_id,
                    "--message",
                    chunk,
                ],
                capture_output=True,
                text=True,
                cwd=str(workspace),
                timeout=remaining,
                check=False,
            )
            stdout += result.stdout
            stderr += result.stderr
            exit_code = result.returncode
            if result.returncode not in (0, -1) and not timed_out:
                break
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout += _ensure_text(exc.stdout)
            stderr += _ensure_text(exc.stderr)
            break
        except FileNotFoundError as exc:
            stderr += f"openclaw command not found: {exc}"
            break

    transcript = _load_transcript(agent_id, session_id, start_time)
    execution_time = time.time() - start_time

    status = "success"
    if timed_out:
        status = "timeout"
    if not transcript:
        status = "error"
    if exit_code not in (0, -1) and not timed_out:
        status = "error"
    if stderr and "openclaw command not found" in str(stderr):
        status = "error"

    return {
        "agent_id": agent_id,
        "status": status,
        "transcript": transcript,
        "workspace": str(workspace),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_time": execution_time,
        "stdout": stdout,
        "stderr": stderr,
    }
