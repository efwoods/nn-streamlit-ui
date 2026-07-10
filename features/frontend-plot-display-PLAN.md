# f-frontend-plot-display

Render data-analysis artifacts (plots, reports) in the Streamlit frontend:
(1) **immediately** show any plot the avatar creates during a turn, and
(2) browse/display **every** artifact already persisted in the avatar's
`data_created` namespace.

## Verified current state (2026-07-10)

- Artifacts live in the LangGraph store under `(<user_id>, <assistant_id>,
  "data_created")` — store `prefix` = `<user_id>.<assistant_id>.data_created`.
  Confirmed real data for avatar `cd8ddcc4…`: `/plot.png` (base64, ~92 KB) and
  `/report.md` (utf-8). Ingested sources sit in the sibling `.data_ingested`
  namespace (28 items).
- `persist_created_artifact` (`analysis_tools.py:345`) already writes each file
  as `{content, encoding: base64|utf-8}` — but does **not** store `content_type`.
- The frontend (`frontend/studio_chat_app.py`) renders images only for the
  avatar reference image and user uploads (`st.image` at ~443/459). There is
  **no path** that surfaces an assistant-produced artifact into the chat.
- Namespaces are per-user **and** per-avatar (not per-thread): the gallery is
  avatar-wide and survives across conversations.

## Design overview

Two-tier delivery, both reading the same `data_created` namespace:

- **Immediate (per-turn):** capture a watermark timestamp before sending a
  message; after the turn's `done`, fetch artifacts created since the watermark
  and render them inline in that assistant bubble.
- **Gallery (avatar-wide):** an "Artifacts" panel lists every `data_created`
  item for the selected avatar and renders/downloads each on demand.

Metadata-only listing + lazy per-key byte fetch, so a large/many-image gallery
never ships every base64 blob at once.

## Backend changes (`src/api/webapp.py` + `analysis_tools.py`)

1. **Persist `content_type`** in `persist_created_artifact`
   (`analysis_tools.py:345`): infer via `mimetypes.guess_type(candidate_path.name)`
   at save time and store it in the record. Backfill is unnecessary — the list
   endpoint infers MIME from the key extension when `content_type` is absent, so
   the two existing artifacts still render.

2. **`GET /avatar/{assistant_id}/artifacts`** — list metadata only.
   - Auth: `get_current_user_or_anonymous_user` → `user_id`.
   - Namespace: `created_namespace(user_id, assistant_id)`; `store.asearch(ns,
     limit=…)`.
   - Returns `[{key, content_type (stored or inferred), encoding, size,
     created_at}]`, newest first. No `content` field.

3. **`GET /avatar/{assistant_id}/artifacts/content?key=/plot.png`** — one
   artifact's bytes.
   - base64 → raw bytes with `Content-Type` from stored/inferred MIME and
     `Cache-Control`, so `st.image` / `<img>` can consume it directly; utf-8
     artifacts return text with the right type. Validate `key` is one of the
     namespace's keys (no traversal).

   Both endpoints are added to the FastAPI app that `langgraph.json` already
   serves, so they route through `api.neuralnexus.site` exactly like `/message`.

## Frontend changes (`frontend/studio_chat_app.py`)

1. **API helpers**
   - `list_created_artifacts(assistant_id) -> list[dict]` → GET list.
   - `artifact_content_url(assistant_id, key) -> str` → the `…/artifacts/content`
     URL (with `api_key`) for `st.image`, or `fetch_artifact_bytes(...)` when the
     Streamlit deployment can't hand an authed URL to the browser (fetch bytes
     server-side and pass to `st.image`).

2. **`render_artifact(meta, assistant_id)`** — MIME-routed:
   - `image/*` → `st.image` (svg via `st.image`/inline HTML).
   - `text/markdown` → `st.markdown`; other `text/*`/`application/json` →
     `st.code` / `st.json` (small) or download.
   - else → `st.download_button`.
   - Always a caption (key + created_at) and a download button.

3. **Immediate per-turn display**
   - Before `api_send_message`, record `turn_started_at = now(UTC)`.
   - After the stream returns `done`, call `list_created_artifacts` and keep
     items with `created_at >= turn_started_at`; store their metadata on the
     assistant message dict (`message["artifacts"] = [...]`).
   - The chat renderer draws `message["artifacts"]` under the assistant text, so
     Streamlit reruns keep showing them (history-persisted, not transient).

4. **Artifacts gallery panel** (sidebar expander or a tab)
   - "🖼 Artifacts (N)" for the selected avatar; lists all `data_created` items
     via `list_created_artifacts`, each drawn with `render_artifact`.
   - Manual **Refresh**; auto-refresh once after each completed turn.
   - Empty state: "No artifacts yet — ask the avatar to analyze data and plot."

## Data flow (immediate case)

```
user msg ──▶ POST /message (SSE)
             avatar: execute → write plot.png → persist_created_artifact
                                                   └─ store: data_created//plot.png
           ◀── done
frontend: GET /avatar/{id}/artifacts  (filter created_at ≥ turn_started_at)
          render_artifact(/plot.png) ─▶ GET …/artifacts/content?key=/plot.png ─▶ st.image
```

## Edge cases & decisions

- **MIME when unknown:** infer from extension; unknown → `application/octet-stream`
  + download button (never a broken `st.image`).
- **Large images:** metadata list carries `size`; content fetched lazily per
  rendered item; consider a soft cap / "show" toggle above e.g. 5 MB.
- **De-dupe across turns:** a re-persisted key (same name) updates in place; the
  per-turn filter is by `created_at`, the gallery is keyed by `key`.
- **Auth in Streamlit Cloud:** if handing an authed URL to the browser `<img>`
  is not acceptable, fetch bytes server-side in the helper and pass to
  `st.image` (no token in DOM). Pick one during implementation.
- **Anonymous users:** endpoints work the same; namespace just keys off the
  anonymous user id.

## Task checklist

- [ ] `analysis_tools.py`: store `content_type` in `persist_created_artifact`.
- [ ] `webapp.py`: `GET /avatar/{assistant_id}/artifacts` (metadata list).
- [ ] `webapp.py`: `GET /avatar/{assistant_id}/artifacts/content` (bytes).
- [ ] `studio_chat_app.py`: list/content helpers + `render_artifact`.
- [ ] `studio_chat_app.py`: per-turn watermark → inline artifacts on the message.
- [ ] `studio_chat_app.py`: avatar-wide Artifacts gallery panel.
- [ ] Verify end-to-end against the two live artifacts (`/plot.png`, `/report.md`).

## Verification

Drive it against real data already in the store: select avatar `cd8ddcc4…`, open
the Artifacts panel → `/plot.png` renders as an image and `/report.md` as
markdown; then ask the avatar to produce a new plot and confirm it appears inline
in that turn without a manual refresh.
