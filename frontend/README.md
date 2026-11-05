# Developer Support Agent: React frontend

A browser UI for the FastAPI backend in this repository (`python -m app.api`).
It talks to the real API only. Every endpoint, field and status code it uses is
documented in the root [README, section 11](../README.md#11-rest-api-fastapi).

Stack: React 19, Vite 8, TypeScript 7, Material UI 9, Axios, React Router,
react-markdown + remark-gfm, remark-math + rehype-katex (KaTeX). Tests: Vitest, React Testing Library, jsdom.

> **Local development only.** The API has no authentication. Run both servers
> on your own machine.

## Quick start

Requirements: Node 22.12+ or 24+ (Vitest's minimum; tested with Node 24.14 and
npm 11.9) and the Python backend from the root README.

```powershell
# Terminal 1: backend (repository root)
.venv\Scripts\Activate.ps1
python -m app.api                 # http://127.0.0.1:8000, ollama mode from .env
# or: python -m app.api --mode local   (rule-based, instant answers, no LLM)

# Terminal 2: frontend
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

Open **http://localhost:5173**. Use exactly that address: it is one of the two
origins the API allows by default (`API_CORS_ORIGINS`).

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server on port 5173 with hot reload |
| `npm run build` | Type-check (`tsc -b`) and build into `dist/` |
| `npm run preview` | Serve `dist/` on port 5173 (same CORS origin as dev) |
| `npm run lint` | oxlint |
| `npm run typecheck` | TypeScript only |
| `npm test` | All tests once (no backend or Ollama needed) |
| `npm run test:watch` | Tests in watch mode |

Ports are **strict**: if 5173 is taken, Vite stops with "Port 5173 is already in
use" instead of moving to 5174, because the API would then reject every request
(CORS). Free the port, or use another one and allow it on the backend:

```powershell
$env:API_CORS_ORIGINS = "http://localhost:5180"; python -m app.api     # backend
npx vite --port 5180                                                    # frontend
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend base URL. Set it in `frontend/.env.local` (git-ignored); see `.env.example` |

Vite reads it at startup, so restart `npm run dev` after changing it.

## Features

- **Chat** with the agent. Replies are rendered as GitHub-flavoured Markdown
  (headings, lists, tables, inline code and code blocks). Raw HTML in replies is
  never rendered.
- **Math** (KaTeX): `$…$` inline, `$$…$$` or a ```` ```math ```` block for
  display. Inline code and all other code blocks stay code, even if they contain
  `$` or LaTeX. Spaces inside the delimiters are allowed (qwen3 writes
  `$ ax^2 $`). Dollar amounts such as "$5 and $10" stay text: a `$…$` pair is
  not math when a digit follows the closing `$`, or when it starts with a digit
  and ends with a space. Invalid LaTeX is shown in red instead of breaking the
  reply, and KaTeX's `trust` option is off, so commands such as `\href` are not
  rendered.
- **Tool calls**: each assistant turn starts with one expandable card per tool
  the backend actually ran. Collapsed, a card shows the tool name, its arguments
  on one line (hidden on phones) and a Success/Failed status. Expanded, it shows
  the arguments and the result (or the tool's error message), with dedicated
  views for the calculator, date/time and knowledge-base results, a JSON
  fallback for anything else, and a "Show raw output" toggle.
- **Restored tool calls**: after a refresh, the history's tool entries are shown
  as the same cards inside their turn, with the recorded name, arguments and
  status. The result section says it isn't available, because the API doesn't
  keep it.
- **Fallback chip**: when a reply starts with `[LLM unavailable:` (Ollama failed
  and the rule-based backend answered), a "Fallback response" chip is shown. The
  reply text itself is not changed.
- **Sessions**:
  - A session is created on your first message, not on page load.
  - Its id is kept in `localStorage` (`dsa.session`), and after a refresh the
    conversation is reloaded from `GET /sessions/{id}/history`.
  - **Reset** clears the conversation but keeps the session. **New conversation**
    deletes it.
- **Long turns**:
  - A pending bubble shows the elapsed time; a CPU-only `qwen3:4b` turn took
    2–10 minutes in testing.
  - **Stop waiting** stops this page from waiting, but the server still finishes
    the turn. **Check for reply** then loads it from history.
- **States**: loading skeletons, an empty conversation with example prompts, an
  offline banner (from `GET /health`, polled every 30 s), and errors for
  404/409/422/503/network with Retry where it makes sense.
- **Tools page** (`/tools`): the tool catalogue from `GET /tools`.
- **Layout**: the chat column is up to 1040 px wide and an assistant turn up to
  840 px, so tool cards and code have room while lines stay readable. All tool
  cards in a turn share the reply's width and spacing. Code blocks, tables and
  display math scroll inside their bubble, so the page never scrolls sideways.
  Checked at 360, 390, 820, 1280 and 1600 px.
- Light and dark themes follow the OS.

## How the app handles the API

| Situation | API behaviour | What the UI does |
|---|---|---|
| Refresh during a running turn | History returns `409` | Shows "still answering…" and polls history every 10 s |
| Session expired, deleted or API restarted | `404` | Clears the stored id and explains. A failed send can be retried in a new conversation |
| Second request in the same session | `409` | "Conversation busy" error. The UI never sends twice: the send button is disabled and there is a synchronous guard |
| Model busy or session limit reached | `503` | Warning with Retry |
| Invalid message | `422` | Checked first in the browser with the same rules: trimmed, 1–2000 characters counted as code points like Python |
| API down or a CORS rejection | No response | Offline banner and "Cannot reach the API" (browsers report both the same way) |

## Known limitations

These come from the backend; the frontend doesn't work around them.

- **History is less detailed than live replies.** After a refresh, tool cards
  show the name, arguments and status from the history summary
  (`name(args) -> ok`), but not the result. The API stores nothing more.
- **Timestamps differ slightly.** History timestamps are server times recorded
  when the turn finished; live messages use browser time.
- **No streaming or progress, and no cancel.** "Stop waiting" can't stop the
  server's turn, so the session stays busy (`409`) until it ends.
- **`Retry-After` can't be read.** The API doesn't expose it via CORS, so the UI
  suggests about 10 seconds.
- **Fallback detection relies on the reply text prefix**; there is no separate
  field.
- **`/health` doesn't contact Ollama**, so "ollama · qwen3:4b" means configured,
  not necessarily running.
- **Rule-based replies are plain text with hard line breaks.** Rendered as
  Markdown, some lines join (for example an article title and its first
  sentence).
- **One stored conversation per browser.** There is no endpoint to list sessions.
  Two tabs share the conversation, and the second one gets `409` while the
  first is waiting.

## Project structure

```
frontend/
├── index.html, vite.config.ts, tsconfig*.json, .oxlintrc.json, .env.example
└── src/
    ├── main.tsx, App.tsx, theme.ts
    ├── api/            types.ts (backend schemas), client.ts (axios), errors.ts (ApiError), agentApi.ts
    ├── conversation/   reducer.ts, useConversation.ts, ConversationProvider.tsx, context.ts, types.ts
    ├── hooks/          useHealth.ts, useTools.ts
    ├── utils/          storage, validation, fallback, history, toolResults, remarkDollarGuard
    ├── components/
    │   ├── layout/     AppLayout, BackendStatusChip
    │   ├── chat/       MessageList, MessageBubble, MarkdownContent, ChatInput, PendingTurn, EmptyConversation
    │   ├── tools/      ToolCallCard (live and restored), ToolResultView
    │   └── feedback/   ErrorAlert, ConfirmDialog
    ├── pages/          ChatPage, ToolsPage, NotFoundPage
    └── test/           setup.ts, fixtures.ts (real backend responses), render.tsx
```

## Tests

`npm test` runs the suite with mocked API calls. It needs no backend, no Ollama
and no network. It covers the error mapping for each status code, endpoint
paths and bodies, validation, history parsing, Markdown and math rendering
(including code blocks and dollar amounts that must not become math), tool cards
(collapsed and expanded, success and failure, live and restored), turn grouping,
the full chat flow (send, duplicate-submit guard, fallback chip, 503 retry, 404
recovery, stop waiting, restore after refresh, 409 polling, reset and new
conversation), the Tools page and the offline banner.

`src/api/agentApi.live.test.ts` is **opt-in** and runs against a real backend.
It uses the real axios client from a jsdom page with origin
`http://localhost:5174`, so CORS is enforced as in a browser:

```powershell
# Terminal 1 (repository root)
$env:API_CORS_ORIGINS = "http://localhost:5174"; python -m app.api --mode local --port 8001
# Terminal 2 (frontend)
$env:LIVE_API_URL = "http://127.0.0.1:8001"; npx vitest run src/api/agentApi.live.test.ts
# With an ollama-mode API (real model, takes minutes):
$env:LIVE_OLLAMA_API_URL = "http://127.0.0.1:8002"; npx vitest run src/api/agentApi.live.test.ts
```
