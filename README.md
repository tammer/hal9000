# hal9000

Deal pipeline tooling for Antler portfolio companies. Scripts ingest meeting transcripts and emails into Google Drive deal folders, generate Claude investment summaries, build a portfolio status table, and publish a static website.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env` and fill in the required values (see [Environment variables](#environment-variables) below). All scripts load `.env` automatically via `python-dotenv`.

Run commands from the repo root with the virtualenv activated:

```bash
source .venv/bin/activate
python run_pipeline.py
```

## Environment variables

| Variable | Required by | Description |
|----------|-------------|-------------|
| `GOOGLE_DRIVE_BASE` | Most scripts | Canada shared-drive root containing `deals/`, `portcos/`, `ai-generated/`, and `facts.md` |
| `WEBSITE_BASE` | `generate_website.py` | Parent directory where `website/` output is written |
| `GROQ_API_KEY` | Transcript fetch, emails, summarizer, `founders.py`, `daily_summary.py`, `daily_summary_portco.py`, `main.py`, `get_facts`, `consolidator.py` | Groq API key |
| `GROQ_MODEL` | Optional | Groq model (default: `llama-3.3-70b-versatile`) |
| `ANTHROPIC_API_KEY` | `claude_summary.py`, `chat.py`, `researcher.py` | Anthropic API key |
| `ANTHROPIC_MODEL` | Optional | Default Anthropic model for `chat.py` |
| `MEETGEEK_API_KEY` | Transcript fetch scripts | MeetGeek API key |
| `MEETGEEK_TEAM_ID` | `fetch_all_transcripts.py` | MeetGeek team ID |
| `MEETGEEK_API_BASE` | Optional | Override MeetGeek API base URL |
| `MAIL_IMAP_HOST` | `process_emails.py` | IMAP server hostname |
| `MAIL_IMAP_PORT` | Optional | IMAP port (default: 993) |
| `MAIL_ADDRESS` | `process_emails.py` | Mailbox address |
| `MAIL_PASSWORD` | `process_emails.py` | Mailbox password |
| `BRAVE_SEARCH_API_KEY` | `get_facts.py`, `researcher.py` | Brave Search API key |

## Drive layout

`GOOGLE_DRIVE_BASE` is the Canada shared-drive root. Deals and portcos are peer trees; shared pipeline outputs live under `ai-generated/`.

```
GOOGLE_DRIVE_BASE/                 # …/Shared drives/Canada
├── facts.md                       # team context for meeting_roundup.py
├── people/                        # ignored by tooling
├── ai-generated/                  # shared outputs
│   ├── status.md                  # portfolio table (summarizer.py)
│   └── dailies/
│       ├── deals/YYYY-MM-DD.json
│       ├── portcos/YYYY-MM-DD.json
│       └── meetgeeks/YYYY-MM-DD.json
├── deals/
│   └── Mobi/
│       ├── Founders.md            # founders + LinkedIn (founders.py)
│       ├── pitch-deck.pdf         # source documents (top-level files)
│       ├── contents.json          # optional file index (generate_contents.py)
│       ├── emails/
│       │   └── email_20260710_143000_Subject.txt
│       ├── transcripts/
│       │   └── Meeting+Title_sentences_2026-07-10T15_00_00Z__meeting-uuid.txt
│       └── ai-generated/
│           ├── deal.json
│           ├── identity.json      # company/people/aliases/domains for matching
│           ├── summary.md
│           └── deal.html          # optional (main.py)
└── portcos/
    └── Central-Agent/
        ├── Founders.md
        ├── emails/
        ├── transcripts/
        └── ai-generated/
            ├── portco.json
            ├── identity.json
            └── summary.md
```

Most scripts read **top-level files only** in each company folder (not recursive). Supported formats include `.txt`, `.md`, `.pdf`, `.docx`, and other text-readable files. Claude summary scripts also read files under `emails/` and `transcripts/`.

**CLI path conventions:**

- Deal scripts (`process_deal.py`, `claude_summary2.py`, …): folder name under `deals/` (e.g. `Mobi`)
- Portco scripts (`process_portco.py`, `generate_portco_report.py`): folder name under `portcos/` (e.g. `Central-Agent`)
- Ingest CLIs (`fetch_transcripts.py`): `deals/<folder>` or `portcos/<folder>`

---

## Pipeline

### `run_pipeline.py`

Runs the full deal pipeline end to end.

```bash
python run_pipeline.py [options]
```

**Steps:**

1. **Founders** — `founders.py --all` (writes/updates top-level `Founders.md`; skips when complete)
2. **Fetch transcripts** — `fetch_all_transcripts.py`
3. **Process emails** — `process_emails.py`
4. **Meeting roundup** — `meeting_roundup.py`
5. **Claude summaries** — `claude_summary2.py` for every deal folder
6. **Portco reports** — `generate_portco_report.py` for every portco folder (refreshes `portco.json`, writes `summary.md`)
7. **Daily summary** — `daily_summary.py` then `daily_summary_portco.py`
8. **Summarizer** — `summarizer.py` (builds `ai-generated/status.md`)
9. **Website** — `generate_website.py`
10. **Deploy** — `website_deploy.py`

**Options:**

| Flag | Description |
|------|-------------|
| `--day {today,yesterday}` | Target calendar day for fetch cutoff, meeting roundup, and daily summaries. Default: yesterday before 16:30 local, otherwise today |
| `--dry-run` | Pass through to fetch and email steps; no files written, emails not marked read |
| `--cutoff-date DATE` | Override fetch cutoff (`YYYY-MM-DD`). Default: the resolved `--day` value |
| `--skip-founders` | Skip step 1 |
| `--skip-fetch` | Skip step 2 |
| `--skip-emails` | Skip step 3 |
| `--skip-meeting-roundup` | Skip step 4 |
| `--skip-claude` | Skip step 5 |
| `--skip-portco` | Skip step 6 |
| `--skip-daily-summary` | Skip step 7 |
| `--skip-summarizer` | Skip step 8 |
| `--skip-website` | Skip step 9 |
| `--skip-deploy` | Skip step 10 |
| `--confirm` | Ask yes/no before running each step |

**Examples:**

```bash
# Full pipeline (day chosen by 16:30 local rule)
python run_pipeline.py

# Explicitly run for today or yesterday
python run_pipeline.py --day today
python run_pipeline.py --day yesterday

# Preview ingest steps without writing
python run_pipeline.py --dry-run

# Rebuild summaries and website only
python run_pipeline.py --skip-fetch --skip-emails

# Summaries for yesterday, but fetch since an earlier cutoff
python run_pipeline.py --day yesterday --cutoff-date 2026-07-01
```

Before Claude summaries, the pipeline scans deal folders and prints notes for empty folders or folders with no readable source documents.

---

## Founders

### `founders.py`

Identifies founders and LinkedIn profile URLs for a deal or portco folder. Writes a human-readable `Founders.md` at the company folder root (visible to `collect_documents` and later Claude steps).

```bash
python founders.py deals/Mobi
python founders.py portcos/Central-Agent
python founders.py --all
python founders.py --all --refresh
```

**Algo:**

1. If `Founders.md` exists and is **complete**, skip (unless `--refresh`).
2. Otherwise extract founders from primary materials only (top-level docs, `emails/`, `transcripts/`; never `ai-generated/`), excluding `Founders.md` itself.
3. Fill blanks only — never overwrite an existing LinkedIn URL or `unknown`.
4. Resolve still-missing LinkedIns with Brave search first, HTTP title validation, and Groq Compound only as a validated fallback.
5. Write `Founders.md` with `Status: complete` or `incomplete`.

**Complete** means either `Status: complete` (manual stop), or ≥1 founder with first+last name and every founder has a `linkedin.com/in/...` URL or the literal `unknown`.

**Example `Founders.md`:**

```markdown
# Founders

Company: Acme Inc
Status: incomplete

## Jane Doe
- LinkedIn: https://www.linkedin.com/in/jane-doe

## John Smith
- LinkedIn: 
```

| Flag | Description |
|------|-------------|
| `--all` | Process every folder under `deals/` and `portcos/` |
| `--refresh` | Re-run even when complete (still fill-blanks only) |
| `--skip-web-search` | Extract from materials only; skip Compound |
| `--model` | Override Groq extraction model |
| `--compound-model` | Override Compound model (default: `groq/compound`) |

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`

Pipeline step 1 runs `founders.py --all`. Use `--skip-founders` to skip it.

---

## Data ingestion

### `fetch_all_transcripts.py`

Fetches team MeetGeek meetings since a cutoff date, matches each meeting to a folder under `deals/` or `portcos/`, and writes relevant transcripts.

Matching is two-stage:

1. **Shortlist** — score folders from persisted `ai-generated/identity.json` (company, people, aliases, email domains) against meeting title/attendees/emails/transcript excerpt.
2. **Confirm** — LLM picks at most one folder from the shortlist using rich identity + a context brief from the deal summary. First-name-only hits never auto-file.

```bash
python fetch_all_transcripts.py [--cutoff-date DATE] [--dry-run] [--refresh-identity]
```

| Flag | Description |
|------|-------------|
| `--cutoff-date DATE` | Include meetings on or after this date (`YYYY-MM-DD`). Default: 2 days ago |
| `--dry-run` | Report actions without writing files |
| `--reprocess` | Ignore the processed-meetings log and re-analyze all meetings |
| `--refresh-identity` | Force rebuild of each folder's `ai-generated/identity.json` |

**Output:** transcript `.txt` files in the matched folder's `transcripts/` directory (under deals or portcos). Also writes/updates `ai-generated/identity.json` per folder when missing or stale relative to `summary.md`.

**Requires:** `GROQ_API_KEY`, `MEETGEEK_API_KEY`, `MEETGEEK_TEAM_ID`, `GOOGLE_DRIVE_BASE`

---

### `fetch_transcripts.py`

Fetches recent MeetGeek transcripts for a **single** company folder (last 8 days). Path must be rooted under `deals/` or `portcos/`.

```bash
python fetch_transcripts.py <deals|portcos>/<folder> [--dry-run] [--refresh-identity]
```

**Examples:**

```bash
python fetch_transcripts.py deals/Mobi
python fetch_transcripts.py portcos/Central-Agent
```

Loads or builds `ai-generated/identity.json` (company, people, aliases, domains, product blurb, context brief), then scores each recent meeting for relevance against that rich identity. Writes matching transcripts as `.txt` files under the folder's `transcripts/` directory.

**Requires:** `GROQ_API_KEY`, `MEETGEEK_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `process_emails.py`

Fetches unread inbox mail, matches each message to a folder under `deals/` or `portcos/`, and saves it as a `.txt` file. Successfully written messages are marked as read.

Uses the same `identity.json` catalog as transcript fetch. Unique **strong** programmatic hits (company, full name, alias, email domain) can auto-match; first-name-only hits fall through to the LLM.

```bash
python process_emails.py [--dry-run] [--refresh-identity]
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Report matches without writing files or marking messages read |
| `--refresh-identity` | Force rebuild of each folder's `ai-generated/identity.json` |

**Output:** `email_<timestamp>_<subject>.txt` files in the matched folder's `emails/` directory (under deals or portcos).

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`, `MAIL_IMAP_HOST`, `MAIL_ADDRESS`, `MAIL_PASSWORD`

---

### `process_deal.py`

Generate or refresh `ai-generated/deal.json` for a deal folder (per-file Claude metadata, mtime cache).

```bash
python process_deal.py <relative_path>
```

**Example:**

```bash
python process_deal.py Mobi
```

**Requires:** `ANTHROPIC_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `process_portco.py`

Same as `process_deal.py`, but for a portfolio-company folder under `portcos/`. Writes `ai-generated/portco.json`.

```bash
python process_portco.py <folder>
```

**Example:**

```bash
python process_portco.py Central-Agent
```

**Output:** `portcos/<folder>/ai-generated/portco.json`

**Requires:** `ANTHROPIC_API_KEY`, `GOOGLE_DRIVE_BASE`

---

## Summarization and publishing

### `claude_summary.py`

Generates an investment summary from a deal folder's source documents (recursively, excluding `ai-generated/`) using Claude.

```bash
python claude_summary.py <relative_path> [--dry-run]
```

| Flag | Meaning |
|------|---------|
| `--dry-run` | List documents that would be summarized without calling the API or writing output |

**Example:**

```bash
python claude_summary.py Mobi
python claude_summary.py Mobi --dry-run
```

**Output:** `ai-generated/summary.md` in the deal folder.

Before summarizing, prints the relative paths of all documents that will be sent to Claude. Skips regeneration if no source files have changed since the last summary. Prints token usage and estimated cost to stderr.

**Requires:** `ANTHROPIC_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `claude_summary2.py`

Generates an investment summary from `ai-generated/deal.json` using Claude (refreshes deal metadata first unless `--dry-run`).

```bash
python claude_summary2.py <relative_path> [--dry-run]
```

**Output:** `ai-generated/summary.md` in the deal folder.

**Requires:** `ANTHROPIC_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `generate_portco_report.py`

Same flow as `claude_summary2.py` for a portfolio company under `portcos/`: refreshes `portco.json`, then writes `ai-generated/summary.md` using `portco_report_prompt.md`.

```bash
python generate_portco_report.py <folder> [--dry-run]
```

**Example:**

```bash
python generate_portco_report.py Central-Agent
```

**Output:** `portcos/<folder>/ai-generated/summary.md`

**Requires:** `ANTHROPIC_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `summarizer.py`

Reads every deal's `ai-generated/summary.md`, extracts structured fields (product, founders, status) with Groq, and writes a portfolio status table.

```bash
python summarizer.py
```

**Output:** `GOOGLE_DRIVE_BASE/ai-generated/status.md`.

Deals without a summary are skipped. Failures for individual deals are logged as warnings; the script still writes the table for successful extractions.

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `daily_summary.py`

For each deal folder under `GOOGLE_DRIVE_BASE/deals/`, reads `ai-generated/deal.json`, keeps only entries whose `created_at` calendar date matches the given day, and asks Groq to infer what happened with that deal. Deals with no matching entries are omitted.

```bash
python daily_summary.py [YYYY-MM-DD]
```

Date is optional; defaults to yesterday.

**Example:**

```bash
python daily_summary.py
python daily_summary.py 2026-07-17
```

**Output:** writes `GOOGLE_DRIVE_BASE/ai-generated/dailies/deals/YYYY-MM-DD.json`:

```json
[
  {
    "deal": "Endo",
    "summary": "Today, Tammer and Alex met the team and talked about PMF"
  }
]
```

Creates `ai-generated/dailies/deals/` if needed. Importable as `generate_daily_summary(day)` (accepts a `date` or `YYYY-MM-DD` string; returns the list). Progress and the written path go to stderr.

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE` (and existing `ai-generated/deal.json` files from `process_deal.py`)

---

### `daily_summary_portco.py`

Same as `daily_summary.py`, but for portfolio-company folders under `portcos/`. Reads each folder's `ai-generated/portco.json` and writes under `ai-generated/dailies/portcos/`. Portcos with no matching entries are omitted. If the portcos root is missing, prints a warning and writes an empty list.

```bash
python daily_summary_portco.py [YYYY-MM-DD]
```

Date is optional; defaults to yesterday before 16:30 local, otherwise today (same as `daily_summary.py`).

**Example:**

```bash
python daily_summary_portco.py
python daily_summary_portco.py 2026-07-17
```

**Output:** writes `GOOGLE_DRIVE_BASE/ai-generated/dailies/portcos/YYYY-MM-DD.json`:

```json
[
  {
    "portco": "Central-Agent",
    "summary": "Tammer and Alex met the team and talked about PMF"
  }
]
```

Creates `ai-generated/dailies/portcos/` if needed. Importable as `generate_daily_summary_portco(day)` (accepts a `date` or `YYYY-MM-DD` string; returns the list). Progress and the written path go to stderr.

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE` (and existing `ai-generated/portco.json` files from `process_portco.py`)

---

### `generate_website.py`

Builds a static HTML site from `ai-generated/status.md`, daily activity JSON, each deal's `ai-generated/summary.md`, and each portco's `ai-generated/summary.md`.

```bash
python generate_website.py
```

**Output:** `website/index.html`, `website/deals.html`, `website/portcos.html`, `website/dailys.html`, and `website/{Name}.html` under `WEBSITE_BASE` for each deal/portco with a summary. Existing `.html` files in `website/` are removed first. Companies without a summary are skipped with a warning.

**Requires:** `GOOGLE_DRIVE_BASE`, `WEBSITE_BASE` (run `summarizer.py` first)

---

## Deal analysis

### `main.py`

Generates a structured deal analysis HTML page from top-level folder documents using Groq and section templates.

```bash
python main.py <relative_path> [options]
python main.py --list-sections
python main.py <relative_path> --section <slug>
```

| Flag | Description |
|------|-------------|
| `--list-sections` | List available section slugs and titles, then exit |
| `--section SLUG` | Generate one section and print markdown to stdout |

**Examples:**

```bash
python main.py --list-sections
python main.py Mobi
python main.py Mobi --section company
```

**Output:** `ai-generated/deal.html` in the deal folder (full run), or markdown on stdout (`--section`).

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`, and a `templates/` directory with section prompt files.

---

### `generate_contents.py`

Scans top-level files in a deal folder, classifies new files with an LLM, and maintains a `contents.json` index.

```bash
python generate_contents.py <relative_path>
```

**Example:**

```bash
python generate_contents.py Mobi
```

**Output:** `contents.json` in the deal folder.

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `consolidator.py`

Extracts dated Antler team notes from top-level files in a deal folder. One Groq call per file; skips saved emails (`email_*`) and MeetGeek transcripts (`*_sentences_*`). Only keeps internal team notes (not founder materials or raw transcripts).

```bash
python consolidator.py <relative_path>
```

**Example:**

```bash
python consolidator.py Mobi
```

**Output:** JSON printed to stdout:

```json
{
  "files": ["tammer-notes.md", "notes_tk_Jul20.txt"],
  "entries": [
    {
      "datetime": "2026-07-20 00:00:00",
      "author": "Tammer",
      "content": "note text",
      "source": "notes_tk_Jul20.txt"
    }
  ]
}
```

`entries` are sorted by `datetime`. Authors are one of `Tammer`, `Bernie`, `Shambhavi`, `Alex`, `Matt`, `Daphne`, or `unknown`. Dates come from the document or filename when available (month/day without a year assumes the current year); otherwise the file's last-modified time is used.

Supported extensions: `.md`, `.txt`, `.docx`, `.gdoc`.

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `actions.py`

Reads a deal's `ai-generated/summary.md` and, using today's date, reports any outstanding actions the Antler team needs to take. An action is flagged when we promised something by a date that is near/at/past, or when we're still waiting on the other party.

```bash
python actions.py <relative_path>
```

**Example:**

```bash
python actions.py Mobi
```

**Output:** A concise actions report printed to stdout (or `No actions needed.`).

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`, and an existing `ai-generated/summary.md` (run `claude_summary.py` first).

---

## Research utilities

### `get_facts.py`

Finds quantitative facts for a prompt using Brave web search and Groq.

```bash
python get_facts.py "<prompt>"
python get_facts.py --prompt "<prompt>" [--max-searches N] [--verbose|--no-verbose]
```

| Flag | Description |
|------|-------------|
| `--max-searches N` | Maximum search queries (default: 4) |
| `--verbose` / `--no-verbose` | Log progress to stderr (default: on when stderr is a TTY) |

**Output:** JSON array of facts printed to stdout.

**Requires:** `GROQ_API_KEY`, `BRAVE_SEARCH_API_KEY`

---

### `researcher.py`

Runs an iterative research loop: a supervisor LLM calls `get_facts` repeatedly until it can answer the research question.

```bash
python researcher.py "<prompt>" [options]
python researcher.py   # interactive prompt if none given
```

| Flag | Description |
|------|-------------|
| `--model MODEL` | Supervisor model (default: `claude-sonnet-4-6`) |
| `--max-iterations N` | Maximum research rounds (default: 8) |
| `--max-searches N` | Max searches per `get_facts` call (default: 4) |
| `--output PATH` | Save final markdown answer to a file |
| `--verbose` / `--no-verbose` | Log `get_facts` progress |

**Output:** Final markdown answer printed to stdout (and optionally saved with `--output`).

**Requires:** `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `BRAVE_SEARCH_API_KEY`

---

### `chat.py`

Interactive terminal chat with Groq or Anthropic models.

```bash
python chat.py [options]
```

| Flag | Description |
|------|-------------|
| `--provider {auto,groq,anthropic}` | API provider (default: auto-detect from model name) |
| `--model MODEL` | Model name |
| `--system TEXT` | System prompt |
| `--max-tokens N` | Max output tokens for Anthropic (default: 4096) |

**In-chat commands:** `/help`, `/clear`, `/models`, `/model <name>`, `/quit`, `quit`

**Requires:** `GROQ_API_KEY` and/or `ANTHROPIC_API_KEY` depending on provider

---

## Typical workflows

**Daily update (recommended):**

```bash
python run_pipeline.py
```

**Single deal — refresh transcript and summary:**

```bash
python fetch_transcripts.py deals/Mobi
python claude_summary.py Mobi
```

**Preview what would be ingested:**

```bash
python run_pipeline.py --dry-run --skip-claude --skip-summarizer --skip-website
```

**Rebuild website after manual summary edits:**

```bash
python summarizer.py
python generate_website.py
```
