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
| `GOOGLE_DRIVE_BASE` | Most scripts | Root directory containing one subfolder per deal |
| `WEBSITE_BASE` | `generate_website.py` | Parent directory where `website/` output is written |
| `GROQ_API_KEY` | Transcript fetch, emails, summarizer, `main.py`, `get_facts` | Groq API key |
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

## Deal folder layout

Each deal lives in a subfolder of `GOOGLE_DRIVE_BASE`:

```
GOOGLE_DRIVE_BASE/
├── status.md                    # portfolio table (written by summarizer.py)
├── Mobi/
│   ├── pitch-deck.pdf           # source documents (top-level files)
│   ├── email-subject.txt        # emails saved by process_emails.py
│   ├── Meeting+Title_sentences_2026-07-10.txt  # MeetGeek transcripts
│   ├── contents.json            # optional file index (generate_contents.py)
│   └── ai-generated/
│       ├── summary.md           # Claude investment summary
│       └── deal.html            # optional (main.py)
└── ...
```

Most scripts read **top-level files only** in each deal folder (not recursive). Supported formats include `.txt`, `.md`, `.pdf`, `.docx`, and other text-readable files.

Many commands take a `relative_path` argument: the path under `GOOGLE_DRIVE_BASE`. For a top-level deal folder, this is just the folder name (e.g. `Mobi`).

---

## Pipeline

### `run_pipeline.py`

Runs the full deal pipeline end to end.

```bash
python run_pipeline.py [options]
```

**Steps:**

1. **Fetch transcripts** — `fetch_all_transcripts.py`
2. **Process emails** — `process_emails.py`
3. **Claude summaries** — `claude_summary.py` for every deal folder
4. **Summarizer** — `summarizer.py` (builds `status.md`)
5. **Website** — `generate_website.py`

**Options:**

| Flag | Description |
|------|-------------|
| `--dry-run` | Pass through to fetch and email steps; no files written, emails not marked read |
| `--cutoff-date DATE` | Pass through to fetch step (`YYYY-MM-DD`) |
| `--skip-fetch` | Skip step 1 |
| `--skip-emails` | Skip step 2 |
| `--skip-claude` | Skip step 3 |
| `--skip-summarizer` | Skip step 4 |
| `--skip-website` | Skip step 5 |

**Examples:**

```bash
# Full pipeline
python run_pipeline.py

# Preview ingest steps without writing
python run_pipeline.py --dry-run

# Rebuild summaries and website only
python run_pipeline.py --skip-fetch --skip-emails

# Fetch transcripts since a specific date
python run_pipeline.py --cutoff-date 2026-07-01
```

Before Claude summaries, the pipeline scans deal folders and prints notes for empty folders or folders with no readable source documents.

---

## Data ingestion

### `fetch_all_transcripts.py`

Fetches team MeetGeek meetings since a cutoff date, matches each meeting to a deal folder using an LLM, and writes relevant transcripts.

```bash
python fetch_all_transcripts.py [--cutoff-date DATE] [--dry-run]
```

| Flag | Description |
|------|-------------|
| `--cutoff-date DATE` | Include meetings on or after this date (`YYYY-MM-DD`). Default: 2 days ago |
| `--dry-run` | Report actions without writing files |

**Output:** transcript `.txt` files in the matched deal folder.

**Requires:** `GROQ_API_KEY`, `MEETGEEK_API_KEY`, `MEETGEEK_TEAM_ID`, `GOOGLE_DRIVE_BASE`

---

### `fetch_transcripts.py`

Fetches recent MeetGeek transcripts for a **single** deal folder (last 4 days).

```bash
python fetch_transcripts.py <relative_path>
```

**Example:**

```bash
python fetch_transcripts.py Mobi
```

Uses deal documents to extract company/people identity, then scores each recent meeting for relevance. Writes matching transcripts as top-level `.txt` files in the deal folder.

**Requires:** `GROQ_API_KEY`, `MEETGEEK_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `process_emails.py`

Fetches unread inbox mail, matches each message to a deal folder with an LLM, and saves it as a `.txt` file. Successfully written messages are marked as read.

```bash
python process_emails.py [--dry-run]
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Report matches without writing files or marking messages read |

**Output:** `{subject}.txt` files in the matched deal folder.

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`, `MAIL_IMAP_HOST`, `MAIL_ADDRESS`, `MAIL_PASSWORD`

---

## Summarization and publishing

### `claude_summary.py`

Generates an investment summary from a deal folder's top-level source documents using Claude.

```bash
python claude_summary.py <relative_path>
```

**Example:**

```bash
python claude_summary.py Mobi
```

**Output:** `ai-generated/summary.md` in the deal folder.

Skips regeneration if no source files have changed since the last summary. Prints token usage and estimated cost to stderr.

**Requires:** `ANTHROPIC_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `summarizer.py`

Reads every deal's `ai-generated/summary.md`, extracts structured fields (product, founders, status) with Groq, and writes a portfolio status table.

```bash
python summarizer.py
```

**Output:** `status.md` at the root of `GOOGLE_DRIVE_BASE`.

Deals without a summary are skipped. Failures for individual deals are logged as warnings; the script still writes the table for successful extractions.

**Requires:** `GROQ_API_KEY`, `GOOGLE_DRIVE_BASE`

---

### `generate_website.py`

Builds a static HTML site from `status.md` and each deal's `ai-generated/summary.md`.

```bash
python generate_website.py
```

**Output:** `website/index.html` and `website/{DealName}.html` under `WEBSITE_BASE`. Existing `.html` files in `website/` are removed first. Deals without a summary are skipped with a warning.

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
python fetch_transcripts.py Mobi
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
