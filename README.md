# HAL9000 — Deal Analysis

Generate deal analysis reports from documents in a Google Drive folder using Groq.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_DRIVE_BASE="/path/to/your/deals/folder"
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile   # optional, this is the default
```

`GOOGLE_DRIVE_BASE` is the root folder containing deal subfolders. `main.py` resolves paths relative to this root.

## Usage

### Full report

Generate all sections and write `analysis/deal.html` inside the deal folder:

```bash
python main.py "Acme Corp"
```

Output path example:

```
/path/to/deals/Acme Corp/analysis/deal.html
```

### Test one section

Generate a single section and print markdown to stdout. Useful for iterating on template prompts without running the full report:

```bash
python main.py "Acme Corp" --section company
```

Save output to a file:

```bash
python main.py "Acme Corp" --section traction > traction.md
```

### List available sections

```bash
python main.py --list-sections
```

| Slug | Title |
|------|-------|
| `company` | Company |
| `product` | Product |
| `traction` | Traction |
| `why-now-and-right-to-win` | Why Now and Right to Win |
| `recorded-meetings` | Recorded Meetings |
| `tam` | TAM |
| `open-questions-concerns` | Open Questions, Concerns |

## Templates

Section prompts live in `templates/` — one markdown file per section, ordered by numeric prefix (e.g. `01-company.md`). Edit these files to change what the LLM is asked to produce for each section.

The LLM returns markdown; `main.py` converts it to HTML when building the full report.

## Input documents

`main.py` reads **top-level files only** in the deal folder (not subfolders). It supports plain text, PDF, and DOCX files.

### Contents index

Classify top-level files and write `contents.json` in the deal folder (filename, creation date, and document type). Re-runs skip files already listed in `contents.json` and only classify new files:

```bash
python generate_contents.py "Acme Corp"
```

Output path example:

```
/path/to/deals/Acme Corp/contents.json
```
