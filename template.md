# Company

Output a two-column HTML table with exactly these rows (one row per field). Do not add a header row.

| Field | Value |
|---|---|
| Name | company name |
| Founding date | year or date if stated in the documents; otherwise `not available` |
| Funding to date | total raised if stated; otherwise `not available` |
| Founders | founder names as HTML links to LinkedIn when a URL is present in the documents; plain text otherwise. Separate multiple founders with commas. |

Use `<table>`, `<tr>`, and `<td>` only. First column = field label. Second column = value.

# Product

In 2–4 sentences, state what the product is and which customer problem it solves. Stick to facts from the documents. No competitive commentary or assessment.

# Traction

Start with a two-column HTML table for quantitative metrics. Include a row only when the documents state a number:

| Metric | Value |
|---|---|
| Customers | count and type (e.g. accounting firms, end customers) if stated |
| ARR | current ARR if stated |
| MRR | current MRR if stated |
| Pipeline | pipeline or projected deals if stated |
| Projected ARR | future ARR targets with timeframe if stated |

Use `not available` for any metric not found in the documents. Do not guess or infer.

After the table, add at most one short paragraph (2–3 sentences) for qualitative traction only if the documents mention it (e.g. distribution model, growth drivers). Do not repeat numbers already in the table. Do not use bullet lists for metrics.

# Why Now and Right to Win

Report what the founders claim and what the documents support. Do **not** make your own assessment. If the documents do not address a point, just say "not available"


1. **Why now** — market or timing factors mentioned in the documents.
2. **Right to win** — the team's specific advantages, insights, or distribution edge as stated in the documents.

# Recorded Meetings

each document that is a transcripts includes a section at the top that includes date, meeting attendees and a URL for the meeting.

For each meeting, include display these three fields. for the date stamp, just include the date not the time.

# Open Questions, Concerns

List every open question or concern explicitly noted in the documents (e.g. investor notes, "open issues" sections, due-diligence writeups). Use an HTML ordered list (`<ol><li>...</li></ol>`).

- Be complete — do not omit items that appear in the source material.
- You may lightly rephrase for clarity, but do not invent new concerns or expand beyond what was noted.
- If nothing is noted, output exactly: `<p>no concerns noted by Antler team.</p>`

If the documents actually address the question, then include the mitigation in your reporting.