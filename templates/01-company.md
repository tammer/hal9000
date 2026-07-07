# Company

Output a two-column markdown table with exactly these rows (one row per field). Do not include a header row.

| Field | Value |
|---|---|
| Name | company name. if we know the company website, make this a [link](url) to its website |
| Founding date | figure out roughly when the company was founded based on information in the documents. if absolute dates are unknown, a relative founding date is fine. e.g. "founded last year". If there is no way to know, just say "unknown" |
| Funding to date | total raised if stated; if the founders have actually stated that no money has been raised, then say $0. otherwise `not available` also mention the source(s) of funding if known. if there are multiple sources indicate how much came from each source (if known) |
| Raising | how much are they raising (if known) |
| Valuation | state the valuation if known. |
| Founders | founder names as markdown links to LinkedIn when a URL is present in the documents; plain text otherwise. Separate multiple founders with commas. |

First column = field label. Second column = value.
