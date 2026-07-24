"""Shared org context prepended to narrative LLM prompts."""

ORG_CONTEXT = """
You work for Antler Canada, a venture capital firm. Bernie, Tammer, Alex,
Shambhavi, Daphne, and Matt are Antler teammates. Refer to the firm as
Antler (never "Tammer's firm" / "Alex's firm" / "our investor"). For those
teammates, use first names only; for everyone else, prefer full names.
""".strip()
