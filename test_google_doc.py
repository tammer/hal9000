#!/usr/bin/env python3
"""Temporary smoke test: read a Google Doc via a service account."""

from google_docs import read_google_doc

DOCUMENT_ID = "1pdgY3W31AXgvuc86aXmnjNni1anH4R_VpIsVw9dNdbI"


def main() -> None:
    print(read_google_doc(DOCUMENT_ID))


if __name__ == "__main__":
    main()
