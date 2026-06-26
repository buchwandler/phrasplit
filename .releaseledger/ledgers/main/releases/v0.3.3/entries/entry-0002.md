---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 3
entry_id: entry-0002
release_version: v0.3.3
kind: added
summary: Added inline_markup parameter for XHTML inline tag handling
status: accepted
audience: null
scopes: []
source_refs:
  - git:44f5989a7bc70278f113665c34578dcc168f157a
paths:
  - phrasplit/splitter.py
  - tests/test_offsets.py
issues: []
prs: []
sources: []
breaking: false
internal: false
order: 2
---

Adds inline_markup parameter to split_with_offsets() and iter_split_with_offsets(). When
True, the regex backend keeps inline XHTML tags balanced across sentence boundaries:
closing tags after punctuation stay in the previous segment and opening tags after
whitespace start the next segment. Raises ValueError when used with the spaCy backend.
