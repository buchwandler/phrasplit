---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.3.4
kind: changed
summary:
  Default spaCy model selection to highest installed quality tier (trf > lg > md > sm)
  without automatic downloads
status: accepted
audience: null
scopes: []
source_refs:
  - git:2979f6948efb44abf86033f04857ce9ed8d07a32
paths:
  - phrasplit/spacy_models.py
  - phrasplit/splitter.py
  - phrasplit/cli.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---
