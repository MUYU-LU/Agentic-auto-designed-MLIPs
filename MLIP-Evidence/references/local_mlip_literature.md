# Local MLIP Literature

Use this note before reading or citing local MLIP papers.

## Source of truth

The authoritative local literature folder is:

```text
<LOCAL_MLIP_LITERATURE_DIR>
```

Use this folder as the source of truth for local PDF evidence, filenames, and the current paper set.

The folder contains the curated MLIP PDFs, the naming template, and any subfolders such as supplemental materials. The current naming convention is documented in:

```text
<LOCAL_MLIP_LITERATURE_DIR>/命名模板.md
```

## OpenClaw PDF cache

There is currently **no active OpenClaw PDF cache**.

The previous cache path was removed during cleanup because it was only a temporary copy and could become stale:

```text
<OPENCLAW_WORKSPACE>/temp_pdfs
```

Use the source-of-truth folder directly when the local filesystem is available. If a future subagent truly cannot read `<LOCAL_MLIP_LITERATURE_DIR>`, recreate a temporary cache deliberately and document the refresh time. Do not assume `temp_pdfs` exists.

## PDF extraction

The OpenClaw built-in PDF tool may fail on `/mnt/c/...` paths because of media allowlist or authentication limits. For local evidence, use the `MLIP-Evidence` Evidence Execution Environment:

```bash
cd <OPENCLAW_WORKSPACE>
pdftotext -f 1 -l 3 "<LOCAL_MLIP_LITERATURE_DIR>/<paper>.pdf" -
```

The bundled `scripts/extract_local_paper.py` uses `pdftotext` first and `pypdf` only as fallback. A subagent should cite local evidence only after it has actually extracted title, abstract, first pages, or relevant pages from the local PDF.

When reporting evidence, include the exact filename and whether the text came from the source folder or the OpenClaw cache.

## Evidence rules

- Prefer local PDFs over web pages when the paper exists locally.
- Do not infer paper content only from a filename.
- Do not report a paper as locally verified unless local text extraction succeeded.
- If a PDF cannot be parsed, mark it as unreadable instead of pretending it was checked.
- Keep generated notes and summaries outside the literature folder unless explicitly asked to modify the library.
