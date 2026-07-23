# <img src="web/public/favicon.svg" alt="Weirwood Index" height="28"> Weirwood Index

Find the passage you half remember.

[Weirwood Index](https://weirwoodindex.com) is an open-source passage search engine for
*A Song of Ice and Fire*. It combines lexical and semantic retrieval to find scenes from a short
description, a quotation, or a set of remembered terms.

The search pipeline runs on local models and indexed text. It does not use an LLM. The public
repository contains the application and retrieval code, but it does not contain the novels,
searchable indexes, embeddings, or generated passage data.

The project is independent and unofficial. See [LEGAL.md](LEGAL.md) for the project and
source-material boundaries.

## Features

- Hybrid search with BM25 and BGE-M3.
- Paginated lexical search with exact-phrase and proximity signals.
- Filters for book and point-of-view character.
- Paragraph-aware previews centred on the strongest matching text.
- Search across the five main published novels in the hosted service.
- A typed FastAPI contract with generated TypeScript definitions.
- Bot verification and layered rate limits at the edge gateway.
- No user accounts, application database, or stored search history.

## How it works

```text
Browser
  │
  ▼
Edge gateway
  ├─ serves the web application
  ├─ verifies bot-challenge tokens
  ├─ applies search rate limits
  └─ adds the private origin credential
  │
  ▼
FastAPI
  ├─ validates requests and filters
  ├─ runs BM25 lexical search
  └─ runs hierarchical BGE-M3 hybrid search
            │
            ▼
     Private model and index
```

The embedding model loads after the first authenticated hybrid request and remains in memory while
the API process is active. Lexical searches do not load the embedding model.

### Search modes

| Mode | Behaviour |
| --- | --- |
| `hybrid` | Combines BGE-M3 semantic candidates with BM25 and hierarchical chapter signals. |
| `lexical` | Ranks exact phrases, term coverage, proximity, and BM25 results without an embedding model. |

Hybrid search returns up to ten ranked passages. Lexical search is paginated and can enumerate all
matching results.

## Repository layout

```text
src/weirwood_index/   Retrieval, corpus, indexing, and evaluation code
src/weirwood_api/     FastAPI application
web/                  Web client and edge gateway
tests/                Retrieval and API tests
evaluation/           Human-written benchmark definitions
tools/                Contract, benchmark, and training utilities
deploy/compose/       Docker Compose and Caddy configuration
```

## Local setup

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 24
- pnpm 11.10
- A lawfully obtained DRM-free EPUB or text source for each book that you want to index

Install the Python project:

```bash
uv sync --locked --dev
```

Install the web dependencies:

```bash
cd web
pnpm install --frozen-lockfile
cd ..
```

### Prepare a source

Book files are local inputs and are ignored by Git. EPUB is preferred because it preserves real
paragraph boundaries.

Place a source under `data/epub/`, then inspect it before indexing:

```bash
uv run weirwood corpus inspect \
  --source data/epub/a-game-of-thrones.epub
```

The inspection command reports the detected books, chapters, points of view, and source hashes.
Stop if the reported structure does not match the source.

### Build an index

The following example builds a BGE-M3 index for one source:

```bash
uv run weirwood index build \
  --source data/epub/a-game-of-thrones.epub \
  --profile short \
  --model BAAI/bge-m3
```

Add another `--source` argument for each additional book. Indexes are written under
`data/indexes/`, which is also ignored by Git.

The first build downloads the selected model. Text and embedding operations run on the local
machine; the source text is not sent to an LLM or external inference API.

### Search from the command line

```bash
uv run weirwood search "Ned warns Cersei before telling Robert" \
  --index data/indexes/<index-id> \
  --mode hybrid \
  --hierarchical \
  --book agot \
  --top 5
```

Use lexical mode for exact wording:

```bash
uv run weirwood search "dance with me then" \
  --index data/indexes/<index-id> \
  --mode lexical \
  --book agot \
  --top 10
```

Run `uv run weirwood --help` or a subcommand with `--help` for the complete CLI reference.

## Run the application

### API

Copy the example environment and set `WEIRWOOD_INDEX_PATH` to the built index:

```bash
cp .env.example .env
uv run uvicorn weirwood_api.main:app --reload --env-file .env
```

The API exposes:

```text
GET  /health
GET  /ready
GET  /v1/catalog
POST /v1/search
```

`/v1/catalog` and `/v1/search` require the `X-Weirwood-Origin-Token` header. The local Vite proxy
adds the development token automatically.

### Web client

```bash
cd web
cp .env.example .env.local
pnpm dev
```

Open `http://127.0.0.1:5173`. The development server forwards `/api/catalog` and `/api/search` to
the local FastAPI server.

## API contract

FastAPI is the source of truth for the API contract. The OpenAPI document and generated TypeScript
types are tracked so CI can detect drift.

Regenerate both files after an API schema change:

```bash
uv run python tools/export_openapi.py
cd web
pnpm api:types
```

The browser calls only same-origin `/api` routes. The edge gateway verifies the request and forwards
it to the API with a secret origin token. FastAPI rejects an invalid token before it loads the model
or runs a search. The runtime model and index remain private and must not be published in a public
container image.

## Public and private data

The repository may contain:

- Source code, tests, and deployment configuration.
- Original benchmark queries and scene descriptions.
- Chapter identifiers, source hashes, and passage offsets.
- Aggregate evaluation results without retrieved excerpts.

The repository must not contain:

- Raw or normalized book text.
- Generated passages, scene windows, or event records.
- Searchable indexes or embedding arrays.
- Evaluation reports that reproduce book text.
- Downloaded models or trained weights derived from passages.
- Environment files, credentials, or provider secrets.

Ignore rules are a safety boundary, not a substitute for reviewing files and Git history before a
public release.

## License and contact

The original project code and documentation are available under the
[Apache License 2.0](LICENSE). That license does not apply to the novels or other third-party
material.

For project or rights-related questions, email [info@weirwoodindex.com](mailto:info@weirwoodindex.com).
