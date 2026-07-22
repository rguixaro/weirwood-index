# Private runtime bundle

This directory contains the private production index and model files. The self-hosted Compose
deployment mounts it read-only into the API container. Its corpus-derived contents are ignored by
Git.

Expected local layout:

```text
deploy/runtime/
├── index/
│   ├── manifest.json
│   ├── chunks.jsonl
│   ├── embeddings.npy
│   └── any selected enrichment artifacts
└── models/
    └── bge-m3/
        └── the pinned local model snapshot
```

Copy the contents of the selected production index into `index/`, not the index directory itself.
The model directory name must match the final segment of the model ID stored in the index manifest.

Never publish this bundle or a container image that contains it to a public registry.
