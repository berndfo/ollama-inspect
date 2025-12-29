Ollama Inspect
=================

ollama-inspect is a minimalist web UI to inspect the contents of locally cached Ollama models (GGUF) and the related blobs on your machine.

It scans your local Ollama directory (typically `~/.ollama/models`) and lets you:

- See all discovered model manifests (by name/tag)
- Check whether model/template/license/params blobs are present
- View model metadata (the parsed GGUF key/value pairs)
- Open the model template, license text, and parameter presets (if present)
- Query a small JSON API to get keys or items programmatically

Contents
--------
- What is this?
- Requirements
- Installation
- Quick start
- Screenshots
- How it finds your models
- Using the web UI
- API endpoints
- Troubleshooting
- Development


What is this?
-------------
When you pull models with [Ollama](https://ollama.com), it stores model manifests and binary blobs (GGUF, templates, licenses, params) under your user home directory. This tool runs a small local webserver that reads those files and shows a compact, searchable view in your browser. All access to your files is read-only.


Requirements
------------
- Python 3.8 or newer
- Locally installed Ollama with one or more downloaded models (optional but recommended)

Dependencies (automatically installed):
- Flask
- gguf


Installation
------------

### From PyPI

```
pip install ollama-inspect
```

### From Source

```
git clone https://github.com/brainlounge/ollama-inspect.git
cd ollama-inspect
pip install .
```


Usage
-----
Run the local server:

```
ollama-inspect
# or
python -m ollama_inspect
```

You can specify host and port:

```
ollama-inspect --host 0.0.0.0 --port 8080
```

Then open your browser at:

```
http://127.0.0.1:13655/
```

Notes
- `--host` defaults to `127.0.0.1`.
- `--port` defaults to `13655`.


Screenshots
-----------

Home page — models overview

![Home page — models overview](./ollama-inspect-overview.png)

Model metadata — key/value view

![Model metadata — key/value view](./ollama-inspect-keyvalues.png)


How it finds your models
------------------------
By default the app looks under your user home directory, using the same structure as Ollama:

- Manifests: `~/.ollama/models/manifests/registry.ollama.ai/library/` (subfolders per model)
- Blobs: `~/.ollama/models/blobs/` (files named like `sha256-<hex>`)

The home page reads all valid manifests it finds and shows one entry per manifest. It also derives the expected blob filenames from the manifests and checks whether those blobs are present under `blobs/`.

Using the web UI
----------------
Home page (`/`)
- Lists discovered model manifests
- Shows model size (if present in the manifest)
- Provides quick links to:
  - Model Metadata (`/model/metadata?...`)
  - Template (`/model/template?...`)
  - License (`/model/license?...`)
  - Parameter Presets (`/model/params?...`)
- Includes a simple filter box and sortable columns

Model metadata (`/model/metadata`)
- Displays parsed GGUF keys and a preview for each value
- Values that are large or multi‑line can be expanded client‑side

Template (`/model/template`), License (`/model/license`), Params (`/model/params`)
- Open the corresponding text blob, if the manifest references it and the blob exists locally
- Params page attempts to parse JSON first; if not JSON, it falls back to a permissive `key: value` parser

Selecting a model or file
- Most pages accept either:
  - `model=<name[:tag]>` to resolve the correct blob via the manifest, or
  - `filename=sha256-<hex>` to directly reference a specific blob from `~/.ollama/models/blobs/`
- The UI links use `model=...` so you usually don’t need to type anything.


API endpoints
-------------
All endpoints return JSON.

1) `GET /api/keys?filename=sha256-<hex>`
- Returns the list of GGUF keys parsed from a model blob.
- Response schema (simplified):

```
{
  "model_path": "/full/path/to/blob",
  "count": <int>,
  "keys": ["..."],
  "filename": "sha256-..."
}
```

2) `GET /api/items?filename=sha256-<hex>`
- Returns key/value items with a short preview for each value.
- Response schema (simplified):

```
{
  "model_path": "/full/path/to/blob",
  "count": <int>,
  "items": [
    { "key": "...", "value": <any>, "preview": "...", "expandable": <bool> }
  ],
  "filename": "sha256-..."
}
```

Tip: You can find the `filename` by looking in `~/.ollama/models/blobs/` or by clicking a model on the home page and copying the shown filename.


Troubleshooting
---------------
- No models appear on the home page
  - Make sure you have pulled some models with Ollama (`ollama pull llama3`, etc.)
  - Verify manifests exist under `~/.ollama/models/manifests/registry.ollama.ai/library/`
  - Check file permissions for your user account

- “Layer not found” or missing template/license/params links
  - Not every manifest includes all layer types; links only appear when present
  - Some blobs might not be downloaded locally yet

- Cannot read GGUF / parser errors
  - Ensure `gguf` is installed (`pip install -r requirements.txt`)
  - Confirm the referenced `filename` actually points to a GGUF model blob

- Port is already in use
  - Start the app with another port: `ollama-inspect --port 13656`

- Windows paths
  - The app uses your user home directory; the equivalent of `~/.ollama` is typically `C:\\Users\\<you>\\.ollama`


Development
-----------
- Code entrypoint: `ollama_inspect/__main__.py` launches the Flask app defined in `webapp.py`
- HTML templates live in `templates/`
- Path helpers are in `path_utils.py`
- GGUF extraction helpers are in `gguf_utils.py`

Run locally with auto‑reload (optional)
- You can enable Flask debug mode by changing the `debug` flag in `ollama_inspect/__main__.py` (for local development only). Do not enable debug in production.

Formatting & style
- The codebase uses standard Python typing and a lightweight, dependency‑minimal approach. Please keep changes small and focused.


Security & privacy
------------------
- The server only reads files from your local `~/.ollama` directory
- There is no file upload and no network calls to third‑party services
- Run it on `127.0.0.1` unless you explicitly intend to expose it on your network


License
-------
This project is licensed under the Apache License, Version 2.0.

- SPDX-License-Identifier: Apache-2.0
- See the LICENSE file in this repository for the full license text.

Copyright (c) 2025 Bernd Fondermann and contributors
