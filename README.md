# VinFast News Agent

VinFast News Agent is a Python application that retrieves the latest VinFast-related articles from [VnExpress](https://vnexpress.net/), extracts and cleans the full article content, and uses Groq-hosted large language models to produce structured business intelligence for an official VinFast distributor in Vietnam.

The project provides:

- A Streamlit web interface for browsing and analyzing articles interactively.
- A command-line interface (CLI) for scripted or terminal-based analysis.
- A reusable pipeline that can be called from a UI, CLI, or future API service.
- Deterministic article normalization and HTML cleaning.
- LLM-based article classification, relevance scoring, signal extraction, entity extraction, and ambiguity analysis.
- Type-specific analysis lenses such as competitive position, pricing impact, service operations, compliance risk, and sales enablement.
- JSON export of analysis results for downstream reporting or integration.

> **Data source and scope:** The application is currently specialized for VinFast-related content available through VnExpress. It is not a general-purpose crawler or a fully autonomous RAG index.

## Table of Contents

- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Analysis Output](#analysis-output)
- [Article Taxonomy and Lenses](#article-taxonomy-and-lenses)
- [Project Structure](#project-structure)
- [Operational Notes](#operational-notes)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Limitations](#limitations)
- [License](#license)

## How It Works

The application processes an article through the following stages:

1. **Fetch listing** – Queries the VnExpress taxonomy endpoint for the latest VinFast articles using the VinFast taxonomy ID (`1798`).
2. **Normalize metadata** – Converts the source response into a consistent article representation containing the ID, title, lead, URL, thumbnail, timestamps, source, and content hash.
3. **Fetch article detail** – Downloads the selected article page and extracts metadata, paragraphs, figures, tables, tags, author information, and publication time.
4. **Clean content** – Removes scripts, styles, embeds, advertisements, related-content blocks, and other irrelevant fragments before assembling the article text.
5. **Classify the article** – Assigns one primary article type and, when appropriate, a secondary type.
6. **Route analysis lenses** – Selects up to four domain-specific lenses based on the classification.
7. **Analyze with an LLM** – Produces a validated JSON object containing business signals, relevance, entities, quotes, fuzzy ambiguities, and lens-specific fields.
8. **Present or export results** – Displays the result in Streamlit or prints it in the CLI and saves the analysis as `analysis_<article_id>.json`.

## Key Features

- **Interactive article discovery:** Browse the latest 5–50 articles and open the original VnExpress page.
- **Structured analysis:** Results are returned as JSON rather than unstructured prose.
- **Distributor perspective:** Prompts are designed around increasing VinFast vehicle sales and maintaining smooth operations.
- **Evidence-based extraction:** Signals require polarity, strength, action hints, and source quotes.
- **Fuzzy reasoning:** Ambiguous statements can include multiple interpretations with confidence strengths.
- **Resilient HTTP access:** Source requests use retries and exponential backoff for rate limits and transient server errors.
- **Progress and latency reporting:** The UI and CLI expose detail-fetch, classification, analysis, and total processing times.
- **Configurable models:** Classification and analysis models can be selected independently through environment variables.
- **Logging:** Application events and errors are written through the project logger and can be inspected in `logs/app.log`.

## Architecture

```text
                          +----------------------+
                          |  Streamlit UI (app)  |
                          +----------+-----------+
                                     |
                          +----------v-----------+
                          | Pipeline orchestration|
                          | fetch_listing         |
                          | analyze_article       |
                          +----+-------------+----+
                               |             |
                +--------------v--+      +---v----------------+
                | VnExpress tools  |      | Analysis services  |
                | listing API      |      | classification     |
                | detail scraper   |      | lens routing       |
                | normalization    |      | Groq JSON analysis |
                +------------------+      +--------------------+
```

The main orchestration layer is intentionally independent from the Streamlit UI. The same `pipeline.py` functions are used by `app.py` and `main.py`.

## Requirements

- Python 3.10 or newer is recommended.
- Internet access to:
  - The VnExpress API and article pages.
  - The Groq API.
- A Groq API key.
- Python packages listed in [`requirements.txt`](requirements.txt).

The project currently imports the `groq` SDK directly. If it is not already installed in your environment, install it explicitly:

```bash
python -m pip install groq
```

## Installation

1. Clone or copy the project and enter its directory:

   ```bash
   cd AI-agent-RAG
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install groq
   ```

4. Create a local environment file:

   ```bash
   cp .env.template .env
   ```

   On Windows PowerShell:

   ```powershell
   Copy-Item .env.template .env
   ```

5. Edit `.env` and provide valid values as described below.

## Configuration

The application reads configuration from `.env` through `python-dotenv`. Never commit `.env` or expose the API key in source control.

| Variable | Required | Description | Example |
|---|---:|---|---|
| `GROQ_API_KEY` | Yes | API key used for both LLM calls | `gsk_...` |
| `LLM_ANALYZE_MODEL` | Yes | Groq model used for structured article analysis | `llama-3.3-70b-versatile` |
| `LLM_CLASSIFY_MODEL` | Yes | Groq model used for article classification | `llama-3.1-8b-instant` |
| `LLM_TIMEOUT` | No | Request timeout in seconds | `60` |

A typical configuration is:

```dotenv
GROQ_API_KEY=your_groq_api_key
LLM_ANALYZE_MODEL=llama-3.3-70b-versatile
LLM_CLASSIFY_MODEL=llama-3.1-8b-instant
LLM_TIMEOUT=60
```

The analysis and classification modules have fallback model values, but production usage should set both model variables explicitly. The classifier uses a shorter default timeout than the analysis module when its own environment configuration is not overridden.

## Running the Application

### Streamlit interface

Start the interactive web application with:

```bash
streamlit run app.py
```

The interface allows you to:

1. Choose how many articles to fetch, from 5 to 50.
2. Refresh the listing.
3. Select an article and run the analysis pipeline.
4. Review the analysis summary, key signals, lens extensions, entities, fuzzy ambiguities, and cleaned source content.
5. Download the analysis JSON from the results view.

Streamlit normally opens the application at `http://localhost:8501`.

### Command-line interface

Analyze the first article from a listing of five:

```bash
python main.py
```

Fetch ten articles and analyze the third article (zero-based index `2`):

```bash
python main.py --limit 10 --index 2
```

CLI options:

| Option | Default | Description |
|---|---:|---|
| `--limit` | `5` | Number of articles to request; valid range is 1–50 at the source-tool level |
| `--index` | `0` | Zero-based index of the article to analyze |

The CLI prints the classification and analysis, reports latency by stage, lists the selected lenses, and writes the result to:

```text
analysis_<article_id>.json
```

## Analysis Output

The structured analysis generally contains the following top-level fields:

```json
{
  "article_id": "string",
  "article_type": "string",
  "summary_1line": "string",
  "vinfast_stance": {
    "sentiment": 0.0,
    "confidence": 0.0,
    "band": "very_low|low|medium|high|very_high",
    "evidence": ["exact quotation"]
  },
  "relevance_to_distributor": {
    "score": 0.0,
    "band": "very_low|low|medium|high|very_high",
    "reason": "string"
  },
  "entities": {
    "brands": [],
    "models": [],
    "markets": [],
    "people": [],
    "numbers": []
  },
  "key_signals": [],
  "fuzzy_ambiguities": [],
  "meta": {
    "model": "string",
    "analyzed_at": "ISO-8601 timestamp",
    "prompt_version": "string"
  }
}
```

Each key signal is expected to include:

- `signal`
- `polarity`: `positive`, `negative`, or `neutral`
- `strength` and `band`
- `angle`
- `action_hint`
- `source_quote`

The analyzer is instructed not to fabricate competitors, numbers, political inferences, or source quotations. Fields that are not applicable should be returned as `null` rather than guessed.

## Article Taxonomy and Lenses

The classifier supports these article types:

- `sales_ranking`
- `product_launch`
- `pricing_promotion`
- `policy_regulation`
- `recall_service`
- `event_launch`
- `finance_market`
- `partnership`
- `competition`
- `technology`
- `consumer_opinion`
- `macro_market`
- `other`

Each type routes the article to business-oriented lenses. Examples include:

| Article type | Default lenses |
|---|---|
| `sales_ranking` | Competitive position, growth signal |
| `product_launch` | Product positioning, sales enablement |
| `pricing_promotion` | Margin impact, demand lever |
| `policy_regulation` | Compliance risk, demand lever |
| `recall_service` | Reputation risk, service operations |
| `event_launch` | Channel expansion, brand presence |
| `finance_market` | Investor signal, cost pressure |
| `competition` | Threat assessment, counter-positioning |
| `technology` | Product differentiation, training need |
| `consumer_opinion` | Perception signal, objection handling |
| `macro_market` | Demand context, timing signal |

The router deduplicates lenses and caps the prompt at four lenses to avoid unnecessary prompt growth.

## Project Structure

```text
.
├── app.py                    # Streamlit user interface
├── main.py                   # Command-line entry point
├── pipeline.py               # Shared listing and analysis orchestration
├── requirements.txt          # Python dependencies
├── .env.template             # Environment variable template
├── tools/
│   ├── analyze.py            # Groq analysis, prompts, validation, output formatting
│   ├── classify.py           # Article taxonomy and classifier
│   ├── lenses.py             # Lens routing and lens-specific schemas
│   ├── registry.py            # Tool registry and dispatcher
│   ├── vnexpress_detail.py   # Article HTML extraction and cleaning
│   └── vnexpress_tool.py     # VnExpress listing API and normalization
└── utils/
    └── logger.py             # Shared logging configuration
```

## Operational Notes

- VnExpress requests use retries for HTTP `429`, `500`, `502`, `503`, and `504` responses.
- Listing requests are limited to 1–50 articles by the source tool. The Streamlit slider intentionally exposes 5–50.
- Article detail extraction depends on the current VnExpress HTML structure and configured CSS selectors.
- The application makes external network and LLM calls during normal operation; latency and availability therefore depend on those services.
- `logs/app.log` may contain operational details. Review and rotate logs appropriately for your deployment environment.
- Generated `analysis_<article_id>.json` files are local artifacts and should be handled according to your data-retention requirements.

## Troubleshooting

### `Missing GROQ_API_KEY in .env`

Confirm that:

1. The file is named `.env` and is located in the project root.
2. `GROQ_API_KEY` contains a valid key.
3. The application was restarted after changing the environment file.

### `missing_llm_env`

Set both `GROQ_API_KEY` and the model variable required by the failing stage:

```dotenv
LLM_ANALYZE_MODEL=llama-3.3-70b-versatile
LLM_CLASSIFY_MODEL=llama-3.1-8b-instant
```

### `ModuleNotFoundError: No module named 'groq'`

Install the Groq SDK in the active virtual environment:

```bash
python -m pip install groq
```

### Listing or detail fetch errors

Check your network connection and inspect the logged error. VnExpress may change its API response schema or page markup, temporarily rate-limit requests, or reject requests from a particular network.

### Article index out of range

The CLI index is zero-based. Keep `--index` between `0` and `number_of_fetched_articles - 1`.

## Development

The main reusable entry points are:

```python
from pipeline import analyze_article, fetch_listing

listing = fetch_listing(limit=5)
result = analyze_article(listing.articles[0])
```

When extending the project:

1. Keep source-specific HTTP and parsing logic in `tools/`.
2. Keep orchestration in `pipeline.py`.
3. Keep presentation concerns in `app.py` or `main.py`.
4. Preserve the structured JSON contract when changing prompts or schemas.
5. Avoid committing secrets, generated artifacts, virtual environments, or local logs.

No automated test suite is currently included in the repository. Changes should therefore be validated with a local listing fetch, a representative article analysis, and a review of the generated JSON before deployment.

## Limitations

- The current source is VnExpress only.
- The system requires live access to both VnExpress and Groq; it does not provide an offline mode.
- LLM output quality depends on the selected model, prompt version, source content, and API availability.
- The HTML parser is resilient to common markup variations but may require updates when VnExpress changes its page structure.
- The repository does not currently define a production deployment configuration, database, authentication layer, or persistent vector store.
- No license file is currently provided.

## License

No license has been specified for this project yet. Add an explicit license before distributing or publishing the repository.
