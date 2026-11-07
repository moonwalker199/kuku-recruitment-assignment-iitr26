# Kuku FM Promotional Moment Discovery System

An AI-assisted pipeline that analyzes screenplay episodes, discovers potential promotional clips, optimizes clip boundaries, evaluates promotional quality, and produces ranked recommendations.

## 1. Problem Overview

The objective is to identify promotional moments from the first 50 episodes of an English-language microdrama screenplay.

A recommended promo should:

- Capture attention within the first 3 seconds.
- End on an unresolved moment or compelling cliffhanger.
- Be understandable to viewers unfamiliar with the show.
- Have a final duration between 20 and 60 seconds.
- Use exact source timestamps and preserve the original sequence of events.

The system identifies moments and their timestamps. It does not edit video, add music, or generate promotional overlays.

## 2. Pipeline Architecture

The system uses a hybrid Python and LLM-based architecture.

### Stage 1: Screenplay ingestion and parsing

- Reads the original screenplay TXT files.
- Extracts scenes, dialogue, actions, camera directions, sound effects, and timestamps.
- Generates structured JSON representations.
- Preserves source segment IDs and timestamps for traceability.

### Stage 2: Episode-level AI analysis

- Processes each episode using an LLM.
- Maintains a running story summary across episodes to provide narrative context.
- Identifies story developments and potential promotional moments.
- Returns candidate moments with source segment IDs and supporting evidence.

### Stage 3: Candidate preparation and structural shortlisting

- Matches discovered candidates against their original screenplay segments.
- Validates source references.
- Computes general structural signals such as dialogue share, speaker diversity, scene diversity, segment transitions, and segment density.
- Combines structural signals with the model's candidate confidence to prioritize candidates.
- Retains a shortlist for more expensive AI evaluation.

### Stage 4: Boundary optimization

- Generates feasible alternative clip boundaries using the original source segments.
- Uses an LLM to select among the proposed alternatives.
- Validates timestamps, segment continuity, and the 20 to 60 second duration constraint.

### Stage 5: Promotional quality evaluation

Each optimized candidate is evaluated on five dimensions:

| Dimension                   | Weight |
| --------------------------- | -----: |
| Hook within first 3 seconds |    20% |
| Cliffhanger strength        |    25% |
| Standalone clarity          |    20% |
| Emotional impact            |    15% |
| Promotional suitability     |    20% |

The weighted score is calculated in Python rather than delegated to the LLM.

### Stage 6: Deduplication and final ranking

- Compares clips from the same episode using temporal Intersection over Union (IoU).
- Suppresses overlapping recommendations at or above the configured 0.50 threshold.
- Retains the higher-scoring candidate when duplicates are found.
- Ranks the remaining candidates by overall evaluation score.

### Stage 7: Output generation

Produces ranked, accepted, review, and rejected candidate files, along with execution statistics and supporting intermediate outputs.

## 3. Technology Stack

- **Python:** Parsing, validation, candidate processing, scoring, ranking, and output generation.
- **Google Gemini API:** Episode analysis, boundary optimization, and promotional evaluation.
- **OpenAI API:** Alternative provider support in the LLM client.
- **PyYAML:** Configuration management.
- **python-dotenv:** Environment variable loading.
- **tqdm:** Progress display.

The LLM provider and model are configured through `config/config.yaml`.

## 4. Project Structure

```text
Assignment Solution/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                  # Original screenplay TXT files
│   └── processed/            # Parsed episode JSON files
├── outputs/
│   ├── candidates/           # Discovered and shortlisted candidates
│   ├── cache/                # Cached LLM responses and rate-limit state
│   ├── episode_analysis/     # Episode analysis results
│   ├── ranked_promos/        # Final ranked, accepted and review results
│   ├── rejected_candidates/ # Rejected and suppressed candidates
│   ├── reports/              # Episode-level reports and statistics
│   └── test/                 # Diagnostic and test artifacts
├── src/
│   ├── candidate_generation/
│   ├── feature_extraction/
│   ├── ingestion/
│   ├── llm/
│   ├── ml/
│   ├── preprocessing/
│   ├── ranking/
│   └── validation/
├── tests/
├── main.py
├── validate.py
├── requirements.txt
└── README.md
```

## 5. Setup and Execution

### Requirements

- Python 3.10 or later.
- A Google Gemini API key for the configured provider.
- Internet access for API calls.

### Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### Configure API credentials

Create a `.env` file in the project root.

For Google Gemini:

```env
GOOGLE_API_KEY=your_api_key_here
```

Alternatively, the client also accepts `GEMINI_API_KEY`.

For OpenAI, if using that provider:

```env
OPENAI_API_KEY=your_api_key_here
```

Never commit API credentials to version control.

### Configure the pipeline

Edit `config/config.yaml`.

The current configuration uses:

- Provider: `google`
- Model: `gemini-3.5-flash-lite`
- LLM analysis: enabled
- Maximum candidates requested per episode: 5
- Default shortlist size: 30

Model availability and quotas depend on the provider account and API configuration.

### Run the pipeline

```bash
python main.py
```

Optional arguments:

```bash
python main.py --help
```

The CLI supports custom configuration and input paths, candidate limits, shortlist size, and verbose logging.

### Validate parsed screenplay data

```bash
python validate.py
```

This checks the original TXT files against their parsed JSON representations.

## 6. Output Files

The main deliverables are:

| File                                                   | Description                               |
| ------------------------------------------------------ | ----------------------------------------- |
| `outputs/ranked_promos/ranked_promos.json`             | Ranked shortlist with evaluation results  |
| `outputs/ranked_promos/accepted_promos.json`           | Accepted promotional candidates           |
| `outputs/ranked_promos/review_promos.json`             | Candidates requiring further review       |
| `outputs/rejected_candidates/rejected_candidates.json` | Rejected and suppressed candidate records |
| `outputs/rejected_candidates/rejected_promos.json`     | Rejected promotional candidates           |
| `outputs/reports/discovery_stats.json`                 | Pipeline execution statistics             |

Intermediate candidate files and episode-level reports are also saved under `outputs/`.

## 7. Evaluation and Decision Rules

The evaluation score is a weighted combination of the five quality dimensions.

The current operational decision thresholds are:

- **Accept:** Overall score of at least 0.72, with hook, cliffhanger, and standalone clarity each at least 0.50.
- **Review:** Overall score of at least 0.50 but not meeting the acceptance conditions.
- **Reject:** Overall score below 0.50.

These thresholds are implementation assumptions used to organize recommendations. They are not official Kuku FM acceptance criteria.

## 8. Reliability and Efficiency

The pipeline includes:

- Cached episode analysis, boundary optimization, and evaluation responses.
- Retry handling for transient API failures.
- A shared Gemini rate limiter with configurable operating targets.
- Daily request tracking.
- Usage logging and actual prompt-token usage updates when returned by the API.
- Source segment and timestamp validation.
- Separate tracking of accepted, review, rejected, and duplicate candidates.

The token reservation mechanism uses an estimate before a request and updates it with reported usage after a successful response. It is an approximate local safeguard, not a guarantee of exact provider quota enforcement.

## 9. Validation and Current Run Results

The completed run processed the 50 supplied episodes.

| Metric                               | Result |
| ------------------------------------ | -----: |
| Episodes loaded                      |     50 |
| Episodes analyzed                    |     50 |
| Episode analysis failures            |      0 |
| Candidates discovered                |     64 |
| Candidates shortlisted               |     30 |
| Boundary optimizations successful    |     30 |
| Promo evaluations successful         |     29 |
| Promo evaluations failed             |      1 |
| Accepted recommendations             |     16 |
| Review recommendations               |     13 |
| Rejected recommendations             |      1 |
| Duplicate recommendations suppressed |      0 |

The episode analysis stage used cached responses in the final run. Boundary optimization and evaluation made 60 API calls combined, of which 59 succeeded.

Parser validation confirmed
