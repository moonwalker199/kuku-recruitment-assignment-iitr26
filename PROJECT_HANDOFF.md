# PROJECT HANDOFF: Audio Drama Promotional Moment Recommendation System (Kuku FM Assignment)

**Document Date:** November 6, 2026  
**Target Audience:** ChatGPT / Successor AI Engineer  
**Status:** In-flight handoff; parser & validation verified, architecture code scaffolded, candidate generation evaluation pending.

---

## 1. Project Overview

### 1.1 Objective & Background
This project builds an explainable, end-to-end **Promotional Moment Recommendation System** for episodic audio dramas (in the style of Kuku FM, ReelShort, DramaBox).
- **Core Goal**: Ingest audio drama screenplay scripts (episodes 1 to 50), identify emotionally charged, cohesive dramatic sequences (moments of betrayal, high conflict, plot twists, confessions, cliffhangers), and rank them globally as promotional hooks/trailers (typically 30s–90s in duration).
- **Core Principles**:
  - **No Episode Quota**: An episode may yield 0, 1, or multiple candidate moments. Candidates are scored and ranked globally.
  - **Cohesive Windows**: Moments must be multi-segment dramatic beats (dialogue exchanges + action + sound effects), not isolated single lines.
  - **Explainability & Provenance**: Every recommendation must trace directly back to exact source segment IDs, timestamps, speaker dialogue, and delivery tags.
  - **Deterministic Priority**: High reliance on screenplay structure and parenthetical cues (`delivery`), with an optional, graceful LLM semantic enhancement layer (Gemini/OpenAI) that never breaks the pipeline if unconfigured.

### 1.2 Input Data & Expected Outputs
- **Inputs**:
  - `data/raw/`: 50 raw screenplay `.txt` files (`episode-001-...txt` to `episode-050-...txt`).
  - `data/processed/`: 50 structured screenplay `.json` files (`episode-001-..._parsed.json` to `episode-050-..._parsed.json`).
- **Expected Outputs**:
  - `outputs/candidates/`: Per-episode candidate JSON files.
  - `outputs/ranked_promos/ranked_promos.json`: Globally ranked Top-K recommendations.
  - `outputs/rejected_candidates/rejected_candidates.json`: Candidates filtered by duration, dialogue density, score threshold, or suppressed by Non-Maximum Suppression (NMS).
  - `outputs/reports/`: Human-readable Markdown reports (per-episode and global summary).
  - `data/processed/candidates.json`: Consolidated candidate generation output (requested in current task phase).
  - `outputs/dataset_profile.md` & `outputs/candidate_generation_report.md`: Dataset audit and candidate evaluation reports.

---

## 2. Complete Project Structure

Below is the verified, exact directory tree as of November 6, 2026:

```
Assignment Solution/
├── .env                                       # Environment file for API keys (GOOGLE_API_KEY, OPENAI_API_KEY)
├── .vscode/
│   └── settings.json
├── config/
│   └── config.yaml                            # Central YAML configuration (parameters, weights, LLM, ranking)
├── data/
│   ├── embeddings/                            # (Empty directory)
│   ├── processed/                             # 50 parsed episode JSON files (*_parsed.json)
│   └── raw/                                   # 50 original screenplay TXT files (*.txt)
├── notebooks/                                 # (Empty directory)
├── outputs/
│   ├── candidates/                            # Pre-created output directory for generated candidates
│   ├── ranked_promos/                         # Pre-created output directory for top-ranked promos
│   ├── rejected_candidates/                   # Pre-created output directory for rejected/suppressed candidates
│   └── reports/                               # Pre-created output directory for Markdown reports
├── src/
│   ├── __init__.py
│   ├── config.py                              # YAML config parser, dataclasses, validation, dynamic weight scaler
│   ├── utils.py                               # Logging, timestamp conversion (seconds <-> MM:SS.cc), timers
│   ├── candidate_generation/
│   │   ├── __init__.py
│   │   └── window_generator.py                # Time-anchored sliding window candidate generator
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   ├── lexicons.py                        # ~380 delivery tags taxonomy, conflict/cliffhanger/hook keyword sets
│   │   └── feature_extractor.py               # 12 deterministic normalised features ([0.0, 1.0])
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── episode_loader.py                  # Ingestion loader for data/processed/*_parsed.json
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py                          # Gemini & OpenAI client wrapper with retry & exponential backoff
│   │   ├── prompts.py                         # Structured JSON prompts & response schema validation
│   │   └── semantic_analyzer.py               # Batch processor with score threshold gating & feature merger
│   ├── ml/
│   │   └── __init__.py                        # Reserved stub (intentionally no trained ML model)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── screenplay_parser.py               # Baseline regex parser converting TXT -> JSON
│   ├── ranking/
│   │   ├── __init__.py
│   │   ├── deduplicator.py                    # Temporal IoU Non-Maximum Suppression (NMS) per episode
│   │   ├── explainer.py                       # Grounded evidence extraction & Markdown/JSON report builder
│   │   ├── ranker.py                          # Threshold filter, NMS orchestrator, global Top-K sorter
│   │   └── scorer.py                          # Weighted sum scoring engine with dynamic weight redistribution
│   └── validation/
│       └── __init__.py
├── tests/                                     # Empty directory (unit tests need to be written)
├── main.py                                    # Orchestrator CLI entry point
├── README.md                                  # Empty file (documentation needs to be populated)
├── requirements.txt                           # pyyaml, python-dotenv, google-generativeai, openai, tqdm
└── validate.py                                # Parser verification script (TXT vs JSON parity check)
```

### Distinction of Planned / Missing Files:
- **`data/processed/candidates.json`**: Missing; to be generated by running candidate generation across all 50 episodes.
- **`outputs/dataset_profile.md`**: Missing; to be generated from the dataset audit.
- **`outputs/candidate_generation_report.md`**: Missing; to be generated from candidate evaluation.
- **`src/candidate_generation/candidate_generator.py`**: Currently implemented as `src/candidate_generation/window_generator.py` (needs alias or rename to match requested naming convention).
- **`tests/test_candidate_generator.py`**: Missing; unit tests for candidate generation and edge cases.
- **`README.md`**: Currently empty; needs complete project documentation.

---

## 3. Dataset Details

### 3.1 File Conventions & Volume
- **Raw scripts**: `data/raw/episode-{001..050}-Episode {id}.txt` (50 files).
- **Parsed scripts**: `data/processed/episode-{001..050}-Episode {id}_parsed.json` (50 files).
- **Corpus story**: A cohesive Regency romance/drama serial following Lady Juliana (betrayed at the altar by Gerald for sister Clarissa, disowned by her family, married to a scarred veteran Alexander who is secretly the rightful Duke of Ashburn).

### 3.2 Dataset Statistics (Verified from Corpus Audit)
- **Episodes**: 50
- **Total Scenes**: 100 (Average: 2.0 scenes/episode; range: 1 to 5 scenes)
- **Total Segments**: 3,727 (Average: 74.5 segments/episode; range: 37 to 121 segments)
- **Episode Duration Range**: 70.5s to 200.8s (Average: 123.3s ≈ 2 minutes per episode)
- **Segment Durations**: Average 2.94s (range: 0.30s to 22.20s)
- **Segment Breakdown by Kind**:
  - `camera`: 1,754 (47.1%) — visual framing, angles, camera motions.
  - `dialogue`: 1,487 (39.9%) — character lines with parenthetical acting deliveries.
  - `action`: 372 (10.0%) — physical movements, character gestures, scene atmosphere.
  - `sound_fx`: 114 (3.1%) — audio cues (carriage clatter, string drones, door slams, gunshots).

### 3.3 Top Speakers
1. Juliana / Lady Juliana: 462 lines
2. Alexander / Mysterious Bearded Man / Scarred Bearded Man: 230 lines
3. Catherine: 187 lines
4. Blond Nobleman (Rupert): 53 lines
5. Alexander's Mother: 39 lines
6. Prudence: 38 lines
7. Lord: 36 lines
8. Gerald / The Former Groom: 48 lines

### 3.4 JSON Schema Structure
Each parsed JSON follows this root schema:
```json
{
  "source_file": "episode-001-Episode 1671703.txt",
  "episode": "Episode 1",
  "logline": "Lady Juliana arrives at the cathedral in her late mother's bridal gown...",
  "continuity_notes": "First episode of the series. Establishes Juliana's humiliation...",
  "episode_duration_seconds": 138.0,
  "scene_count": 1,
  "segment_count": 95,
  "counts_by_kind": {
    "dialogue": 33,
    "action": 7,
    "camera": 54,
    "sound_fx": 1
  },
  "scenes": [
    {
      "scene_number": 1,
      "heading": "INT. CATHEDRAL – DAY",
      "start_seconds": 0.0,
      "end_seconds": 138.0,
      "segments": [ ... ]
    }
  ],
  "segments": [ ... ],
  "parse_errors": []
}
```

### 3.5 Segment-Level Schema & Actual Example
```json
{
  "segment_id": "E1_S1_011",
  "scene_number": 1,
  "kind": "dialogue",
  "label": "LADY JULIANA (Desperate, Unbelieving)",
  "speaker": "LADY JULIANA",
  "delivery": "Desperate, Unbelieving",
  "start": "00:15.50",
  "end": "00:17.50",
  "start_seconds": 15.5,
  "end_seconds": 17.5,
  "duration_seconds": 2.0,
  "text": "Gerald is waiting for me.",
  "source_line": 54
}
```

### 3.6 Critical Data Properties & Irregularities
1. **Wall-Clock Overlap**: Screenplay directions frequently overlap in time (e.g. `camera` 00:00.00–00:04.80 overlaps with `action` 00:01.00–00:04.80). Candidates **must** be grouped by wall-clock time intervals, not pure segment indexing.
2. **Camera Dominance**: 47.1% of segments are camera directions. In promotional audio clips, camera cues carry zero acoustic value. Windows must not be selected purely on visual cues; they require dialogue and dramatic action.
3. **Delivery Tags Richness**: 380 unique parenthetical tags exist (e.g., `Cold, Callous`, `Choking on Tears`, `Roaring Defiance`). These are gold-standard signals for emotional intensity and conflict.

---

## 4. Completed Parser and Validation Work

### 4.1 Parser: `src/preprocessing/screenplay_parser.py`
- Converts raw `.txt` into structured JSON.
- **Parsing logic**:
  - Matches header blocks: `EPISODE NUMBER:`, `EPISODE LOGLINE:`, `CONTINUITY NOTES:`.
  - Matches scenes via `SCENE \d+`, `Scene heading:`, `Scene start:`, `Scene end:`.
  - Matches segment blocks via `\*\*(.+?):\*\*` followed by `Start: (MM:SS.cc) | End: (MM:SS.cc)` and body text.
  - Classifies segment type (`camera`, `action`, `sound_fx`, `dialogue`). For dialogue, extracts speaker name and regexes parenthetical `(delivery)`.
  - Generates deterministic IDs: `E{episode_id}_S{scene_number}_{sequence:03d}` (e.g. `E1_S1_011`).

### 4.2 Validator: `validate.py`
- Compares all 50 `.txt` raw files directly against their corresponding `.json` files.
- Checks:
  - Header extraction parity.
  - Scene numbers and headings.
  - Segment count, start/end timestamps, text body integrity, kind classification.
  - Monotonicity and boundary consistency.
- **Latest Verified Result**:
  ```
  TXT files found:       50
  JSON files found:      50
  Files expected:        50
  Files matched:         50
  Total errors:          0
  Total warnings:        0
  RESULT: PASS
  ```
- **Rule**: Parser and validator are verified 100% PASS and must NOT be modified.

---

## 5. Every Existing Python Module

| File | Responsibility | Main Functions / Classes | Dependencies | Implementation Status |
|---|---|---|---|---|
| `validate.py` | TXT vs JSON parser parity verification | `validate_file()`, `main()` | `re`, `json`, `pathlib` | **Verified (PASS)** |
| `src/preprocessing/screenplay_parser.py` | Converts screenplay `.txt` to structured JSON | `parse_script()`, `clean_body()`, `to_seconds()`, `main()` | `re`, `json`, `pathlib`, `argparse` | **Verified (PASS)** |
| `src/config.py` | Configuration dataclasses, YAML loader, constraint validation, weight normaliser | `Config`, `LLMConfig`, `CandidateConfig`, `RankingConfig`, `OutputConfig`, `load_config()`, `apply_overrides()` | `pyyaml`, `dataclasses`, `pathlib` | **Implemented, unverified in full pipeline** |
| `src/utils.py` | Logging format, timestamp conversion (`seconds_to_timestamp`), duration formatting, timer | `setup_logging()`, `seconds_to_timestamp()`, `format_duration()`, `ensure_output_dirs()`, `Timer` | `logging`, `time`, `pathlib` | **Implemented, unverified in full pipeline** |
| `src/ingestion/episode_loader.py` | Loads and verifies all `*_parsed.json` files | `load_episodes()`, `episode_id_from_name()` | `json`, `pathlib` | **Implemented, tested standalone (OK)** |
| `src/candidate_generation/window_generator.py` | Generates wall-clock sliding windows (25s–95s) with scene alignment and segment deduplication | `CandidateWindow`, `WindowGenerator`, `_segments_in_range()`, `_count_scene_boundaries()` | `src.config`, `src.ingestion.episode_loader` | **Implemented, unverified across full corpus** |
| `src/feature_extraction/lexicons.py` | Curated taxonomy of ~380 delivery tags into 7 dramatic groups + hook/cliffhanger keyword tuples | `HIGH_CONFLICT_TAGS`, `DISTRESS_TAGS`, `DEFIANCE_TAGS`, `CLIFFHANGER_TAGS`, `HOOK_KEYWORDS`, `score_delivery_tag()` | Standard library (`frozenset`) | **Implemented, unverified in full pipeline** |
| `src/feature_extraction/feature_extractor.py` | Computes 12 normalised deterministic features in [0.0, 1.0] | `FeatureExtractor`, `_emotional_intensity()`, `_cliffhanger_score()`, `_conflict_score()`, `_tension_arc()`, etc. | `src.candidate_generation.window_generator`, `src.feature_extraction.lexicons` | **Implemented, unverified in full pipeline** |
| `src/llm/client.py` | Optional LLM API caller for Google Gemini & OpenAI with retries | `LLMClient`, `_GeminiClient`, `_OpenAIClient`, `_parse_json_response()` | `google.generativeai` (optional), `openai` (optional), `os`, `json` | **Implemented, unverified with live API** |
| `src/llm/prompts.py` | Formats screenplay excerpt into prompt, parses & validates JSON response | `build_analysis_prompt()`, `validate_llm_response()` | `src.candidate_generation.window_generator`, `src.utils` | **Implemented, unverified with live API** |
| `src/llm/semantic_analyzer.py` | Score threshold gate, batch LLM executor, features merger | `SemanticAnalyzer`, `merge_llm_features()` | `src.config`, `src.llm.client`, `src.llm.prompts` | **Implemented, unverified** |
| `src/ranking/scorer.py` | Weighted scoring engine combining active features | `Scorer` (`score()`, `score_breakdown()`) | `src.config` | **Implemented, unverified** |
| `src/ranking/deduplicator.py` | Per-episode Non-Maximum Suppression (NMS) based on temporal IoU | `Deduplicator`, `_iou()`, `group_by_episode()` | Standard library | **Implemented, unverified** |
| `src/ranking/ranker.py` | Threshold filtering, per-episode NMS triggering, global Top-K ranking | `Ranker` (`rank()`, `apply_score_to_candidates()`) | `src.config`, `src.ranking.deduplicator` | **Implemented, unverified** |
| `src/ranking/explainer.py` | Evidence extraction and grounded reasoning generation for JSON & Markdown | `Explainer`, `write_episode_report()`, `write_global_report()` | `src.candidate_generation.window_generator`, `src.utils`, `src.feature_extraction.lexicons` | **Implemented, unverified** |
| `main.py` | Full CLI orchestrator connecting stages 1 to 10 | `run()`, `main()`, `_build_parser()` | All `src.*` modules | **Implemented, unverified end-to-end** |

### Potential Incompleteness / Naming Deviations:
1. **Module Name Discrepancy**: The prompt asks for `src/candidate_generation/candidate_generator.py`. The existing module is named `src/candidate_generation/window_generator.py`. ChatGPT should either alias `CandidateGenerator = WindowGenerator` or rename/wrap it so both conventions work seamlessly.
2. **Missing Standalone Candidate Output**: `data/processed/candidates.json` is not yet saved to disk.
3. **Tests**: Directory `tests/` is completely empty.

---

## 6. Configuration and Dependencies

### 6.1 `config/config.yaml`
```yaml
pipeline:
  top_k: 50
  llm_enabled: true
  random_seed: 42

candidate_generation:
  window_step_seconds: 5
  window_durations: [15, 25, 35, 50, 65, 80]
  min_duration_seconds: 25.0
  max_duration_seconds: 95.0
  min_dialogue_segments: 3
  include_scene_aligned_windows: true
  include_scene_pair_windows: true

llm:
  provider: "google"
  model: "gemini-1.5-flash"
  temperature: 0.1
  max_output_tokens: 512
  timeout_seconds: 30
  max_retries: 3
  retry_delay_seconds: 2.0
  llm_score_threshold: 0.25

feature_weights:
  emotional_intensity: 0.15
  cliffhanger_score: 0.15
  conflict_score: 0.12
  tension_arc: 0.10
  lexical_hook_score: 0.08
  revelation_score: 0.07
  dialogue_density: 0.05
  protagonist_presence: 0.04
  speaker_variety: 0.03
  position_in_episode: 0.02
  sound_fx_density: 0.01
  scene_boundary_bonus: 0.01
  llm_promotional_suitability: 0.10
  llm_hook_potential: 0.05
  llm_context_independence: 0.02

ranking:
  min_score_threshold: 0.30
  diversity:
    enabled: true
    iou_threshold: 0.50

output:
  candidates_dir: "outputs/candidates"
  ranked_promos_dir: "outputs/ranked_promos"
  rejected_candidates_dir: "outputs/rejected_candidates"
  reports_dir: "outputs/reports"
  save_all_candidates: true
  generate_markdown: true
  generate_global_report: true
  max_evidence_segments: 5
```

### 6.2 Installed Dependencies (Python 3.10)
- `pyyaml>=6.0.0` (Installed: 6.0.3)
- `python-dotenv>=1.0.0` (Installed: 1.2.4)
- `tqdm>=4.65.0` (Installed: 4.68.2)
- `google-generativeai` & `openai` (Listed in requirements; LLM calls must remain optional and fail-safe if uninstalled).

---

## 7. Actual Execution Status

### What Has Actually Run and Succeeded:
1. **`python validate.py`**:
   - **Result**: `RESULT: PASS`. All 50 TXT files match their JSON files with 0 errors and 0 warnings.
2. **Corpus profiling scripts**:
   - Executed inline Python scripts to compute exact corpus statistics (3,727 segments, 100 scenes, delivery tag lists, duration stats).
3. **Module Import Sanity Check**:
   - Executed `python -c "import src.config, src.utils, ..."`: All 13 modules imported successfully without syntax or import errors.

### What Has NOT Yet Run or Succeeded:
1. **`python main.py --no-llm`**:
   - Attempted via PowerShell with bash chaining `&&` which resulted in a PowerShell syntax error (`&& is not a valid statement separator`). The command was **not** executed cleanly end-to-end.
2. **Candidate Generation across all 50 episodes**:
   - `data/processed/candidates.json` does NOT exist yet.
   - `outputs/dataset_profile.md` does NOT exist yet.
   - `outputs/candidate_generation_report.md` does NOT exist yet.
3. **Unit Tests**:
   - Zero test files have been written or run.
4. **Final Ranked Outputs**:
   - `outputs/ranked_promos/ranked_promos.json` does NOT exist yet.

---

## 8. Known Problems, Risks & Edge Cases

1. **PowerShell vs Bash Chaining**:
   - On this Windows environment, commands run under PowerShell. Chaining with `&&` or piping to `head` fails. Use standard PowerShell syntax or run commands as separate invocations.
2. **Naming Convention**:
   - The user requested `src/candidate_generation/candidate_generator.py`. The file was created as `src/candidate_generation/window_generator.py`. Create `candidate_generator.py` (either implementing the generator or importing and re-exporting `WindowGenerator` as `CandidateGenerator`) to prevent confusion.
3. **Camera-Heavy Window Risk**:
   - 47% of segments are camera directions. In `window_generator.py`, there is a check `len(dialogue) < min_dialogue_segments (3)`. However, there should also be a ratio check (e.g. `dialogue_duration + action_duration` must comprise at least 40% of the candidate's duration) to ensure camera segments don't dominate the time window.
4. **Candidate Serialization Schema**:
   - The user prompt specifies that each candidate must contain:
     - `candidate_id` (or `window_id`)
     - `episode_id` and `episode`
     - `scene_number` (or `scene_numbers`)
     - `start_seconds` and `end_seconds` (and formatted strings)
     - `duration` (`duration_seconds`)
     - `source_segment_ids` (`segment_ids`)
     - `combined_screenplay_text` (concatenated text of included dialogue and action)
     - `relevant_speakers` (list of speakers in the window)
     - `segment_type_distribution` (counts of dialogue, action, camera, sound_fx)
   - Ensure the candidate generator dict representation includes all these exact fields.
5. **No Supervised ML**:
   - Do NOT attempt to train an ML model in `src/ml/`. There are no ground-truth engagement labels.

---

## 9. Remaining Work

### Critical Tasks (Phase 1 & Phase 2 & Phase 3)
1. **Dataset Profile (`outputs/dataset_profile.md`)**:
   - Write a standalone script or run the audit to generate the markdown report with all 8 requested statistics.
2. **Candidate Generator Standardisation**:
   - Expose `src/candidate_generation/candidate_generator.py` with the exact requested candidate schema.
   - Add filter to reject camera-dominated windows.
3. **Run Candidate Generation**:
   - Run candidate generation across all 50 episodes.
   - Save the consolidated candidates to `data/processed/candidates.json`.
4. **Candidate Evaluation Report (`outputs/candidate_generation_report.md`)**:
   - Document candidates generated, distributions, filter stats, and at least 10 representative examples.
5. **Unit Tests**:
   - Implement `tests/test_candidate_generator.py` covering:
     - Valid window generation
     - Scene boundary enforcement
     - Minimum dialogue segment filtering
     - Duration bounds (min/max duration)
     - Edge cases (single-scene vs multi-scene, zero candidates, short episodes)
   - Execute tests using `pytest` or `unittest`.

### Downstream Tasks (Subsequent Phases)
- Hook up semantic analysis (`src/llm/`) with live API key if available, or confirm rule-based fallback.
- Run full ranking pipeline end-to-end (`main.py --no-llm`).
- Verify outputs in `outputs/ranked_promos/` and `outputs/reports/`.
- Write comprehensive `README.md`.

---

## 10. NEXT ACTIONS FOR CHATGPT

When taking over this project, execute these exact steps in order:

### Step 1: Inspect Key Files
1. Read `validate.py` (to understand data schema guarantees).
2. Read `src/candidate_generation/window_generator.py` (to review existing window generation logic).
3. Read `config/config.yaml` (to review current parameter settings).

### Step 2: Implement Phase 1 (Dataset Audit)
Run a script to generate `outputs/dataset_profile.md` covering:
- Total episodes, scenes, segments, kind breakdown, duration distributions, speaker frequencies, and timestamp continuity notes.

### Step 3: Implement Phase 2 (`src/candidate_generation/candidate_generator.py`)
Ensure `candidate_generator.py`:
- Contains a class `CandidateGenerator`.
- Implements `generate_candidates(episode_dict) -> list[dict]`.
- Implements fields: `candidate_id`, `episode_id`, `episode`, `scene_numbers`, `start_seconds`, `end_seconds`, `start_time`, `end_time`, `duration_seconds`, `source_segment_ids`, `combined_screenplay_text`, `relevant_speakers`, `segment_type_distribution`.
- Rejects windows that exceed camera threshold (>60% camera time) or have fewer than `min_dialogue_segments`.

### Step 4: Implement Phase 3 (Generation & Reporting)
1. Write a runner script that processes all 50 episodes from `data/processed/` using `CandidateGenerator`.
2. Save output to `data/processed/candidates.json`.
3. Generate `outputs/candidate_generation_report.md` showing:
   - Total candidates generated
   - Candidate counts per episode
   - Duration histogram/distribution
   - Filtered candidates count & rejection reasons
   - 10 diverse representative candidate moments with exact screenplay quotes.

### Step 5: Write & Run Unit Tests
1. Create `tests/test_candidate_generator.py`.
2. Run tests via `python -m unittest discover tests` or `pytest tests`.
3. Verify all tests pass.

---

## 11. Final Handoff Summary

| Component | Status | Evidence |
|---|---|---|
| Screenplay Parser | **Verified** | `validate.py` reports 100% PASS (50/50 files, 0 errors, 0 warnings). |
| Parser Validator | **Verified** | Executed and confirmed passing. |
| Ingestion Module | **Verified** | `src/ingestion/episode_loader.py` imports and loads all 50 episodes without error. |
| Window / Candidate Generation | **Implemented, unverified** | `src/candidate_generation/window_generator.py` exists; not yet executed across corpus to output `candidates.json`. |
| Feature Extraction & Lexicons | **Implemented, unverified** | `src/feature_extraction/feature_extractor.py` and `lexicons.py` exist and import cleanly; end-to-end extraction not verified. |
| LLM Semantic Layer | **Implemented, unverified** | `src/llm/client.py`, `prompts.py`, `semantic_analyzer.py` implemented with retries and fallbacks; live API calls not yet executed. |
| Scoring & Ranking | **Implemented, unverified** | `src/ranking/scorer.py`, `deduplicator.py`, `ranker.py` implemented; full ranking not yet executed. |
| Explainer & Reporting | **Implemented, unverified** | `src/ranking/explainer.py` implemented; report generation not yet executed. |
| Orchestrator (`main.py`) | **Implemented, unverified** | Syntax valid and imports cleanly; full run not yet completed. |
| Dataset Profile Report | **Missing** | `outputs/dataset_profile.md` not yet created. |
| Consolidated Candidates JSON | **Missing** | `data/processed/candidates.json` not yet generated. |
| Candidate Generation Report | **Missing** | `outputs/candidate_generation_report.md` not yet created. |
| Unit Tests | **Missing** | `tests/` directory is empty. |
| Documentation (`README.md`) | **Missing** | Root `README.md` is currently empty. |
