import os
import re
import json
import hmac
import uuid
import threading
from pathlib import Path

import streamlit as st

from src.config import load_config
from src.preprocessing.screenplay_parser import parse_script
from main import run


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Kuku Assignment",

    layout="wide",
)

# --------------------------------------------------
# CUSTOM UI THEME
# --------------------------------------------------

st.markdown(

    """
    <style>

    /* Overall theme */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #000107;
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
        letter-spacing: -0.3px;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #A7A7AF !important;
    }

        /* Reduce empty space below the top navbar */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Reduce top spacing of the first content element */
    .block-container > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #17181D;
        border-right: 1px solid #303139;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.5rem !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    /* Tooltips */
    [data-testid="stTooltipContent"] {
        max-width: 260px !important;
        width: 260px !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }

    /* Upload metrics and result metrics */
    [data-testid="stMetric"] {
        background: #17181D;
        border: 1px solid #303139;
        border-radius: 10px;
        padding: 16px 12px;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        width: 100%;
        justify-content: center;
        text-align: center;
    }

    /* Center upload metric cards */
    .upload-metrics [data-testid="stHorizontalBlock"] {
        justify-content: center;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
        border: 1px solid #44454D;
    }

    .stButton > button[kind="primary"] {
        background: #EC1C24;
        color: #FFFFFF;
        border: 1px solid #EC1C24;
    }

    .stButton > button[kind="primary"]:hover {
        background: #C9161D;
        border-color: #C9161D;
        color: #FFFFFF;
    }

    /* Inputs */
    input,
    textarea,
    [data-baseweb="select"] > div {
        background: #24252B !important;
        color: #FFFFFF !important;
        border-color: #44454D !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #B8B8C0;
        font-weight: 600;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #EC1C24 !important;
        border-bottom-color: #EC1C24 !important;
    }

    /* Expanders and dividers */
    [data-testid="stExpander"] {
        background: #17181D;
        border: 1px solid #303139;
        border-radius: 8px;
    }

    hr {
        border-color: #303139;
    }

    /* Links */
    a {
        color: #FF555B !important;
    }
    /* Sidebar opening button */
    /* Replace Streamlit sidebar toggle icon with hamburger */
    [data-testid="collapsedControl"] svg {
        display: none !important;
    }

    [data-testid="collapsedControl"] button {
        position: relative;
    }

    [data-testid="collapsedControl"] button::after {
        content: "☰";
        font-size: 23px;
        font-weight: 400;
        color: inherit;
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }

        /* Center upload metrics: label, value and delta */
    [data-testid="stMetric"] {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center !important;
        text-align: center !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        width: 100%;
        display: flex;
        justify-content: center !important;
        text-align: center !important;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #303139;
        border-radius: 8px;
        overflow: hidden;
    }
    </style>

    """,
    unsafe_allow_html=True,
)


MAX_FILES = 70
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_TOTAL_SIZE = 50 * 1024 * 1024

PROJECT_ROOT = Path(__file__).resolve().parent
JOBS_ROOT = PROJECT_ROOT / "outputs" / "streamlit_jobs"

if "pipeline_lock" not in globals():
    pipeline_lock = threading.Lock()


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def get_secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def authenticate():
    app_password = get_secret("APP_PASSWORD")

    if not app_password:
        st.error(
            "APP_PASSWORD is not configured. "
            "Add it to Streamlit Secrets before using the app."
        )
        st.stop()

    if st.session_state.get("authenticated", False):
        return

    st.title("AI Drama Promo Generation Pipeline")
    st.subheader("Authentication required")
    st.info(
    "The application password is provided in the submission "
    "Google Drive folder. You can find it in either "
    "'1. Solution Doc.pdf' or '2. Links.pdf', "
    "both submitted through the Google Form. "
    "The password protection was added to prevent unauthorized "
    "API calls, as the application is publicly deployed."
)

    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0

    if st.session_state.login_attempts >= 5:
        st.error(
            "Too many failed attempts. "
            "Refresh the session to try again."
        )
        st.stop()

    entered_password = st.text_input(
        "Application password",
        type="password",
    )

    if st.button("Unlock application", type="primary"):
        if hmac.compare_digest(
            str(entered_password),
            str(app_password),
        ):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.session_state.login_attempts += 1
            st.error("Incorrect password.")

    st.stop()


authenticate()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def safe_filename(filename):
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:150] or "episode.txt"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def read_json(path):
    path = Path(path)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def show_json_download(label, data, filename):
    if data is None:
        st.info(f"{label} is not available.")
        return

    content = json.dumps(data, indent=2, ensure_ascii=False)

    st.download_button(
        label=f"Download {label}",
        data=content,
        file_name=filename,
        mime="application/json",
        key=f"download_{filename}",
    )


def validate_uploads(uploaded_files):
    if not uploaded_files:
        return "Upload at least one TXT file."

    if len(uploaded_files) > MAX_FILES:
        return f"You can upload at most {MAX_FILES} files."

    total_size = 0

    for uploaded_file in uploaded_files:
        file_size = len(uploaded_file.getvalue())
        total_size += file_size

        if file_size == 0:
            return f"{uploaded_file.name} is empty."

        if file_size > MAX_FILE_SIZE:
            return (
                f"{uploaded_file.name} exceeds "
                "the 5 MB per-file limit."
            )

    if total_size > MAX_TOTAL_SIZE:
        return "Combined upload size exceeds the 50 MB limit."

    return None


def parse_uploaded_files(uploaded_files, job_dir):
    raw_dir = job_dir / "raw"
    processed_dir = job_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    parse_summary = []

    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        original_name = safe_filename(uploaded_file.name)

        raw_path = raw_dir / f"{index:03d}_{original_name}"
        raw_path.write_bytes(uploaded_file.getvalue())

        parsed = parse_script(raw_path)

        if not isinstance(parsed, dict):
            raise ValueError(
                f"Parser returned an invalid result "
                f"for {uploaded_file.name}."
            )

        if (
            not parsed.get("episode")
            or not isinstance(parsed.get("segments"), list)
        ):
            raise ValueError(
                f"Parsing failed for {uploaded_file.name}: "
                "missing episode metadata or segments."
            )

        parsed_filename = (
            f"{index:03d}_{Path(original_name).stem}_parsed.json"
        )

        parsed_path = processed_dir / parsed_filename
        write_json(parsed_path, parsed)

        parse_errors = parsed.get("parse_errors") or []

        parse_summary.append({
            "file": uploaded_file.name,
            "episode": parsed.get("episode"),
            "scenes": parsed.get(
                "scene_count",
                len(parsed.get("scenes", [])),
            ),
            "segments": parsed.get(
                "segment_count",
                len(parsed["segments"]),
            ),
            "parse_errors": len(parse_errors),
        })

    return processed_dir, parse_summary




def display_result_tab(title, data, filename):
    st.subheader(title)

    if data is None:
        st.info("No output file was generated for this category.")
        return

    if isinstance(data, list):
        st.caption(f"{len(data)} records")

        if data:
            display_data = []

            for item in data:
                row = dict(item)

                # Put the final rank first in the ranked table
                if title == "Ranked promo recommendations":
                    rank_key = (
                        "final_rank"
                        if "final_rank" in row
                        else "rank"
                        if "rank" in row
                        else None
                    )

                    if rank_key:
                        row = {
                            rank_key: row.pop(rank_key),
                            **row,
                        }

                # Render nested values as readable JSON
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(
                            value,
                            ensure_ascii=False,
                            indent=2,
                        )

                display_data.append(row)

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No records in this category.")
    else:
        st.json(data)

    # Download original, unmodified JSON
    show_json_download(title, data, filename)

# --------------------------------------------------
# MAIN INTERFACE
# --------------------------------------------------


st.title("AI Drama Promo Generator")

st.markdown(
    """
    Discover high-potential promotional moments from screenplay
    episodes using a multi-stage AI pipeline.

    Upload episodes, configure your model, and generate
    ranked promo recommendations.
    """
)

st.divider()



# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header("Pipeline settings")
    st.subheader("Model connection")

    provider_options = {
        "Google Gemini": "google",
        "OpenAI": "openai",
        "OpenAI-compatible API": "openai_compatible",
    }

    selected_provider_label = st.selectbox(
        "API protocol",
        list(provider_options.keys()),
    )

    selected_provider = provider_options[
        selected_provider_label
    ]

    if selected_provider == "google":

        default_model = "gemini-3.5-flash-lite"

        model_options = [
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "Custom model",
        ]

    elif selected_provider == "openai":

        default_model = "gpt-4o-mini"

        model_options = [
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "Custom model",
        ]

    else:

        default_model = ""

        model_options = [
            "Custom model",
        ]

    selected_option = st.selectbox(
        "Model",
        model_options,
        index=0,
        key=f"model_options_{selected_provider}",
    )

    if selected_option == "Custom model":
        selected_model = st.text_input(
            "Model ID",
            placeholder="Enter the exact model ID",
            key=f"custom_model_{selected_provider}",
        )
    else:
        selected_model = selected_option

    base_url = None

    if selected_provider in (
        "openai",
        "openai_compatible",
    ):

        base_url_input = st.text_input(
            "Base URL (optional)",
            placeholder=(
                "https://api.example.com/v1"
                if selected_provider == "openai_compatible"
                else "Leave blank for the official OpenAI API"
            ),
            help=(
                "For an OpenAI-compatible provider, enter its "
                "API endpoint. Leave blank to use the default "
                "OpenAI endpoint."
            ),
        )

        base_url = base_url_input.strip() or None



    max_candidates = st.number_input(
            "Candidates per episode",
            min_value=1,
            max_value=10,
            value=5,
        )
    
    shortlist_size = st.number_input(
            "Final shortlist size",
            min_value=5,
            max_value=100,
            value=30,
        )
    
    st.caption(
            "The selected model and endpoint must support the "
            "configured JSON response workflow."
        )
    
    
    st.markdown("<div style='height: 4px'></div>", unsafe_allow_html=True)
    st.subheader("Bring Your Own Key")

    if selected_provider == "google":
        server_api_key = (
            get_secret("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        key_placeholder = "Paste your Google Gemini API key"

    elif selected_provider == "openai":
        server_api_key = (
            get_secret("OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        key_placeholder = "Paste your OpenAI API key"

    else:
        server_api_key = None
        key_placeholder = "Paste your provider API key"

    user_api_key = st.text_input(
        "Add Your Own API Key",
        type="password",
        placeholder=key_placeholder,
        help=(
            "Enter your own API key to authenticate with the "
            "selected provider."
        ),
        key="user_api_key_input",
    )

    st.caption(
        "API usage through your own key is billed directly "
        "to your account by the selected provider."
    )

    st.warning(
        "Your key is passed to the selected API client for the "
        "pipeline run. The app does not intentionally save it "
        "to project files or pipeline output files."
    )

    

    if user_api_key.strip():
        st.success("Your own API key is ready to use.")

    elif server_api_key:
        st.info(
            "No personal key entered. The application's "
            "configured server-side key will be used instead."
        )

    else:
        st.warning(
            "Add your own API key to enable pipeline execution."
        )


    
# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.subheader("1. Upload screenplay episodes")

st.caption(
    "Upload English screenplay TXT files. "
    "You can upload fewer than 50 episodes for testing."
)

uploaded_files = st.file_uploader(
    "Choose screenplay files",
    type=["txt"],
    accept_multiple_files=True,
    help="Upload up to 70 TXT files, with a maximum of 5 MB per file.",
)

if uploaded_files:
    total_size = sum(
        len(file.getvalue())
        for file in uploaded_files
    )

    st.markdown('<div class="upload-metrics">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Files uploaded", len(uploaded_files))
    col2.metric("Total size", f"{total_size / (1024 * 1024):.2f} MB")
    # col3.metric("Expected files", "50")

    st.markdown("</div>", unsafe_allow_html=True)

    if len(uploaded_files) < 50:
        st.caption(
            "Testing mode: fewer than 50 episodes uploaded."
        )

    with st.expander("View uploaded filenames"):
        for file in uploaded_files:
            st.write(f"📄 {file.name}")

#run button


# --------------------------------------------------
# RUN BUTTON
# --------------------------------------------------

st.divider()

button_label = (
    "Discover again"
    if st.session_state.get("last_job_dir")
    else "Parse and discover promos"
)

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:
    run_button = st.button(
        button_label,
        type="primary",
        use_container_width=True,
        disabled=(
            not uploaded_files
            or not selected_model.strip()
        ),
    )

    # st.caption(
    #     "Runs the complete pipeline on the currently uploaded episodes."
    # )

if uploaded_files:
    st.caption(
        "Processing may take several minutes depending on the "
        "number of episodes and model response times."
    )


# --------------------------------------------------
# PIPELINE EXECUTION
# --------------------------------------------------

if run_button:

    upload_error = validate_uploads(uploaded_files)

    effective_api_key = (
        user_api_key.strip()
        or server_api_key
        or None
    )

    if upload_error:

        st.error(upload_error)

    elif not effective_api_key:

        st.error(
            "No API key is available. "
            "Enter a key in the sidebar or configure "
            "a supported server-side key."
        )

    elif (
        selected_provider == "openai_compatible"
        and not base_url
    ):

        st.error(
            "Enter the Base URL for the OpenAI-compatible API."
        )

    elif not pipeline_lock.acquire(blocking=False):

        st.error(
            "Another pipeline run is currently in progress. "
            "Please try again after it finishes."
        )

    else:

        try:

            job_id = uuid.uuid4().hex
            job_dir = JOBS_ROOT / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            with st.status(
                "Running promo discovery pipeline...",
                expanded=True,
            ) as status:

                # ----------------------------------------
                # STEP 1: PARSING
                # ----------------------------------------

                st.write(
                    "Step 1: Parsing and validating uploaded episodes..."
                )

                processed_dir, parse_summary = parse_uploaded_files(
                    uploaded_files,
                    job_dir,
                )

                write_json(
                    job_dir / "parse_summary.json",
                    parse_summary,
                )

                total_parse_errors = sum(
                    item["parse_errors"]
                    for item in parse_summary
                )

                st.write(
                    f"Parsed {len(parse_summary)} episode files. "
                    f"Parser-reported errors: {total_parse_errors}."
                )

                # ----------------------------------------
                # STEP 2: MODEL CONFIGURATION
                # ----------------------------------------

                st.write(
                    "Step 2: Configuring the selected model..."
                )

                cfg = load_config()

                for stage in (
                    "episode_analysis",
                    "boundary_optimization",
                    "promo_evaluation",
                ):

                    llm_cfg = cfg.get_llm_config(stage)

                    llm_cfg.provider = selected_provider
                    llm_cfg.model = selected_model.strip()
                    llm_cfg.base_url = base_url

                # ----------------------------------------
                # STEP 3: PIPELINE EXECUTION
                # ----------------------------------------

                st.write(
                    "Step 3: Running the promo discovery pipeline..."
                )

                PIPELINE_STAGES = [
                    ("loading", "Episode ingestion", "Loading and validating screenplay files"),
                    ("episode_analysis", "Episode understanding & candidate discovery", "Analyzing story context and extracting promotional moments"),
                    ("preparation", "Candidate preparation", "Validating source segments and calculating initial signals"),
                    ("shortlisting", "Candidate shortlisting", "Selecting diverse candidates for further processing"),
                    ("boundary_optimization", "Boundary optimization", "Refining promotional clip start and end points"),
                    ("promo_evaluation", "Promo evaluation", "Evaluating hook, cliffhanger, clarity and emotional impact"),
                    ("ranking", "Ranking and decision classification", "Classifying candidates into accepted, review and rejected"),
                    ("saving", "Structured output generation", "Writing JSON results and execution statistics"),
                ]

                stage_order = [item[0] for item in PIPELINE_STAGES]
                stage_progress = {
                    stage: {"current": 0, "total": 0, "message": "Waiting", "status": "pending"}
                    for stage in stage_order
                }

                progress_container = st.container()
                overall_progress = progress_container.progress(0)
                stage_placeholder = progress_container.empty()
                activity_placeholder = progress_container.empty()

                def render_pipeline_progress():
                    completed = sum(
                        1 for s in stage_progress.values()
                        if s["status"] == "complete"
                    )
                    fraction = min(1.0, max(0.0, completed / len(stage_order)))
                    overall_progress.progress(fraction)

                    stage_lines = []
                    for s_id, s_title, _ in PIPELINE_STAGES:
                        st_info = stage_progress[s_id]
                        status_sym = "⏳"
                        if st_info["status"] == "complete":
                            status_sym = "✅"
                        elif st_info["status"] == "running":
                            status_sym = "🔄"
                        elif st_info["status"] == "failed":
                            status_sym = "❌"

                        msg = st_info["message"]
                        if msg and msg != "Waiting":
                            stage_lines.append(f"{status_sym} **{s_title}**: {msg}")
                        else:
                            stage_lines.append(f"{status_sym} **{s_title}**")

                    stage_placeholder.markdown("\n\n".join(stage_lines))

                def on_progress(event):
                    stage = event.get("stage")
                    stage_aliases = {
                        "boundary_complete": "boundary_optimization",
                        "evaluation_complete": "promo_evaluation",
                    }
                    stage = stage_aliases.get(stage, stage)
                    if stage not in stage_progress:
                        return

                    event_status = event.get("status", "running")
                    stage_progress[stage].update({
                        "current": event.get("current", 0),
                        "total": event.get("total", 0),
                        "message": event.get("message", ""),
                        "status": event_status,
                    })
                    render_pipeline_progress()
                    activity_placeholder.caption(event.get("message", ""))

                render_pipeline_progress()

                run_kwargs = {
                    "input_dir": str(processed_dir),
                    "max_candidates_per_episode": int(max_candidates),
                    "shortlist_size": int(shortlist_size),
                    "output_root": job_dir / "outputs",
                    "cache_root": job_dir / "cache",
                    "api_key": effective_api_key,
                    "progress_callback": on_progress,
                }

                stats = run(
                    cfg,
                    **run_kwargs,
                )

                for stage_key in stage_order:
                    if stage_progress[stage_key]["status"] != "failed":
                        stage_progress[stage_key]["status"] = "complete"
                render_pipeline_progress()

                write_json(
                    job_dir / "execution_stats.json",
                    stats,
                )

                status.update(
                    label="Pipeline completed successfully.",
                    state="complete",
                    expanded=False,
                )

            st.session_state.last_job_dir = str(job_dir)
            st.session_state.last_job_id = job_id
            st.session_state.last_parse_summary = parse_summary
            st.session_state.last_stats = stats

            st.success(f"Completed job: {job_id}")

        except Exception as exc:

            for stage_key, s_data in stage_progress.items():
                if s_data["status"] == "running":
                    s_data["status"] = "failed"
            render_pipeline_progress()

            st.error(f"Pipeline failed: {exc}")
            st.exception(exc)

        finally:

            pipeline_lock.release()


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if st.session_state.get("last_job_dir"):

    job_dir = Path(st.session_state.last_job_dir)
    output_dir = job_dir / "outputs"

    st.divider()
    st.header("2. Results dashboard")
    st.caption(
        "Review ranked promotional candidates, evaluation decisions "
        "and pipeline execution statistics."
    )

    stats = st.session_state.get("last_stats", {})

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Episodes analysed",
        stats.get("episodes_analyzed", 0),
    )

    metric_columns[1].metric(
        "Candidates discovered",
        stats.get("total_discovered_candidates", 0),
    )

    metric_columns[2].metric(
        "Candidates shortlisted",
        stats.get("shortlisted_candidates", 0),
    )

    metric_columns[3].metric(
        "API calls",
        stats.get("api_calls", 0)
        + stats.get("boundary_api_calls", 0)
        + stats.get("evaluation_api_calls", 0),
    )

    st.caption(
        f"Job ID: {st.session_state.last_job_id}"
    )

    # ----------------------------------------
    # PARSING SUMMARY
    # ----------------------------------------

    with st.expander("Parsing summary"):

        st.dataframe(
            st.session_state.last_parse_summary,
            use_container_width=True,
            hide_index=True,
        )

        show_json_download(
            "Parsing summary",
            st.session_state.last_parse_summary,
            "parse_summary.json",
        )

    # ----------------------------------------
    # LOAD OUTPUTS
    # ----------------------------------------

    ranked = read_json(
        output_dir / "ranked_promos" / "ranked_promos.json"
    )

    accepted = read_json(
        output_dir / "ranked_promos" / "accepted_promos.json"
    )

    review = read_json(
        output_dir / "ranked_promos" / "review_promos.json"
    )

    rejected = read_json(
        output_dir / "rejected_candidates" / "rejected_promos.json"
    )

    # ----------------------------------------
    # RESULT TABS
    # ----------------------------------------

    tabs = st.tabs([
        "Ranked promos",
        "Accepted",
        "Human review",
        "Rejected",
        "Execution stats",
    ])

    with tabs[0]:
        display_result_tab(
            "Ranked promo recommendations",
            ranked,
            "ranked_promos.json",
        )

    with tabs[1]:
        display_result_tab(
            "Accepted promos",
            accepted,
            "accepted_promos.json",
        )

    with tabs[2]:
        display_result_tab(
            "Promos requiring human review",
            review,
            "review_promos.json",
        )

    with tabs[3]:
        display_result_tab(
            "Rejected candidates",
            rejected,
            "rejected_promos.json",
        )

    with tabs[4]:
        display_result_tab(
            "Execution statistics",
            stats,
            "execution_stats.json",
        )

    st.caption(
        "Results are stored in this app instance's job directory. "
        "Download any outputs you need to retain."
    )
