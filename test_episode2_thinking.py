
import logging

from dotenv import load_dotenv

from src.config import load_config
from src.ingestion.episode_loader import load_episodes
from src.llm.semantic_analyzer import SemanticAnalyzer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

load_dotenv(".env")

cfg = load_config("config/config.yaml")
episodes = load_episodes("data/processed")

target_episode_numbers = {
    f"Episode {n}" for n in range(11, 21)
}

target_episodes = [
    ep for ep in episodes
    if str(ep.get("episode", "")).strip() in target_episode_numbers
]

if len(target_episodes) != 10:
    raise RuntimeError(
        f"Expected 10 episodes, found {len(target_episodes)}."
    )


if len(target_episodes) != 10:
    raise RuntimeError(
        f"Expected 10 episodes, found {len(target_episodes)}."
    )

print("Model:", cfg.llm_stages.episode_analysis.model)
print("Thinking level:", cfg.llm_stages.episode_analysis.thinking_level)
print("Episodes:", [ep["episode"] for ep in target_episodes])

analyzer = SemanticAnalyzer(
    cfg,
cache_dir="outputs/test/thinking_low_11_20_cache",
)

if not analyzer.is_available:
    raise RuntimeError("Gemini client could not be initialized.")

results = analyzer.analyze_episodes(
    target_episodes,
    max_candidates_per_episode=5,
)

print("\n========== TEST RESULTS ==========")

successes = 0
failures = 0
total_candidates = 0

for result in results:
    episode = result.get("episode")
    status = result.get("analysis_status")
    candidates = len(result.get("candidate_moments", []))

    if status == "success":
        successes += 1
    else:
        failures += 1

    total_candidates += candidates

    print(
        f"{episode}: {status} | "
        f"Candidates: {candidates}"
    )

print("\nSuccessful episodes:", successes)
print("Failed episodes:", failures)
print("Total candidates:", total_candidates)
print("API calls:", analyzer.stats["api_calls"])
print("Cache hits:", analyzer.stats["episodes_cached"])

print("==================================")

if successes == 5 and failures == 0:
    print("EPISODES 6-10 TEST: PASS")
else:
    print("EPISODES 6-10 TEST: PARTIAL/FAIL")
