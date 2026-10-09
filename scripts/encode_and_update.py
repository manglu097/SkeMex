import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json

from medmem_agent.evolution.config import load_evo_config
from medmem_agent.evolution.embeddings import EmbeddingManager
from medmem_agent.evolution.encode import TrajectoryEncoder
from medmem_agent.evolution.governance import SkillGovernance
from medmem_agent.evolution.storage import SkillStore
from medmem_agent.evolution.trajectory import TrajectoryBuilder
from medmem_agent.evolution.utility import UtilityUpdater


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a trajectory file and update the skill library")
    parser.add_argument("--trajectory_json", required=True)
    parser.add_argument("--evo_default_config", default="configs/evolution/default.json")
    parser.add_argument("--evo_override_config", default="configs/evolution/online.json")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    evo_config = load_evo_config(args.evo_default_config, args.evo_override_config)
    store = SkillStore(repo_root=".", evo_config=evo_config)
    store.ensure_initialized()
    evo_provider = evo_config.api.provider
    evo_api_base = evo_config.api.api_base
    evo_api_key = args.api_key or evo_config.api.api_key
    embeddings = EmbeddingManager(
        cache_path=str(store.embeddings_path),
        model=evo_config.retrieval.embedding_model,
        api_base=evo_api_base,
        api_key=evo_api_key,
        global_query_cache_path=str(store.global_query_cache_path),
    )
    encoder = TrajectoryEncoder(store=store, evo_config=evo_config, provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key)
    governance = SkillGovernance(store=store, embeddings=embeddings, evo_config=evo_config, provider=evo_provider, api_base=evo_api_base, api_key=evo_api_key)
    updater = UtilityUpdater(store=store, evo_config=evo_config)

    payload = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))
    trajectory = TrajectoryBuilder.build(
        sample=payload.get("sample", {}),
        result=payload.get("result", {}),
        steps_history=payload.get("steps_history", []),
        task_category=payload.get("task_category", []),
        injected_skill_ids=payload.get("injected_skill_ids", []),
        eval_result=payload.get("eval_result"),
        history_file=payload.get("history_file", ""),
    )
    encode_result = encoder.encode(trajectory)
    governance.apply_encode_result(encode_result)
    updater.update(trajectory, encode_result)


if __name__ == "__main__":
    main()
