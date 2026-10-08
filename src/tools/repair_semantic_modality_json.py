# 修复sem emb中的Error，重新发送http请求调用api
import argparse
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml


PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from models.lsdcdr_gpt52_te3large import (  # noqa: E402
    BASE_TGT_ITEM_SYSTEM_PROMPT,
    BASE_USER_SYSTEM_PROMPT,
    CONCISE_ITEM_SYSTEM_PROMPT,
    CONCISE_USER_SYSTEM_PROMPT,
    FINEGRAINED_ITEM_SYSTEM_PROMPT,
    FINEGRAINED_USER_SYSTEM_PROMPT,
    _build_all_user_prompt_strings,
    _build_item_prompt_string,
    _collect_domain_items,
    _collect_domain_user_items,
    _run_prompt_summarization_multiprocess,
    _user_item_profile_to_string,
)
from models.lsdcdr_prompt_fewertags import (  # noqa: E402
    FEWERTAGS_ITEM_SYSTEM_PROMPT,
    FEWERTAGS_USER_SYSTEM_PROMPT,
)
from models.lsdcdr_prompt_paragraph import (  # noqa: E402
    PARAGRAPH_ITEM_SYSTEM_PROMPT,
    PARAGRAPH_USER_SYSTEM_PROMPT,
)


MODEL_DEFAULTS = {
    "LSDCDR_GPT52_TE3Large": {
        "modality": "CrossDomain_semantics_both_tag_gpt52_te3large_src_only_tgt_only",
        "user_prompt": BASE_USER_SYSTEM_PROMPT,
        "item_prompt": BASE_TGT_ITEM_SYSTEM_PROMPT,
    },
    "LSDCDR_Qwen3Max_Qwen3Emb8B": {
        "modality": "CrossDomain_semantics_both_tag_qwen3max_qwen3emb8b_src_only_tgt_only",
        "user_prompt": BASE_USER_SYSTEM_PROMPT,
        "item_prompt": BASE_TGT_ITEM_SYSTEM_PROMPT,
    },
    "LSDCDR_Prompt_Concise": {
        "modality": "CrossDomain_semantics_both_tag_prompt_concise_src_only_tgt_only",
        "user_prompt": CONCISE_USER_SYSTEM_PROMPT,
        "item_prompt": CONCISE_ITEM_SYSTEM_PROMPT,
    },
    "LSDCDR_Prompt_FineGrained": {
        "modality": "CrossDomain_semantics_both_tag_prompt_finegrained_src_only_tgt_only",
        "user_prompt": FINEGRAINED_USER_SYSTEM_PROMPT,
        "item_prompt": FINEGRAINED_ITEM_SYSTEM_PROMPT,
    },
    "LSDCDR_Prompt_FewerTags": {
        "modality": "CrossDomain_semantics_both_tag_prompt_fewertags_src_only_tgt_only",
        "user_prompt": FEWERTAGS_USER_SYSTEM_PROMPT,
        "item_prompt": FEWERTAGS_ITEM_SYSTEM_PROMPT,
    },
    "LSDCDR_Prompt_Paragraph": {
        "modality": "CrossDomain_semantics_both_tag_prompt_paragraph_src_only_tgt_only",
        "user_prompt": PARAGRAPH_USER_SYSTEM_PROMPT,
        "item_prompt": PARAGRAPH_ITEM_SYSTEM_PROMPT,
    },
}


def _load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def _load_config(model, dataset, domains):
    configs_dir = PROJECT_SRC / "configs"
    files = [
        configs_dir / "overall.yaml",
        configs_dir / "dataset" / f"{dataset}.yaml",
        configs_dir / "model" / f"{model}.yaml",
    ]

    merged = {}
    modalities = []
    hyper_parameters = []
    for path in files:
        data = _load_yaml(path)
        if data.get("modalities"):
            modalities.extend(data["modalities"])
        if data.get("hyper_parameters"):
            hyper_parameters.extend(data["hyper_parameters"])
        merged.update(data)

    merged["model"] = model
    merged["dataset"] = dataset
    merged["domains"] = domains
    merged["modalities"] = modalities
    merged["hyper_parameters"] = hyper_parameters
    return merged


def _format_split_value(value):
    return str(value)


def _resolve_data_path(config):
    data_path = Path(config.get("data_path", "../data/"))
    if data_path.is_absolute():
        return data_path
    return (PROJECT_SRC / data_path).resolve()


def _joint_path(config):
    data_path = _resolve_data_path(config)
    joint_dataset_name = "+".join(config["domains"])
    dataset_type = "only_overlap_users" if config["only_overlap_users"] else "all_users"
    split_dir = (
        f"WarmValid{_format_split_value(config['warm_valid_ratio'])}_"
        f"WarmTest{_format_split_value(config['warm_test_ratio'])}_"
        f"ColdValid{_format_split_value(config['t_cold_valid'])}_"
        f"ColdTest{_format_split_value(config['t_cold_test'])}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
    )
    if config["only_overlap_users"]:
        split_dir += f"_{config['k_cores']}cores"
    return data_path / config["dataset"] / joint_dataset_name / dataset_type / split_dir


def _get_modality(config, modality_name):
    for modality in config.get("modalities", []):
        if modality.get("name") == modality_name:
            return modality
    raise ValueError(f"Cannot find modality '{modality_name}' in merged config.")


def _parse_gz(path):
    with gzip.open(path, "r") as g:
        for line in g:
            line = line.replace(b"true", b"True").replace(b"false", b"False")
            yield eval(line)


def _load_metadata(path, valid_items):
    metadata = {}
    for info in _parse_gz(path):
        asin = info.get("asin")
        if asin in valid_items:
            metadata[asin] = info
    return metadata


def _load_reviews(path, id_mapping, interaction_df):
    valid_users = set(id_mapping["user2id"].keys())
    valid_items = set(id_mapping["item2id"].keys())
    interaction_pairs = set(
        (id_mapping["id2user"][u], id_mapping["id2item"][i])
        for u, i in interaction_df[["user", "item"]].to_numpy()
    )

    reviews = {}
    for inter in _parse_gz(path):
        user = inter["reviewerID"]
        item = inter["asin"]
        if user not in valid_users or item not in valid_items:
            continue
        if (user, item) not in interaction_pairs:
            continue
        reviews[(user, item)] = inter.get("reviewText", "")
    return reviews


def _load_id_mapping(joint_path):
    with (joint_path / "id_mapping.json").open("r", encoding="utf-8") as f:
        id_mapping = json.load(f)
    return id_mapping


def _load_interaction(joint_path):
    import pandas as pd

    train_src = pd.read_pickle(joint_path / "train_src.pkl")
    train_tgt = pd.read_pickle(joint_path / "train_tgt.pkl")
    return {"src": train_src, "tgt": train_tgt}


def _load_raw_data(config, id_mapping, interaction=None, need_reviews=False):
    data_path = _resolve_data_path(config)
    reviews = {"src": {}, "tgt": {}}
    metadata = {}

    for role, domain in zip(["src", "tgt"], config["domains"]):
        raw_dir = data_path / config["dataset"] / domain / "raw"
        metadata_path = raw_dir / f"meta_{domain}.json.gz"
        reviews_path = raw_dir / f"reviews_{domain}_5.json.gz"
        valid_items = set(id_mapping[role]["item2id"].keys())

        print(f"[load] metadata {role}: {metadata_path}")
        metadata[role] = _load_metadata(metadata_path, valid_items)

        if need_reviews:
            if interaction is None:
                raise ValueError("interaction data is required when repairing failed users.")
            print(f"[load] reviews {role}: {reviews_path}")
            reviews[role] = _load_reviews(reviews_path, id_mapping[role], interaction[role])

    return reviews, metadata


def _build_failed_user_prompts(failed_users, interaction, id_mapping, reviews, metadata):
    user_item_profiles = {}
    for role in ["src", "tgt"]:
        _collect_domain_user_items(
            interaction_df=interaction[role],
            domain=role,
            id_mapping=id_mapping,
            metadata=metadata,
            reviews=reviews,
            user_item_dict=user_item_profiles,
        )

    missing = [user for user in failed_users if user not in user_item_profiles]
    if missing:
        raise ValueError(f"Cannot rebuild prompts for {len(missing)} failed users, examples={missing[:5]}")

    profile_strings = {
        user: _user_item_profile_to_string(user_item_profiles[user])
        for user in failed_users
    }
    return _build_all_user_prompt_strings(profile_strings)


def _build_failed_item_prompts(failed_items, domain, id_mapping, metadata):
    item_profiles = {}
    _collect_domain_items(
        domain=domain,
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles,
    )

    missing = [item for item in failed_items if item not in item_profiles]
    if missing:
        raise ValueError(f"Cannot rebuild prompts for {len(missing)} failed {domain} items, examples={missing[:5]}")

    return {
        item: _build_item_prompt_string(item_profiles[item])
        for item in failed_items
    }


def _repair_one_section(
    modality_data,
    data_key,
    error_key,
    prompts,
    system_prompt,
    config,
    modality,
    args,
):
    failed = list((modality_data.get(error_key) or {}).keys())
    if not failed:
        print(f"[skip] {error_key}: no failed entries")
        return 0, 0

    retry_prompts = {key: prompts[key] for key in failed if key in prompts}
    missing_prompt = sorted(set(failed) - set(retry_prompts))
    if missing_prompt:
        raise ValueError(f"Missing prompts for {len(missing_prompt)} entries in {error_key}: {missing_prompt[:5]}")

    print(f"[retry] {error_key}: {len(retry_prompts)} entries with model={modality.get('llm_chat_model')}")
    results, errors = _run_prompt_summarization_multiprocess(
        prompts=retry_prompts,
        system_prompt=system_prompt,
        model_name=modality.get("llm_chat_model", "gpt-4o"),
        api_key=config["openai_api_key"],
        base_url=modality.get("llm_base_url") or config.get("openai_base_url"),
        max_retry=args.max_retry,
        num_workers=args.num_workers,
        timeout=args.request_timeout,
        openai_max_retries=args.openai_max_retries,
    )

    modality_data.setdefault(data_key, {})
    current_errors = dict(modality_data.get(error_key) or {})
    for key, tags in results.items():
        modality_data[data_key][key] = tags
        current_errors.pop(key, None)
    for key, err in errors.items():
        current_errors[key] = err
    modality_data[error_key] = current_errors

    print(f"[done] {error_key}: fixed={len(results)}, still_failed={len(errors)}")
    return len(results), len(errors)


def _write_json_with_backup(path, data):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup_path)

    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)
    print(f"[save] updated: {path}")
    print(f"[save] backup : {backup_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retry failed user/item LLM tag summaries in a LSDCDR semantic modality metadata JSON."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS.keys()))
    parser.add_argument("--dataset", default="Amazon2014")
    parser.add_argument("--domains", nargs=2, default=["Clothing_Shoes_and_Jewelry", "Sports_and_Outdoors"])
    parser.add_argument("--metadata_json", default=None, help="Optional direct path to the modality metadata JSON.")
    parser.add_argument("--modality_name", default=None, help="Optional override for the modality name.")
    parser.add_argument(
        "--sections",
        default="users,tgt_items,src_items",
        help="Comma-separated subset of users,tgt_items,src_items.",
    )
    parser.add_argument("--max_retry", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--request_timeout", type=float, default=180)
    parser.add_argument("--openai_max_retries", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = _load_config(args.model, args.dataset, args.domains)
    defaults = MODEL_DEFAULTS[args.model]
    modality_name = args.modality_name or defaults["modality"]
    modality = _get_modality(config, modality_name)

    joint_path = _joint_path(config)
    metadata_path = Path(args.metadata_json) if args.metadata_json else (
        joint_path / "modality_emb" / f"{modality_name}_metadata.json"
    )
    metadata_path = metadata_path.resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot find metadata JSON: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        modality_data = json.load(f)

    sections = {item.strip() for item in args.sections.split(",") if item.strip()}
    failed_users = list((modality_data.get("error_users") or {}).keys()) if "users" in sections else []
    failed_src_items = list((modality_data.get("error_items_src") or {}).keys()) if "src_items" in sections else []
    failed_tgt_items = list((modality_data.get("error_items_tgt") or {}).keys()) if "tgt_items" in sections else []

    print(f"[path] joint    : {joint_path}")
    print(f"[path] metadata : {metadata_path}")
    print(
        "[count] failed users/src_items/tgt_items = "
        f"{len(failed_users)}/{len(failed_src_items)}/{len(failed_tgt_items)}"
    )

    if not (failed_users or failed_src_items or failed_tgt_items):
        print("[ok] no failed entries to repair")
        return

    id_mapping = _load_id_mapping(joint_path)
    need_reviews = bool(failed_users)
    interaction = _load_interaction(joint_path) if need_reviews else None
    reviews, metadata = _load_raw_data(config, id_mapping, interaction, need_reviews=need_reviews)

    fixed_total = 0
    still_failed_total = 0

    if failed_users:
        prompts = _build_failed_user_prompts(failed_users, interaction, id_mapping, reviews, metadata)
        fixed, still_failed = _repair_one_section(
            modality_data,
            data_key="data_users",
            error_key="error_users",
            prompts=prompts,
            system_prompt=defaults["user_prompt"],
            config=config,
            modality=modality,
            args=args,
        )
        fixed_total += fixed
        still_failed_total += still_failed

    if failed_src_items:
        prompts = _build_failed_item_prompts(failed_src_items, "src", id_mapping, metadata)
        fixed, still_failed = _repair_one_section(
            modality_data,
            data_key="data_items_src",
            error_key="error_items_src",
            prompts=prompts,
            system_prompt=defaults["item_prompt"],
            config=config,
            modality=modality,
            args=args,
        )
        fixed_total += fixed
        still_failed_total += still_failed

    if failed_tgt_items:
        prompts = _build_failed_item_prompts(failed_tgt_items, "tgt", id_mapping, metadata)
        fixed, still_failed = _repair_one_section(
            modality_data,
            data_key="data_items_tgt",
            error_key="error_items_tgt",
            prompts=prompts,
            system_prompt=defaults["item_prompt"],
            config=config,
            modality=modality,
            args=args,
        )
        fixed_total += fixed
        still_failed_total += still_failed

    print(f"[summary] fixed={fixed_total}, still_failed={still_failed_total}")
    if args.dry_run:
        print("[dry-run] not writing metadata JSON")
        return

    _write_json_with_backup(metadata_path, modality_data)


if __name__ == "__main__":
    main()
