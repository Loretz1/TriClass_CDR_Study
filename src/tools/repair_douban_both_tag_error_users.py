"""Repair failed user tags in legacy Douban both-tag modality metadata.

This tool only retries entries in ``error_users``. Existing user tags, item
tags, embeddings, and dataset splits are left untouched.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml


PROJECT_SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_SRC.parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from models.llm_diff_interaction_as_condition_mask_both_tag_ema_douban import (  # noqa: E402
    CROSSDOMAIN_USER_SYSTEM_PROMPT,
    _build_all_user_prompt_strings,
    _collect_domain_user_items,
    _extract_string_list,
    _user_item_profile_to_string,
)
from utils.data_preprocessing.douban_modality_processor import (  # noqa: E402
    DoubanModalityProcessor,
)


def _load_yaml(path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader) or {}


def _load_api_config(args):
    dataset_config = _load_yaml(PROJECT_SRC / "configs" / "dataset" / "Douban.yaml")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or dataset_config.get("openai_api_key")
    base_url = (
        args.base_url
        or os.environ.get("OPENAI_BASE_URL")
        or dataset_config.get("openai_base_url")
    )
    if not api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    return api_key, base_url


def _load_joint_data(joint_path):
    with (joint_path / "id_mapping.json").open("r", encoding="utf-8") as file:
        id_mapping = json.load(file)
    interaction = {
        "src": pd.read_pickle(joint_path / "train_src.pkl"),
        "tgt": pd.read_pickle(joint_path / "train_tgt.pkl"),
    }
    return id_mapping, interaction


def _load_raw_data(data_root, domains, id_mapping, interaction, joint_path):
    config = {"model": "LLM_Diff_interaction_as_condition_mask_both_tag_EMA_douban"}
    processor = DoubanModalityProcessor(config, domains, id_mapping, interaction, str(joint_path))
    reviews = {}
    metadata = {}

    for role, domain in zip(("src", "tgt"), domains):
        raw_path = data_root / "Douban" / domain / "raw"
        reviews[role], _, metadata[role] = processor._load_raw(
            str(raw_path / "reviews.json.gz"),
            str(raw_path / "user_meta.json.gz"),
            str(raw_path / "item_meta.json.gz"),
            id_mapping[role],
            interaction[role],
        )
    return reviews, metadata


def _build_failed_profiles(failed_users, interaction, id_mapping, reviews, metadata):
    profiles = {}
    for role in ("src", "tgt"):
        _collect_domain_user_items(
            interaction_df=interaction[role],
            domain=role,
            id_mapping=id_mapping,
            metadata=metadata,
            reviews=reviews,
            user_item_dict=profiles,
        )

    profiles_by_string_id = {str(user): profile for user, profile in profiles.items()}
    missing = [user for user in failed_users if user not in profiles_by_string_id]
    if missing:
        raise ValueError(
            f"Cannot reconstruct {len(missing)} failed users; examples={missing[:5]}"
        )
    return {user: profiles_by_string_id[user] for user in failed_users}


def _take_balanced(items, limit):
    if limit <= 0 or len(items) <= limit:
        return items
    left = limit // 2
    right = limit - left
    return items[:left] + items[-right:]


def _fallback_profile(profile, max_items_per_domain, max_review_chars, labels_only):
    result = {"src": [], "tgt": []}
    for role in ("src", "tgt"):
        items = _take_balanced(profile.get(role, []), max_items_per_domain)
        for item in items:
            cleaned = copy.deepcopy(item)
            review = cleaned.get("review")
            if labels_only:
                cleaned["review"] = "None"
            elif isinstance(review, str) and len(review) > max_review_chars:
                cleaned["review"] = review[:max_review_chars]
            result[role].append(cleaned)
    return result


def _profile_to_prompt(profile):
    profile_string = _user_item_profile_to_string(profile)
    return next(iter(_build_all_user_prompt_strings({"user": profile_string}).values()))


def _prompt_variants(profile, previous_error, args):
    previous_error = str(previous_error).lower()
    safe_profile = _fallback_profile(
        profile,
        max_items_per_domain=args.max_items_per_domain,
        max_review_chars=args.max_review_chars,
        labels_only=False,
    )
    labels_only_profile = _fallback_profile(
        profile,
        max_items_per_domain=args.max_items_per_domain,
        max_review_chars=0,
        labels_only=True,
    )

    original = _profile_to_prompt(profile)
    safe = _profile_to_prompt(safe_profile)
    labels_only = _profile_to_prompt(labels_only_profile)

    if "sensitive" in previous_error:
        variants = [("labels_only", labels_only)]
    elif "context_length_exceeded" in previous_error or "maximum context length" in previous_error:
        variants = [("truncated", safe), ("labels_only", labels_only)]
    else:
        variants = [("original", original), ("truncated", safe), ("labels_only", labels_only)]

    unique = []
    seen = set()
    for name, prompt in variants:
        if prompt not in seen:
            unique.append((name, prompt))
            seen.add(prompt)
    return unique


def _repair_one_user(user, profile, previous_error, api_key, base_url, args):
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.request_timeout,
        max_retries=0,
    )
    last_error = previous_error

    for variant_name, prompt in _prompt_variants(profile, previous_error, args):
        for attempt in range(args.max_retry):
            try:
                response = client.chat.completions.create(
                    model=args.chat_model,
                    messages=[
                        {"role": "system", "content": CROSSDOMAIN_USER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = response.choices[0].message.content.strip()
                tags = _extract_string_list(text)
                if tags is not None:
                    return user, tags, None, variant_name
                last_error = f"Unparseable response: {text[:500]}"
            except Exception as error:  # API failures must remain visible in metadata.
                last_error = f"{type(error).__name__}: {error}"

            if attempt + 1 < args.max_retry:
                time.sleep(min(args.retry_delay * (2**attempt), 30.0))

    return user, None, last_error, None


def _atomic_write_with_backup(path, data):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    temporary = path.with_name(f"{path.name}.tmp")
    shutil.copy2(path, backup)
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
    os.replace(temporary, path)
    return backup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retry only error_users in legacy Douban both-tag metadata."
    )
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--domains", nargs=2, default=["Book", "Music"])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--chat-model", default="gpt-4o-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--max-retry", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-items-per-domain", type=int, default=100)
    parser.add_argument("--max-review-chars", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = Path(args.metadata_json).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        modality_data = json.load(file)

    failed = dict(modality_data.get("error_users") or {})
    print(f"[path] metadata: {metadata_path}")
    print(f"[count] error_users={len(failed)}")
    print(
        "[count] error_items_src/error_items_tgt="
        f"{len(modality_data.get('error_items_src') or {})}/"
        f"{len(modality_data.get('error_items_tgt') or {})}"
    )
    if not failed:
        print("[ok] no failed users to repair")
        return

    joint_path = metadata_path.parent.parent
    id_mapping, interaction = _load_joint_data(joint_path)
    reviews, metadata = _load_raw_data(
        Path(args.data_root).expanduser().resolve(),
        args.domains,
        id_mapping,
        interaction,
        joint_path,
    )
    profiles = _build_failed_profiles(
        list(failed), interaction, id_mapping, reviews, metadata
    )

    for user, previous_error in failed.items():
        variants = _prompt_variants(profiles[user], previous_error, args)
        lengths = ", ".join(f"{name}:{len(prompt)}" for name, prompt in variants)
        print(f"[plan] user={user} variants={lengths}")

    if args.dry_run:
        print("[dry-run] prompts reconstructed; no API requests or writes performed")
        return

    api_key, base_url = _load_api_config(args)
    fixed = {}
    remaining = dict(failed)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                _repair_one_user,
                user,
                profiles[user],
                previous_error,
                api_key,
                base_url,
                args,
            ): user
            for user, previous_error in failed.items()
        }
        for future in as_completed(futures):
            user, tags, error, variant = future.result()
            if tags is not None:
                fixed[user] = tags
                remaining.pop(user, None)
                print(f"[fixed] user={user} variant={variant} tags={len(tags)}")
            else:
                remaining[user] = error
                print(f"[failed] user={user} error={error}")

    modality_data.setdefault("data_users", {}).update(fixed)
    modality_data["error_users"] = remaining
    backup = _atomic_write_with_backup(metadata_path, modality_data)
    print(f"[save] backup={backup}")
    print(f"[summary] fixed={len(fixed)} still_failed={len(remaining)}")
    if remaining:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
