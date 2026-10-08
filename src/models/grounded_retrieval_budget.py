"""Retrieval comparisons matched to each user's original union size."""

import fcntl
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.grounded_static_confidence import load_config, configure_logging
from models.grounded_triclass_pseudo import (TriClassPseudoSupervisor, histories,
    isolate_teacher_edges, load_semantics, rank_top, read_json, support_labels, digest_json)
from models.target_pattern_pseudo import (BRIDGE_TYPE, TargetPatternBridge, atomic_json,
    atomic_npz, dataset_split_directory, file_sha256, load_split, mapping_fingerprint, config_scalar)

SRC = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SRC / "configs/model/TriClassRetrievalBudget.yaml"
VARIANTS = ("semantic", "user_g", "popularity", "union_rrf", "trusted")
PROTOCOL = "match_original_union_per_user_fixed_final_k_uniform_loss"
LOGGER = logging.getLogger("retrieval-budget")


def load_experiment(path=DEFAULT_CONFIG):
    with Path(path).open() as handle:
        result = yaml.safe_load(handle)
    if result["variants"] != list(VARIANTS) or result["candidate_budget"] != "match_original_union_per_user":
        raise ValueError("Require five variants matched to the existing per-user union.")
    if result["final_top_k"] <= 0 or result["rrf_constant"] <= 0 or result["loss_weighting"] != "uniform":
        raise ValueError("Invalid fixed-slot, uniformly weighted comparison.")
    return result


def locations(experiment, task, run_dir, device="cpu"):
    config = load_config(str(SRC / experiment["reference_config"]), "Amazon2014",
                         experiment["tasks"][task]["domains"], device)
    root = Path(experiment["data_path"])
    config["data_path"] = str((root if root.is_absolute() else SRC / root).resolve())
    split = Path(dataset_split_directory(config))
    return config, split, split / "grounded_triclass", Path(run_dir) / "cache" / task


def check_candidates(items, counts, num_items):
    counts = np.broadcast_to(np.asarray(counts), (len(items),))
    if items.ndim != 2 or np.any(items < 0) or np.any(items > num_items):
        raise ValueError("Invalid candidate shape/range.")
    if not np.array_equal((items > 0).sum(1), counts):
        raise ValueError("Per-user candidate counts differ across variants.")
    if any(len(np.unique(row[row > 0])) != count for row, count in zip(items, counts)):
        raise ValueError("Duplicates do not count toward the retrieval budget.")


def truncate_per_user(rankings, counts):
    if len(rankings) != len(counts) or np.any(np.asarray(counts) > rankings.shape[1]):
        raise ValueError("Rankings cannot fill the matched budgets.")
    return np.where(np.arange(rankings.shape[1])[None] < np.asarray(counts)[:, None], rankings, 0)


def rrf_scores(items, features, sources, constant):
    """Unlearned rank fusion of the original three routes."""
    result = np.zeros(items.shape, dtype=np.float64)
    for row in range(len(items)):
        for bit, column in ((1, 0), (2, 1), (4, 3)):
            pos = np.flatnonzero((items[row] > 0) & ((sources[row] & bit) != 0))
            pos = pos[np.lexsort((items[row, pos], -features[row, pos, column]))]
            result[row, pos] += 1.0 / (constant + np.arange(1, len(pos) + 1))
    return result


def fixed_top(items, scores, k):
    if items.shape != scores.shape or not np.isfinite(scores).all() or np.any((items > 0).sum(1) < k):
        raise ValueError("Invalid fixed-size selection inputs.")
    result = np.empty((len(items), k), dtype=np.int64)
    for row in range(len(items)):
        pos = np.flatnonzero(items[row] > 0)
        result[row] = items[row, pos[np.lexsort((items[row, pos], -scores[row, pos]))[:k]]]
    return result


def quality(users, items, target_histories, neighbors):
    labels = support_labels(users, items, target_histories, neighbors)
    valid = items > 0
    denominators = np.asarray([max(1, len(target_histories.get(int(u), set()))) for u in users])
    return dict(users=len(users), pairs=int(valid.sum()), mean_items_per_user=float(valid.sum(1).mean()),
        user_coverage=float(valid.any(1).mean()), hit_precision=float((labels[valid] == 0).mean()),
        recall=float(((labels == 0).sum(1) / denominators).mean()), near_fraction=float((labels[valid] == 1).mean()))


def retrieve_singles(users, budgets, queries, sem_items, bridge, collab_items, counts, config):
    width = int(max(budgets))
    if np.count_nonzero(counts[1:]) < width:
        raise ValueError("G/Popularity cannot fill the matched budgets.")
    device, batch = config["device"], config["candidates"]["batch_size"]
    semantic = F.normalize(torch.as_tensor(np.array(sem_items), dtype=torch.float32, device=device), dim=-1)
    collaborative = F.normalize(torch.as_tensor(np.array(collab_items[1:]), dtype=torch.float32, device=device), dim=-1)
    trained = torch.as_tensor(counts[1:] > 0, device=device)
    available = np.flatnonzero(counts[1:] > 0) + 1
    popularity = available[np.lexsort((available, -counts[available]))[:width]]
    result = {name: np.empty((len(users), width), dtype=np.int64) for name in VARIANTS[:3]}
    bridge.eval()
    with torch.no_grad():
        for start in range(0, len(users), batch):
            query = torch.as_tensor(queries[users[start:start + batch]], dtype=torch.float32, device=device)
            sem = query @ semantic.T
            mapped = (bridge(query) @ collaborative.T).masked_fill(~trained[None], -torch.inf)
            result["semantic"][start:start + len(query)] = rank_top(sem, width).cpu().numpy() + 1
            result["user_g"][start:start + len(query)] = rank_top(mapped, width).cpu().numpy() + 1
            result["popularity"][start:start + len(query)] = popularity
            if start == 0 or start // batch % 100 == 0:
                LOGGER.info("Matched single-route retrieval %d/%d", start + len(query), len(users))
    return {name: truncate_per_user(items, budgets) for name, items in result.items()}


def common_negative_exclusions(candidates):
    result = np.zeros((len(next(iter(candidates.values()))), sum(x.shape[1] for x in candidates.values())), dtype=np.int64)
    for row in range(len(result)):
        items = np.unique(np.concatenate([x[row] for x in candidates.values()]))
        items = items[items > 0]
        result[row, :len(items)] = items
    return result


def build(experiment, task, run_dir, device="auto"):
    config, split, base, output = locations(experiment, task, run_dir, device)
    output.mkdir(parents=True, exist_ok=True)
    configure_logging(str(output))
    torch.set_num_threads(experiment["num_threads"])
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, spec = read_json(base / "manifest.json"), read_json(base / "build_spec.json")
        if (manifest["teacher_target_users"] != "target_only" or manifest["cold_evaluation_labels_used"]
                or manifest["domains"] != config["domains"]):
            raise ValueError("Frozen teacher identity/isolation mismatch.")
        _, all_users, mapping, src, tgt = load_split(config)
        no, nu, ni = len(all_users["overlap_users"]), len(mapping["src"]["id2user"]) - 1, len(mapping["tgt"]["id2item"]) - 1
        if mapping_fingerprint(mapping) != manifest["mapping_sha256"]:
            raise ValueError("Item/user mapping changed.")
        files = [split / "train_src.pkl", split / "train_tgt.pkl", base / "manifest.json", base / "build_spec.json"]
        for name in ("train_src", "train_tgt"):
            if file_sha256(str(split / (name + ".pkl"))) != manifest[name + "_sha256"]:
                raise ValueError("Original training data changed.")
        for name in ("target_item_neighbors.npy", "target_only_teachers/target_lightgcn_item_embeddings.npy",
                     "target_only_teachers/user_semantic_to_collaborative_bridge.pt", "overlap_candidates.npz",
                     "cold_candidates.npz", "overlap_support.npz", "cold_support.npz"):
            path = base / name
            if file_sha256(str(path)) != manifest["artifacts"][name]:
                raise ValueError("Frozen artifact changed: " + name)
            files.append(path)
        semantic_dir = split / config["source_subdir"]
        for name, expected in spec["semantic_files"].items():
            path = semantic_dir / name
            if file_sha256(str(path)) != expected:
                raise ValueError("Semantic artifact changed: " + name)
            files.append(path)
        identity = dict(experiment=experiment, task=task, protocol=PROTOCOL,
            source_sha256={str(p): file_sha256(str(p)) for p in files}, builder_sha256=file_sha256(__file__),
            common_sha256=file_sha256(str(SRC / "models/grounded_triclass_pseudo.py")))
        identity["build_sha256"] = digest_json(identity)
        if (output / "build_spec.json").exists() and read_json(output / "build_spec.json") != identity:
            raise ValueError("Immutable inputs changed; use a fresh directory.")
        atomic_json(str(output / "build_spec.json"), identity)
        if (output / "manifest.json").exists():
            for name, digest in read_json(output / "manifest.json")["artifacts"].items():
                if file_sha256(str(output / name)) != digest:
                    raise ValueError("Finished cache modified.")
            return output
        queries, sem_items = load_semantics(str(semantic_dir), nu, no, ni)
        collab = np.load(base / "target_only_teachers/target_lightgcn_item_embeddings.npy")
        checkpoint = torch.load(base / "target_only_teachers/user_semantic_to_collaborative_bridge.pt", map_location="cpu")
        if checkpoint["bridge_type"] != BRIDGE_TYPE:
            raise ValueError("Expected the frozen user-level G.")
        bridge = TargetPatternBridge(*(checkpoint[k] for k in ("input_dim", "hidden_dim", "output_dim", "dropout")))
        bridge.load_state_dict(checkpoint["state_dict"])
        bridge.to(config["device"])
        counts = np.bincount(isolate_teacher_edges(tgt, no)["item"].to_numpy(), minlength=ni + 1)
        target_histories = histories(tgt.loc[tgt.user <= no])
        neighbors = np.load(base / "target_item_neighbors.npy")
        artifacts = []
        diagnostics = dict(protocol=PROTOCOL, variants={}, final_top_k=experiment["final_top_k"],
                           original_three_route_candidates_unchanged=True, classifier_retrained=False)
        etas = sorted({float(v[1]) for v in experiment["tasks"][task]["backbones"].values()})
        for scope in ("overlap", "cold"):
            with np.load(base / (scope + "_candidates.npz")) as cached:
                users, union, features, sources = (cached[k] for k in ("user_ids", "item_ids", "features", "sources"))
            with np.load(base / (scope + "_support.npz")) as support:
                if not np.array_equal(users, support["user_ids"]) or not np.array_equal(union, support["item_ids"]):
                    raise ValueError("Original union and confidence are not aligned.")
                probabilities = support["probabilities"]
                codes = support["split_codes"] if scope == "overlap" else None
            budgets = (union > 0).sum(1)
            if budgets.min() < experiment["final_top_k"]:
                raise ValueError("Original candidates cannot fill the final supervision slots.")
            ranking_file = output / (scope + "_singles.npz")
            if ranking_file.exists():
                with np.load(ranking_file) as saved:
                    singles = {name: saved[name] for name in VARIANTS[:3]}
                    if not np.array_equal(users, saved["user_ids"]):
                        raise ValueError("Partial cache has different users.")
            else:
                singles = retrieve_singles(users, budgets, queries, sem_items, bridge, collab, counts, config)
                atomic_npz(str(ranking_file), user_ids=users, **singles)
            artifacts.append(ranking_file.name)
            candidates = dict(singles, union_rrf=union, trusted=union)
            fused_scores = rrf_scores(union, features, sources, experiment["rrf_constant"])
            if scope == "cold":
                exclusions = common_negative_exclusions(dict(singles, union=union))
                atomic_npz(str(output / "cold_negative_exclusions.npz"), user_ids=users, item_ids=exclusions)
                artifacts.append("cold_negative_exclusions.npz")
            for variant in VARIANTS:
                items = candidates[variant]
                check_candidates(items, budgets, ni)
                for eta in etas:
                    k = experiment["final_top_k"]
                    if variant == "trusted":
                        selected = fixed_top(items, probabilities[..., 0] + eta * probabilities[..., 1], k)
                    elif variant == "union_rrf":
                        selected = fixed_top(items, fused_scores, k)
                    else:
                        selected = items[:, :k].copy()
                    check_candidates(selected, k, ni)
                    name = f"{scope}_{variant}_eta{eta}.npz"
                    atomic_npz(str(output / name), user_ids=users, candidate_ids=items, candidate_counts=budgets,
                        item_ids=selected, loss_weights=np.ones(selected.shape, dtype=np.float32))
                    artifacts.append(name)
                    if scope == "overlap":
                        test = codes == 2
                        diagnostics["variants"][variant + "_eta" + str(eta)] = dict(
                            candidate_test=quality(users[test], items[test], target_histories, neighbors),
                            selected_test=quality(users[test], selected[test], target_histories, neighbors))
            diagnostics[scope] = dict(users=len(users), candidate_min=int(budgets.min()), candidate_max=int(budgets.max()),
                candidate_mean=float(budgets.mean()), candidate_pairs=int(budgets.sum()), final_pairs=len(users) * experiment["final_top_k"])
        atomic_json(str(output / "diagnostics.json"), diagnostics)
        artifacts.append("diagnostics.json")
        out = {k: manifest[k] for k in ("dataset", "domains", "mapping_sha256", "num_users_src", "num_users_overlap",
                                      "num_items_tgt", "train_src_sha256", "train_tgt_sha256")}
        out.update(protocol=PROTOCOL, cold_evaluation_labels_used=False, candidate_budget="match_original_union_per_user",
            final_top_k=experiment["final_top_k"], loss_weighting="uniform", original_union_unchanged=True,
            original_confidence_unchanged=True, negative_exclusions="common_union_of_all_methods",
            etas=etas, variants=list(VARIANTS), build_sha256=identity["build_sha256"],
            artifacts={name: file_sha256(str(output / name)) for name in artifacts})
        atomic_json(str(output / "manifest.json"), out)
        LOGGER.info("Completed per-user matched cache %s", output)
        return output


class FixedBudgetSupervisor(TriClassPseudoSupervisor):
    """Existing sampler/loss with matched candidates, slots and negative pools."""

    def __init__(self, config, dataloader):
        self.device = config["device"]
        self.rng = np.random.default_rng(int(config["pseudo_sampler_seed"]))
        dataset = dataloader.dataset
        self.num_items_tgt = dataset.num_items_tgt
        root = Path(config["retrieval_cache_path"])
        manifest = read_json(root / "manifest.json")
        expected = dict(dataset=config["dataset"], domains=list(config["domains"]), mapping_sha256=mapping_fingerprint(dataset.id_mapping),
            num_users_src=dataset.num_users_src, num_users_overlap=dataset.num_users_overlap,
            num_items_tgt=dataset.num_items_tgt, protocol=PROTOCOL, loss_weighting="uniform", cold_evaluation_labels_used=False)
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("Retrieval cache protocol/split mismatch.")
        split = Path(dataset_split_directory(config))
        for field in ("train_src", "train_tgt"):
            if file_sha256(str(split / (field + ".pkl"))) != manifest[field + "_sha256"]:
                raise ValueError("Training split changed.")
        variant = config["retrieval_variant"]
        eta = float(config_scalar(config["near_support_weight"], 0))
        name = f"cold_{variant}_eta{eta}.npz"
        for filename in (name, "cold_negative_exclusions.npz"):
            if file_sha256(str(root / filename)) != manifest["artifacts"][filename]:
                raise ValueError("Supervision checksum mismatch.")
        with np.load(root / name) as a:
            self.user_ids, self.candidates, self.items, self.weights, budgets = (
                a[k] for k in ("user_ids", "candidate_ids", "item_ids", "loss_weights", "candidate_counts"))
        if not np.array_equal(self.user_ids, np.arange(dataset.num_users_overlap + 1, dataset.num_users_src + 1)):
            raise ValueError("Cold user coverage changed.")
        self.top_k = manifest["final_top_k"]
        check_candidates(self.candidates, budgets, self.num_items_tgt)
        check_candidates(self.items, self.top_k, self.num_items_tgt)
        if not np.all(self.weights == 1) or self.weights.shape != self.items.shape:
            raise ValueError("Require equal per-pair loss weights.")
        if any(not set(s).issubset(set(c)) for s, c in zip(self.items, self.candidates)):
            raise ValueError("Selected positives must be retrieved candidates.")
        history = histories(dataset.df["train_tgt"])
        mapping = dataset.id_mapping
        self.target_positive = {int(u): history.get(mapping["tgt"]["user2id"].get(mapping["src"]["id2user"][int(u)]), set())
                                for u in self.user_ids}
        with np.load(root / "cold_negative_exclusions.npz") as a:
            if not np.array_equal(a["user_ids"], self.user_ids):
                raise ValueError("Shared negative exclusions have different users.")
            exclusions = a["item_ids"]
        self.forbidden = [set(map(int, row[row > 0])) | self.target_positive[int(u)] for u, row in zip(self.user_ids, exclusions)]
        if any(len(blocked) >= self.num_items_tgt or not set(selected).issubset(blocked)
               for blocked, selected in zip(self.forbidden, self.items)):
            raise ValueError("Invalid common negative pool.")
        LOGGER.info("Matched budget variant=%s users=%d candidates_mean=%.2f final=%d weights=1", variant,
                    len(self.user_ids), float(budgets.mean()), self.top_k)
