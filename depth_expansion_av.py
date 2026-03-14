
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans
from tqdm import tqdm

from util import load_json, save_json, makedir, parse_args

from adaptive_breath_expansion_av import (
    load_clip_features,
    fuse_features,
    find_closest_points_per_cluster,
)


def build_relevance_scores_for_clusters(tree_node, cluster_ids_x, clip_relevance):
    """
    Map clip_relevance (ordered by tree_node) to relevance_scores[cluster_id].

    tree_node: temporally sorted centroid clip indices
    cluster_ids_x: list of length N, cluster_ids_x[i] = cluster of clip i
    clip_relevance: list of length len(tree_node), score per centroid
    """
    num_clusters = max(cluster_ids_x) + 1 if cluster_ids_x else 0
    relevance_scores = [1] * num_clusters
    for i in range(len(tree_node)):
        clip_idx = tree_node[i]
        if clip_idx < len(cluster_ids_x):
            cluster_id = cluster_ids_x[clip_idx]
            relevance_scores[cluster_id] = clip_relevance[i] if i < len(clip_relevance) else 1
    return relevance_scores


def hierarchical_clustering_av(
    fused_feats,
    cluster_ids_x,
    relevance_scores,
    num_subclusters=4,
    num_subsubclusters=4,
    device=None,
):
    """
    Build hierarchical cluster structure based on breadth clusters and relevance.

    - Score 1: flat list of clip indices (no subclustering)
    - Score 2: dict of subcluster_id -> [clip indices]
    - Score 3: dict of subcluster_id -> {subsubcluster_id -> [clip indices]}

    fused_feats: [N, D] tensor (fused visual+audio embeddings)
    cluster_ids_x: list of length N
    relevance_scores: list indexed by cluster_id
    """
    if device is None:
        device = fused_feats.device
    fused_feats = fused_feats.to(device)

    clusters = {i: {} for i in range(max(cluster_ids_x) + 1)}

    for cluster_id in set(cluster_ids_x):
        primary_indices = [i for i, c in enumerate(cluster_ids_x) if c == cluster_id]

        score = relevance_scores[cluster_id] if cluster_id < len(relevance_scores) else 3

        if len(primary_indices) < 2:
            clusters[cluster_id] = primary_indices
            continue

        subset = fused_feats[primary_indices]

        if score == 1:
            clusters[cluster_id] = primary_indices
            continue

        k_sub = min(num_subclusters, len(primary_indices))
        cluster_ids, _ = kmeans(
            X=subset,
            num_clusters=k_sub,
            distance="cosine",
            device=device,
        )
        cluster_ids = cluster_ids.cpu().tolist()

        if score == 2:
            clusters[cluster_id] = {
                i: [primary_indices[j] for j in range(len(primary_indices)) if cluster_ids[j] == i]
                for i in range(k_sub)
            }
            continue

        clusters[cluster_id] = {}
        for sub_id in range(k_sub):
            sub_indices = [primary_indices[j] for j in range(len(primary_indices)) if cluster_ids[j] == sub_id]
            if len(sub_indices) < 2:
                if len(sub_indices) == 1:
                    clusters[cluster_id][sub_id] = {0: sub_indices}
                continue

            subsubset = fused_feats[sub_indices]
            k_subsub = min(num_subsubclusters, len(sub_indices))
            subsub_ids, _ = kmeans(
                X=subsubset,
                num_clusters=k_subsub,
                distance="cosine",
                device=device,
            )
            subsub_ids = subsub_ids.cpu().tolist()

            clusters[cluster_id][sub_id] = {}
            for subsub_id in range(k_subsub):
                final_idx = [sub_indices[j] for j in range(len(sub_indices)) if subsub_ids[j] == subsub_id]
                clusters[cluster_id][sub_id][subsub_id] = final_idx

    return clusters


def cosine_similarity(points, centroid):
    """Cosine distance (1 - similarity) for representative selection."""
    points_norm = F.normalize(points.float(), dim=1)
    centroid_norm = F.normalize(centroid.unsqueeze(0).float(), dim=1)
    return 1 - torch.mm(points_norm, centroid_norm.T).squeeze()


def find_closest_points_in_temporal_order_subsub(x, clusters, relevance_scores):
    """
    Select one representative clip per leaf. Returns temporally sorted list.
    """
    closest_indices = []

    for cluster_id, cluster_data in clusters.items():
        relevance = relevance_scores[cluster_id] if cluster_id < len(relevance_scores) else 3

        if isinstance(cluster_data, list):
            if len(cluster_data) == 0:
                continue
            points = x[torch.tensor(cluster_data, dtype=torch.long, device=x.device)]
            centroid = points.mean(dim=0)
            distances = cosine_similarity(points, centroid)
            if distances.numel() > 0:
                idx = torch.argmin(distances).item()
                closest_indices.append(int(cluster_data[idx]))

        elif isinstance(cluster_data, dict):
            if relevance == 1:
                primary_indices = []
                for sub_data in cluster_data.values():
                    if isinstance(sub_data, dict):
                        for lst in sub_data.values():
                            if lst:
                                primary_indices.extend(lst)
                    elif isinstance(sub_data, list) and sub_data:
                        primary_indices.extend(sub_data)

                if primary_indices:
                    points = x[torch.tensor(primary_indices, dtype=torch.long, device=x.device)]
                    centroid = points.mean(dim=0)
                    distances = cosine_similarity(points, centroid)
                    if distances.numel() > 0:
                        idx = torch.argmin(distances).item()
                        closest_indices.append(int(primary_indices[idx]))
                continue

            if relevance in (2, 3):
                primary_indices = []
                for sub_data in cluster_data.values():
                    if isinstance(sub_data, dict):
                        for lst in sub_data.values():
                            if lst:
                                primary_indices.extend(lst)
                    elif isinstance(sub_data, list) and sub_data:
                        primary_indices.extend(sub_data)

                if primary_indices:
                    points = x[torch.tensor(primary_indices, dtype=torch.long, device=x.device)]
                    centroid = points.mean(dim=0)
                    distances = cosine_similarity(points, centroid)
                    if distances.numel() > 0:
                        idx = torch.argmin(distances).item()
                        closest_indices.append(int(primary_indices[idx]))

                for sub_id, subclusters in cluster_data.items():
                    if isinstance(subclusters, dict):
                        for subsub_id, indices in subclusters.items():
                            if not indices:
                                continue
                            points = x[torch.tensor(indices, dtype=torch.long, device=x.device)]
                            centroid = points.mean(dim=0)
                            distances = cosine_similarity(points, centroid)
                            if distances.numel() > 0:
                                idx = torch.argmin(distances).item()
                                closest_indices.append(int(indices[idx]))
                    elif isinstance(subclusters, list) and subclusters:
                        points = x[torch.tensor(subclusters, dtype=torch.long, device=x.device)]
                        centroid = points.mean(dim=0)
                        distances = cosine_similarity(points, centroid)
                        if distances.numel() > 0:
                            idx = torch.argmin(distances).item()
                            closest_indices.append(int(subclusters[idx]))

    closest_indices.sort()
    return closest_indices


def launch():
    args = parse_args()

    breadth_path = args.breadth_path
    if not os.path.exists(breadth_path):
        raise FileNotFoundError(f"breadth_expansion not found: {breadth_path}")

    breadth_data = load_json(breadth_path)
    if "data" in breadth_data:
        breadth_data = breadth_data["data"]

    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, args.output_filename)
    output_by_quid_path = output_path.replace(".json", "_by_quid.json")

    ukeys = [k for k, v in breadth_data.items() if isinstance(v, dict)]
    missing = [
        k for k in ukeys
        if not all(f in breadth_data[k] for f in ("tree_node", "cluster_ids_x", "clip_relevance"))
    ]
    if missing:
        print(f"[warn] Skipping {len(missing)} entries missing tree_node/cluster_ids_x: {missing[:5]}...")
        ukeys = [k for k in ukeys if k not in missing]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_data = []
    by_quid = {}

    pbar = tqdm(total=len(ukeys))
    for quid in ukeys:
        entry = breadth_data[quid]
        uid = entry["uid"]
        tree_node = entry["tree_node"]
        cluster_ids_x = entry["cluster_ids_x"]
        clip_relevance = entry["clip_relevance"]
        w_v = entry.get("final_visual_weight", 0.5)
        w_a = entry.get("final_audio_weight", 0.5)

        visual_feats, audio_feats = load_clip_features(uid, args.clip_feat_path)
        fused = fuse_features(visual_feats, audio_feats, w_v, w_a)

        if len(cluster_ids_x) != fused.size(0):
            print(f"[warn] quid {quid}: cluster_ids_x length {len(cluster_ids_x)} != clip count {fused.size(0)}, skipping")
            pbar.update(1)
            continue

        relevance_scores = build_relevance_scores_for_clusters(
            tree_node, cluster_ids_x, clip_relevance
        )

        clusters_info = hierarchical_clustering_av(
            fused,
            cluster_ids_x,
            relevance_scores,
            num_subclusters=args.num_subclusters,
            num_subsubclusters=args.num_subsubclusters,
            device=device,
        )

        sorted_clip_indices = find_closest_points_in_temporal_order_subsub(
            fused, clusters_info, relevance_scores
        )

        all_data.append({
            "name": quid,
            "sorted_values": sorted_clip_indices,
            "relevance": relevance_scores,
        })
        by_quid[quid] = {
            "sorted_values": sorted_clip_indices,
            "relevance": relevance_scores,
        }
        pbar.update(1)

    pbar.close()

    save_json(all_data, output_path)
    save_json(by_quid, output_by_quid_path)
    print(f"Saved to {output_path} and {output_by_quid_path}")


if __name__ == "__main__":
    launch()
