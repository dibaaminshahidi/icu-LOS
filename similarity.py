import torch
import faiss

# def compute_similarity_edges(x, k=10, min_similarity = None):

#   if not isinstance(x, torch.Tensor):
#       x = torch.tensor(x, dtype=torch.float32)
#   x = x.float()
#   x_np = x.detach().cpu().numpy()

#   faiss.normalize_L2(x_np)
#   index = faiss.IndexFlatIP(x_np.shape[1])
#   index.add(x_np)
#   D, I = index.search(x_np, k + 1)

#   I, D = I[:, 1:], D[:, 1:]

#   src = torch.arange(x.size(0)).unsqueeze(1).repeat(1, k).flatten()
#   dst = torch.tensor(I.flatten(), dtype=torch.long)
#   edge_index = torch.stack([src, dst])
#   edge_attr = torch.tensor(D.flatten(), dtype=torch.float32).unsqueeze(1)

#   return edge_index, edge_attr

def compute_similarity_edges(x, k_pos=10, k_neg=10, min_abs_similarity=None):

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.float()

    x_np = x.detach().cpu().numpy().astype(np.float32)
    faiss.normalize_L2(x_np)

    index = faiss.IndexFlatIP(x_np.shape[1])
    index.add(x_np)

    n = x_np.shape[0]

    Dp, Ip = index.search(x_np, k_pos + 1)
    Ip, Dp = Ip[:, 1:], Dp[:, 1:]  # drop self

    src_p = torch.arange(n).unsqueeze(1).repeat(1, k_pos).flatten()
    dst_p = torch.from_numpy(Ip.reshape(-1)).long()
    sim_p = torch.from_numpy(Dp.reshape(-1)).float()  # positive-ish

    x_neg_np = -x_np
    Dn, In = index.search(x_neg_np, k_neg + 1)
    In, Dn = In[:, 1:], Dn[:, 1:]


    src_n = torch.arange(n).unsqueeze(1).repeat(1, k_neg).flatten()
    dst_n = torch.from_numpy(In.reshape(-1)).long()
    sim_n = -torch.from_numpy(Dn.reshape(-1)).float()  

    src = torch.cat([src_p, src_n], dim=0)
    dst = torch.cat([dst_p, dst_n], dim=0)
    sim = torch.cat([sim_p, sim_n], dim=0)

    if min_abs_similarity is not None:
        mask = sim.abs() >= float(min_abs_similarity)
        src, dst, sim = src[mask], dst[mask], sim[mask]

    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = sim.unsqueeze(1)

    return edge_index, edge_attr



def aggregate_neighbor_embeddings(data, source_type, relation, target_type, agg='mean'):

    edge_index = data[source_type, relation, target_type].edge_index
    src, dst = edge_index

    num_target_nodes = data[target_type].x.size(0)

    valid_indices_mask = (dst < num_target_nodes)
    filtered_src = src[valid_indices_mask]
    filtered_dst = dst[valid_indices_mask]

    if filtered_src.numel() == 0:
        return torch.zeros((data[source_type].x.shape[0], data[target_type].x.shape[1]), device=data[source_type].x.device)

    emb = data[target_type].x[filtered_dst]

    agg_emb = torch.zeros((data[source_type].x.shape[0], emb.shape[1]), device=emb.device)
    agg_emb.index_add_(0, filtered_src, emb)

    counts = torch.bincount(filtered_src, minlength=data[source_type].x.shape[0]).clamp(min=1)
    if agg == 'mean':
        agg_emb = agg_emb / counts.unsqueeze(1)

    return agg_emb

def add_similarity_edges(data, node_type, k_pos=10, k_neg=10, min_abs_similarity=None, embeddings=None, relation_name='SIMILAR_TO'):

    x = embeddings if embeddings is not None else data[node_type].x
    edge_index, edge_attr = compute_similarity_edges(x, k_pos=k_pos, k_neg=k_neg, min_abs_similarity=min_abs_similarity)

    edge_type = (node_type, relation_name, node_type)
    data[edge_type].edge_index = edge_index
    data[edge_type].edge_attr = edge_attr
    data[edge_type].edge_attr_name = {'similarity': [0,0], 'sign': [1,1]}

    return data

from collections import defaultdict
from torch.nn.functional import normalize

def build_similarity_edges_from_edge_attr(
    data,
    edge_type,
    new_edge_name="SIM_BY_EDGE",
    k=20,
    device="cpu",
    normalize_vectors=True,
    add_reverse_edges=True,
):

    src_type, _, dst_type = edge_type
    edge = data[edge_type]

    edge_index = edge.edge_index
    edge_attr = edge.edge_attr

    src_ids = edge_index[0]
    dst_ids = edge_index[1]

    d = edge_attr.size(1)

    groups = defaultdict(list)
    for e, dst_id in enumerate(dst_ids.tolist()):
        groups[dst_id].append(e)

    row_list = []
    col_list = []
    weight_list = []

    for dst_id, edge_list in groups.items():

        if len(edge_list) < 2:
            continue

        local_src = src_ids[edge_list]
        local_attr = edge_attr[edge_list].clone()

        if normalize_vectors:
            local_attr = normalize(local_attr, p=2, dim=1)

        arr = local_attr.detach().cpu().numpy().astype('float32')

        index = faiss.IndexFlatIP(d)
        index.add(arr)

        sim_scores, neighbors = index.search(arr, min(k, len(edge_list)))

        for i in range(len(edge_list)):
            s_i = local_src[i].item()

            for j in range(1, neighbors.shape[1]):
                s_j = local_src[neighbors[i][j]].item()
                sim = sim_scores[i][j]

                if sim <= 0:
                    continue

                row_list.append(s_i)
                col_list.append(s_j)
                weight_list.append(sim)

                if add_reverse_edges:
                    row_list.append(s_j)
                    col_list.append(s_i)
                    weight_list.append(sim)

    if len(row_list) == 0:
        print("No similarity edges produced.")
        return data

    sim_edge_index = torch.tensor([row_list, col_list], dtype=torch.long)
    sim_edge_attr  = torch.tensor(weight_list, dtype=torch.float32).unsqueeze(1)


    data[src_type, new_edge_name, src_type].edge_index = sim_edge_index
    data[src_type, new_edge_name, src_type].edge_attr  = sim_edge_attr
    data[src_type, new_edge_name, src_type].edge_attr_name = {'similarity': [0,0]}

    return data
