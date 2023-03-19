from typing import Tuple

import torch
from torch import Tensor


# return 2 predicted ranks (replacing head and trail)
def accuracy_for_one(
    ent_emb: Tensor, 
    rel_emb: Tensor, 
    triplet: Tensor,
    p_norm: float
) -> Tuple[float, float]:
  num_nodes = ent_emb.shape[0]

  head = ent_emb[triplet[0]]
  label = rel_emb[triplet[1]]
  tail = ent_emb[triplet[2]]

  correct_diss = torch.norm(head + label - tail, p=p_norm)

  # calculating all dissimilarities with replaced head
  diss_with_head_replaced = torch.norm(
      ent_emb + (label - tail).repeat(num_nodes, 1),
      p=p_norm, dim=1
  )

  # calculating all dissimilarities with replaced tail
  diss_with_tail_replaced = torch.norm(
      (head + label).repeat(num_nodes, 1) - ent_emb,
      p=p_norm, dim=1
  )

  # rank among triplets with replaced head
  rank_heads = torch.sum(diss_with_head_replaced < correct_diss)
  # rank among triplets with replaced tail
  rank_tails = torch.sum(diss_with_tail_replaced < correct_diss)

  return (rank_heads.item(), rank_tails.item())

# return MR and Hits@10 (without filtering)
def mean_accuracy(
    ent_emb: Tensor, 
    rel_emb: Tensor, 
    triplets: Tensor,
    p_norm: float
) -> Tuple[float, float]:
  ranks_sum = 0
  hits10 = 0

  for triplet in triplets.t():
    r1, r2 = accuracy_for_one(ent_emb, rel_emb, triplet, p_norm)
    ranks_sum += r1 + r2
    hits10 += (r1 < 10) + (r2 < 10)
  
  ranks_sum /= triplets.shape[1] * 2
  hits10 /= triplets.shape[1] * 2

  return ranks_sum, hits10