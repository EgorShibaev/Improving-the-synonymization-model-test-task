import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple

# takes head, tails and number of nodes
# returns corrupded heads and tails
def corrupted_triplets(
    head: Tensor, 
    tail: Tensor, 
    nodes: int
) -> Tuple[Tensor, Tensor]:
  num_triplets = head.shape[0]
  # choose what to corrupt head or tail
  head_or_tail = torch.randint(0, 2, (num_triplets,)).bool()

  # new values
  rand_inds = torch.randint(0, nodes, size=(num_triplets,))
  head_cor = torch.where(head_or_tail, head, rand_inds)
  tail_cor = torch.where(head_or_tail, rand_inds, tail)

  return (head_cor, tail_cor)


class TransE(nn.Module):

  def __init__(
      self, 
      embedding_size: int,
      margin: float, 
      num_nodes: int, 
      num_diff_edges: int, 
      p_norm: float # norm parameter
  ):
    super().__init__()

    # saving hyperparameters 
    self.p_norm = p_norm
    self.margin = margin
    self.num_nodes = num_nodes
    self.num_diff_edges = num_diff_edges

    # bounds for initialization
    rand_lower_bound = -6 * np.sqrt(embedding_size)
    rand_upper_bound =  6 * np.sqrt(embedding_size)
    # embeddings for relations
    self.rel_embs = torch.zeros((num_diff_edges, embedding_size))
    self.rel_embs.uniform_(rand_lower_bound, rand_upper_bound)
    self.rel_embs = F.normalize(self.rel_embs, dim=1)
    self.rel_embs = nn.Parameter(self.rel_embs.requires_grad_())

    # embeddings for enteties
    self.ent_embs = torch.zeros((num_nodes, embedding_size))
    self.ent_embs.uniform_(rand_lower_bound, rand_upper_bound)
    self.ent_embs = nn.Parameter(self.ent_embs.requires_grad_())


  # take triplet (h, l, t) and return dissimilarity of h + l and t  
  def forward(self, head_ind: Tensor, label: Tensor, tail_ind: Tensor):
    head = self.ent_embs[head_ind]
    tail = self.ent_embs[tail_ind]

    head = F.normalize(head, dim=1)
    tail = F.normalize(tail, dim=1)

    rel = self.rel_embs[label]

    return torch.norm(head + rel - tail, p=self.p_norm, dim=1)
  
  
  def loss(self, head_ind: Tensor, label: Tensor, tail_ind: Tensor):
    head_cor_ind, tail_cor_ind = corrupted_triplets(
        head_ind, 
        tail_ind, 
        self.num_nodes
    )

    diss_valid = self(head_ind, label, tail_ind)
    diss_corrupted = self(head_cor_ind, label, tail_cor_ind)

    # sum over all max(0, margin + valid - corrupted)
    return F.margin_ranking_loss(
      diss_valid,
      diss_corrupted,
      target=-torch.ones_like(head_ind), # -1 assumes that first input should be ranked lower than second
      margin=self.margin,
      reduction='sum' # sum over all losses in batch
    )


  def get_entities_embs(self) -> Tensor:
    return F.normalize(self.ent_embs, dim=1)
  
  
  def get_relations_embs(self) -> Tensor:
    return self.rel_embs.clone()