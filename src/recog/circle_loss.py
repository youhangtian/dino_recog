import torch 
import torch.nn.functional as F
from torch import distributed

class CircleLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: int,
        m: float,
        embedding_size: int, 
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False
    ):
        super().__init__()
        assert distributed.is_initialized(), 'must initialize distributed'
        self.world_size = distributed.get_world_size()

        self.gamma: int = gamma
        self.m: float = m
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.fp16: bool = fp16
        self.num_sample: int = int(self.sample_rate * num_classes)

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_classes, self.embedding_size)))

    def sample(self, local_labels, gather_labels):
        with torch.no_grad():
            unique_labels = torch.unique(gather_labels, sorted=True).cuda()
            if self.num_sample - unique_labels.size(0) >= 0:
                perm = torch.rand(size=[self.num_classes]).cuda()
                perm[unique_labels] = 2.0
                index = torch.topk(perm, k=self.num_sample).indices.cuda()
                index = index.sort().values.cuda()
            else:
                index = unique_labels

            weight_index = index
            labels = torch.searchsorted(index, local_labels)

        return self.weight[weight_index], labels 

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        batch_size = local_embeddings.size(0)
        local_labels = local_labels.long()
        _gather_labels = [torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)]
        distributed.all_gather(_gather_labels, local_labels)
        gather_labels = torch.cat(_gather_labels)

        if self.sample_rate < 1:
            weight, labels = self.sample(local_labels, gather_labels)
        else:
            weight = self.weight
            labels = local_labels

        with torch.cuda.amp.autocast(self.fp16):
            norm_weight = F.normalize(weight)
            logits = F.linear(local_embeddings, norm_weight)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        tensor = torch.arange(self.num_sample).long().to(labels.device)
        p_index = tensor.unsqueeze(0) == labels.unsqueeze(1)
        sp = logits[p_index]
        
        '''
        with torch.no_grad():
            sp.arccos_()
            logits.arccos_()
            logits[p_index] = sp + self.m
            logits.cos_()
        logits = logits * self.gamma 

        loss = F.cross_entropy(logits, labels)
        return loss 
        '''
        
        n_index = p_index.logical_not()
        sn = logits[n_index]

        alpha_p = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m 

        logits[p_index] = alpha_p * (sp - delta_p) * self.gamma 
        logits[n_index] = alpha_n * (sn - delta_n) * self.gamma 

        logits_diff = logits - logits[p_index].unsqueeze(1)
        loss = torch.logsumexp(logits_diff, dim=1).mean()
        return loss
    