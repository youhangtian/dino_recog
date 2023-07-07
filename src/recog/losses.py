import torch 

def get_loss(name, margin_list=None):
    if name == 'arcface':
        if margin_list:
            return ArcFace(margin_list[0], margin_list[1])
        else:
            return ArcFace()
    elif name == 'cosface':
        if margin_list:
            return CosFace(margin_list[0], margin_list[1])
        else:
            return CosFace()
    else:
        return None 


class ArcFace(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s 
        self.margin = margin 

    def forward(self, logits, labels):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin 
            logits[index, labels[index].view(-1)] = final_target_logit 
            logits.cos_()
        logits = logits * self.s 
        return logits 

class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(CosFace, self).__init__()
        self.s = s 
        self.margin = margin 

    def forward(self, logits, labels):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.margin 
        logits[index, labels[index].view(-1)] = final_target_logit 
        logits = logits * self.s 
        return logits 
