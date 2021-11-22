import torch
import numpy as np
import torch.nn.functional as F

def form_triplets(inA, inB):
        
    '''
    Form triplets from two tensors of embeddings. It is assumed that the embeddings at corresponding batch positions are similar
    and all other batch positions are dissimilar 
    
    i.e inA[i] ~ inB[i] and inA[i] !~ inB[j] for all i =! j
    '''
    
    b, emb_size = inA.shape
    perms = b**2
    
    labels = [0]*perms; sim_idxs = [(0 + i*b) + i for i in range(b)]
    for idx in sim_idxs:
        labels[idx] = 1
    labels = torch.Tensor(labels)
        
    labels = labels.type(torch.BoolTensor).to(inA.device)
    anchors = inA.repeat(b, 1)[~labels]
    negatives = torch.cat([inB[i,:].repeat(b,1) for i in range(b)])[~labels]
    positives = inB.repeat(b, 1)[~labels]

    return(anchors, positives, negatives)


def form_pairs(inA, inB):
    
    '''
    Form pairs from two tensors of embeddings. It is assumed that the embeddings at corresponding batch positions are similar
    and all other batch positions are dissimilar 
    
    i.e inA[i] ~ inB[i] and inA[i] !~ inB[j] for all i =! j
    '''
    
    b, emb_size = inA.shape
    perms = b**2
    
    labels = [0]*perms; sim_idxs = [(0 + i*b) + i for i in range(b)]
    for idx in sim_idxs:
        labels[idx] = 1
    labels = torch.Tensor(labels)
    
    return(inA.repeat(b, 1), torch.cat([inB[i,:].repeat(b,1) for i in range(b)]), labels.type(torch.LongTensor).to(inA.device))


class NTXent(torch.nn.Module):
    
    '''
    Modified from: https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/
    '''
    
    def __init__(self, 
                 batch_size, 
                 temperature=0.5,
                 device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
        self.device = device
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class ContrastiveLoss(torch.nn.Module):
    
    
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    
    args:
        distance (function): A function that returns the distance between two tensors - should be a valid metric over R; default= L2 distance
        margin (scalar): The margin value between positive and negative class ; default=1.0
        miner (function, optional): A function that calculates similarity labels [0,1] on the input if no labels are explicitly provided - should return (embs1, embs2, labels)
    """

    
    def __init__(self,
                 distance = lambda x,y: torch.pow(x-y, 2).sum(1),
                 margin=1.0,
                 mode='pairs',
                 batch_size=None,
                 temperature=0.5):
        
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.mode = mode
        
        if self.mode == 'ntxent':
            assert batch_size is not None, "Must specify batch size to use Ntxent Loss"
            self.ntxent = NTXent(batch_size = batch_size, temperature = temperature)
        
    
    
    
    def forward(self, x, y):
        
        if self.mode == 'pairs':
            return(self.forward_pairs(x, y))
        
        elif self.mode == 'triplets':
            return(self.forward_triplets(x, y))
        
        elif self.mode == 'ntxent':
            return(self.forward_ntxent(x, y))
    
    
    def forward_triplets(self, x, y):
        a, p, n = form_triplets(x,y)
        return(torch.nn.functional.triplet_margin_with_distance_loss(a,p,n, margin=self.margin, distance_function=self.distance))
    
    def forward_ntxent(self, x, y):
        return(self.ntxent(x, y))
    
    def forward_pairs(self, x, y, label=None):
        '''
        Return the contrastive loss between two similar or dissimilar outputs
        
        Args:
            x (torch.Tensor) : The first input tensor (B, N)
            y (torch.Tensor) : The second input tensor (B,N)
            label (torch.Tensor, optional) : A tensor with elements either 0 or 1 indicating dissimilar or similar (B, 1)
        '''
        
        assert x.shape==y.shape, str(x.shape) + "does not match input 2: " + str(y.shape)
        
        x, y, label = form_pairs(x,y)
        
        distance = self.distance(x,y)
        
        # When the label is 1 (similar) - the loss is the distance between the embeddings
        # When the label is 0 (dissimilar) - the loss is the distance between the embeddings and a margin
        loss_contrastive = torch.mean((label) * distance +
                                      (1-label) * torch.clamp(self.margin - distance, min=0.0))


        return loss_contrastive