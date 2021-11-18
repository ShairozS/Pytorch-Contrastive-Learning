import torch
import pandas as pd
import numpy as np

def extract_embeddings(dl, model, n, device='cuda'):
    '''
    Use a test dataloader and torch model to extract N embeddings and 
    organize them into a dictionary with the GT label for plotting purposes
    '''
        
    model.eval()
    m = model.to(device)
    
    embs = None
    labs = None
    tot_cnt = 0
    
    for idx, dat in enumerate(dl):
        
        # Extract relevent data points from batch
        x1 = dat['x1'] # (B,C,W,H)
        x2 = dat['x2'] # (B,C,W,H)
        labels = dat['labels'].type(torch.LongTensor) # (B,1)
        
        # Send inputs to correct device
        x1 = x1.to(device); x2 = x2.to(device)
        
        # Pass inputs through model
        emb1 = m(x1); emb2 = m(x2) #(B, EMB_SIZE)
        
        # Add to running lists
        if embs is None:
            embs = torch.cat([emb1, emb2])
        else:
            embs = torch.cat([embs, emb1, emb2])
        
        # Add coloring to list
        if labs is None:
            labs = torch.cat([labels, labels])
        else:
            labs = torch.cat([labs, labels, labels])

        # Check count of points so far
        curr_cnt = x1.shape[0] + x2.shape[0]
        if tot_cnt + curr_cnt >= n:
            break
        tot_cnt += curr_cnt
    
    # Convert to numpy array and seperate coords for plotting
    embs = [np.array(e.detach()) for e in embs]
    embs_x = [e[0] for e in embs]; embs_y = [e[1] for e in embs]
    labs = np.array([x.item() for x in labs])
    
    df = pd.DataFrame({"X": embs_x, "Y": embs_y, "Label": labs})
    
    return(df)