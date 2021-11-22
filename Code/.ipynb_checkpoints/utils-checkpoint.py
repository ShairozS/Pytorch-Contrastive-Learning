import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def extract_embeddings(test_dataloader, model, N, reduce_to_dimension=2, device='cuda'):
    
    '''
    Use a test dataloader and torch model to extract N embeddings, reduce them to 
    reduce_to_dimension dimensions, then organize them into a dataframe with the GT labels
    
    output:
    
        | Embs   | Label
    -------------------
        [0.2,...]   3
        [0.1,...]   5
        [0.9,...]   4
        ...
        ..
        
    '''
        
    model.eval()
    m = model.to(device)
    
    embs = None
    labs = None
    tot_cnt = 0
    
    for idx, dat in enumerate(test_dataloader):
        
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
        if tot_cnt + curr_cnt >= N:
            break
        tot_cnt += curr_cnt
    
    # If there's only two dimensions in the embeddings we can directly use them
    embs = [np.array(e.cpu().detach()) for e in embs]
    emb_size = embs[0].shape[0]
    
    assert reduce_to_dimension is not None and reduce_to_dimension <= emb_size, "reduce_to_dimension must be integer <= emb dimension"
    if emb_size > reduce_to_dimension:
        from sklearn.decomposition import PCA
        embs = np.array(embs)
        print("performing PCA to reduce embeddings to " + str(reduce_to_dimension) + " dimensions")
        pca = PCA(n_components = reduce_to_dimension)
        embs = pca.fit_transform(embs)
        print(str(pca.explained_variance_ratio_.sum()) + " % variance explained using PCA")

    
    # When embeddings are 2-dimensional
    #if reduce_to_
    #embs_x = [e[0] for e in embs]; embs_y = [e[1] for e in embs]
    labs = [x.item() for x in labs]
    df = pd.DataFrame({"Emb": list(embs), "Label": labs})
    #df = pd.DataFrame({"X": embs_x, "Y": embs_y, "Label": labs})
    
    return(df)


def plot_embeddings(emb_df):
    '''
    Plot the DataFrame from extract_embeddings() in 2 dimensions 
    and color by label
    '''
    embs = list(emb_df['Emb'])
    assert embs[0].shape[0] == 2, "Embeddings must be reduced to dimension 2, use reduce_to_dimension param in extract_embeddings"
    
    embs_x = [e[0] for e in embs]; embs_y = [e[1] for e in embs]
    labs = list(emb_df['Label'])
    
    import seaborn as sns
    sns.set_style("darkgrid")
    sns.relplot(x=embs_x, y=embs_y, hue=labs, palette="deep", alpha=0.7, s=75)
    plt.title("Test Embeddings")