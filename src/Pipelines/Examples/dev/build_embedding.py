import os, sys
import torch
from torch.utils.data import random_split
import numpy as np
import pytorch_lightning as pl
from LightningModules.Embedding.layerless_embedding import LayerlessEmbedding, EmbeddingInferenceCallback
from LightningModules.Embedding.utils import get_best_run, build_edges, res, graph_intersection
device = "cuda" if torch.cuda.is_available() else "cpu"


def post_process(pl_module, load_dir, save_dir, train_split):
    print("Training finished, running inference to filter graphs...")

    # By default, the set of examples propagated through the pipeline will be train+val+test set
    datatypes = ["train", "val", "test"]
    [os.makedirs(os.path.join(save_dir, datatype), exist_ok=True) for datatype in datatypes]
    
    loadsets = load_datasets(load_dir, train_split)
    
    total_length = sum([len(dataset) for dataset in loadsets])
    batch_incr = 0

    pl_module.eval()
    with torch.no_grad():
        for set_idx, (datatype, dataset) in enumerate(zip(datatypes, loadsets)):
            for batch_idx, event in enumerate(dataset):
#                 print(event)
                percent = (batch_incr / total_length) * 100
                sys.stdout.flush()
                sys.stdout.write(f'{percent:.01f}% inference complete \r')
                if (not os.path.exists(os.path.join(save_dir, datatype, event[-4:]))):
                    batch = torch.load(event, map_location=torch.device('cpu'))
                    data = batch.to(pl_module.device) #Is this step necessary??
                    data = construct_downstream(data, pl_module)
                    save_downstream(data, pl_module, datatype, save_dir)

                batch_incr += 1

def construct_downstream(batch, pl_module):
   
    if 'ci' in pl_module.hparams['regime']:
        spatial = pl_module(torch.cat([batch.cell_data, batch.x], axis=-1))
    else:
        spatial = pl_module(batch.x)
    e_spatial = build_edges(spatial, 1.7, 500, res)  
    e_bidir = torch.cat([batch.layerless_true_edges.to(device), 
                           torch.stack([batch.layerless_true_edges[1], batch.layerless_true_edges[0]], axis=1).T.to(device)], axis=-1) 

    # Remove duplicate edges by distance from vertex
    R_dist = torch.sqrt(batch.x[:,0]**2 + batch.x[:,2]**2)
    e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]

    e_spatial, y = graph_intersection(e_spatial, e_bidir)  

    # Re-introduce random direction, to avoid training bias
    random_flip = torch.randint(2, (e_spatial.shape[1],)).bool()
    e_spatial[0, random_flip], e_spatial[1, random_flip] = e_spatial[1, random_flip], e_spatial[0, random_flip]

    batch.embedding = spatial.cpu().detach()
    
    batch.e_radius = e_spatial.cpu()
    batch.y = torch.from_numpy(y).float()
    
    return batch

def save_downstream(batch, pl_module, datatype, save_dir):

    with open(os.path.join(save_dir, datatype, batch.event_file[-4:]), 'wb') as pickle_file:
        torch.save(batch, pickle_file)
        
def load_datasets(input_dir, train_split, seed = 0):
    '''
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    '''
    torch.manual_seed(seed)
    all_events = os.listdir(input_dir)
    all_events = sorted([os.path.join(input_dir, event) for event in all_events])
    train_events, val_events, test_events = random_split(all_events, train_split)

    return train_events, val_events, test_events
        
        

def main():

# ================================== Embedding Building ==========================
    run_label = "hd6lqvip"
    wandb_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/wandb_data"
    best_run_path = get_best_run(run_label,wandb_dir)

    chkpnt = torch.load(best_run_path)
    model = LayerlessEmbedding(chkpnt["hyper_parameters"])
    model = model.load_from_checkpoint(best_run_path)
    model = model.to(device)
    
    load_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/trackml/feature_store_endcaps"
    save_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/trackml_processed/embedding_processed/0_pt_cut_endcaps/"

    train_split = [8700, 50, 50]
    post_process(model, load_dir, save_dir, train_split)

if __name__=="__main__":
    main()
