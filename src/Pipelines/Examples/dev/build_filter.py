import os, sys
import torch
from torch.utils.data import random_split
import numpy as np
import pytorch_lightning as pl
from LightningModules.Embedding.layerless_embedding import LayerlessEmbedding, EmbeddingInferenceCallback
from LightningModules.Embedding.utils import get_best_run, build_edges, res, graph_intersection
from LightningModules.Filter.utils import stringlist_to_classes
from LightningModules.Filter.vanilla_filter import VanillaFilter, FilterInferenceCallback
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"


def post_process(pl_module, load_dir, save_dir):
    print("Training finished, running inference to filter graphs...")

    # By default, the set of examples propagated through the pipeline will be train+val+test set
    datatypes = ["train", "val", "test"]
    [os.makedirs(os.path.join(save_dir, datatype), exist_ok=True) for datatype in datatypes]
    
    input_dirs = [os.path.join(load_dir, datatype) for datatype in datatypes]
    loadsets = [load_dataset(input_dir) for input_dir in input_dirs]
    
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

    emb = (None if (pl_module.hparams["emb_channels"] == 0)
           else batch.embedding)  # Does this work??
    
    sections = 8
    cut_list = []
    for j in range(sections):
#         print(j)
        subset_ind = torch.chunk(torch.arange(batch.e_radius.shape[1]), sections)[j]
        output = pl_module(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius[:, subset_ind], emb).squeeze() if ('ci' in pl_module.hparams["regime"]) else pl_module(batch.x, batch.e_radius[:, subset_ind], emb).squeeze()
        cut = F.sigmoid(output) > pl_module.hparams["filter_cut"]
        cut_list.append(cut)
#     print("Predicted!")
    y_pid = batch.pid[batch.e_radius[0]] == batch.pid[batch.e_radius[1]]
    cut_list = torch.cat(cut_list)
    batch.edge_index = batch.e_radius[:, cut_list]
    batch.e_radius = None
    batch.embedding = None
    if "pid" not in pl_module.hparams["regime"]:
        batch.y = batch.y[cut_list]
    else:
        batch.y = None
    batch.y_pid = y_pid[cut_list]
    
    return batch

def save_downstream(batch, pl_module, datatype, save_dir):

    with open(os.path.join(save_dir, datatype, batch.event_file[-4:]), 'wb') as pickle_file:
        torch.save(batch, pickle_file)


def load_dataset(input_dir):
    all_events = os.listdir(input_dir)
    all_events = sorted([os.path.join(input_dir, event) for event in all_events])

    return all_events
        
        

def main():

# ================================== Graph Building ==========================
    run_label = "8qov0g7o"
    wandb_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/wandb_data"
    best_run_path = get_best_run(run_label,wandb_dir)

    chkpnt = torch.load(best_run_path)
    model = VanillaFilter(chkpnt["hyper_parameters"])
    best_run_path = get_best_run(run_label,wandb_dir)
    model = model.load_from_checkpoint(best_run_path)
    model = model.to(device)
    
    model.hparams["filter_cut"] = 0.18
    
    load_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/trackml_processed/embedding_processed/0_pt_cut_endcaps"
    save_dir = "/global/cscratch1/sd/danieltm/ExaTrkX/trackml_processed/filter_processed/0_pt_cut_endcaps_connected_high_eff"
    
    post_process(model, load_dir, save_dir)

if __name__=="__main__":
    main()
