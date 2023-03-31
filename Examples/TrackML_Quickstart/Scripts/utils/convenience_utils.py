import os
import sys
import shutil

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from bokeh.io import output_notebook, show
from bokeh.plotting import figure, row
from bokeh.models import ColumnDataSource
from bokeh.palettes import viridis
from bokeh.models.annotations import Label
output_notebook()

from sklearn.metrics import roc_auc_score  
from matplotlib import pyplot as plt

sys.path.append("../../")
from Pipelines.TrackML_Example.LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def headline(message):
    buffer_len = (80 - len(message))//2 if len(message) < 80 else 0
    return "-"*buffer_len + ' ' + message + ' ' + '-'*buffer_len

def delete_directory(dir):
    if os.path.isdir(dir):
        for files in os.listdir(dir):
            path = os.path.join(dir, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)

def get_example_data(configs):

    metric_learning_configs = configs['metric_learning_configs']

    model = LayerlessEmbedding(metric_learning_configs)
    model.setup(stage='fit')
    training_example = model.trainset[0]

    example_hit_inputs = model.get_input_data(training_example)
    example_hit_df = pd.DataFrame(example_hit_inputs.numpy())

    return example_hit_df, training_example


def get_training_metrics(trainer):

    log_file = os.path.join(trainer.logger.log_dir , 'metrics.csv')
    metrics = pd.read_csv(log_file, sep=',')
    train_metrics = metrics[ ~ metrics['train_loss'].isna() ][['epoch', 'train_loss']]
    train_metrics['epoch'] -= 1
    val_metrics = metrics[ ~ metrics['val_loss'].isna() ][['val_loss', 'eff', 'pur', 'current_lr', 'epoch']]
    metrics = pd.merge(left=train_metrics, right=val_metrics, how='inner', on='epoch')

    return metrics

def plot_training_metrics(metrics):

    p1 = figure(title='Training validation loss', x_axis_label='Epoch', y_axis_label='Loss', y_axis_type="log")

    source = ColumnDataSource(metrics)

    cmap = viridis(3)

    for idx, y in enumerate(['train_loss', 'val_loss']):
        p1.circle(y=y, x='epoch', source=source, color=cmap[idx], legend_label=y)
        p1.line(x='epoch', y=y, source=source, color=cmap[idx], legend_label=y)


    p2 = figure(title='Purity on validation set', x_axis_label='Epoch', y_axis_label='Purity')
    p2.circle(y='pur', x='epoch', source=source, color=cmap[0], legend_label='Purity')
    p2.line(x='epoch', y='pur', source=source, color=cmap[0], legend_label='Purity')

    p3 = figure(title='Efficiency on validation set', x_axis_label='Epoch', y_axis_label='Efficiency')
    p3.circle(y='eff', x='epoch', source=source, color=cmap[0], legend_label='Efficiency')
    p3.line(x='epoch', y='eff', source=source, color=cmap[0], legend_label='Efficiency')

    show(row([p1,p2, p3]))

def plot_neighbor_performance(model):

    all_radius = np.arange(0.001, 0.15, 0.005)
    results = { 'eff': [], 'pur': [], 'loss': [], 'radius': all_radius }
    model.to(device)
    test_data = model.testset[0].to(device)

    with torch.no_grad():
        for r in all_radius:
            test_results = model.shared_evaluation(
                test_data, 0, r, 1000, log=False
            )
            for key in results:
                if key not in test_results: continue
                results[key].append( test_results[key].cpu().numpy() )
    results = pd.DataFrame(results)

    source = ColumnDataSource(results)
    cmap = viridis(3)
    titles = ['Efficiency', 'Purity', 'Loss'] 
    figures = []
    x='radius'
    for idx, y in enumerate(['eff', 'pur', 'loss']):
        figures.append( figure(title=titles[idx], x_axis_label=x, y_axis_label=y) )
        figures[-1].circle(y=y, x=x, source=source, color=cmap[0], legend_label=y)
        figures[-1].line(x=x, y=y, source=source, color=cmap[0], legend_label=y)
        y_val = results[y][(results[x] - model.hparams["r_test"]).abs().idxmin()].item()
        label = Label(x=model.hparams["r_test"], y=y_val, x_offset=10, y_offset=-10, text=f"@ radius = {model.hparams['r_test']}, \n" + y + " = "+str(round(y_val, 3)), border_line_color='black', border_line_alpha=1.0,
        background_fill_color='white', background_fill_alpha=0.8)
        figures[-1].add_layout(label)

    show(row(figures))

def plot_true_graph(sample_data, num_tracks=100):

    p = figure(title='Truth graph', x_axis_label='x', y_axis_label='y', height=800, width=800) 
 
    true_edges = sample_data.signal_true_edges
    true_unique, true_lengths = sample_data.pid[true_edges[0]].unique(return_counts=True)
    pid = sample_data.pid
    r, phi, z = sample_data.cpu().x.T
    x, y = r * np.cos(phi * np.pi), r * np.sin(phi * np.pi)
    cmap = viridis(num_tracks)
    source = ColumnDataSource(dict(x=x.numpy(), y=y.numpy()))
    p.circle(x='x', y='y', source=source, color=cmap[0], size=1, alpha=0.1)

    for i, track in enumerate(true_unique[true_lengths >= 5][:num_tracks]):
        # Get true track plot
        track_true_edges = true_edges[:, pid[ true_edges[0]] == track ]
        X_edges, Y_edges = x[track_true_edges].numpy(), y[track_true_edges].numpy()
        X = np.concatenate(X_edges)
        Y = np.concatenate(Y_edges)

        p.circle(X, Y, color=cmap[i], size=5)
        p.multi_line(X_edges.T.tolist(), Y_edges.T.tolist(), color=cmap[i])
        
    show(p)

def plot_predicted_graph(model):

    # from matplotlib import pyplot as plt
    test_data = model.testset[0].to(device)
    test_results = model.to(device).shared_evaluation(test_data.to(device), 0, model.hparams["r_test"], 1000, log=False)

    p = figure(title='Truth graphs', x_axis_label='x', y_axis_label='y', height=500, width=500) 
    q = figure(title='Predicted graphs', x_axis_label='x', y_axis_label='y', height=500, width=500) 

    true_edges = test_results['truth_graph']
    true_unique, true_lengths = test_data.pid[true_edges[0]].unique(return_counts=True)
    pred_edges = test_results['preds']
    pid = test_data.pid
    r, phi, z = test_data.cpu().x.T
    x, y = r * np.cos(phi * np.pi), r * np.sin(phi * np.pi)
    cmap = viridis(11)
    source = ColumnDataSource(dict(x=x.numpy(), y=y.numpy()))
    p.circle(x='x', y='y', source=source, color=cmap[0], size=1, alpha=0.1)
    q.circle(x='x', y='y', source=source, color=cmap[0], size=1, alpha=0.1)

    for i, track in enumerate(true_unique[true_lengths >= 10][:10]):
        # Get true track plot
        track_true_edges = true_edges[:, pid[ true_edges[0]] == track ]
        X_edges, Y_edges = x[track_true_edges].numpy(), y[track_true_edges].numpy()
        X = np.concatenate(X_edges)
        Y = np.concatenate(Y_edges)

        p.circle(X, Y, color=cmap[i], size=5)
        p.multi_line(X_edges.T.tolist(), Y_edges.T.tolist(), color=cmap[i])

        track_pred_edges = pred_edges[:, (pid[pred_edges] == track).any(0)]

        X_edges, Y_edges = x[track_pred_edges].numpy(), y[track_pred_edges].numpy()
        X = np.concatenate(X_edges)
        Y = np.concatenate(Y_edges)

        q.circle(X, Y, color=cmap[i], size=5)
        q.multi_line(X_edges.T.tolist(), Y_edges.T.tolist(), color=cmap[i])
        
    show(row([p,q]))

def plot_track_lengths(model):

    all_true_edges = []
    all_pred_edges = []
    test_data = model.testset[0].to(device)
    signal_true_edges = test_data.signal_true_edges
    test_results = model.to(device).shared_evaluation(test_data.to(device), 0, model.hparams["r_test"], 1000, log=False)
    pred_edges = test_results['preds']
    pid = test_data.pid
    for track_id in test_data.pid.unique():
        e = signal_true_edges[:, pid[ signal_true_edges[0]] == track_id ]
        true_edges = pid[ e[0]] == pid[e[1]]
        all_true_edges.append( true_edges.sum().cpu().numpy() )

        e = pred_edges[:, pid[ pred_edges[0]] == track_id ]
        true_edges = pid[ e[0]] == pid[e[1]]
        all_pred_edges.append( true_edges.sum().cpu().numpy() )

    histogram = np.histogram(all_true_edges, bins=20, range=(0,20))

    pred_histogram = np.histogram(all_pred_edges, bins=200, range=(0,200))

    true_histogram = pd.DataFrame(
        dict(
            low = histogram[1][:-1],
            high = histogram[1][1:],
            true_hist= histogram[0],
        )
    )

    pred_histogram = pd.DataFrame(
        dict(
            low = pred_histogram[1][:-1],
            high = pred_histogram[1][1:],
            pred_hist = pred_histogram[0]
        )
    )

    p1 =  figure(title='Histogram of true track lengths', x_axis_label='Edges', y_axis_label='Count', height=400, width=400) 
    p2 =  figure(title='Histogram of predicted track lengths', x_axis_label='Edges', y_axis_label='Count', height=400, width=400) 
    p1.quad(bottom=0, top='true_hist', left='low', right='high', source=ColumnDataSource(true_histogram))
    p2.quad(bottom=0, top='pred_hist', left='low', right='high', source=ColumnDataSource(pred_histogram))
    show(row([p1,p2]))

def plot_graph_sizes(model):

    graph_sizes = []
    model = model.to(device)
    with torch.no_grad():
        for data in tqdm(model.trainset):
            results = model.shared_evaluation(data.to(device), 0, 0.12, 100, log=False)
            graph_sizes.append(results['preds'].shape[1])

    # Make histogram of graph sizes
    plt.figure(figsize=(10,5))
    plt.hist(graph_sizes);
    plt.title('Histogram of predicted graph sizes');
    plt.xlabel('Number of edges');

def plot_edge_performance(model):

    all_cuts = np.arange(0.001, 1., 0.02)
    results = { 'eff': [], 'pur': [], 'score cut': all_cuts }
    model.to(device)
    test_data = model.testset[0].to(device)

    with torch.no_grad():
        test_results = model.shared_evaluation(
                test_data, 0, log=False
            )

        auc = roc_auc_score(test_results["truth"].cpu(), test_results["score"].cpu())
        
        for cut in all_cuts:
            preds = test_results["score"] > cut
            edge_positive = preds.sum().float()
            edge_true = test_results["truth"].sum().float()
            edge_true_positive = (
                (test_results["truth"].bool() & preds).sum().float()
            )          

            results["eff"].append( (edge_true_positive / max(1, edge_true)).cpu().numpy() )
            results["pur"].append( (edge_true_positive / max(1, edge_positive)).cpu().numpy() )
    results = pd.DataFrame(results)

    source = ColumnDataSource(results)
    cmap = viridis(3)
    titles = ['Efficiency', 'Purity'] 
    figures = []
    x='score cut'
    for idx, y in enumerate(['eff', 'pur']):
        figures.append( figure(title=titles[idx], x_axis_label=x, y_axis_label=y) )
        figures[-1].circle(y=y, x=x, source=source, color=cmap[0], legend_label=y)
        figures[-1].line(x=x, y=y, source=source, color=cmap[0], legend_label=y)
        y_val = results[y][(results[x] - 0.5).abs().idxmin()].item()
        label = Label(x=0.1, y=y_val, x_offset=10, y_offset=-10, text="@ score cut = 0.5, \n" + y + " = "+str(round(y_val, 3)) + "\n AUC: "+str(auc), border_line_color='black', border_line_alpha=1.0,
        background_fill_color='white', background_fill_alpha=0.8)
        figures[-1].add_layout(label)

    show(row(figures))