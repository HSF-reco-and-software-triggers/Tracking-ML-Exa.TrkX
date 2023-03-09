import matplotlib.pyplot as plt
import mplhep as hep
import torch
import numpy as np
hep.style.use(hep.style.ATLAS)

def plot_score(outputs, stage='val', **kwargs):
    fig, ax = plt.subplots(1,1)
    score = torch.cat([o['score'] for o in outputs]).detach().numpy()
    truth = torch.cat([o['truth'] for o in outputs]).detach().numpy()
    ax2 = ax.twinx()
    hist1 = hep.histplot(np.histogram(score[truth==0], bins=50, range=(0,1)), label=f'{stage}_negative', ax=ax, color='c', **kwargs.get('histplot_args', {}))
    hist2 = hep.histplot(np.histogram(score[truth==1], bins=50, range=(0,1)), label=f'{stage}_positive', ax=ax2, color='m', **kwargs.get('histplot_args', {}))
    ax.set_xlabel('score')
    ax.set_ylim((0, kwargs.get('negative_ylim', 7e6)))
    ax2.set_ylim((0, kwargs.get('positive_ylim', 2e5)))
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    if kwargs.get('title'): ax.set_title(kwargs['title'])
    plt.tight_layout()
    return fig, ax