# Truth Definitions

## TrackML Tracking



## ITk Tracking

### Edge-wise Truth

In training the various stages of the pipeline, two definitions of pair-wise or edge-wise truth are used. The first is the simplest: `pid_truth`. If two spacepoints share a particle ID (`pid`), an edge between them is given `pid_truth=1`. Otherwise `pid_truth=0`. This also shows up in the library as `y_pid`, as a toggle that can be turned on/off in GNN training. This truth definition is useful when a graph *has already been constructed*, as in GNN training. 

The other definition of truth is `modulewise_truth`. This is based on the concept of a module-wise truth graph, which will be useful to visualize.

<figure>
  <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/truth_graph.png"/>
  <figcaption>A cartoon of a truth graph. Each blue line is a module here (note the dashed layers of the barrel - the layers are made of many small modules)</figcaption>
</figure>

A truth graph is simply some target graph that tries to represent the underlying physics in some way. It must be opinionated, and there is clearly not a single, correct way to define it. That said, for the ITk geometry, we define it by:
1. Order all spacepoints of a track by increasing distance from its creation vertex
2. Connect each **pair** of spacepoints in this sequence, as an edge in a graph
3. (Module-wise definition) Where two spacepoints are on the **same module**, do not join them. Instead, form all pair combinations with the preceding and succeeding spacepoints. 

To phrase this module-wise defintion differently, for the set of spacepoints {x_i} and modules {m_i} in the diagram, we order all spacepoints as

```
[(x_0, m_0), (x_1, m_1), (x_2, m_2), (x_3, m_3), ((x_4, x_5), m_4), (x_6, m_5)]
```

Therefore our edge list becomes

```
(x_0, x_1), (x_1, x_2), (x_2, x_3), (x_3, x_4), (x_3, x_5), (x_4, x_6), (x_5, x_6)
```

This is achieved quite efficiently in code with:
```
signal_list = (
    signal.groupby(
        ["particle_id", "barrel_endcap", "layer_disk", "eta_module", "phi_module"],
        sort=False,
    )["hit_id"]
    .agg(lambda x: list(x))
    .groupby(level=0)
    .agg(lambda x: list(x))
)

true_edges = []
for row in signal_list.values:
    for i, j in zip(row[:-1], row[1:]):
        true_edges.extend(list(itertools.product(i, j)))
```