# Data & Graph Manipulation

## Data Processing

### Constructing Truth Graph

The purpose of a "truth graph" is explained in [Truth Definitions](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/performance/truth_definitions/). Here we explain how it is constructed.

1. Begin with a set of hit IDs (HIDs) with associated particle IDs (PIDs). For ITk, this set **has duplicated HIDs**. This is because two particles may leave a hit in a very coincident space on a module. In that case, we cannot be sure whether the hit is from a single particle, or from many - it is called "merged" in the latter case. We only know a hit is "merged" from ground truth. Therefore, there is not a unique map from HIDs to PIDs, and we may have many PIDs associated with a single HID. We preserve this many-many map in the input dataframe, in order to construct the truth graph.
2. Calculate the distance of each hit from its creation vertex, and order the entire set by this distance. We now have sequential ordering.
3. Group the entries of this set by PID and module ID. The module ID is whatever combination of identifiers gives a unique module. In the case of ITk, this is `[barrel_endcap, layer_disk, eta_module, phi_module]`. In the case of TrackML, this is `[volume_id, layer_id, module_id]`. 
4. For each particle, iterate through pairs in the list of hits. Since we group by modules, this list is of the form `[[0], [1], [2, 3], [4], ...]`, such that where there is more than one entry we take a `product` of the combinations. E.g. `[1] x [2, 3] = [1, 2], [1, 3]`.
5. Concatenate all these pairs into an edge list for the full truth graph.
6. **Throw away duplicates of HIDs**. This is a hack and liable to change. This makes referencing hits very simple, since now there is only one hit ID per node, and only one PID per hit. This contradicts our knowledge of merged hits, but since only ~0.2% of hits are merged, we accept this approximation for now.


## Graph Processing

### Graph Intersection

Consider a "prediction graph" and a "target graph". How do we compare the two? The simplest way is to treat each edge as a prediction, and see which edges overlap (true positives) and which do not (false positives). This can be done very efficiently with sparse arrays.

1. Convert predicted `e_p` and truth `e_t` edge lists into sparse arrays, with either `scipy.sparse` or `cupy.sparse`
2. Define an intersection array by
```
intersection = e_p.multiply(e_t) - ((e_p - e_t) > 0)
```
This is simple boolean logic: The first value gives 1 for true positives, and 0 otherwise. The second value gives 1 for false positives, and 0 otherwise. Combining them gives us a sparse array of 1 (for true positives) and -1 (for false positives). Thus when we convert back to a (2, N) edge list, we take `intersection > 0` to be a true edge, and `intersection < 0` to be a fake edge.