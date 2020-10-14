# import faiss
# res = faiss.StandardGpuResources()
import torch
import scipy as sp
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def graph_intersection(pred_graph, truth_graph):
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    l1 = pred_graph.cpu().numpy()
    l2 = truth_graph.cpu().numpy()
    e_1 = sp.sparse.coo_matrix((np.ones(l1.shape[1]), l1), shape=(array_size, array_size)).tocsr()
    e_2 = sp.sparse.coo_matrix((np.ones(l2.shape[1]), l2), shape=(array_size, array_size)).tocsr()
    e_intersection = (e_1.multiply(e_2) - ((e_1 - e_2)>0)).tocoo()

    new_pred_graph = torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col])).long().to(device)
    y = e_intersection.data > 0

    return new_pred_graph, y

def build_edges(spatial, r_max, k_max, res, return_indices=False):

    index_flat = faiss.IndexFlatL2(spatial.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    spatial_np = spatial.cpu().detach().numpy()
    gpu_index_flat.add(spatial_np)

    D, I = search_index_pytorch(gpu_index_flat, spatial, k_max)

    D, I = D[:,1:], I[:,1:]
    ind = torch.Tensor.repeat(torch.arange(I.shape[0]), (I.shape[1], 1), 1).T.to(device)
    edge_list = torch.stack([ind[D <= r_max**2], I[D <= r_max**2]])

    if return_indices:
        return edge_list, D, I, ind
    else:
        return edge_list


def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I

def stringlist_to_classes(stringlist):
    return [getattr(sys.modules[__name__], class_string)() for class_string in stringlist]