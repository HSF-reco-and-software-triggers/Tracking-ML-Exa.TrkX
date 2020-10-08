import time
import numpy as np
import pandas as pd
import logging

#####################################################
#                   UTILD PANDAS                    #
#####################################################
def select_min(test_val, current_val):
    return min(test_val, current_val)

def select_max(test_val, current_val):
    if current_val == -1:
        return test_val
    else:
        return max(test_val, current_val)

def find_ch0_min(cells_in, nb_hits):
    cell_idx = cells_in.index.values.reshape(-1,1)
    cells = cells_in[['hit_id', 'ch0']].values
    where_min = find_ch0_property(cells, nb_hits, select_min, 10**8)
    return where_min

def find_ch0_max(cells_in, nb_hits):
    cells = cells_in[['hit_id', 'ch0']].values
    where_max = find_ch0_property(cells, nb_hits, select_max, -10**8)
    return where_max

def find_ch0_property(cells, nb_hits, comparator, init_val):
    nb_cells = cells.shape[0]
    cells = sort_cells_by_hit_id(cells)

    hit_property = [init_val] * nb_hits
    cell_property = [0] * nb_cells
    cell_values = cells[:,2].tolist()
    hit_ids = cells[:,1].tolist()

    hit_property_id = 0
    current_hit_id = hit_ids[0]
    for i, (h, v) in enumerate(zip(hit_ids, cell_values)):
        if h > current_hit_id:
            hit_property_id += 1
            current_hit_id = h
        hit_property[hit_property_id] = comparator(v, hit_property[hit_property_id])

    hit_property_id = 0
    current_hit_id = hit_ids[0]
    for i, (h, v) in enumerate(zip(hit_ids, cell_values)):
        if h > current_hit_id:
            hit_property_id += 1
            current_hit_id = h
        if v == hit_property[hit_property_id]:
            cell_property[i] = 1

    original_order = np.argsort(cells[:,0])
    cell_property = np.array(cell_property, dtype=bool)[original_order]
    return cell_property

def sort_cells_by_hit_id(cells):
    orig_order = np.arange(len(cells)).reshape(-1,1)
    cells = np.concatenate((orig_order, cells),1)
    sort_idx = np.argsort(cells[:,1]) # Sort by hit ID
    cells = cells[sort_idx]
    return cells

#################################################
#                   EXTRACT DIR                 #
#################################################

#####################################################
#                   UTILD PANDAS                    #
#####################################################
def select_min(test_val, current_val):
    return min(test_val, current_val)

def select_max(test_val, current_val):
    if current_val == -1:
        return test_val
    else:
        return max(test_val, current_val)

def find_ch0_min(cells_in, nb_hits):
    cell_idx = cells_in.index.values.reshape(-1,1)
    cells = cells_in[['hit_id', 'ch0']].values
    where_min = find_ch0_property(cells, nb_hits, select_min, 10**8)
    return where_min

def find_ch0_max(cells_in, nb_hits):
    cells = cells_in[['hit_id', 'ch0']].values
    where_max = find_ch0_property(cells, nb_hits, select_max, -10**8)
    return where_max

def find_ch0_property(cells, nb_hits, comparator, init_val):
    nb_cells = cells.shape[0]
    cells = sort_cells_by_hit_id(cells)

    hit_property = [init_val] * nb_hits
    cell_property = [0] * nb_cells
    cell_values = cells[:,2].tolist()
    hit_ids = cells[:,1].tolist()

    hit_property_id = 0
    current_hit_id = hit_ids[0]
    for i, (h, v) in enumerate(zip(hit_ids, cell_values)):
        if h > current_hit_id:
            hit_property_id += 1
            current_hit_id = h
        hit_property[hit_property_id] = comparator(v, hit_property[hit_property_id])

    hit_property_id = 0
    current_hit_id = hit_ids[0]
    for i, (h, v) in enumerate(zip(hit_ids, cell_values)):
        if h > current_hit_id:
            hit_property_id += 1
            current_hit_id = h
        if v == hit_property[hit_property_id]:
            cell_property[i] = 1

    original_order = np.argsort(cells[:,0])
    cell_property = np.array(cell_property, dtype=bool)[original_order]
    return cell_property

def sort_cells_by_hit_id(cells):
    orig_order = np.arange(len(cells)).reshape(-1,1)
    cells = np.concatenate((orig_order, cells),1)
    sort_idx = np.argsort(cells[:,1]) # Sort by hit ID
    cells = cells[sort_idx]
    return cells

#################################################
#                   EXTRACT DIR                 #
#################################################

def local_angle(cell, module):
    n_u = max(cell['ch0']) - min(cell['ch0']) + 1
    n_v = max(cell['ch1']) - min(cell['ch1']) + 1
    l_u = n_u * module.pitch_u.values   # x
    l_v = n_v * module.pitch_v.values   # y
    l_w = 2   * module.module_t.values  # z
    return (l_u, l_v, l_w)


def extract_rotation_matrix(module):
    rot_matrix = np.matrix( [[ module.rot_xu.values[0], module.rot_xv.values[0], module.rot_xw.values[0]],
                             [  module.rot_yu.values[0], module.rot_yv.values[0], module.rot_yw.values[0]],
                             [  module.rot_zu.values[0], module.rot_zv.values[0], module.rot_zw.values[0]]])
    return rot_matrix, np.linalg.inv(rot_matrix)


def cartesion_to_spherical(x, y, z):
    r3 = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z/r3)
    return r3, theta, phi

def theta_to_eta(theta):
    return -np.log(np.tan(0.5*theta))

def extract_dir_old(hits, cells, detector):
    angles = []

#     for ii in range(hits.shape[0]):
#         if (ii%5000)==0:
#             print(ii)
#         hit = hits.iloc[ii]
#         cell = cells[cells.hit_id == hit.hit_id]
#         module = detector[(detector.volume_id == hit.volume_id)
#                                 & (detector.layer_id == hit.layer_id)
#                                 & (detector.module_id == hit.module_id)]

    cells_by_hit = cells.groupby('hit_id')
    detector_by_module = detector.groupby(['volume_id', 'layer_id', 'module_id'])

    for hit in hits.itertuples():

        if (hit.Index%5000)==0:
            print("{} out of {}".format(hit.Index,hits.shape[0]))
        cell = cells_by_hit.get_group(hit.hit_id)
        module = detector_by_module.get_group(hit[5:8])

        l_x, l_y, l_z = local_angle(cell, module)
        # convert to global coordinates
        module_matrix, module_matrix_inv = extract_rotation_matrix(module)
        g_matrix = module_matrix * [l_x, l_y, l_z]
        _, g_theta, g_phi = cartesion_to_spherical(g_matrix[0][0], g_matrix[1][0], g_matrix[2][0])
        _, l_theta, l_phi = cartesion_to_spherical(l_x[0], l_y[0], l_z[0])
        # to eta and phi...
        l_eta = theta_to_eta(l_theta)
        g_eta = theta_to_eta(g_theta[0, 0])
        lx, ly, lz = l_x[0], l_y[0], l_z[0]
        angles.append([int(hit.hit_id), l_eta, l_phi, lx, ly, lz, g_eta, g_phi[0, 0]])
    df_angles = pd.DataFrame(angles, columns=['hit_id', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi'])
    hits = hits.merge(df_angles, on='hit_id', how='left')
    return hits

def get_all_local_angles(hits, cells, detector):
    direction_count_u = cells.groupby(['hit_id']).ch0.agg(['min', 'max'])
    direction_count_v = cells.groupby(['hit_id']).ch1.agg(['min', 'max'])
    nb_u = direction_count_u['max'] - direction_count_u['min'] + 1
    nb_v = direction_count_v['max'] - direction_count_v['min'] + 1

    vols = hits['volume_id'].values
    layers = hits['layer_id'].values
    modules = hits['module_id'].values

    pitch = detector['pixel_size']
    thickness = detector['thicknesses']

    pitch_cells = pitch[vols, layers, modules]
    thickness_cells = thickness[vols, layers, modules]

    l_u = nb_u * pitch_cells[:,0]
    l_v = nb_v * pitch_cells[:,1]
    l_w = 2*thickness_cells
    return l_u, l_v, l_w

def get_all_rotated(hits, detector, l_u, l_v, l_w):
    vols = hits['volume_id'].values
    layers = hits['layer_id'].values
    modules = hits['module_id'].values
    rotations = detector['rotations']
    rotations_hits = rotations[vols, layers, modules]
    u = l_u.values.reshape(-1,1)
    v = l_v.values.reshape(-1,1)
    w = l_w.reshape(-1,1)
    dirs = np.concatenate((u,v,w),axis=1)

    dirs = np.expand_dims(dirs, axis=2)
    vecRot = np.matmul(rotations_hits, dirs).squeeze(2)
    return vecRot

def extract_dir_new(hits, cells, detector):
    l_u, l_v, l_w = get_all_local_angles(hits, cells, detector)
    g_matrix_all = get_all_rotated(hits, detector, l_u, l_v, l_w)
    hit_ids = hits['hit_id'].values.tolist()

    l_u = l_u.values.tolist()
    l_v = l_v.values.tolist()
    l_w = l_w.tolist()
    angles = []
    for ii in range(hits.shape[0]):
        #if ((ii+1)%50000)==0:
        #    print(ii)
        l_x = [l_u[ii]]
        l_y = [l_v[ii]]
        l_z = [l_w[ii]]
        # convert to global coordinates
        g_matrix = g_matrix_all[ii].reshape(-1,1)
        g_matrix = np.matrix(g_matrix)
        _, g_theta, g_phi = cartesion_to_spherical(g_matrix[0][0], g_matrix[1][0], g_matrix[2][0])
        _, l_theta, l_phi = cartesion_to_spherical(l_x[0], l_y[0], l_z[0])
        # to eta and phi...
        l_eta = theta_to_eta(l_theta)
        g_eta = theta_to_eta(g_theta[0, 0])
        lx, ly, lz = l_x[0], l_y[0], l_z[0]
        angles.append([int(hit_ids[ii]), l_eta, l_phi, lx, ly, lz, g_eta, g_phi[0, 0]])
    df_angles = pd.DataFrame(angles, columns=['hit_id', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi'])
    hits = hits.merge(df_angles, on='hit_id', how='left')
    return hits

def check_diff(h1, h2, name):
    n1 = h1[name].values
    n2 = h2[name].values
    print(name, max(np.absolute(n1-n2)))

def extract_dir(hits, cells, detector_orig, detector_proc):
    # print(cells.keys())
    # print(hits.shape, cells.shape)
    # hits_subset = hits[:100]
    # cells_subset = cells.loc[cells['hit_id'].isin(hits_subset['hit_id'])]
    # print(hits_subset.shape, cells_subset.shape, '\n')
    # t0 = time.time()
    # # h1 = extract_dir_old(hits_subset, cells, detector_orig)
    # h1 = extract_dir_old(hits, cells, detector_orig)
    # t1 = time.time()
    # print("\nnb2\n")
    # # h2 = extract_dir_new(hits_subset, cells_subset, detector_proc)
    # h2 = extract_dir_new(hits, cells, detector_proc)
    # t2 = time.time()
    # print("{:8.3f}s for old".format(t1-t0))
    # print("{:8.3f}s for new".format(t2-t1))
    # for n in ['x', 'y', 'z', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']:
    #     check_diff(h1, h2, n)
    # # eta1 = h1['geta'].values
    # # eta2 = h2['geta'].values
    # # print(max(eta1-eta2))
    # print("made it to end")
    # print(h1[:10])
    # print(h2[:10])
    # exit()

    return extract_dir_new(hits, cells, detector_proc)
    # return extract_dir_old(hits, cells, detector_orig)
