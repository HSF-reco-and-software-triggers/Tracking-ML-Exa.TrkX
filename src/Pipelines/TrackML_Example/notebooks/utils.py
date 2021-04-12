import os


def get_best_run(run_label, wandb_save_dir):
    for (root_dir, dirs, files) in os.walk(wandb_save_dir + "/wandb"):
        if run_label in dirs:
            run_root = root_dir

    best_run_base = os.path.join(run_root, run_label, "checkpoints")
    best_run = os.listdir(best_run_base)
    best_run_path = os.path.join(best_run_base, best_run[0])

    return best_run_path


def evaluate_set_metrics(r, dataset, model, chkpnt, test_size):
    cluster_total_positive, cluster_total_true, cluster_total_true_positive = 0, 0, 0
    for i, batch in enumerate(test_dataset[:test_size]):
        data = batch.to(device)
        if "ci" in chkpnt["hyper_parameters"]["regime"]:
            spatial = model(torch.cat([data.cell_data, data.x], axis=-1))
        else:
            spatial = model(data.x)
        e_spatial = build_edges(spatial, r, 1024, res)
        e_bidir = torch.cat(
            [
                batch.layerless_true_edges.to(device),
                torch.stack(
                    [batch.layerless_true_edges[1], batch.layerless_true_edges[0]],
                    axis=1,
                ).T.to(device),
            ],
            axis=-1,
        )
        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=1, reduction="mean"
        )
        #         print("Loss:", loss.item())
        #         total_loss += loss.item()

        # Cluster performance
        cluster_true = 2 * len(batch.layerless_true_edges[0])
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        cluster_total_true_positive += cluster_true_positive
        cluster_total_positive += cluster_positive
        cluster_total_true += cluster_true
    #         if i % 5 == 0:
    #             print(i+1, "validated")

    cluster_eff = cluster_total_true_positive / max(cluster_total_true, 1)
    cluster_pur = cluster_total_true_positive / max(cluster_total_positive, 1)

    return cluster_eff, cluster_pur


def evaluate_set_root(r, dataset, model, chkpnt, test_size=1, goal=0.96, fom="eff"):
    eff, pur = evaluate_set_metrics(r, dataset, model, chkpnt, test_size)

    if fom == "eff":
        return eff - goal

    elif fom == "pur":
        return pur - goal


def model_evaluation(model, test_dataset, test_size=10, fom="eff", fixed_value=0.96):

    # Seed solver with one batch, then run on full test dataset
    sol = root(
        evaluate_set_root,
        args=(test_dataset, model, chkpnt, 1, fixed_value, fom),
        x0=0.5,
        x1=1.0,
    )
    print("Seed solver complete, radius:", sol.root)
    sol = root(
        evaluate_set_root,
        args=(test_dataset, model, chkpnt, test_size, fixed_value, fom),
        x0=sol.root,
        x1=sol.root * 0.9,
    )
    print("Final solver complete, radius:", sol.root)

    # Return ( (efficiency, purity), radius_size)
    return (
        evaluate_set_metrics(sol.root, test_dataset, model, chkpnt, test_size),
        sol.root,
    )
