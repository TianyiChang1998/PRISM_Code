from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import numpy as np
import os


def apply_gmm(reduced_features, num_clusters, means_init=None):
    gmm = GMM(n_components=num_clusters, covariance_type="diag", means_init=means_init)
    gmm_clusters = gmm.fit(reduced_features)
    labels = gmm_clusters.predict(reduced_features)
    return gmm_clusters, labels


def GMM_by_layer(
    intensity_fra,
    G_layer,
    num_per_layer,
    channel_to_use,
    centroid_init_dict,
):
    GMM_dict = dict()
    intensity_fra["label"] = [-1] * len(intensity_fra)

    for layer in range(G_layer):
        centroids_init = centroid_init_dict[layer]
        filtered_data = intensity_fra[intensity_fra["G_layer"] == layer]
        reduced_features = filtered_data[channel_to_use]
        gmm, gmm_labels = apply_gmm(
            reduced_features,
            num_clusters=len(centroids_init),
            means_init=centroids_init,
        )
        GMM_dict[layer] = gmm
        intensity_fra.loc[filtered_data.index, "label"] = gmm_labels + int(
            layer * num_per_layer + 1
        )

    return intensity_fra, GMM_dict


def GMM_visualization(
    intensity_fra,
    G_layer,
    num_per_layer,
    GMM_dict,
    centroid_init_dict,
    RYB_xy_transform,
    out_path_dir,
):
    s = 1 / np.log2(len(intensity_fra))
    alpha = 1 / np.log2(len(intensity_fra))

    fig, ax = plt.subplots(nrows=1, ncols=G_layer, figsize=(5.5 * G_layer, 5))
    for layer in range(G_layer):
        ax_tmp = ax if G_layer < 2 else ax[layer]

        data = intensity_fra[intensity_fra["G_layer"] == layer]
        data = data[data["label"] != -1]
        gmm = GMM_dict[layer]

        ax_tmp.scatter(
            data["X_coor_gaussian"],
            data["Y_coor_gaussian"],
            c=data["label"],
            marker=".",
            alpha=alpha,
            s=s,
        )
        centroids = gmm.means_ @ RYB_xy_transform

        centroid_init = centroid_init_dict[layer] @ RYB_xy_transform
        ax_tmp.scatter(
            centroid_init[:, 0], centroid_init[:, 1], color="cyan", s=1.5, alpha=0.7
        )

        for i, centroid in enumerate(centroids):
            ax_tmp.text(
                centroid[0],
                centroid[1],
                i + int(layer * num_per_layer + 1),
                fontsize=12,
                color="black",
                ha="center",
                va="center",
            )
        ax_tmp.scatter(centroids[:, 0], centroids[:, 1], color="r", s=1.5, alpha=0.7)
        ax_tmp.set_xlim([-1, 1])
        ax_tmp.set_ylim([-1, 1.5])

        ax_tmp.set_title(f"G={layer}")

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_path_dir, "mousebrain_scatter_GMM_cluster_by_layer.jpg"),
        bbox_inches="tight",
        # dpi=300,
    )

    for i, ax in enumerate(ax.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        plt.savefig(os.path.join(out_path_dir, rf"layer{i+1}.jpg"), bbox_inches=bbox)

    plt.close(fig)


def visualization(intensity_fra,
                  G_layer=2,
                  num_per_layer=15,
                  out_path_dir='./',):
    s = 1 / np.log2(len(intensity_fra))
    alpha = 1 / np.log2(len(intensity_fra))

    fig, ax = plt.subplots(nrows=1, ncols=G_layer, figsize=(5.5 * G_layer, 5))
    for layer in range(G_layer):
        data = intensity_fra[intensity_fra['G_layer'] == layer]

        ax_tmp = ax if G_layer < 2 else ax[layer]
        ax_tmp.scatter(data['X_coor_gaussian'], data['Y_coor_gaussian'], c=data['label'], marker='.', alpha=alpha, s=s,)
        ax_tmp.set_xlim([-1, 1])
        ax_tmp.set_ylim([-1, 1.5])

        for i in range(1 + layer * num_per_layer, 1 + (layer+1) * num_per_layer):
            data_tmp = data[data['label']==i]
            cen_tmp = np.mean(data_tmp[['X_coor_gaussian', 'Y_coor_gaussian']], axis=0)
            ax_tmp.text(cen_tmp[0], cen_tmp[1], i, fontsize=12, color='black', ha='center', va='center')

        ax_tmp.set_title(f'G={layer}')
    plt.tight_layout()

    plt.savefig(out_path_dir, bbox_inches="tight")
    plt.close(fig)