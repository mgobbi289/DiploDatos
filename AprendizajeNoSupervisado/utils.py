import numpy as np
import matplotlib.pyplot as plt

from matplotlib.cm import nipy_spectral
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, MeanShift


def silhouette_analysis_KM(DF, range_clusters, X, Y):

    X_index = DF.columns.get_loc(X)
    Y_index = DF.columns.get_loc(Y)

    for n_clusters in range_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1.
        ax1.set_xlim([-1, 1])
        # The (n_clusters + 1) * 10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly
        ax1.set_ylim([0, len(DF) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 123 for reproducibility
        clusterer = KMeans(n_clusters=n_clusters, random_state=123)
        cluster_labels = clusterer.fit_predict(DF)

        # The silhouette_score gives the average value for all the samples
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(DF, cluster_labels)
        print(f'Para n_clusters: {n_clusters}, el silhouette_score promedio es: {silhouette_avg}')

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(DF, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]

            y_upper = y_lower + size_cluster_i

            color = nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0,
                              ith_cluster_silhouette_values,
                              facecolor=color,
                              edgecolor=color,
                              alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10 # 10 for the 0 samples

        ax1.set_title('Silhouette Score')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Cluster')

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

        ax1.set_yticks([]) # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # The 2nd subplot showing the actual clusters formed
        colors = nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(DF[X], DF[Y], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, X_index], centers[:, Y_index], marker='o', c='white', alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[X_index], c[Y_index], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

        ax2.set_title('Data Visualization')
        ax2.set_xlabel(X)
        ax2.set_ylabel(Y)

        plt.suptitle((f'Silhouette Analysis for #Clusters = {n_clusters}'), fontsize=14, fontweight='bold')

    plt.show()


def silhouette_analysis_MS(DF, range_BW, X, Y):

    X_index = DF.columns.get_loc(X)
    Y_index = DF.columns.get_loc(Y)

    for bw in range_BW:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Initialize the clusterer with bw value
        clusterer = MeanShift(bandwidth=bw, bin_seeding=True)
        cluster_labels = clusterer.fit_predict(DF)
        n_clusters = max(cluster_labels) + 1

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1.
        ax1.set_xlim([-1, 1])
        # The (n_clusters + 1) * 10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly
        ax1.set_ylim([0, len(DF) + (n_clusters + 1) * 10])

        # The silhouette_score gives the average value for all the samples
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(DF, cluster_labels)
        print(f'Para bandwidth: {bw}, el silhouette_score promedio es: {silhouette_avg}')

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(DF, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]

            y_upper = y_lower + size_cluster_i

            color = nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0,
                              ith_cluster_silhouette_values,
                              facecolor=color,
                              edgecolor=color,
                              alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10 # 10 for the 0 samples

        ax1.set_title('Silhouette Score')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Cluster')

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

        ax1.set_yticks([]) # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # The 2nd subplot showing the actual clusters formed
        colors = nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(DF[X], DF[Y], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, X_index], centers[:, Y_index], marker='o', c='white', alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[X_index], c[Y_index], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

        ax2.set_title('Data Visualization')
        ax2.set_xlabel(X)
        ax2.set_ylabel(Y)

        plt.suptitle((f'Silhouette Analysis for BW = {bw}'), fontsize=14, fontweight='bold')

    plt.show()

