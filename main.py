# spotify_project.py
import os, sys, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")

# --------------------------
# Utility helpers
# --------------------------
def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# --------------------------
# Pipeline
# --------------------------
def pipeline(csv_path, chosen_k=None, save_plots=True, out_dir="outputs", random_state=42):
    safe_mkdir(out_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {csv_path} -> shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # identify columns
    track_col = find_col(df, ["track_name", "track", "name", "title"])
    artist_col = find_col(df, ["track_artist", "artist_name", "artist", "artists"])
    genre_col = find_col(df, ["playlist_genre", "genre", "genres"])
    playlist_col = find_col(df, ["playlist_name", "playlist"])

    print(f"\nUsing columns -> track: {track_col}, artist: {artist_col}, genre: {genre_col}, playlist: {playlist_col}")

    # 1) Preprocessing
    df = df.drop_duplicates().reset_index(drop=True)
    numeric_cols_expected = ["danceability","energy","loudness","speechiness","acousticness",
                             "instrumentalness","liveness","valence","tempo","duration_ms"]
    numeric_cols = [c for c in numeric_cols_expected if c in df.columns]
    if len(numeric_cols) == 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nNumeric features used:", numeric_cols)

    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.dropna(how="all")
    df = df.loc[df_numeric.index].reset_index(drop=True)
    df_numeric = df_numeric.reset_index(drop=True)
    df_numeric = df_numeric.fillna(df_numeric.median())

    scaler = StandardScaler()
    X = scaler.fit_transform(df_numeric.values)

    df.to_csv(os.path.join(out_dir, "spotify_cleaned.csv"), index=False)
    print(f"Saved cleaned dataset: {out_dir}/spotify_cleaned.csv")

    # --------------------------
    # 2) EDA & Visualizations
    # --------------------------
    if save_plots:
        # Histograms
        try:
            df_numeric.hist(figsize=(14,10), bins=30)
            plt.suptitle("Feature Distributions")
            plt.tight_layout(rect=[0,0.03,1,0.95])
            plt.savefig(os.path.join(out_dir, "feature_distributions.png"), dpi=150)
            plt.close()
        except Exception as e:
            print("Could not save distributions:", e)

        # Boxplots
        try:
            fig, ax = plt.subplots(figsize=(14,6))
            df_numeric.plot.box(ax=ax, rot=45)
            plt.title("Feature Boxplots")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "feature_boxplots.png"), dpi=150)
            plt.close()
        except Exception as e:
            print("Could not save boxplots:", e)

        # Correlation heatmap
        try:
            corr = df_numeric.corr()
            plt.figure(figsize=(10,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Correlation Matrix (numeric features)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
            plt.close()
        except Exception as e:
            print("Could not save correlation heatmap:", e)

        print(f"Saved EDA plots to {out_dir}/")

    print("\nSimple insights (feature means):")
    print(df_numeric.mean().round(3).to_string())

    # --------------------------
    # 3) Clustering
    # --------------------------
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Not enough samples to cluster (need >= 2 rows).")

    max_k = min(10, n_samples)
    Ks = list(range(2, max_k + 1))
    inertias, sil_scores = [], []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
        except Exception:
            sil = np.nan
        sil_scores.append(sil)

    if save_plots:
        try:
            fig, ax1 = plt.subplots(figsize=(8,5))
            ax1.plot(Ks, inertias, 'bo-', label='inertia')
            ax1.set_xlabel('k')
            ax1.set_ylabel('Inertia', color='b')
            ax2 = ax1.twinx()
            ax2.plot(Ks, sil_scores, 'r--', label='silhouette')
            ax2.set_ylabel('Silhouette', color='r')
            plt.title('Elbow & Silhouette')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "elbow_silhouette.png"), dpi=150)
            plt.close()
        except Exception as e:
            print("Could not save elbow plot:", e)

    if chosen_k is None:
        valid = [(Ks[i], s) for i,s in enumerate(sil_scores) if not np.isnan(s)]
        chosen_k = max(valid, key=lambda x:x[1])[0] if valid else min(4, n_samples)
    chosen_k = max(2, min(chosen_k, n_samples))
    print(f"\nChosen k = {chosen_k}")

    kmeans = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X)
    df['Cluster'] = labels

    print("\nCluster counts:")
    print(df['Cluster'].value_counts().sort_index().to_string())

    # --------------------------
    # 4) PCA Visualization
    # --------------------------
    try:
        pca = PCA(n_components=2, random_state=random_state)
        pcs = pca.fit_transform(X)
        df['_pc1'], df['_pc2'] = pcs[:,0], pcs[:,1]

        if save_plots:
            # basic PCA plot (color by cluster)
            plt.figure(figsize=(10,7))
            sc = plt.scatter(df['_pc1'], df['_pc2'], c=df['Cluster'], cmap='tab10', alpha=0.7, s=35)
            plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Clusters (PCA 2D)')
            plt.colorbar(sc, label='Cluster')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "clusters_pca.png"), dpi=150)
            plt.close()

            # PCA colored by genre
            if genre_col:
                plt.figure(figsize=(10,7))
                sns.scatterplot(data=df, x='_pc1', y='_pc2', hue=genre_col, style='Cluster', palette='tab10', s=50, alpha=0.7)
                plt.title("PCA: Clusters vs Genres")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "clusters_pca_genre.png"), dpi=150)
                plt.close()

            # PCA colored by playlist
            if playlist_col:
                plt.figure(figsize=(10,7))
                sns.scatterplot(data=df, x='_pc1', y='_pc2', hue=playlist_col, style='Cluster', s=50, alpha=0.7)
                plt.title("PCA: Clusters vs Playlists")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "clusters_pca_playlist.png"), dpi=150)
                plt.close()

    except Exception as e:
        print("PCA or cluster plotting failed:", e)

    # --------------------------
    # 5) Cluster analysis vs Genre & Playlist
    # --------------------------
    if genre_col and genre_col in df.columns:
        crosstab = pd.crosstab(df['Cluster'], df[genre_col])
        crosstab.to_csv(os.path.join(out_dir, "cluster_vs_genre.csv"))
        print(f"\nSaved cluster vs genre crosstab -> {out_dir}/cluster_vs_genre.csv")

        if save_plots:
            crosstab.plot(kind="bar", stacked=True, figsize=(12,6), colormap="tab20")
            plt.title("Clusters vs Playlist Genres")
            plt.xlabel("Cluster"); plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "clusters_vs_genres.png"), dpi=150)
            plt.close()

    if playlist_col and playlist_col in df.columns:
        crosstab = pd.crosstab(df['Cluster'], df[playlist_col])
        crosstab.to_csv(os.path.join(out_dir, "cluster_vs_playlist.csv"))
        print(f"Saved cluster vs playlist crosstab -> {out_dir}/cluster_vs_playlist.csv")

        if save_plots:
            crosstab.plot(kind="bar", stacked=True, figsize=(12,6), colormap="tab20c")
            plt.title("Clusters vs Playlists")
            plt.xlabel("Cluster"); plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "clusters_vs_playlists.png"), dpi=150)
            plt.close()

    # Save dataset with cluster labels
    df.to_csv(os.path.join(out_dir, "spotify_with_clusters.csv"), index=False)
    print(f"\nSaved clustered dataset -> {out_dir}/spotify_with_clusters.csv")

    return df, track_col, artist_col, genre_col, playlist_col, numeric_cols

# --------------------------
# Recommendation utility
# --------------------------
def recommend(df, track_col, artist_col, song_name, n=5):
    if track_col is None or track_col not in df.columns:
        raise ValueError("Track column not found in dataset.")
    matches = df[df[track_col].str.lower() == song_name.lower()]
    if matches.empty:
        candidates = df[df[track_col].str.lower().str.contains(song_name.lower(), na=False)]
        if candidates.empty:
            raise ValueError(f"Song '{song_name}' not found.")
        match = candidates.iloc[0]
        print(f"Using approximate match: {match[track_col]} by {match.get(artist_col, 'unknown')}")
    else:
        match = matches.iloc[0]

    cluster_id = match['Cluster']
    pool = df[df['Cluster'] == cluster_id]
    pool = pool[pool[track_col] != match[track_col]]

    k = min(n, max(1, pool.shape[0]))
    recs = pool.sample(k, replace=False) if pool.shape[0] > 0 else pd.DataFrame(columns=[track_col, artist_col, 'Cluster'])
    cols = [c for c in [track_col, artist_col, 'Cluster'] if c in recs.columns]
    return recs[cols]

# --------------------------
# Main CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotify Genre Segmentation Project")
    parser.add_argument("--csv", type=str, default="spotify_dataset.csv", help="CSV path")
    parser.add_argument("--k", type=int, default=None, help="Optional number of clusters")
    parser.add_argument("--no-plots", action="store_true", help="Disable saving plots")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    save_plots = not args.no_plots

    try:
        df, track_col, artist_col, genre_col, playlist_col, numeric_cols = pipeline(
            args.csv, chosen_k=args.k, save_plots=save_plots, out_dir=args.out
        )
    except Exception as e:
        print("Pipeline error:", e)
        sys.exit(1)

    if track_col:
        try:
            user = input("\nEnter a song name to get recommendations (type 'exit' to quit): ").strip()
            while user.lower() != 'exit':
                try:
                    recs = recommend(df, track_col, artist_col, user, n=5)
                    if recs.empty:
                        print("No recommendations available for this track.")
                    else:
                        print(f"\nRecommendations for '{user}':")
                        print(recs.to_string(index=False))
                except Exception as e:
                    print("Error:", e)
                user = input("\nEnter another song name or 'exit' to quit: ").strip()
        except KeyboardInterrupt:
            print("\nExiting recommendations.")
    else:
        print("No track column detected; cannot run interactive recommendations.")
