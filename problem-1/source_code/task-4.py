import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==========================================
# STEP 1: DEFINE WORD CLUSTERS TO VISUALIZE
# ==========================================
# Replace these with the specific words/categories from your assignment prompt
word_clusters = {
    'Departments': ['cse', 'electrical', 'mechanical', 'biosciences', 'physics', 'mathematics'],
    'Academic Programs': ['btech', 'mtech', 'phd', 'curriculum', 'semester', 'thesis'],
    'Campus Life': ['hostel', 'library', 'sports', 'mess', 'laboratory', 'students']
}

# Colors for the different clusters
colors = ['red', 'blue', 'green']

def plot_embeddings(model, title, method='tsne'):
    """
    Extracts word embeddings, reduces dimensionality, and plots them.
    """
    words = []
    embeddings = []
    cluster_labels = []
    
    # Extract words and their vectors if they exist in the model's vocabulary
    for i, (cluster_name, cluster_words) in enumerate(word_clusters.items()):
        for word in cluster_words:
            if word in model.wv:
                words.append(word)
                embeddings.append(model.wv[word])
                cluster_labels.append(colors[i])
            else:
                print(f"Word '{word}' not in vocabulary. Skipping.")

    if not embeddings:
        print(f"No valid words found in the model for {title}.")
        return

    embeddings = np.array(embeddings)

    # ==========================================
    # STEP 2: DIMENSIONALITY REDUCTION
    # ==========================================
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        # Perplexity must be less than the number of samples
        perplexity_value = min(5, len(embeddings) - 1) 
        reducer = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # ==========================================
    # STEP 3: PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with cluster colors
    for i, (cluster_name, color) in enumerate(zip(word_clusters.keys(), colors)):
        # Plot a dummy point for the legend
        plt.scatter([], [], c=color, label=cluster_name)
        
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, edgecolors='k', s=100)
    
    # Annotate points with the actual words
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_embeddings[i, 0] + 0.5, reduced_embeddings[i, 1] + 0.5), fontsize=10)

    plt.title(f"{title} ({method.upper()} Projection)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ==========================================
# STEP 4: EXECUTE VISUALIZATIONS
# ==========================================
# NOTE: Make sure model_cbow and model_sg are already trained and in memory!

print("Generating t-SNE for CBOW...")
# plot_embeddings(model_cbow, "CBOW Word Embeddings", method='tsne')

print("Generating t-SNE for Skip-gram...")
# plot_embeddings(model_sg, "Skip-gram Word Embeddings", method='tsne')

# You can also change method='tsne' to method='pca' to see the linear projection!