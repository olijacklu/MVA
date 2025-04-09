import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_layer_weights(models_by_language, title="Layer Weight Analysis", save_path=None):
    languages = list(models_by_language.keys())
    num_layers = len(next(iter(models_by_language.values())).layer_weights)

    weights_matrix = np.zeros((len(languages), num_layers))

    for i, (lang, model) in enumerate(models_by_language.items()):
        weights_matrix[i] = model.layer_weights.detach().cpu().numpy()

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(weights_matrix, cmap="YlOrRd",
                     xticklabels=range(num_layers),
                     yticklabels=languages)

    plt.xlabel('Layer')
    plt.ylabel('Language')
    plt.title(f"Distribution of Layer Weights By Language")

    cbar = ax.collections[0].colorbar

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')

    plt.show()

    return weights_matrix
