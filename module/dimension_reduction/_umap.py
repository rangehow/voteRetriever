import umap.umap_ as umap
import matplotlib.pyplot as plt

class UMAP:
    def __init__(self,n_components=2):
        self.n_components = n_components
        self.umap = umap.UMAP(n_components=n_components)

    def fit_transform(self,data):
        return self.umap.fit_transform(data)

    def plot(self,data,labels):
        """_summary_

        Args:
            data (_type_): _description_
            labels (_type_): 
        """
        embedding = self.fit_transform(data)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
        plt.colorbar()
        plt.savefig("umap.png")

if __name__ == "__main__":
    import numpy as np
    data = np.random.rand(100, 10)
    labels = np.random.randint(0, 3, 100)
    umap = UMAP()
    umap.plot(data, labels)
