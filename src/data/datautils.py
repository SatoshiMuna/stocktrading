import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
    
def visualize_pca_of_stockprice(tensor_inputs, tensor_targets, dim=2):
    # Close price
    close = [ti[:,3].numpy().copy() for ti in tensor_inputs]
    category = [tt.numpy().copy() for tt in tensor_targets[1]]
    t = [0 if c[0]==1 else 1 for c in category]
    # PCA
    pca = PCA()
    pca.fit(close)
    # Get top-'dim' scores of each component
    score = pd.DataFrame(pca.transform(close))
    s = score.iloc[:,0:dim].assign(cls=t)
    s.columns = ['x','y','cls'] if dim == 2 else ['x','y','z','cls']
    print(s)
    print(pca.explained_variance_ratio_)

    fig, ax = plt.subplots()
    groups = s.groupby('cls')
    if dim >= 3:
        ax = fig.add_subplot(projection='3d')
        for name, group in groups:
            ax.plot(group.x, group.y, group.z, marker='o', linestyle='', ms=8, label=name)
    elif dim == 2:
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, label=name)
    ax.legend()
    plt.show()