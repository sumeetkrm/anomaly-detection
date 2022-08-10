import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from main import *
from sklearn.decomposition import PCA

args = arg_parser()
n_train, n_components, d = args.n_train, args.n_components, args.patch_size
folder_name = 'results/n_components=%d,n_train=%d,patch_size=%d/' % (n_components, n_train, d)

train_data = load_train_ecg(num=n_train, patch_size=d, class_id=0)

pca = PCA(n_components=n_components)
pca.fit(train_data)
var = pca.explained_variance_ratio_

print(var)
plot_data([x for x in range(n_components)], var, path=folder_name+'pca.png', ylabel='var')