import numpy as np
import joblib
from tqdm import tqdm
from main import *

args = arg_parser()
n_train, n_test, n_components, d = args.n_train, args.n_test, args.n_components, args.patch_size
folder_name = 'results/n_components=%d,n_train=%d,patch_size=%d/' % (n_components, n_train, d)

def mmse(model, A, sigma = 1e-5):
    cost = 0
    var_noise = sigma * np.eye(A.shape[0])
    for j in range(model.means_.shape[0]):
        var_j = model.covariances_[j]
        m_j = np.trace(var_j - var_j @ A.T @ np.linalg.inv(A @ var_j @ A.T + var_noise) @ A @ var_j)
        cost += model.weights_[j] * m_j
    return cost

def optimize_mat(model, A_):
    A = A_.copy()
    costs = []
    for it in tqdm(range(20)):
        cnt = 0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                # flip the bit only if mmse is reduced
                cost = mmse(model, A)
                A[i, j] = 1 - A[i, j]
                cnt += 1
                if cost <= mmse(model, A):
                    A[i, j] = 1 - A[i, j]
                    cnt -= 1
        # break the loop if no improvement
        if cnt == 0:
            break
        costs.append(mmse(model, A))
    return A

def calc_psnr(model, A, test_data):
    reconstruction = np.empty(test_data.shape)
    for j in range(len(test_data)):
        x = test_data[j]
        x_hat = x.copy()
        for i in range(x.shape[0]//d):
            y = A @ x[i*d:(i+1)*d]
            x_hat[i*d:(i+1)*d] = decode(model, A, y)
        reconstruction[j] = x_hat
    val, _, _, _ = PSNR(test_data, reconstruction)
    return val

# Load the model from the file
model = joblib.load(folder_name + 'model.pkl')
test_data = load_test_ecg(num=n_test, patch_size=d)

folder_name += 'mat/'

if os.path.isdir(folder_name):
    for f in os.listdir(folder_name):
        continue
        if not os.path.isdir(folder_name + f):
            os.remove(folder_name + f)
else:
    os.makedirs(folder_name)

for _ in range(1):
    A = np.random.binomial(1, 0.5, size=(d, d))
    # B = optimize_mat(model, A)
    ms = [d*(i+1)//10 for i in range(10)]
    # ms = [10, 20]
    psnr1 = []
    psnr2 = []

    for m in ms:
        psnr1.append(calc_psnr(model, A[:m], test_data))
        B = optimize_mat(model, A[:m])
        np.save(folder_name + '%d.npy' % m, B)
        psnr2.append(calc_psnr(model, B, test_data))
    
    print(psnr1)
    print(psnr2)