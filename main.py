import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
from tqdm import tqdm
import pandas as pd
import argparse
import os

max_signal_length = 100

def arg_parser():
    parser = argparse.ArgumentParser(description='compressed sensing using GMMs')
    parser.add_argument('--n_train', default=10000, type=int, help='num of training samples')
    parser.add_argument('--train', default=False, type=bool, help='to train or not')
    parser.add_argument('--matrix', default=False, type=bool, help='use designed matrix')
    parser.add_argument('--n_test', default=1000, type=int, help='num of testing samples')
    parser.add_argument('--n_components', default=10, type=int, help='num of mixture components')
    parser.add_argument('--test_class', default=-1, type=int, help='test class id')
    parser.add_argument('--n_init', default=10, type=int, help='num of initializations')
    parser.add_argument('--patch_size', default=60, type=int, help='patch size')
    parser.add_argument('--num_peaks', default=2, type=int, help='num of peaks')
    parser.add_argument('--peak_width', default=5, type=int, help='peak width')
    return parser.parse_args()

def weighted_l2(v, mat):
    if np.linalg.matrix_rank(mat) < mat.shape[0]:
        # handles the case where the matrix is not invertible
        m = v.T @ np.linalg.pinv(mat) @ v
    else:
        m = v.T @ np.linalg.inv(mat) @ v
    return m

def decode(model, A, y, sigma = 1e-7):
    x_hat = np.empty(model.means_.shape)
    cost = []
    var_noise = sigma * np.eye(A.shape[0])

    for j in range(model.means_.shape[0]):
        var_j, mu_j = model.covariances_[j], model.means_[j]
        x_hat_j = var_j @ A.T @ np.linalg.inv(A @ var_j @ A.T + var_noise) @ (y - A @ mu_j) + mu_j
        # print(np.linalg.det(var_j))
        # print(var_j)
        cost_j = weighted_l2(y - A @ x_hat_j, var_noise) + weighted_l2(x_hat_j - mu_j, var_j) + np.log(np.linalg.det(var_j))
        # print(np.linalg.det(var_j))
        x_hat[j] = x_hat_j
        cost.append(cost_j)

    j = np.argmin(cost)
    # print(j)
    return x_hat[j]

def PSNR(x_, x_t):
    assert x_.shape == x_t.shape
    peak = np.max(x_) - np.min(x_)
    mse = np.mean(((x_ - x_t)/peak)**2, axis=1)
    psnr = -10*np.log10(mse)
    return np.mean(psnr), np.min(psnr), np.max(psnr), np.std(psnr)

def plot_data(xs, ys, path = 'PSNR.png', ylabel='Avg PSNR'):
    plt.plot(xs, ys, marker="o")
    plt.xlabel('# of measurements')
    plt.ylabel(ylabel)
    plt.grid(True)
    for x,y in zip(xs,ys):
        label = f"({x},{round(y,2)})"
        plt.annotate(label, (x,y), textcoords="offset points", xytext=(-5,10), ha='center')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# dell
def load_housing(path='data/california_housing_train.csv', num=1000, patch_size=20):
    repeat_times = 10
    data = np.genfromtxt(path, delimiter=',' , usecols=7, skip_header=1)
    data = data/np.amax(data) #normalize
    data = np.repeat(data, repeat_times) #allow repeats
    data = data[:data.shape[0]//patch_size*patch_size]
    data = data.reshape(-1, patch_size)
    np.random.shuffle(data)
    data = data[:num]
    return data

# dell
def load_ecg(path='data/ecg_small.csv', num=1000, patch_size=20):
    data = pd.read_csv(path)
    data = data.dropna(axis = 0, how ='any')
    data = (data-data.min())/(data.max()-data.min())
    data = data.to_numpy()
    data = data[:data.shape[0]//patch_size*patch_size]
    data = data.reshape(-1, patch_size, 4).T.reshape(4*patch_size, -1).T
    np.random.shuffle(data)
    data = data[:num]
    return data

def load_accel(path='data/accelerometer-1/trace3/Measurer.csv', num=1000, patch_size=20):
    data = pd.read_csv(path).iloc[:, 1:4].to_numpy()
    data = data[:data.shape[0]//patch_size*patch_size]
    data = data.reshape(-1, patch_size, 3).T.reshape(3*patch_size, -1).T
    # np.random.shuffle(data)
    # data = data[:num]
    data = data[-num:]
    return data

def load_train_ecg(path='data/ecg/mitbih_train.csv', num=10000, patch_size=20, class_id=-1):
    data = pd.read_csv(path, header=None)
    # select examples of only 1 class
    if class_id >= 0:
        data = data.loc[data[187] == class_id]
    data = np.array(data.values)
    data = data[:,:max_signal_length//patch_size*patch_size]
    data = data.reshape(-1, patch_size)
    np.random.shuffle(data)
    data = data[:num]
    # data = 2*data-1
    return data

def load_test_ecg(path='data/ecg/mitbih_test.csv', num=1000, patch_size=20, class_id=-1):
    data = pd.read_csv(path, header=None)
    # select examples of only 1 class
    if class_id >= 0:
        data = data.loc[data[187] == class_id]
    data = np.array(data.values)
    data = data[:,:max_signal_length//patch_size*patch_size]
    # np.random.seed(2)
    np.random.shuffle(data)
    data = data[:num]
    # data = 2*data-1
    return data

def create_pulse(num=100, signal_length=20, peak_width=5, num_peaks=2):
    def check(arr):
        return (np.array([arr[i+1]-arr[i] for i in range(len(arr)-1)]) >= peak_width).all()
    data = np.zeros((num, signal_length))
    for i in tqdm(range(num)):
        n = np.random.randint(2, 1+num_peaks)
        l = sorted(np.random.choice([signal_length//num_peaks*i for i in range(num_peaks)], n, replace=False))
        # l = sorted(np.random.choice([peak_width*i for i in range(signal_length//peak_width)], n, replace=False))
        # l = sorted(np.random.randint(0, signal_length-peak_width+1, n))
        while(not check(l)):
            l = sorted(np.random.randint(0, signal_length-peak_width+1, n))
        for k in l:
            # data[i,k:k+np.random.randint(2,peak_width+1)] = np.random.uniform(0,1)
            # data[i,k:k+peak_width] = 1
            data[i,k:k+peak_width] = np.random.uniform(0,1)
    return data


if __name__ == "__main__":
    
    args = arg_parser()
    n_train, n_test, n_components, cnt = args.n_train, args.n_test, args.n_components, 5
    n_init, d, to_train, use_mat = args.n_init, args.patch_size, args.train, args.matrix
    folder_name = 'results/n_components=%d,n_train=%d,patch_size=%d/' % (n_components, n_train, d)

    # train_data = create_pulse(num=n_train, signal_length=d, peak_width=10, num_peaks=d//10)
    train_data = load_train_ecg(num=n_train, patch_size=d)
    # train_data = load_accel(num=n_train, patch_size=d)
    # train_data = load_ecg(num=n_train, patch_size=d)
    # train_data = load_housing(num=n_train, patch_size=d)
    print(train_data.shape)
    # for i in range(3):
    #     plt.plot(train_data[i])
    #     plt.show()
    #     plt.savefig("train_%d.png" % i)
    #     plt.close()

    if to_train:
        model = GaussianMixture(n_components=n_components, n_init=n_init, verbose=1, max_iter=200, init_params='random')
        model.fit(train_data)
        
        if os.path.isdir(folder_name):
            for f in os.listdir(folder_name):
                if not os.path.isdir(folder_name + f):
                    os.remove(folder_name + f)
        else:
            os.makedirs(folder_name)

        # Save the model as a pickle in a file
        joblib.dump(model, folder_name + 'model.pkl')

    else:
        # Load the model from the file
        model = joblib.load(folder_name + 'model.pkl')

    for var in model.covariances_:
        print(np.linalg.det(var), np.linalg.matrix_rank(var))
        # print(np.linalg.det(np.linalg.inv(var)))
    for var in model.weights_:
        print(var)
    # exit()

    # test_data = create_pulse(num=n_test, signal_length=d, peak_width=10, num_peaks=d//10)
    test_data = load_test_ecg(num=n_test, patch_size=d)
    # test_data = load_accel(num=n_test, patch_size=d)
    # test_data = load_ecg(num=n_test, patch_size=d)
    # test_data = load_housing(num=n_test, patch_size=d)

    # d = 3*d
    A_ = np.random.binomial(1, 0.5, size=(d, d))

    ms = [d*(i+1)//10 for i in range(10)]
    psnr, psnr_min, psnr_max, psnr_std = [], [], [], []
    val_err, err_std = [], []

    for m in tqdm(ms):
        if use_mat:
            A = np.load(folder_name + 'mat/%d.npy' % m)
        else:
            A = A_[:m, :]
        mm = m//5*4
        reconstruction = np.empty(test_data.shape)
        patch_err = []
        cnt1 = cnt
        for j in range(len(test_data)):
            x = test_data[j]
            x_hat = np.zeros(x.shape)
            for i in range(x.shape[0]//d):
                y = A @ x[i*d:(i+1)*d]
                x_hat[i*d:(i+1)*d] = decode(model, A[:mm], y[:mm])
                patch_err.append(100*np.mean(np.square(y[mm:] - A[mm:] @ x_hat[i*d:(i+1)*d])))
            # cs example
            if cnt1 > 0:
                plt.plot(x, label='Original', color='C0')
                plt.plot(x_hat, label='Reconstructed', color='C1')
                # plt.plot(x[:20], label='Original', color='C0')
                # plt.plot(x[20:40], label='Original', color='C0')
                # plt.plot(x[40:], label='Original', color='C0')
                # plt.plot(x_hat[:20], label='Reconstructed', color='C1')
                # plt.plot(x_hat[20:40], label='Reconstructed', color='C1')
                # plt.plot(x_hat[40:], label='Reconstructed', color='C1')
                plt.legend()
                plt.savefig(folder_name + 'cs_%d_%d.png' % (cnt1, mm))
                plt.close()
                cnt1 -= 1
            reconstruction[j] = x_hat
        val, min_val, max_val, std_val = PSNR(test_data, reconstruction)
        psnr.append(val)
        psnr_min.append(min_val)
        psnr_max.append(max_val)
        psnr_std.append(std_val)
        val_err.append(np.mean(patch_err))
        err_std.append(np.std(patch_err))

    print(psnr)
    # print(psnr_min)
    # print(psnr_max)
    print(psnr_std)
    print(val_err)
    print(err_std)
    plot_data([x//5*4 for x in ms], psnr, path=folder_name + 'psnr.png')
    plot_data([x//5*4 for x in ms], val_err, path=folder_name + 'val_err.png', ylabel='Validation error (1e-2)')
    # print(model.covariances_)
    # print(model.means_)
    # exit()
    
    batch = 20
    for ii in range(len(test_data)//batch):
        avg_cost = 0
        for i in range(ii,ii+batch):
            x = test_data[i]
            cost = []
            for j in range(model.means_.shape[0]):
                var_j, mu_j = model.covariances_[j], model.means_[j]
                cost_j = weighted_l2(x - mu_j, var_j) + np.log(np.linalg.det(var_j))
                cost.append(cost_j)
            k = np.argmin(cost)
            # print(cost[k])
            avg_cost += cost[k]
        avg_cost /= batch
        print(avg_cost)


    test_data = create_pulse(num=n_test, signal_length=d, peak_width=10, num_peaks=d//10)
    # test_data = load_test_ecg(num=n_test, patch_size=d)
    psnr, psnr_min, psnr_max, psnr_std = [], [], [], []
    val_err, err_std = [], []

    # print(test_data.sum(axis=0))
    # exit()

    for m in tqdm(ms):
        if use_mat:
            A = np.load(folder_name + 'mat/%d.npy' % m)
        else:
            A = A_[:m, :]
        mm = m//5*4
        reconstruction = np.empty(test_data.shape)
        patch_err = []
        cnt1 = cnt
        for j in range(len(test_data)):
            x = test_data[j]
            x_hat = np.zeros(x.shape)
            for i in range(x.shape[0]//d):
                y = A @ x[i*d:(i+1)*d]
                x_hat[i*d:(i+1)*d] = decode(model, A[:mm], y[:mm])
                patch_err.append(100*np.mean(np.square(y[mm:] - A[mm:] @ x_hat[i*d:(i+1)*d])))
            # cs example
            if cnt1 > 0:
                plt.plot(x, label='Original')
                plt.plot(x_hat, label='Reconstructed')
                plt.legend()
                plt.savefig(folder_name + 'csx_%d_%d.png' % (cnt1, mm))
                plt.close()
                cnt1 -= 1
            reconstruction[j] = x_hat
        val, min_val, max_val, std_val = PSNR(test_data, reconstruction)
        psnr.append(val)
        psnr_min.append(min_val)
        psnr_max.append(max_val)
        psnr_std.append(std_val)
        val_err.append(np.mean(patch_err))
        err_std.append(np.std(patch_err))

    print(psnr)
    print(val_err)
    plot_data([x//5*4 for x in ms], psnr, path=folder_name + 'psnrx.png')
    plot_data([x//5*4 for x in ms], val_err, path=folder_name + 'val_errx.png', ylabel='Validation error (1e-2)')


    N = train_data.shape[0]
    # N = n_train
    batch = 25
    for ii in range(len(test_data)//batch):
        avg_cost = 0
        ks = []
        for i in range(ii,ii+batch):
            x = test_data[i]
            cost = []
            for j in range(model.means_.shape[0]):
                var_j, mu_j = model.covariances_[j], model.means_[j]
                cost_j = weighted_l2(x - mu_j, var_j) + np.log(np.linalg.det(var_j))
                cost.append(cost_j)
            k = np.argmin(cost)
            ks.append(k)
            # print(cost[k])
            avg_cost += cost[k]
        avg_cost /= batch
        print(avg_cost)
        print(ks)

        if avg_cost < 0:
            for i in range(ii,ii+batch):
                k = ks[i-ii]
                n = N*model.weights_[k]
                # print(k)
                # update rule
                z = (x-model.means_[k]).reshape(1,d)
                var = z.T @ z
                # var = np.diag((z.reshape(-1) ** 2))
                # print(z.shape, var.shape)
                # print(np.linalg.matrix_rank(var))
                model.covariances_[k] = n*(model.covariances_[k])/(n+1) + n*var/((n+1)**2)
                model.means_[k] += (x-model.means_[k])/(n+1)
                model.weights_ *= N
                model.weights_[k] += 1
                model.weights_ /= N+1
                N += 1
        else:
            print("new component")
            mu = np.mean(test_data[ii:ii+batch], axis=0)
            z = (test_data[ii:ii+batch] - mu)
            # var = z.T @ z / batch
            var = np.diag((z ** 2).mean(axis=0))
            var += 1e-6 * np.eye(d)
            # print("var shape", var.shape)
            print("var rank", np.linalg.matrix_rank(var))
            model.means_ = np.append(model.means_, mu.reshape(1,-1), axis=0)
            model.covariances_ = np.append(model.covariances_, var.reshape(1,d,d), axis=0)
            model.weights_ = np.append(model.weights_, [batch/N], axis=0)
            model.weights_ *= N/(N+batch)
            N += batch

    for var in model.covariances_:
        print(np.linalg.det(var), np.linalg.matrix_rank(var))


    test_data = create_pulse(num=n_test, signal_length=d, peak_width=10, num_peaks=d//10)
    # test_data = load_test_ecg(num=n_test, patch_size=d)
    psnr, psnr_min, psnr_max, psnr_std = [], [], [], []
    val_err, err_std = [], []

    for m in tqdm(ms):
        if use_mat:
            A = np.load(folder_name + 'mat/%d.npy' % m)
        else:
            A = A_[:m, :]
        mm = m//5*4
        reconstruction = np.empty(test_data.shape)
        patch_err = []
        cnt1 = cnt
        for j in range(len(test_data)):
            x = test_data[j]
            x_hat = np.zeros(x.shape)
            for i in range(x.shape[0]//d):
                y = A @ x[i*d:(i+1)*d]
                x_hat[i*d:(i+1)*d] = decode(model, A[:mm], y[:mm])
                patch_err.append(100*np.mean(np.square(y[mm:] - A[mm:] @ x_hat[i*d:(i+1)*d])))
            # cs example
            if cnt1 > 0:
                plt.plot(x, label='Original')
                plt.plot(x_hat, label='Reconstructed')
                plt.legend()
                plt.savefig(folder_name + 'csy_%d_%d.png' % (cnt1, mm))
                plt.close()
                cnt1 -= 1
            reconstruction[j] = x_hat
        val, min_val, max_val, std_val = PSNR(test_data, reconstruction)
        psnr.append(val)
        psnr_min.append(min_val)
        psnr_max.append(max_val)
        psnr_std.append(std_val)
        val_err.append(np.mean(patch_err))
        err_std.append(np.std(patch_err))

    print(psnr)
    print(val_err)
    plot_data([x//5*4 for x in ms], psnr, path=folder_name + 'psnry.png')
    plot_data([x//5*4 for x in ms], val_err, path=folder_name + 'val_erry.png', ylabel='Validation error (1e-2)')


    test_data = load_test_ecg(num=n_test, patch_size=d)
    # test_data = create_pulse(num=n_test, signal_length=d, peak_width=10, num_peaks=d//10)
    psnr, psnr_min, psnr_max, psnr_std = [], [], [], []
    val_err, err_std = [], []

    for m in tqdm(ms):
        if use_mat:
            A = np.load(folder_name + 'mat/%d.npy' % m)
        else:
            A = A_[:m, :]
        mm = m//5*4
        reconstruction = np.empty(test_data.shape)
        patch_err = []
        cnt1 = cnt
        for j in range(len(test_data)):
            x = test_data[j]
            x_hat = np.zeros(x.shape)
            for i in range(x.shape[0]//d):
                y = A @ x[i*d:(i+1)*d]
                x_hat[i*d:(i+1)*d] = decode(model, A[:mm], y[:mm])
                patch_err.append(100*np.mean(np.square(y[mm:] - A[mm:] @ x_hat[i*d:(i+1)*d])))
            # cs example
            if cnt1 > 0:
                plt.plot(x, label='Original')
                plt.plot(x_hat, label='Reconstructed')
                plt.legend()
                plt.savefig(folder_name + 'csz_%d_%d.png' % (cnt1, mm))
                plt.close()
                cnt1 -= 1
            reconstruction[j] = x_hat
        val, min_val, max_val, std_val = PSNR(test_data, reconstruction)
        psnr.append(val)
        psnr_min.append(min_val)
        psnr_max.append(max_val)
        psnr_std.append(std_val)
        val_err.append(np.mean(patch_err))
        err_std.append(np.std(patch_err))

    print(psnr)
    print(val_err)
    plot_data([x//5*4 for x in ms], psnr, path=folder_name + 'psnrz.png')
    plot_data([x//5*4 for x in ms], val_err, path=folder_name + 'val_errz.png', ylabel='Validation error (1e-2)')