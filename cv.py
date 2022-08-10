import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from main import *

args = arg_parser()
n_train, n_test, class_id = args.n_train, args.n_test, args.test_class
n_components, use_mat, d = args.n_components, args.matrix, args.patch_size
folder_name = 'results/n_components=%d,n_train=%d,patch_size=%d/' % (n_components, n_train, d)

if use_mat:
    A = np.load(folder_name + 'mat/%d.npy' % 40)
    # B = np.load(folder_name + 'mat/%d.npy' % 10)
    th = 0.03
else:
    A = np.random.binomial(1, 0.5, size=(20, d))
    B = np.random.binomial(1, 0.5, size=(10, d))
    th = 0.03

from PIL import Image
im = Image.fromarray(A.astype(np.uint8)*255)
im.save("matrix.png")
exit()

model = joblib.load(folder_name + 'model.pkl')
df = pd.DataFrame()

print("val err threshold:", th)

for c_id in [0, 4]:
    test_data = load_test_ecg(num=n_test, patch_size=d, class_id=c_id)
    psnr, recon_err, cnt, l1, l2 = [], [], 0, [], []

    for j in range(len(test_data)):
        x = test_data[j]
        x_hat = x.copy()
        for i in range(x.shape[0]//d):
            y = A @ x[i*d:(i+1)*d]
            x_hat[i*d:(i+1)*d] = decode(model, A, y)
            err = np.mean(np.square(B @ (x_hat[i*d:(i+1)*d] - x[i*d:(i+1)*d])))
            recon_err.append(err)
            snr = -10*np.log10(np.mean(np.square(x_hat[i*d:(i+1)*d] - x[i*d:(i+1)*d])))
            psnr.append(snr)
            if err > th:
                cnt += 1
                l1.append(snr)
            else:
                l2.append(snr)

    print("--------------------------(%d)--------------------------" % c_id)
    print("% flagged:", 100*cnt/(n_test*x.shape[0]//d))
    print("psnr:", np.mean(psnr), np.std(psnr))
    print("flagged psnr:", np.mean(l1), np.std(l1))
    print(np.max(l1))
    print("non-flagged psnr:", np.mean(l2), np.std(l2))
    print(np.min(l2))
    print("val err:", np.mean(recon_err), np.std(recon_err))
    df[c_id] = recon_err

df.plot.density()
plt.savefig(folder_name + 'recon_err.png')