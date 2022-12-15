# anomaly-detection

BTP

- Since the data folder is too large to track, please make a local copy of the same. It has been added to the .gitignore file

- <code>gmm_recon.ipynb</code> assumptions
    - Gaussian sensing matrices
    - We know the number of components required to model the source GMM
    - Even for initialisation, we know approximately the range of the values of the GMM parameters

- <code>ecg_recon.ipynb</code> has the code for testing the update algorithm on ECG data. Noe that the mechanism of drift introduction remains the same of only adding an error to the mean vectors of each of the componenets.
