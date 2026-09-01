import os
import json
import h5py
import numpy as np
from pathlib import Path

def gkern(kernlen=21, nsig=3):
    import numpy
    import scipy.stats as st
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = numpy.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = numpy.diff(st.norm.cdf(x))
    kernel_raw = numpy.sqrt(numpy.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def simulate_and_save_continuous_data(
    n: int = 512,
    image_size: int = 32,
    quadrant_size: int = 16,
    nsig: float = 5.0,
    noise_std: float = 0.01,
    mf_noise_std: float = 0.1,
    cf_exo: float = 0.1,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    out_dir: str = "sim_data"
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine split sizes
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    n_train = n - n_val - n_test

    # ---- TRAIN: Biased ----
    labels_train = np.random.uniform(0, 1, size=n_train)
    # cf_train = labels_train + np.random.normal(0, cf_exo, size=n_train)
    cf_train = np.random.uniform(0, 1, size=n_train) + np.random.normal(0, cf_exo, size=n_train)
    mf_train = labels_train + np.random.normal(0, mf_noise_std, size=n_train)

    kernel = gkern(quadrant_size, nsig)
    images_train = np.empty((n_train, image_size, image_size), dtype=np.float32)
    for i in range(n_train):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        img[0:quadrant_size, 0:quadrant_size] = kernel * (np.exp(mf_train[i]) - 1) * 3
        img[quadrant_size:image_size, quadrant_size:image_size] = kernel * (np.exp(cf_train[i]) - 1) * 3
        img += np.random.normal(0, noise_std, size=(image_size, image_size))
        images_train[i] = img

    cf_train_std = (cf_train - cf_train.mean()) / cf_train.std()
    mf_train_std = (mf_train - mf_train.mean()) / mf_train.std()
    label_c_train = (labels_train - labels_train.mean()) / labels_train.std()

    # ---- VAL: Unbiased ----
    labels_val = np.random.uniform(0, 1, size=n_val)
    cf_val = np.random.uniform(cf_train.min(), cf_train.max(), size=n_val)
    mf_val = labels_val + np.random.normal(0, mf_noise_std, size=n_val)

    images_val = np.empty((n_val, image_size, image_size), dtype=np.float32)
    for i in range(n_val):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        img[0:quadrant_size, 0:quadrant_size] = kernel * (np.exp(mf_val[i]) - 1) * 3
        img[quadrant_size:image_size, quadrant_size:image_size] = kernel * (np.exp(cf_val[i]) - 1) * 3
        img += np.random.normal(0, noise_std, size=(image_size, image_size))
        images_val[i] = img

    cf_val_std = (cf_val - cf_val.mean()) / cf_val.std()
    mf_val_std = (mf_val - mf_val.mean()) / mf_val.std()
    label_c_val = (labels_val - labels_val.mean()) / labels_val.std()

    # ---- TEST: Unbiased ----
    labels_test = np.random.uniform(0, 1, size=n_test)
    cf_test = np.random.uniform(cf_train.min(), cf_train.max(), size=n_test)
    mf_test = labels_test + np.random.normal(0, mf_noise_std, size=n_test)

    images_test = np.empty((n_test, image_size, image_size), dtype=np.float32)
    for i in range(n_test):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        img[0:quadrant_size, 0:quadrant_size] = kernel * (np.exp(mf_test[i]) - 1) * 3
        img[quadrant_size:image_size, quadrant_size:image_size] = kernel * (np.exp(cf_test[i]) - 1) * 3
        img += np.random.normal(0, noise_std, size=(image_size, image_size))
        images_test[i] = img

    cf_test_std = (cf_test - cf_test.mean()) / cf_test.std()
    mf_test_std = (mf_test - mf_test.mean()) / mf_test.std()
    label_c_test = (labels_test - labels_test.mean()) / labels_test.std()

    # ---- Save everything in one HDF5 ----
    h5_path = out_path / "sim_all.h5"
    split = {"train": [], "val": [], "test": []}
    unique_id = 0
    with h5py.File(h5_path, "w") as f:
        # Train
        for i in range(n_train):
            key = str(unique_id)
            grp = f.create_group(key)
            grp.create_dataset("img", data=images_train[i])
            grp.create_dataset("label_c", data=label_c_train[i])
            grp.create_dataset("label", data=label_c_train[i])
            grp.create_dataset("cf", data=cf_train[i])
            grp.create_dataset("mf", data=mf_train[i])
            grp.create_dataset("cf_std", data=cf_train_std[i])
            grp.create_dataset("mf_std", data=mf_train_std[i])
            split["train"].append(key)
            unique_id += 1

        # Val
        for i in range(n_val):
            key = str(unique_id)
            grp = f.create_group(key)
            grp.create_dataset("img", data=images_val[i])
            grp.create_dataset("label_c", data=label_c_val[i])
            grp.create_dataset("label", data=label_c_val[i])
            grp.create_dataset("cf", data=cf_val[i])
            grp.create_dataset("mf", data=mf_val[i])
            grp.create_dataset("cf_std", data=cf_val_std[i])
            grp.create_dataset("mf_std", data=mf_val_std[i])
            split["val"].append(key)
            unique_id += 1

        # Test
        for i in range(n_test):
            key = str(unique_id)
            grp = f.create_group(key)
            grp.create_dataset("img", data=images_test[i])
            grp.create_dataset("label_c", data=label_c_test[i])
            grp.create_dataset("label", data=label_c_test[i])
            grp.create_dataset("cf", data=cf_test[i])
            grp.create_dataset("mf", data=mf_test[i])
            grp.create_dataset("cf_std", data=cf_test_std[i])
            grp.create_dataset("mf_std", data=mf_test_std[i])
            split["test"].append(key)
            unique_id += 1

    # ---- Save the split info ----
    split_file = out_path / "split.json"
    with open(split_file, "w") as f:
        json.dump(split, f, indent=4)

    return f"Simulated regression data saved to: {out_path}"


def simulate_and_save_continuous_data_3blobs(
    n: int = 512,
    image_size: int = 32,
    quadrant_size: int = 16,
    nsig: float = 5.0,
    noise_std: float = 0.01,
    mf_noise_std: float = 0.2,
    cf_exo: float = 0.5,
    nf_noise_std: float = 0.5,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    out_dir: str = "sim_data"
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine split sizes
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    n_train = n - n_val - n_test

    kernel = gkern(quadrant_size, nsig)

    # # ---- TRAIN: Unbiased ----
    # labels_train = np.random.uniform(0, 1, size=n_train)
    # nf_train = np.random.uniform(0, 1, size=n_train)  # NEW: unbiased nf
    # cf_train = np.random.uniform(0, 1, size=n_train)  # NEW: unbiased cf
    # mf_train = labels_train + np.random.normal(0, mf_noise_std, size=n_train)

    # ---- TRAIN: Biased ----
    # --- Noise scale functions ---
    def mf_noise_scale(labels):
        """High noise at edges [0,0.3333] U [0.6666,1], low in the middle."""
        scale = np.ones_like(labels) * 1.0
        scale[(labels <= 0.25) | (labels >= 0.75)] = 0.1
        return scale

    def nf_noise_scale(labels):
        """Low noise when label in [0,0.3333], high otherwise."""
        scale = np.ones_like(labels) * 1.0  # high default
        scale[labels <= 0.3333] = 0.1       # low in first third
        return scale

    def cf_noise_scale(labels):
        """Low noise when label in [0.6666,1], high otherwise."""
        scale = np.ones_like(labels) * 1.0  # high default
        scale[labels >= 0.6666] = 0.1       # low in last third
        return scale

    # --- Feature generation ---
    labels_train = np.random.uniform(0, 1, size=n_train)
    nf_train = (labels_train +
                np.random.normal(0, nf_noise_std * nf_noise_scale(labels_train), size=n_train))

    cf_train = (labels_train +
                np.random.normal(0, cf_exo * cf_noise_scale(labels_train), size=n_train))

    mf_train = labels_train + np.random.normal(0, mf_noise_std * mf_noise_scale(labels_train), size=n_train)

    # NEW: define nf (placeholder relation)
    images_train = np.empty((n_train, image_size, image_size), dtype=np.float32)
    for i in range(n_train):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        # Top-left: mf
        img[0:quadrant_size, 0:quadrant_size] = kernel * (np.exp(mf_train[i]) - 1) * 3
        # Bottom-right: cf
        img[quadrant_size:image_size, quadrant_size:image_size] = kernel * (np.exp(cf_train[i]) - 1) * 3
        # Top-right: nf (NEW blob)
        img[0:quadrant_size, quadrant_size:image_size] = kernel * (np.exp(nf_train[i]) - 1) * 3
        img += np.random.normal(0, noise_std, size=(image_size, image_size))
        images_train[i] = img

    cf_train_std = (cf_train - cf_train.mean()) / cf_train.std()
    mf_train_std = (mf_train - mf_train.mean()) / mf_train.std()
    nf_train_std = (nf_train - nf_train.mean()) / nf_train.std()
    label_c_train = (labels_train - labels_train.mean()) / labels_train.std()

    # # ---- VAL: Unbiased ----
    # labels_val = np.random.uniform(0, 1, size=n_val)
    # nf_val = np.random.uniform(nf_train.min(), nf_train.max(), size=n_val)  # NEW: unbiased nf
    # cf_val = np.random.uniform(cf_train.min(), cf_train.max(), size=n_val)  # NEW: unbiased cf
    # mf_val = labels_val + np.random.normal(0, mf_noise_std, size=n_val)

    # ---- VAL: Biased ----
    labels_val = np.random.uniform(0, 1, size=n_val)
    nf_val = (labels_val +
               np.random.normal(0, nf_noise_std * nf_noise_scale(labels_val), size=n_val))
    cf_val = (labels_val +
               np.random.normal(0, cf_exo * cf_noise_scale(labels_val), size=n_val))
    mf_val = labels_val + np.random.normal(0, mf_noise_std * mf_noise_scale(labels_val), size=n_val)

    images_val = np.empty((n_val, image_size, image_size), dtype=np.float32)
    for i in range(n_val):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        img[0:quadrant_size, 0:quadrant_size] = kernel * (np.exp(mf_val[i]) - 1) * 3
        img[quadrant_size:image_size, quadrant_size:image_size] = kernel * (np.exp(cf_val[i]) - 1) * 3
        img[0:quadrant_size, quadrant_size:image_size] = kernel * (np.exp(nf_val[i]) - 1) * 3
        img += np.random.normal(0, noise_std, size=(image_size, image_size))
        images_val[i] = img

    cf_val_std = (cf_val - cf_val.mean()) / cf_val.std()
    mf_val_std = (mf_val - mf_val.mean()) / mf_val.std()
    nf_val_std = (nf_val - nf_val.mean()) / nf_val.std()
    label_c_val = (labels_val - labels_val.mean()) / labels_val.std()

    # ---- TEST: Unbiased ----
    common_conf_test = np.random.uniform(0, 1, size=n_test)
    nf_test = np.random.uniform(nf_train.min(), nf_train.max(), size=n_test)  # NEW: unbiased nf
    labels_test = common_conf_test + np.random.normal(0, 0.1, size=n_test)
    cf_test = np.random.uniform(cf_train.min(), cf_train.max(), size=n_test)
    mf_test = labels_test + np.random.normal(0, mf_noise_std, size=n_test)

    images_test = np.empty((n_test, image_size, image_size), dtype=np.float32)
    for i in range(n_test):
        img = np.zeros((image_size, image_size), dtype=np.float32)
        img[0:quadrant_size, 0:quadrant_size] = kernel * (np.exp(mf_test[i]) - 1) * 3
        img[quadrant_size:image_size, quadrant_size:image_size] = kernel * (np.exp(cf_test[i]) - 1) * 3
        img[0:quadrant_size, quadrant_size:image_size] = kernel * (np.exp(nf_test[i]) - 1) * 3
        img += np.random.normal(0, noise_std, size=(image_size, image_size))
        images_test[i] = img

    cf_test_std = (cf_test - cf_test.mean()) / cf_test.std()
    mf_test_std = (mf_test - mf_test.mean()) / mf_test.std()
    nf_test_std = (nf_test - nf_test.mean()) / nf_test.std()
    label_c_test = (labels_test - labels_test.mean()) / labels_test.std()

    # ---- Save everything in one HDF5 ----
    h5_path = out_path / "sim_all.h5"
    split = {"train": [], "val": [], "test": []}
    unique_id = 0
    with h5py.File(h5_path, "w") as f:
        # Train
        for i in range(n_train):
            key = str(unique_id)
            grp = f.create_group(key)
            grp.create_dataset("img", data=images_train[i])
            grp.create_dataset("label_c", data=label_c_train[i])
            grp.create_dataset("label", data=label_c_train[i])
            grp.create_dataset("cf", data=cf_train[i])
            grp.create_dataset("mf", data=mf_train[i])
            grp.create_dataset("nf", data=nf_train[i])       # NEW
            grp.create_dataset("cf_std", data=cf_train_std[i])
            grp.create_dataset("mf_std", data=mf_train_std[i])
            grp.create_dataset("nf_std", data=nf_train_std[i]) # NEW
            split["train"].append(key)
            unique_id += 1

        # Val
        for i in range(n_val):
            key = str(unique_id)
            grp = f.create_group(key)
            grp.create_dataset("img", data=images_val[i])
            grp.create_dataset("label_c", data=label_c_val[i])
            grp.create_dataset("label", data=label_c_val[i])
            grp.create_dataset("cf", data=cf_val[i])
            grp.create_dataset("mf", data=mf_val[i])
            grp.create_dataset("nf", data=nf_val[i])       
            grp.create_dataset("cf_std", data=cf_val_std[i])
            grp.create_dataset("mf_std", data=mf_val_std[i])
            grp.create_dataset("nf_std", data=nf_val_std[i]) 
            split["val"].append(key)
            unique_id += 1

        # Test
        for i in range(n_test):
            key = str(unique_id)
            grp = f.create_group(key)
            grp.create_dataset("img", data=images_test[i])
            grp.create_dataset("label_c", data=label_c_test[i])
            grp.create_dataset("label", data=label_c_test[i])
            grp.create_dataset("cf", data=cf_test[i])
            grp.create_dataset("mf", data=mf_test[i])
            grp.create_dataset("nf", data=nf_test[i])      
            grp.create_dataset("cf_std", data=cf_test_std[i])
            grp.create_dataset("mf_std", data=mf_test_std[i])
            grp.create_dataset("nf_std", data=nf_test_std[i]) 
            split["test"].append(key)
            unique_id += 1

    # ---- Save the split info ----
    split_file = out_path / "split.json"
    with open(split_file, "w") as f:
        json.dump(split, f, indent=4)

    return f"Simulated regression data with 3 blobs saved to: {out_path}"


if __name__ == "__main__":
    # Example usage
    # simulate_and_save_continuous_data(n=10000, val_frac=0.15, test_frac=0.15)
    simulate_and_save_continuous_data_3blobs(n=10000, val_frac=0.15, test_frac=0.15)
