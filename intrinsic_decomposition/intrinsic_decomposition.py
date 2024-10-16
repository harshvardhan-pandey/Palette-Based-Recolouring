import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import cv2
import sys

def intrinsic_decomposition(rgb_im: np.ndarray, wd: int, iter_num:int, rho: float) -> tuple[np.ndarray, np.ndarray]:
    epsilon = 1e-10
    ntsc_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2LUV)[:,:,0] / 100  # Using L channel from Lab color space
    rgb_im = rgb_im.astype(float)/255 + epsilon
    n, m, _ = rgb_im.shape
    img_size = n * m

    print('Compute Weight matrix ...')
    # Normalize RGB triplets
    rgb_im2 = np.sum(rgb_im**2, axis=2)
    rgb_im_c = rgb_im / np.sqrt(rgb_im2[:,:,np.newaxis])

    # Compute local windows efficiently
    y, x = np.mgrid[-wd:wd+1, -wd:wd+1]
    window = (y**2 + x**2 <= wd**2) & (y**2 + x**2 > 0)
    offsets = np.column_stack((y[window], x[window]))

    # Compute intensity differences
    intensity_diff = np.zeros((n, m, len(offsets)))
    for i, (dy, dx) in enumerate(offsets):
        rolled = np.roll(ntsc_im, (dy, dx), axis=(0, 1))
        intensity_diff[:,:,i] = (rolled - ntsc_im)**2

    # Compute color differences
    color_diff = np.zeros((n, m, len(offsets)))
    for i, (dy, dx) in enumerate(offsets):
        rolled = np.roll(rgb_im_c, (dy, dx), axis=(0, 1))
        dot_product = np.sum(rolled * rgb_im_c, axis=2)
        color_diff[:,:,i] = np.arccos(np.clip(dot_product, -1, 1))

    # Compute weights
    c_var = np.var(ntsc_im)
    csig = max(c_var * 0.6, 0.000002)
    c_var_color = np.var(color_diff)
    csig_color = max(c_var_color * 0.6, 0.000002)

    weights = np.exp(-(intensity_diff / csig + color_diff**2 / csig_color**2))
    weights_sum = np.sum(weights, axis=2, keepdims=True)
    weights /= weights_sum + epsilon

    # Prepare for sparse matrix construction
    pixel_indices = np.arange(img_size).reshape(n, m)
    rows = np.repeat(pixel_indices[:,:,np.newaxis], len(offsets), axis=2).ravel()
    cols = (pixel_indices[:,:,np.newaxis] + offsets[:,0] * m + offsets[:,1]).ravel()
    valid_indices = (cols >= 0) & (cols < img_size)
    
    # Create sparse weight matrix
    w = sparse.csr_matrix((weights.ravel()[valid_indices], 
                           (rows[valid_indices], cols[valid_indices])), 
                           shape=(img_size, img_size))

    r = 0.5 * np.ones((img_size, 3))
    inv_s = 2 * np.ones(img_size)
    rgb_im_flat = rgb_im.reshape(img_size, 3)
    rgb_im2_flat = rgb_im2.reshape(img_size)

    for iter in range(iter_num):
        print(f"Iteration {iter + 1}")
        s_i = inv_s[:, np.newaxis] * rgb_im_flat
        
        sum_r = w.dot(r) + s_i
        r = (1 - rho) * r + 0.5 * rho * sum_r
        r = np.clip(r, epsilon, 1)
        
        inv_s = (1 - rho) * inv_s + rho * np.sum(rgb_im_flat * r, axis=1) / rgb_im2_flat
        inv_s = np.maximum(1, inv_s)

    reflectance = r.reshape(n, m, 3)
    shading = (1 / inv_s).reshape(n, m)
    shading = np.repeat(shading[:, :, np.newaxis], 3, axis=2)

    return reflectance, shading

if __name__ == "__main__":
    path = sys.argv[1]
    bgr_im = cv2.imread(path)
    rgb_im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB)
    reflectance, shading = intrinsic_decomposition(rgb_im, 7, 100, 1.9)

    _, axs = plt.subplots(2, 2)
    axs[0][0].imshow(rgb_im)
    axs[0][1].imshow(reflectance)
    axs[1][0].imshow(shading)
    axs[1][1].imshow(reflectance * shading)
    plt.show()
