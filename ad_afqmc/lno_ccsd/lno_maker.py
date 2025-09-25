import numpy as np

def thouless_trans(t1):
    q, r = np.linalg.qr(t1)
    u_ai = r.T
    u_ji = q
    u_occ = np.vstack((u_ji,u_ai))
    u, _, _ = np.linalg.svd(u_occ)
    return u
