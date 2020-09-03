import numpy as np
from concurrent.futures import ProcessPoolExecutor
A_list = [np.random.rand(2000, 2000) for i in range(10)]

with ProcessPoolExecutor(max_workers=4) as pool:
    pool.map(np.linalg.svd, A_list)