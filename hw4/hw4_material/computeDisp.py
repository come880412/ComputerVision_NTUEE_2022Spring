import numpy as np
import cv2.ximgproc as xip

def census_cost(local_binary_L, local_binary_R):
    disparity = np.sum(np.abs(local_binary_L - local_binary_R), axis=1)

    return disparity

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    window_size = 3
    pad_size = window_size // 2


    img_L, img_R = np.zeros((h + pad_size * 2, w + pad_size * 2, ch), dtype=np.float32), np.zeros((h + pad_size * 2, w + pad_size * 2, ch), dtype=np.float32)
    img_L[1:-1, 1:-1, :], img_R[1:-1, 1:-1, :] = Il, Ir
    
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    
            
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    cost_list_L = np.zeros((h, w, max_disp), dtype=np.float32)
    cost_list_R = np.zeros((h, w, max_disp), dtype=np.float32)
    local_binary_IL, local_binary_IR = [], []
    for i in range(h):
        for j in range(w):
            tmp_local_binary_IL = img_L[i : i + window_size, j : j + window_size, :].copy()
            tmp_local_binary_IR = img_R[i : i + window_size, j : j + window_size, :].copy()
            for k in range(ch):
                middle_value = tmp_local_binary_IL[pad_size, pad_size, k]
                tmp_local_binary_IL[:, :, k] = np.where(tmp_local_binary_IL[:, :, k] >= middle_value, 0, 1)
                middle_value = tmp_local_binary_IR[pad_size, pad_size, k]
                tmp_local_binary_IR[:, :, k] = np.where(tmp_local_binary_IR[:, :, k] >= middle_value, 0, 1)
            local_binary_IL.append(tmp_local_binary_IL)
            local_binary_IR.append(tmp_local_binary_IR)

    local_binary_IL = np.array(local_binary_IL).reshape(h, w, -1) # (h, w, 27)
    local_binary_IR = np.array(local_binary_IR).reshape(h, w, -1) # (h, w, 27)
    for i in range(h):
        for j in range(w):
            # left to right
            if j < max_disp - 1: 
                local_binary_L = local_binary_IL[i, j].copy()[np.newaxis, :]
                local_binary_R = np.flip(local_binary_IR[i, : j+1].copy(), 0)
                disparity = census_cost(local_binary_L, local_binary_R)
                cost_list_L[i, j, :j+1] = disparity
                cost_list_L[i, j, j+1:] = cost_list_L[i, j, j]
            else:
                local_binary_L = local_binary_IL[i, j].copy()[np.newaxis, :]
                local_binary_R = np.flip(local_binary_IR[i, (j - max_disp + 1): j + 1].copy(), 0)
                disparity = census_cost(local_binary_L, local_binary_R)
                cost_list_L[i, j, :] = disparity
            
            # right to left
            if j + max_disp > w:
                local_binary_L = local_binary_IL[i, j : w].copy()
                local_binary_R = local_binary_IR[i, j].copy()[np.newaxis, :]
                disparity = census_cost(local_binary_L, local_binary_R)
                cost_list_R[i, j, :w - j] = disparity
                cost_list_R[i, j, w - j:] = cost_list_R[i, j, w - j - 1]
            else:
                local_binary_L = local_binary_IL[i, j : j + max_disp].copy()
                local_binary_R = local_binary_IR[i, j].copy()[np.newaxis, :]
                disparity = census_cost(local_binary_L, local_binary_R)
                cost_list_R[i, j, :] = disparity
            
    for d in range(max_disp):
        cost_list_L[:, :, d] = xip.jointBilateralFilter(Il, cost_list_L[:, :, d], 20, 10, 10)
        cost_list_R[:, :, d] = xip.jointBilateralFilter(Ir, cost_list_R[:, :, d], 20, 10, 10)  

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    winner_L = np.argmin(cost_list_L, axis=2)
    winner_R = np.argmin(cost_list_R, axis=2)

    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    for i in range(h):
        for j in range(w):
            if winner_L[i, j] == winner_R[i, j - winner_L[i, j]]:
                continue
            else:
                winner_L[i, j]=-1
    
    for i in range(h):
        for j in range(w):
            
            if winner_L[i, j] == -1:
                l_idx = j - 1
                r_idx = j + 1
                while l_idx >= 0 and winner_L[i, l_idx] == -1:
                    l_idx -= 1
                
                if l_idx < 0:
                    FL = 100000000
                else:
                    FL = winner_L[i, l_idx]

                while r_idx < w and winner_L[i, r_idx] == -1:
                    r_idx += 1

                if r_idx > w - 1:
                    FR = 100000000
                else:
                    FR = winner_L[i, r_idx]
                winner_L[i, j] = min(FL, FR)

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), winner_L.astype(np.uint8), 18, 1)
    return labels.astype(np.uint8)
    