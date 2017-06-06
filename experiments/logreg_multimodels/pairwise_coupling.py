import numpy as np


CONVERGENCE_LIMIT = 10**(-6)


def recount_probs_by_pwc(predictions, count_mtrx):

    new_predictions = np.zeros((predictions.shape[0], 3))
    for row_idx, row in enumerate(predictions):
        empir_mtrx = np.asarray(row.reshape(3, 3))
        weighted_empir_mtrx = count_mtrx * empir_mtrx
        np.fill_diagonal(weighted_empir_mtrx, 0)

        phat = empir_mtrx.sum(axis=0)/3

        theor_mtrx = np.zeros((3, 3))
        converged = [False, False, False]
        while not all(converged):
            for j in range(3):
                for k in range(3):
                    if j == k:
                        continue
                    theor_mtrx[j, k] = phat[j] / (phat[j] + phat[k])
                ratio = (np.sum(weighted_empir_mtrx[j,:]/np.sum(theor_mtrx[j, :] * count_mtrx[j,:])))
                old_p = phat[j]
                phat[j] = phat[j]*ratio
                phat[j] = phat[j]/phat.sum()
                converged[j] = abs(phat[j] - old_p) < CONVERGENCE_LIMIT
        new_predictions[row_idx,:] = phat
    return new_predictions
