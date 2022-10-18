import numpy as np


class DropRows:

    """ This Class is used to compute which rows of the input data X
    needs to be rejected. It does not process X but rather returns a mask
    (list of booleans) with the same shape as input data X.
    """

    def __init__(self, resources):
        self.columns = resources["drop_rows"]["columns"]

    def compute_valid_mask(self, X_numeric, X_non_numeric):
        rows_to_compute = np.ones(len(X_numeric), dtype=bool)
        for column in self.columns:
            # important to check in non numeric first because dates can be in both
            #  but are not normalized yet
            if column in X_non_numeric.column_index:
                rows_to_compute *= np.where(X_non_numeric[:, column] == None, False, True)
            else:
                rows_to_compute *= np.where(np.isnan(X_numeric[:, column]), False, True)
        return rows_to_compute

    def __repr__(self):
        return "DropRows({})".format(", ".join(self.columns))
