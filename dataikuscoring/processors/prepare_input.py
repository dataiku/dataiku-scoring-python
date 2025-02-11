import numpy as np

from ..utils import IndexedMatrix


class PrepareInput:

    """This class breaks the input matrix into two matrices as described below
    to optimize the feature computations (i.e. the preprocessings).

    It handles the 4 supported input types: List[Dict], List[List], numpy.ndarray
    and pandas.DataFrame.

    The goal is to allocate memory in advance for the computed features in a
    X_numeric matrix. This matrix as a dtype=np.float64 which allows fast matrix
    computations with numpy.
    The first columns of X_numeric corresponds to the final preprocessed features.
    Then the missing numerical input columns are added (if they were not already
    among the features).

    A X_non_numeric matrix is also created to store the categorical input data and dates
    before normalization.

    We use the IndexedMatrix class to be able to use explicit column name when getting
    or setting values in the matrix.
    """

    def __init__(self, resources):

        self.input_column_names = [column_name for column_name, _ in resources["columns"]]

        self.feature_columns = resources["feature_columns"]  # used in tests

        numeric_input = [column_name for column_name, column_type in resources["columns"]
                         if column_type in ["date", "dateonly", "datetimenotz"] or resources["per_feature"][column_name]["type"] == "NUMERIC"]
        non_numeric_input = [column_name for column_name, column_type in resources["columns"]
                             if column_type in ["date", "dateonly", "datetimenotz"] or resources["per_feature"][column_name]["type"] != "NUMERIC"]

        self.column_index_numeric = {
            column: index for (index, column) in enumerate(set(self.feature_columns + numeric_input))  # using set to have unicity
        }
        self.column_index_non_numeric = {column: index for (index, column) in enumerate(non_numeric_input)}

    def process(self, X):
        # Initialize numeric matrix
        X_numeric = IndexedMatrix(
            matrix=np.empty((len(X), len(self.column_index_numeric)), dtype=np.float64),
            column_index=self.column_index_numeric
        )
        X_numeric[:] = np.nan

        # Initialize non_numeric matrix
        X_non_numeric = IndexedMatrix(
            matrix=np.empty((len(X), len(self.column_index_non_numeric)), dtype="object"),
            column_index=self.column_index_non_numeric
        )
        X_non_numeric[:] = None

        get_column_copy = None
        if isinstance(X, (list, np.ndarray)):  # type is List[List] or numpy array or List[dict]
            data = X[0]
            if not isinstance(data, dict):  # type is List[List] or numpy array
                if isinstance(data, (list, np.ndarray)):
                    if len(data) != len(self.input_column_names):
                        raise ValueError(
                            ("Invalid input size, got n_columns={} instead of {}. "
                             "Expected columns (ordered): {}").format(
                                len(data), len(self.input_column_names), ", ".join(self.input_column_names)))
                    get_column_copy = lambda X, index_column, column: np.array(X)[:, index_column]

            else:  # Type is List[Dict] because we handled validation in check_input_data
                get_column_copy =  lambda X, index_column, column: np.array([x.get(column) for x in X])
        else:  # Type is Dataframe because we handled validation in check_input_data
            missing_columns = set(self.input_column_names).difference(set(X.columns))
            if len(missing_columns) > 0:
                raise ValueError("Missing column(s) in input DataFrame: {}".format(
                    ",".join(missing_columns)))
            get_column_copy = lambda X, index_column, column: X[column].values

        # Fill the input columns data into the right matrices
        for (index_column, column) in enumerate(self.input_column_names):
            data = get_column_copy(X, index_column, column)
            # important to check in non numeric first because dates can be in both
            #  but are not normalized yet
            if column in X_non_numeric.column_index:
                if np.issubdtype(data.dtype, np.number):  # if data is numeric convert nan to None
                    X_non_numeric[:, column] = np.where(np.isnan(data), None, data)
                else:  # we have to convert empty string and nan to None
                    X_non_numeric[:, column] = np.where(data.astype(str) == "", None, data)
                    X_non_numeric[:, column] = np.where([x is np.nan for x in X_non_numeric[:, column]], None, X_non_numeric[:, column])
            else:
                if np.issubdtype(data.dtype, np.number):  # if data is not numeric check empty string
                    X_numeric[:, column] = data
                else:  # None are converted to NaN, we have to convert empty strings
                    X_numeric[:, column] = np.where(data.astype(str) == "", np.nan, data.astype('object'))

        return X_numeric, X_non_numeric
