class IndexedMatrix:
    """Data structure to handle column operations efficiently.

    It a simplified pd.DataFrame not using pandas but that will dynamically
    replace the column name with the proper column index when slicing so that
    you can do things like:

        matrix = np.array([['3', 'orange', '0.7'], ['2', 'blue', '0.5']])
        index_column = {'color': 1, 'size': 0, 'ratio': 2}
        X = IndexedMatrix(matrix, index_column)
        X[:, "color"] = ["yellow", "green"]
        X[:, ["ratio", "color"]] = [['1', "orange"], ['0.003', 'pink']]

    matrix: 2D np.ndarray
    column_index: dict column -> index of the column
    """

    def __init__(self, matrix, column_index):
        self.matrix = matrix
        self.column_index = column_index
        assert set(column_index.values()) == set(range(matrix.shape[1])), (
            "Index and matrix mismatch, values expected to match range({}) exactly".format(matrix.shape[1]))

    def _remap_key(self, key):
        """In case of column selection, remap the columns using self.column_index

        The validity of rows is handled by numpy.
        An error on a column will raise a KeyError.
        """
        remapped_key = key
        if isinstance(key, tuple):  # false is self is called with a single slice self[:]
            if isinstance(key[1], list):  # Slicing on more than one column
                remapped_key = (key[0], [self.column_index[column] for column in key[1]])
            else:
                remapped_key = (key[0], self.column_index[key[1]])
        return remapped_key

    def __getitem__(self, key):
        return self.matrix[self._remap_key(key)]

    def __setitem__(self, key, value):
        self.matrix[self._remap_key(key)] = value

    def __len__(self):
        return len(self.matrix)

    def select_rows(self, rows_mask):
        """Create a new IndexedMatrix from a subselection of rows from self.matrix

        rows_mask is a numpy array of booleans with size len(self)
        """
        return IndexedMatrix(self.matrix[rows_mask, :], self.column_index)

    def __repr__(self):
        lookup = {v: k for k, v in self.column_index.items()}
        sorted_columns = [lookup[i] for i in range(len(self.column_index))]
        return "column_index: {} \n{}".format(sorted_columns, self.matrix)
