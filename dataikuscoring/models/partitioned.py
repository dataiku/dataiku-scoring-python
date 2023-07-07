from datetime import datetime
import re
import numpy as np

from .common import PredictionModelMixin, ProbabilisticModelMixin


def get_dimension_value(value, dimension_name, dimension_type, dimensions_values=None):
    """Extract the exact partition dimension value from the dimension type.

    Args:
        :param str/int/float/datetime value: value of the partition dimension, can be of many form (str, date, timestamp, ...)
        :param str dimension_name: name of the partition dimension
        :param str dimension_type: type of the partition dimension
        :param dict dimensions_values: dict of list of dimension values per dimension name. Defaults to None.

        :return: str
    """    
    if dimension_type == "DISCRETE":
        return value
    else:
        return _get_time_dimension_value(value, dimension_name, dimension_type, dimensions_values=dimensions_values)


def _get_time_dimension_value(value, dimension_name, dimension_type, dimensions_values=None):
    """
    This is basically a copy of what is done in Java (PartitionedPipeline.getTimeDimensionValue).
    """
    def _get_time_dimension_value_from_timestamp(value, dimension_type, milliseconds=True):
        datetime_value = datetime.fromtimestamp(value / 1000) if milliseconds else datetime.fromtimestamp(value)
        return _get_time_dimension_value_from_datetime(datetime_value, dimension_type)
    
    def _get_time_dimension_value_from_datetime(value, dimension_type):
        dimension_value = ""
        if dimension_type == "HOUR":
            dimension_value = "-%02d" % value.hour
        if dimension_type in ("HOUR", "DAY"):
            dimension_value = "-%02d" % value.day + dimension_value
        if dimension_type in ("HOUR", "DAY", "MONTH"):
            dimension_value = "-%02d" % value.month + dimension_value
        if dimension_type in ("HOUR", "DAY", "MONTH", "YEAR"):
            dimension_value = "%02d" % value.year + dimension_value
        return dimension_value

    if isinstance(value, (str)):
        if dimensions_values and value in dimensions_values[dimension_name]:
            return value
        elif dimension_type == "YEAR" and re.match(r"^\d{4}$", value):
            # a YEAR value of format "yyyy" should not be interpreted as an epoch even though it can be parsed as an int
            return value
        elif value.isdigit() or (value[0] in ("-", "+") and value[1:].isdigit()):
            epoch = int(value)
            return _get_time_dimension_value_from_timestamp(epoch, dimension_type)
        elif re.match(
            r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(([+-]\d\d:\d\d)|Z)?$",
            value,
        ):  # "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"
            return _get_time_dimension_value_from_timestamp(
                (
                    datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ") - datetime(1970, 1, 1)
                ).total_seconds(), dimension_type, milliseconds=False
            )
        else:
            return value

    elif isinstance(value, (float, int)):
        return _get_time_dimension_value_from_timestamp(value, dimension_type)
    
    elif isinstance(value, datetime):  # works also with pd.Timestamp
        return _get_time_dimension_value_from_datetime(value, dimension_type)

    else:
        raise ValueError("Unknown dimension value type for {} / {} : {}".format(dimension_name, dimension_type, type(value)))


class PartitionedModel(object):

    def __init__(self, models, resources):
        self.models = models
        self.dimensions = resources["meta"]["partitioning"]
        self.dimensions_types = ["DISCRETE"] * len(self.dimensions) if resources["meta"]["partitioningTypes"] is None else resources["meta"]["partitioningTypes"]
        self.dimensions_values = {dimension: [] for dimension in self.dimensions}
        for partition in self.models.keys():
            for partition_dimension, dimension in zip(partition.split("|"), self.dimensions):
                self.dimensions_values[dimension].append(partition_dimension)

    def get_partition_for_dict(self, data):
        for dimension in self.dimensions:
            assert dimension in data, (
                "Data is missing dimension {} required for partitioning".format(dimension))

        partition = "|".join([
            str(get_dimension_value(
                data[dimension], dimension, dimension_type, dimensions_values=self.dimensions_values
            ))
            for dimension, dimension_type in zip(self.dimensions, self.dimensions_types)
        ])

        return partition

    def get_partition_for_list(self, data):
        input_column_names = self.prepare_input.input_column_names
        partition = "|".join([
            str(get_dimension_value(
                data[input_column_names.index(dimension)], dimension, dimension_type, dimensions_values=self.dimensions_values
            ))
            for dimension, dimension_type in zip(self.dimensions, self.dimensions_types)
        ])

        return partition

    def _compute(self, X, method):
        """ Separate in sub batch per model and then sort to match the original order """
        results = []
        indices = []

        # Compute the partition per row
        if isinstance(X, (list, np.ndarray)):
            row_0 = X[0]
            if isinstance(row_0, (list, np.ndarray)):
                partitions = np.array([self.get_partition_for_list(row) for row in X])
            else:
                partitions = np.array([self.get_partition_for_dict(row) for row in X])
        else:  # it is a dataframe
            X.reset_index()
            partitions = np.array([self.get_partition_for_dict(row.to_dict()) for (_, row) in X.iterrows()])

        # Dispatch the rows to their models and get the computation result
        for partition, model in self.models.items():
            indices_tmp = np.where(partitions == partition)[0]
            if len(indices_tmp) == 0:
                continue

            indices.extend(indices_tmp)

            if isinstance(X, (list, np.ndarray)):
                selected_rows = [X[i] for i in indices_tmp]
            else:
                selected_rows = X.iloc[indices_tmp, :]
            results.extend(getattr(model, method)(selected_rows))

        # Rows without which partition has no model
        indices_no_model = np.where(~np.isin(partitions, list(self.models.keys())))[0]
        if len(indices_no_model) > 0:
            indices.extend(indices_no_model)
            results.extend([None] * len(indices_no_model))

        return [result for result, _ in sorted(zip(results, indices), key=lambda x: x[1])]

    def _predict(self, X):
        return np.array(self._compute(X, "_compute_predict"))

    def _describe(self):
        description = "PartitionedModel \n" + "\n\n".join(["*** Partition {} ***\n {}".format(
            partition, model._describe()) for partition, model in self.models.items()])
        return description

    @property
    def prepare_input(self):
        return list(self.models.values())[0].prepare_input


class ClassificationPartitionedModel(PartitionedModel, PredictionModelMixin, ProbabilisticModelMixin):

    @property
    def classes(self):
        return list(self.models.values())[0].classes

    def _predict_proba(self, X):
        return {label: np.array([row[label] if row is not None else None for row in self._compute(X, "_predict_proba_list_dict")])
                for label in self.classes}


class RegressionPartitionedModel(PartitionedModel, PredictionModelMixin):
    pass
