import datetime
import numpy as np


from .preprocessor import Preprocessor


class DatetimeCyclical(Preprocessor):

    FILENAME = "datetime_cyclical"

    def __init__(self, parameters):
        self.mapping = parameters["mapping"]

    def truncate_datetime(self, period, date):
        if period == "MINUTE":
            return date.replace(second=0)
        if period == "HOUR":
            return date.replace(minute=0, second=0)
        if period == "DAY":
            return date.replace(hour=0, minute=0, second=0)
        if period == "WEEK":
            return date.replace(hour=0, minute=0, second=0) - datetime.timedelta(days=date.weekday())
        if period == "MONTH":
            return date.replace(day=1, hour=0, minute=0, second=0)
        if period == "QUARTER":
            quarter = ((date.month - 1) // 3) + 1
            return date.replace(month=quarter * 3 - 2, day=1, hour=0, minute=0, second=0)
        if period == "YEAR":
            return date.replace(month=1, day=1, hour=0, minute=0, second=0)

    def process(self, X_numeric, X_non_numeric):
        origin = datetime.datetime(1900, 1, 1, 0, 0, 0)
        for column, periods in self.mapping.items():
            timestamps = X_numeric[:, column] * 1000.0
            dates = [origin + datetime.timedelta(milliseconds=timestamp) for timestamp in timestamps]

            for period in periods:
                second_subperiods = np.array([(date - self.truncate_datetime(period, date)).total_seconds() for date in dates])
                trigo_args = second_subperiods * 2 * 3.141592653589793 / PERIOD[period]
                prefix = "datetime_cyclical:{}:{}:".format(column, period.lower())
                X_numeric[:, prefix + "sin"] = np.sin(trigo_args)
                X_numeric[:, prefix + "cos"] = np.cos(trigo_args)

        return X_numeric, X_non_numeric

    def __repr__(self):
        return "DatetimeCyclical({})".format(str(self.mapping))


PERIOD = {
    "MINUTE": 60,
    "HOUR": 3600,
    "DAY": 86400,
    "WEEK": 604800,
    "MONTH": 2678400,
    "QUARTER": 7948800,
    "YEAR": 31622400
}
