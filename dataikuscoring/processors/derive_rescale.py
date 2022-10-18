import os
import json

from .derive import Derive
from .rescale import Rescale


class DeriveRescale(Rescale):
    """Handle models where both derivatives generation an rescaling is used.

    In that case, the rescaling parameters contains rescaling parameters for both
    the rescaled features and their generated derivatibes.
    However, rescaling is applied before feature generation (and feature generation
    uses scaled features).
    This module adds a Rescale processing only on the generated derivatives and will
    be applied right after the derivatives are generated.
    """

    @classmethod
    def load_parameters(cls, resources_folder):
        filename_rescale = os.path.join(resources_folder, "{}.json".format(Rescale.FILENAME))
        filename_derive = os.path.join(resources_folder, "{}.json".format(Derive.FILENAME))
        parameters = None
        if os.path.isfile(filename_rescale) and os.path.isfile(filename_derive):
            with open(filename_rescale) as f:
                parameters_rescale = json.load(f)
            with open(filename_derive) as f:
                parameters_derive = json.load(f)

            # Select column in rescaler which corresponds to derivatives
            derived_rescale_columns = [
                column_rescale for column_rescale in parameters_rescale["columns"] if column_rescale in [
                    derivative.format(derived_column) for derived_column in parameters_derive["columns"]
                    for derivative in Derive.DERIVATIVES_NAMES.values()
                ]
            ]
            parameters = {
                "columns": derived_rescale_columns,
                "shifts": [shift for column, shift in zip(parameters_rescale["columns"], parameters_rescale["shifts"])
                           if column in derived_rescale_columns],
                "inv_scales": [inv_scale for column, inv_scale in zip(parameters_rescale["columns"], parameters_rescale["inv_scales"])
                               if column in derived_rescale_columns],
            }

        return parameters
