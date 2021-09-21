#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import sys

from geoapps.io.MagneticScalar import MagneticScalarParams

from .base_inversion import InversionDriver


def start_inversion(filepath=None):
    """Starts inversion with parameters defined in input file."""

    params = MagneticScalarParams.from_path(filepath)
    driver = MagneticScalarDriver(params)
    driver.run()


class MagneticScalarDriver(InversionDriver):
    def __init__(self, params: MagneticScalarParams):
        super().__init__(params)

    def run(self):
        super().run()


if __name__ == "__main__":

    filepath = sys.argv[1]
    start_inversion(filepath)