#
# The MIT License
#
# Copyright (c) 2022 Interstellar Technologies Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import sys
import shutil
import pandas as pd
from math import sin, cos, tan, atan, atan2, sqrt, pi


def distance_vincenty(lat_origin, lon_origin, lat_target, lon_target):

    itr_limit = 5000
    Ra = 6378137.0
    f = 1.0 / 298.257223563
    Rb = Ra * (1.0 - f)

    lat1 = lat_origin * pi / 180.0
    lon1 = lon_origin * pi / 180.0
    lat2 = lat_target * pi / 180.0
    lon2 = lon_target * pi / 180.0

    if lon2 - lon1 == 0.0:
        return 0.0

    U1 = atan((1.0 - f) * tan(lat1))
    U2 = atan((1.0 - f) * tan(lat2))
    diff_lon = lon2 - lon1

    sin_sigma = 0.0
    cos_sigma = 0.0
    sigma = 0.0
    sin_alpha = 0.0
    cos_alpha = 0.0
    cos_2sigma_m = 0.0
    coeff = 0.0

    lamda = diff_lon
    for _ in range(itr_limit):
        sin_sigma = (cos(U2) * sin(lamda)) ** 2 + (
            cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(lamda)
        ) ** 2
        sin_sigma = sqrt(sin_sigma)
        cos_sigma = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(lamda)
        sigma = atan2(sin_sigma, cos_sigma)

        sin_alpha = cos(U1) * cos(U2) * sin(lamda) / sin_sigma
        cos_alpha = sqrt(1.0 - sin_alpha**2)

        cos_2sigma_m = cos_sigma - 2.0 * sin(U1) * sin(U2) / cos_alpha**2

        coeff = f / 16.0 * cos_alpha**2 * (4.0 + f * (4.0 - 3.0 * cos_alpha**2))
        lamda_itr = lamda
        lamda = diff_lon + (1.0 - coeff) * f * sin_alpha * (
            sigma
            + coeff
            * sin_sigma
            * (cos_2sigma_m + coeff * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m))
        )

        if abs(lamda - lamda_itr) < 1e-12:
            break

    u_squr = cos_alpha**2 * (Ra**2 - Rb**2) / Rb**2
    A = 1.0 + u_squr / 16384.0 * (
        4096.0 + u_squr * (-768.0 + u_squr * (320.0 - 175.0 * u_squr))
    )
    B = u_squr / 1024.0 * (256.0 + u_squr * (-128.0 + u_squr * (74.0 - 47.0 * u_squr)))
    delta_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigma_m
            + 0.25
            * B
            * (
                cos_sigma * (-1.0 + 2.0 * cos_2sigma_m**2)
                - (1.0 / 6.0)
                * B
                * cos_2sigma_m
                * (-3.0 + 4.0 * sin_sigma**2)
                * (-3.0 + 4.0 * cos_2sigma_m**2)
            )
        )
    )

    downrange = Rb * A * (sigma - delta_sigma)
    # alpha1 = atan2(cos(U2) * sin(lamda), (cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(lamda)))  # observer to target azimuth
    # alpha2 = atan2(cos(U1) * sin(lamda), (-sin(U1) * cos(U2) + cos(U1) * sin(U2) * cos(lamda)))  # target to observer azimuth
    return downrange


def add_downrange(df):
    lon_origin = df["lon"].iat[0]
    lat_origin = df["lat"].iat[0]
    df["downrange"] = [
        distance_vincenty(lat_origin, lon_origin, latlon[0], latlon[1])
        for latlon in df.loc[:, ["lat", "lon"]].to_numpy()
    ]
    return df


if __name__ == "__main__":
    df_in = pd.read_csv(sys.argv[1])
    df_out = add_downrange(df_in)
    shutil.copy(sys.argv[1], sys.argv[1] + ".backup")
    df_out.to_csv(sys.argv[1])
