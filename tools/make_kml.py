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
import re
import pandas as pd
import simplekml


if len(sys.argv) < 2:
    print("USAGE: python make_kml.py [filename]")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

kml = simplekml.Kml()
ppi_folder = kml.newfolder(name="PPI")
iip_folder = kml.newfolder(name="IIP")

ppi_line = ppi_folder.newlinestring(
    name="PPI",
    coords=df.loc[:, ["lon", "lat", "altitude"]].to_numpy(),
    altitudemode="absolute",
)
iip_line = iip_folder.newlinestring(
    name="IIP",
    coords=df.dropna(subset=["lon_IIP"]).loc[:, ["lon_IIP", "lat_IIP"]].to_numpy(),
    tessellate=1,
)

df_event = df.dropna(subset=["event"])
for _, v in df_event.iterrows():
    ppi_event = ppi_folder.newpoint(
        name=v["event"],
        coords=[v[["lon", "lat", "altitude"]]],
        altitudemode="absolute",
    )
for _, v in df_event.dropna(subset=["lon_IIP"]).iterrows():
    iip_event = iip_folder.newpoint(
        name=v["event"],
        coords=[v[["lon_IIP", "lat_IIP"]]],
    )

kml.save(re.sub("\.[a-zA-Z0-9]+$", ".kml", sys.argv[1]))
