import sys
import re
import numpy as np
import pandas as pd
import simplekml


if len(sys.argv) < 2:
    print("USAGE: python make_kml.py [filename]")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

kml = simplekml.Kml()
ppi_folder = kml.newfolder(name="PPI")
iip_folder = kml.newfolder(name="IIP")

ppi_line = ppi_folder.newlinestring(name="PPI", coords=df.loc[:,["lon", "lat", "alt"]].to_numpy(), altitudemode="absolute")
iip_line = iip_folder.newlinestring(name="IIP", coords=df.dropna(subset=["lon_IIP"]).loc[:,["lon_IIP", "lat_IIP"]].to_numpy(), tessellate=1)

df_event = df.dropna(subset=["event"])
for _,v in df_event.iterrows():
    ppi_event = ppi_folder.newpoint(name=v["event"], coords=[v[["lon", "lat", "alt"]]], altitudemode="absolute")
for _,v in df_event.dropna(subset=["lon_IIP"]).iterrows():
    iip_event = iip_folder.newpoint(name=v["event"], coords=[v[["lon_IIP", "lat_IIP"]]])

kml.save(re.sub("\.[a-zA-Z0-9]+$", ".kml", sys.argv[1]))