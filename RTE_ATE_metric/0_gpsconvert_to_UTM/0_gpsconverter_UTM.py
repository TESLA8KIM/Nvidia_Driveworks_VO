import pandas as pd
import pyproj

#GPS 데이터를 m 단위로 변환하기 위함 : UTM 단위로 변환 코드

def transform_coords(row):
    inProj = pyproj.Proj('epsg:4326')  # WGS84
    outProj = pyproj.Proj('epsg:32652')  # UTM Zone 52N
    transformer = pyproj.Transformer.from_proj(inProj, outProj)
    
    try:
        lat, lon = float(row[1]), float(row[0])
        if not -180 <= lon <= 180 or not -90 <= lat <= 90:
            print(f"Illegal latitude or longitude at row {row.name}: {lat}, {lon}")
            return pd.Series([None, None])
        utm_x, utm_y = transformer.transform(lat, lon)
        if utm_x == float('inf') or utm_y == float('inf'):
            print(f"Conversion resulted in infinity at row {row.name}: {lat}, {lon}")
            return pd.Series([None, None])
        return pd.Series([utm_x, utm_y])
    except ValueError:
        print(f"Cannot convert {row[1]}, {row[0]} to float at row {row.name}.")
        return pd.Series([None, None])

df = pd.read_excel('/mnt/disk1/joonoh/JKIM_2023/joonoh_gist_around.xlsx', engine='openpyxl', header=None)

df[['UTM_x', 'UTM_y']] = df.apply(transform_coords, axis=1)

df.to_excel('GPS_UTM_converted_around_final1.xlsx', index=False)
