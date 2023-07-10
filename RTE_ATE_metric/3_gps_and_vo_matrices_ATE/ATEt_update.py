import numpy as np
import pandas as pd

#For ATE (translation error)

#Translation error: Translation error를 구하기 위해서는 VO와 GPS의 데이터 값들을 바로 넣어서 계산한다.

# Load data from Excel
df = pd.read_excel('/mnt/disk1/joonoh/JKIM_2023/GPS_AND_CAMERA.xlsx')

# Extract the VO and GPS coordinates
vo_coords = df[['x1', 'y1']].to_numpy() #vo
gps_coords = df[['x2', 'y2']].to_numpy() #gps

# Ensure that VO and GPS have the same number of points
assert len(vo_coords) == len(gps_coords), "VO and GPS must have the same number of points"

# Calculate ATE for each step
ate_list = []
for vo, gps in zip(vo_coords, gps_coords):
    ate = np.linalg.norm(vo - gps)
    ate_list.append(ate)

# Convert to a DataFrame
ate_df = pd.DataFrame(ate_list, columns=['ATE'])

# Write to an Excel file
ate_df.to_excel('ATE_results_around_V2_test1.xlsx', index=False)

# Print the results
for i, ate in enumerate(ate_list):
    print(f'ATE for step {i+1}: {ate}')
