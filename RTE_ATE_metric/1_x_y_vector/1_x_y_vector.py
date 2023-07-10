import pandas as pd
import numpy as np
import math

#이전 점(x1,y1)과 현재 점(x2,y2)의 세타(각도) 및 이동량을 추출하여, 해당 값들을 통해 4*4 행렬로 만들어주는 코드!!
#GPS와 VO 데이터를 가지고 4*4 행렬로 만들어주는 코드
#4*4 행렬 => 회전 행렬 및 변환 행렬을 합쳐서 만든 행렬

# Read the Excel file
df = pd.read_excel('/mnt/disk1/joonoh/JKIM_2023/GPS_AND_CAMERA.xlsx', header=98) # header=98

# Separate the data into different dataframes for each coordinate pair
df1 = df[['x1', 'y1']].values
df2 = df[['x2', 'y2']].values

def calculate_polar_coordinates_and_displacement(dataframe):
    # Initialize lists to hold the polar coordinates
    theta_list = []
    Tx_list = []
    Ty_list = []
    
    # Calculate the polar coordinates and displacement for each point
    for i in range(1, len(dataframe)):
        dx = dataframe[i][0] - dataframe[i-1][0]
        dy = dataframe[i][1] - dataframe[i-1][1]
        theta = math.atan2(dy, dx) # Result will be in radians
        
        Tx_list.append(dx)
        Ty_list.append(dy)
        theta_list.append(theta)
        
    return theta_list, Tx_list, Ty_list

def create_4x4_matrix(theta, Tx, Ty):
    # Initialize a list to hold the matrices
    matrices = []
    
    # Create a 4x4 matrix for each set of values
    for t, tx, ty in zip(theta, Tx, Ty):
        matrix = np.array([[np.cos(t), -np.sin(t), 0, tx],
                           [np.sin(t), np.cos(t), 0, ty],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        matrices.append(matrix)
        
    return matrices

theta1, Tx1, Ty1 = calculate_polar_coordinates_and_displacement(df1)
theta2, Tx2, Ty2 = calculate_polar_coordinates_and_displacement(df2)

# Convert radians to degrees
#theta1 = [math.degrees(t) for t in theta1]
#theta2 = [math.degrees(t) for t in theta2]

# Add polar coordinates to dataframe
df1_polar = pd.DataFrame({'theta1': theta1, 'Tx1': Tx1, 'Ty1': Ty1})
df2_polar = pd.DataFrame({'theta2': theta2, 'Tx2': Tx2, 'Ty2': Ty2})

# Save the polar coordinates to an excel file
#df1_polar.to_excel("df1_polar.xlsx")
#df2_polar.to_excel("df2_polar.xlsx")

print(df1_polar)
print(df2_polar)

# Create the 4x4 matrices
matrices1 = create_4x4_matrix(theta1, Tx1, Ty1)
matrices2 = create_4x4_matrix(theta2, Tx2, Ty2)

# Convert each list of matrices into a single dataframe
df1_matrices = pd.concat([pd.DataFrame(m) for m in matrices1], keys=range(len(matrices1)))
df2_matrices = pd.concat([pd.DataFrame(m) for m in matrices2], keys=range(len(matrices2)))

# Save the dataframes to excel files
df1_matrices.to_excel("df1_matrices_straight_V2.xlsx")  #VO 4*4 행렬 생성
df2_matrices.to_excel("df2_matrices_straight_V2.xlsx")  #GPS 4*4 행렬 생성