import numpy as np
import pandas as pd
import cv2

# Read the Excel files
gps_matrices = pd.read_excel('/mnt/disk1/joonoh/JKIM_2023/gps_matrices.xlsx', header=None) #GPS 4*4 행렬 불러옴
vo_matrices = pd.read_excel('/mnt/disk1/joonoh/JKIM_2023/vo_matrices.xlsx', header=None)   #VO 4*4 행렬 불러옴

# Convert to numpy arrays and reshape
gps_matrices = gps_matrices.to_numpy().reshape(-1, 4, 4)
vo_matrices = vo_matrices.to_numpy().reshape(-1, 4, 4)

def calculate_errors(gps_matrices, vo_matrices, delta):
    rotation_errors = []
    translation_errors = []
    
    for i in range(len(gps_matrices) - delta - 1):
        # Calculate Fi
        Fi = np.linalg.inv(gps_matrices[i]) @ gps_matrices[i+delta] @ np.linalg.inv(vo_matrices[i]) @ vo_matrices[i+delta]
        
        # Extract the rotation matrix and translation vector from Fi
        R_Fi = Fi[:3, :3]
        t_Fi = Fi[:3, 3]
        
        # Calculate Fi_next
        Fi_next = np.linalg.inv(gps_matrices[i+1]) @ gps_matrices[i+delta+1] @ np.linalg.inv(vo_matrices[i+1]) @ vo_matrices[i+delta+1]

        # Extract the rotation matrix from Fi_next
        R_Fi_next = Fi_next[:3, :3]
        
        # Calculate the relative rotation matrix
        R_rel = R_Fi @ R_Fi_next.T
        
        # Calculate the rotation error
        angle, _ = cv2.Rodrigues(R_rel)
        rotation_error = np.linalg.norm(angle)
        rotation_errors.append(rotation_error)
        
        # Calculate the translation error
        translation_error = np.sqrt(np.sum(t_Fi**2))
        translation_errors.append(translation_error)
    
    return rotation_errors, translation_errors

# Calculate the errors for a step size of 1
rotation_errors, translation_errors = calculate_errors(gps_matrices, vo_matrices, 1)

# Convert to a DataFrame
error_df = pd.DataFrame({
    'Rotation Error': rotation_errors,
    'Translation Error': translation_errors
})

# Write to an Excel file
error_df.to_excel('RTE_results_straight_V2.xlsx', index=False)

# Print the results
for i, (rotation_error, translation_error) in enumerate(zip(rotation_errors, translation_errors)):
    print(f'Error for step {i+1}: Rotation Error={rotation_error}, Translation Error={translation_error}')
