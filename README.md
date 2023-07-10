****How to operate feature tracker****

**Hardware Settings**
1. Ensure Xavier is connected via the LAN port (for SSH).
2. Keep Xavier powered with the power cable.
3. Connect the monitor in the front seat to Xavier using an HDMI cable.
4. Connect the mouse and keyboard to the Xavier's USB port (Mind the power limit).

**Nomachine**
1. Launch the Nomachine App.
2. Select 'tegra'.
3. Enter the login details (ID: nvidia, PW: 1) and press 'OK'.
4. The Xavier screen will now be visible.

**Eclipse IDE(Driveworks)**
1. Launch Eclipse IDE.
2. Open 'sample_feature_tracker/main.cpp'.
3. Press 'Ctrl + B' to compile the file (this creates the executable files).
4. Navigate to 'Project name -> Binaries -> Sample_feature_tracker[none/le]', right click and select 'Run As -> Remote C/C++ Application(Tegra)'.
5. Output :
	1. Console will display the Rotation and Translation matrix.
	2. Monitor : Use Nomachine monitor to view the video working for feature tracking.

**Calculate the RPE and ATE**
	(This part requires Python and Excel files.)

1. **0_gpsconverter_UTM**
	1. This converts GPS coordinates from longitude and latitude to UTMx and UTMy to determine the relative path (in meters).
	2. Open '0_gpsconverter_UTM.py' file and set the Excel path.
	3. 'read_excel': (INPUT) GPS raw data file you have set.
	4. 'to_excel': (OUTPUT) write data files with UTMx and UTMy coordinates.

2. **1_x_y_vector**
	1. Converts VO and GPS data to 4x4 matrices (Rotation and Translation matrices).
	2. (Input) The VO and GPS data in 'GPS_AND_CAMERA.xlsx'.
	3. (Output1) 'df1_matrices_name.xlsx' => VO 4x4 matrix.
	4. (Output2) 'df2_matrices_name.xlsx' => GPS 4x4 matrix.

3. **2_gps_and_vo_matrices_RPE**
	(To get the RPE Translation error and RPE Rotation error)
	1. Copy the VO and GPS 4x4 matrix data to their respective files ('df1_matrices_name.xlsx => vo_matrices.xlsx', 'df2_matrices_name.xlsx => gps_matrices.xlsx').
	2. (Input) vo_matrices.xlsx, gps_matrices.xlsx.
	3. (Output) 'RTE_results_name.xlsx'.

4. **3_gps_and_vo_matrices_ATE**
	(To get the ATE error)
	1. (Input) The VO and GPS data in 'GPS_AND_CAMERA.xlsx'.
	2. (Output) 'ATE_results_name.xlsx'.
