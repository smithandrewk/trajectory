import numpy as np
def preprocess_phone_data(filename):
    """
    params :
        filename : absolute or relative path to raw input file obtained from SensorLog for Apple Watch
        save : save preprocessed file if True (default to True)
        plot : plot data if True (default to True)
    returns : 
        df : pandas dataframe which is a subset of the original data with renamed columns
    description : function takes input filename of raw input file from SensorLog for Apple Watch,
        extracts a subset of the data columns, renames these columns to be terse, optionally plots
        statistics about gyroscopic data
    """
    import pandas as pd
    df = pd.read_csv(filename,names=['loggingTime(txt)',
    'loggingSample(N)',
    'accelerometerTimestamp_sinceReboot(s)',
    'accelerometerAccelerationX(G)',
    'accelerometerAccelerationY(G)',
    'accelerometerAccelerationZ(G)',
    'gyroTimestamp_sinceReboot(s)',
    'gyroRotationX(rad/s)',
    'gyroRotationY(rad/s)',
    'gyroRotationZ(rad/s)',
    'magnetometerTimestamp_sinceReboot(s)',
    'magnetometerX(µT)',
    'magnetometerY(µT)',
    'magnetometerZ(µT)'], header=None)

    column_name_mapping_from_phone_names_to_my_names = {
        'accelerometerTimestamp_sinceReboot(s)': 'acc_t',
        'accelerometerAccelerationX(G)': 'acc_x',
        'accelerometerAccelerationY(G)': 'acc_y',
        'accelerometerAccelerationZ(G)': 'acc_z',
        'gyroTimestamp_sinceReboot(s)': 'gyr_t',
        'gyroRotationX(rad/s)': 'gyr_x',
        'gyroRotationY(rad/s)': 'gyr_y',
        'gyroRotationZ(rad/s)': 'gyr_z',
    }
    df = df[list(column_name_mapping_from_phone_names_to_my_names.keys())]
    df = df.rename(column_name_mapping_from_phone_names_to_my_names, axis=1)
    df['acc_t'] = df['acc_t']-df['acc_t'][0]
    df.to_csv(filename.replace(".csv", "_preprocessed.csv"), index=False)
    
    return df
def process_line(line,device):
    """
    loggingTime(txt),
    loggingSample(N),
    accelerometerTimestamp_sinceReboot(s),
    accelerometerAccelerationX(G),
    accelerometerAccelerationY(G),
    accelerometerAccelerationZ(G),
    gyroTimestamp_sinceReboot(s),
    gyroRotationX(rad/s),
    gyroRotationY(rad/s),
    gyroRotationZ(rad/s),
    magnetometerTimestamp_sinceReboot(s),
    magnetometerX(µT),
    magnetometerY(µT),
    magnetometerZ(µT)
    """
    """watch
    loggingTime(txt),
    locationTimestamp_since1970(s),
    locationLatitude(WGS84),
    locationLongitude(WGS84),
    locationAltitude(m),
    locationSpeed(m/s),
    locationCourse(°),
    locationVerticalAccuracy(m),
    locationHorizontalAccuracy(m),
    locationFloor(Z),
    accelerometerTimestamp_sinceReboot(s),
    accelerometerAc     celerationX(G),
    accelerometerAccelerationY(G),
    accelerometerAccelerationZ(G),
    motionTimestamp_sinceReboot(s),
    motionYaw(rad),
    motionRoll(rad),
    motionPitch(rad),
    motionRotationRateX(rad/s),
    motionRotationRateY(rad/s),
    motionRotationRateZ(rad/s),
    motionUserAccelerationX(G),
    motionUserAccelerati     onY(G),
    motionUserAccelerationZ(G),
    motionAttitudeReferenceFrame(txt),
    motionQuaternionX(R),
    motionQuaternionY(R),
    motionQuaternionZ(R),
    motionQuat     ernionW(R),
    motionGravityX(G),
    motionGravityY(G),
    motionGravityZ(G),
    motionMagneticFieldX(µT),
    motionMagneticFieldY(µT),
    motionMagneticFieldZ(µT),
    motionHeading(°),m
    otionMagneticFieldCalibrationAccuracy(Z),
    activityTimestamp_sinceReboot(s),
    activity(txt),activityActivityConfidence(Z),
    activityActivityStartDate(txt),p
    edometerStartDate(txt),
    pedometerNumberofSteps(N),
    pedometerAverageActivePace(s/m),
    pedometerCurrentPace(s/m),
    pedometerCurrentCadence(steps/s),
    pedometerDistance(m),
    pedometerFloorAscended(N),
    pedometerFloorDescended(N)
    ,pedometerEndDate(txt),
    altimeterTimestamp_sinceReboot(s)
    ,altimeterReset(bool),
    altimeterRelativeAltitude(m),
    altimeterPressure(kPa),
    batteryState(N),
    batteryLevel(R)
    """
    line = line.strip()
    line = line.split(',')
    if(line[0]=="loggingTime(txt)" or line[0]=="acc_t"):
        return 0,0,0
    if(device=="watch"):
        t = float(line[10])
        acc = list(map(float, line[11:14]))
        omega = list(map(float, line[18:21]))
    elif(device=="phone"):
        t = float(line[2])
        omega = list(map(float, line[7:10]))
        acc = list(map(float, line[3:6]))
    else:
        t = None
        acc = None
        omega = None
    return t, omega, np.array(acc)
def get_rotation_matrix_to_rotate_vector_a_to_vector_b(a,b=np.array([0,0,-1])):
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    I = np.eye(3,3)
    v_x = np.array([[0,-v[2],v[1]],
                    [v[2],0,-v[0]],
                    [-v[1],v[0],0]])
    R = I + v_x + (v_x @ v_x * (1/(1+c)))
    return R
def low_pass_filter(signal,Hz,cutoff):
    n = len(signal)
    d = 1 / Hz
    hs = np.fft.rfft(signal)
    fs = np.fft.rfftfreq(n, d)
    y_a = np.abs(hs)/n*2 # Scale accordingly
    x = fs
    y = np.abs(y_a)
    filter = fs<cutoff
    y_filt = hs*filter
    signal_filt = np.fft.irfft(y_filt)
    return signal_filt
def get_rotation_matrix_from_yaw_pitch_roll(roll,pitch,yaw):
    from math import cos,sin
    ## according to right hand rule,
    # yaw = z rotation
    # pitch = y rotation
    # roll = x rotation
    R_x = np.array([[1,0,0],[0,cos(roll),-sin(roll)],[0,sin(roll),cos(roll)]])
    R_y = np.array([[cos(pitch),0,sin(pitch)],[0,1,0],[-sin(pitch),0,cos(pitch)]])
    R_z = np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]])
    R = R_z @ R_y @ R_x
    return R
def simple_moving_average(signal,window_size):
    output = []
    for i in range(len(signal)):
        if(i-(window_size-1)<0):
            window = signal[:i+1]
        else:
            window = signal[i-(window_size-1):i+1]
        val = np.mean(window)
        output.append(val)
    return output
def numerically_integrate_signal(signal,time=[]):
    if(len(time) == 0):
        time = np.arange(0,len(signal),1)
    output = [signal[0]]
    for i in range(1,len(signal)):
        output.append(output[-1]+(time[i]-time[i-1])*signal[i])
    return output