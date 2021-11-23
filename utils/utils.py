import numpy as np
def get_axis_angle(dt,angular_velocity):
    norm = np.linalg.norm(angular_velocity)
    axis = angular_velocity/norm
    angle = dt*norm
    return axis,angle
def plot(Q,length,save=False):
    ## plotting
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.animation as animation
    from pytransform3d import rotations as pr


    def update_lines(step, Q, rot):
        R = pr.matrix_from_quaternion(Q[step])
        print(step,"--",Q[step])
        # Draw new frame
        rot[0].set_data(np.array([0, R[0, 0]]), [0, R[1, 0]])
        rot[0].set_3d_properties([0, R[2, 0]])

        rot[1].set_data(np.array([0, R[0, 1]]), [0, R[1, 1]])
        rot[1].set_3d_properties([0, R[2, 1]])

        rot[2].set_data(np.array([0, R[0, 2]]), [0, R[1, 2]])
        rot[2].set_3d_properties([0, R[2, 2]])

        return rot

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    R = pr.matrix_from_quaternion([1,0,0,0])

    rot = [
        ax.plot([0, 1], [0, 0], [0, 0], c="r", lw=3)[0],
        ax.plot([0, 0], [0, 1], [0, 0], c="g", lw=3)[0],
        ax.plot([0, 0], [0, 0], [0, 1], c="b", lw=3)[0],

        ax.plot([0, R[0, 0]], [0, R[1, 0]], [0, R[2, 0]],
                c="r", lw=3, alpha=0.3)[0],
        ax.plot([0, R[0, 1]], [0, R[1, 1]], [0, R[2, 1]],
                c="g", lw=3, alpha=0.3)[0],
        ax.plot([0, R[0, 2]], [0, R[1, 2]], [0, R[2, 2]],
                c="b", lw=3, alpha=0.3)[0]
    ]
    anim = animation.FuncAnimation(fig, update_lines, length,
                            fargs=(Q, rot),
                            interval=10, blit=False)
    if(save):
        writer = animation.ImageMagickWriter(fps=5,bitrate=1)
        anim.save("anim.mp4",writer=writer)
    plt.show()
def preprocess_watch_data(df):
    column_name_mapping_from_watch_names_to_my_names = {
        'accelerometerTimestamp_sinceReboot(s)':'acc_t',
        'accelerometerAccelerationX(G)':'acc_x',
        'accelerometerAccelerationY(G)':'acc_y',
        'accelerometerAccelerationZ(G)':'acc_z',
        'motionTimestamp_sinceReboot(s)':'gyr_t',
        'motionYaw(rad)':'yaw', 
        'motionRoll(rad)':'roll', 
        'motionPitch(rad)':'pitch',
        'motionRotationRateX(rad/s)':'gyr_x', 
        'motionRotationRateY(rad/s)':'gyr_y',
        'motionRotationRateZ(rad/s)':'gyr_z', 
        'motionUserAccelerationX(G)':'acc_x_2',
        'motionUserAccelerationY(G)':'acc_y_2', 
        'motionUserAccelerationZ(G)':'acc_z_2',
        'motionQuaternionX(R)':'quat_x',
        'motionQuaternionY(R)':'quat_y', 
        'motionQuaternionZ(R)':'quat_z', 
        'motionQuaternionW(R)':'quat_w'
    }
    df = df[list(column_name_mapping_from_watch_names_to_my_names.keys())]
    df = df.rename(column_name_mapping_from_watch_names_to_my_names,axis=1)
    return df

def make_cartesian_axes():
    import matplotlib.pyplot as plt
    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -5, 5, -5, 5
    ticks_frequency = 1

    # Plot points
    plt.figure(1)
    # ax = plt.plot(figsize=(10, 10))

    fig, ax = plt.subplots(figsize=(5, 5))
    
    fig.patch.set_facecolor('#ffffff')
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('$x$', size=14, labelpad=-24, x=1.02)
    ax.set_ylabel('$y$', size=14, labelpad=-21, y=1.02, rotation=0)
    plt.text(0.49, 0.49, r"$O$", ha='right', va='top',
        transform=ax.transAxes,
            horizontalalignment='center', fontsize=14)
    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(ticks=[])
    ax.set_yticks(ticks=[])
    ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax+1), minor=True)
    return fig,ax