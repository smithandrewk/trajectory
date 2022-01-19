
def animate_trajectory(time,bases,trajectory):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.animation as animation
    from pytransform3d.plot_utils import Frame
    from pytransform3d import rotations as pr

    def update_frame(step, n_frames, frame):
        R = bases[step]
        A2B = np.eye(4)
        A2B[:3, :3] = R
        A2B[:3,3] = trajectory[step]
        frame.set_data(A2B)
        return frame


    n_frames = len(time)

    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    frame = Frame(np.eye(4), label="rotating frame", s=0.5)
    frame.add_frame(ax)

    anim = animation.FuncAnimation(
        fig, update_frame, n_frames, fargs=(n_frames, frame), interval=50,
        blit=False)
    anim.save('figures/basic_animation.mp4', fps=100, extra_args=['-vcodec', 'libx264'])

    plt.show()