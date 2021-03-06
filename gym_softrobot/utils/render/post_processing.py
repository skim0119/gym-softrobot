import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from tqdm import tqdm
import matplotlib.animation as manimation


def plot_video(
    plot_params_list: list, video_name="video.mp4", margin=0.2, fps=15, plot3d=False
):

    t = np.array(plot_params_list[0]["time"])
    pos_list=[]
    for plot_params in plot_params_list:
        positions_over_time = np.array(plot_params["position"])
        pos_list.append(positions_over_time)
    total_time = int(np.around(t[..., -1], 1))
    total_frames = fps * total_time
    dump = round(len(t) / total_frames)
    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    with writer.saving(fig, video_name, dpi=300):
        for time in range(1, len(t), dump):
            fig.clf()
            if plot3d:
                ax = plt.axes(projection="3d")
            else:
                ax = plt.axes()
            for positions_over_time in pos_list:
                x = positions_over_time[time][0]
                y = positions_over_time[time][1]
                z = positions_over_time[time][2]

                if plot3d:
                    ax.plot3D(x, y, z)
                    ax.set_xlim(0, 4)
                    ax.set_ylim(-1, 1)
                    ax.set_zlim(-0.05, 0.05)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                else:
                    ax.plot(x, y)
                    ax.set_xlim(-0.5, 2.5)
                    ax.set_ylim(-0.5, 2.5)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_aspect("equal")
            writer.grab_frame()
    plt.close(fig)


def compute_projected_velocity(plot_params: dict, period):

    time_per_period = np.array(plot_params["time"]) / period
    avg_velocity = np.array(plot_params["avg_velocity"])
    center_of_mass = np.array(plot_params["center_of_mass"])

    # Compute rod velocity in rod direction. We need to compute that because,
    # after snake starts to move it chooses an arbitrary direction, which does not
    # have to be initial tangent direction of the rod. Thus we need to project the
    # snake velocity with respect to its new tangent and roll direction, after that
    # we will get the correct forward and lateral speed. After this projection
    # lateral velocity of the snake has to be oscillating between + and - values with
    # zero mean.

    # Number of steps in one period.
    period_step = int(period / (time_per_period[-1] - time_per_period[-2])) + 1
    number_of_period = int(time_per_period.shape[0] / period_step)
    # Center of mass position averaged in one period
    center_of_mass_averaged_over_one_period = np.zeros((number_of_period - 2, 3))
    for i in range(1, number_of_period - 1):
        # position of center of mass averaged over one period
        center_of_mass_averaged_over_one_period[i - 1] = np.mean(
            center_of_mass[(i + 1) * period_step : (i + 2) * period_step]
            - center_of_mass[(i + 0) * period_step : (i + 1) * period_step],
            axis=0,
        )

    # Average the rod directions over multiple periods and get the direction of the rod.
    direction_of_rod = np.mean(center_of_mass_averaged_over_one_period, axis=0)
    direction_of_rod /= np.linalg.norm(direction_of_rod, ord=2)

    # Compute the projected rod velocity in the direction of the rod
    velocity_mag_in_direction_of_rod = np.einsum(
        "ji,i->j", avg_velocity, direction_of_rod
    )
    velocity_in_direction_of_rod = np.einsum(
        "j,i->ji", velocity_mag_in_direction_of_rod, direction_of_rod
    )

    # Get the lateral or roll velocity of the rod after subtracting its projected
    # velocity in the direction of rod
    velocity_in_rod_roll_dir = avg_velocity - velocity_in_direction_of_rod

    # Compute the average velocity over the simulation, this can be used for optimizing snake
    # for fastest forward velocity. We start after first period, because of the ramping up happens
    # in first period.
    average_velocity_over_simulation = np.mean(
        velocity_in_direction_of_rod[period_step * 2 :], axis=0
    )

    return (
        velocity_in_direction_of_rod,
        velocity_in_rod_roll_dir,
        average_velocity_over_simulation[2],
        average_velocity_over_simulation[0],
    )

