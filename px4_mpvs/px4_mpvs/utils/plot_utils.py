import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# matplotlib.use("Agg")  # Use TkAgg backend for interactive plotting


def plot_stats(statistics):
    plot_features(statistics["recorded_features"], statistics["desired_points"])
    plot_weights(statistics["recorded_wp"], statistics["recorded_ws"])




def plot_weights(w_p, w_s, duration=None, lyapunov=False):
    """
    Plot the weights w_p and w_s over time.

    Args:
        w_p (np.ndarray): Weights for the platform.
        w_s (np.ndarray): Weights for the spacecraft.
    """
    fig, ax = plt.subplots()
    time_steps = (
        np.arange(len(w_p)) * duration / len(w_p)
        if duration is not None
        else np.arange(len(w_p))
    )
    ax.plot(time_steps, w_p, label="w_p" if not lyapunov else "Vp_dot", color="blue")
    ax.plot(time_steps, w_s, label="w_s" if not lyapunov else "Vs_dot", color="orange")
    ax.set_xlabel("Time (s)" if duration is not None else "Time Steps")
    ax.set_ylabel("Weight Value" if not lyapunov else "Lyapunov Derivative Value")
    ax.legend()
    plt.title("Weights Over Time" if not lyapunov else "Lyapunov derivative Over Time")
    # plt.show()


def plot_features(features, desired, duration = None):
    """
    Plot the features in 3D space.

    Args:
        features (list of np.ndarray): List of feature points, each row consist of 4 points (x,y) flattened.
        desired (np.ndarray): Desired feature points, each row consist of 4 points (x,y) flattened.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    features = np.array(features)

    p1 = features[1::3, 0:2]
    p2 = features[1::3, 2:4]
    p3 = features[1::3, 4:6]
    p4 = features[1::3, 6:8]

    p1_des = desired[0:2]
    p2_des = desired[2:4]
    p3_des = desired[4:6]
    p4_des = desired[6:8]

    # plot the features
    ax.scatter(p1[:, 0], p1[:, 1], c="r", label="Feature 1", s=2, marker="o")
    ax.scatter(p2[:, 0], p2[:, 1], c="g", label="Feature 2", s=2, marker="o")
    ax.scatter(p3[:, 0], p3[:, 1], c="b", label="Feature 3", s=2, marker="o")
    ax.scatter(p4[:, 0], p4[:, 1], c="y", label="Feature 4", s=2, marker="o")

    # Plot starting points as crosses
    ax.scatter(p1[0, 0], p1[0, 1], c="r", marker="x", s=50, linewidths=1)
    ax.scatter(p2[0, 0], p2[0, 1], c="g", marker="x", s=50, linewidths=1)
    ax.scatter(p3[0, 0], p3[0, 1], c="b", marker="x", s=50, linewidths=1)
    ax.scatter(p4[0, 0], p4[0, 1], c="y", marker="x", s=50, linewidths=1)

    # plot finishing points as big dots
    ax.scatter(
        p1[-1, 0],
        p1[-1, 1],
        c="k",
        marker="o",
        s=30,
        linewidths=1,
        label="Final Features",
    )
    ax.scatter(p2[-1, 0], p2[-1, 1], c="k", marker="o", s=30, linewidths=1)
    ax.scatter(p3[-1, 0], p3[-1, 1], c="k", marker="o", s=30, linewidths=1)
    ax.scatter(p4[-1, 0], p4[-1, 1], c="k", marker="o", s=30, linewidths=1)

    # plot desired points as stars
    ax.scatter(p1_des[0], p1_des[1], c="c", marker="*", s=100, label="Desired Features")
    ax.scatter(p2_des[0], p2_des[1], c="c", marker="*", s=100)
    ax.scatter(p3_des[0], p3_des[1], c="c", marker="*", s=100)
    ax.scatter(p4_des[0], p4_des[1], c="c", marker="*", s=100)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()
    ax.legend()

    plot_feature_errors(features, desired, duration)

    plt.title("Feature Points and Desired Points Plot")

    plt.show()


def plot_feature_errors(features, desired, duration=None):
    """
    Plot the individual feature errors over time with different colors for each feature.

    Args:
        features (list of np.ndarray): List of feature points, each row consist of 4 points (x,y) flattened.
        desired (np.ndarray): Desired feature points, each row consist of 4 points (x,y) flattened.
    """
    features = np.array(features)

    time_steps = (
        np.arange(len(features)) * duration / len(features)
        if duration is not None
        else np.arange(len(features))
    )
    
    # Extract each feature point over time
    p1 = features[:, 0:2]  # Feature 1 (x,y)
    p2 = features[:, 2:4]  # Feature 2 (x,y)
    p3 = features[:, 4:6]  # Feature 3 (x,y)
    p4 = features[:, 6:8]  # Feature 4 (x,y)
    
    # Desired points
    p1_des = desired[0:2]
    p2_des = desired[2:4]
    p3_des = desired[4:6]
    p4_des = desired[6:8]
    
    # Calculate errors for each feature
    error_p1 = p1 - p1_des
    error_p2 = p2 - p2_des
    error_p3 = p3 - p3_des
    error_p4 = p4 - p4_des
    
    # Calculate euclidean distance errors
    error_p1_norm = np.linalg.norm(error_p1, axis=1)
    error_p2_norm = np.linalg.norm(error_p2, axis=1)
    error_p3_norm = np.linalg.norm(error_p3, axis=1)
    error_p4_norm = np.linalg.norm(error_p4, axis=1)
    
    
    # Create subplots for X and Y errors separately
    fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle('Individual Feature Errors Over Time')
    
    # Colors for each feature
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']

    
    # Plot Euclidean distance errors
    axes[0].plot(time_steps, error_p1[:,0], color=colors[0], label='Feature 1 (x)', linewidth=1)
    axes[0].plot(time_steps, error_p2[:,0], color=colors[1], label='Feature 2 (x)', linewidth=1)
    axes[0].plot(time_steps, error_p3[:,0], color=colors[2], label='Feature 3 (x)', linewidth=1)
    axes[0].plot(time_steps, error_p4[:,0], color=colors[3], label='Feature 4 (x)', linewidth=1)
    axes[0].plot(time_steps, error_p1[:,1], color=colors[4], label='Feature 1 (y)', linewidth=1)
    axes[0].plot(time_steps, error_p2[:,1], color=colors[5], label='Feature 2 (y)', linewidth=1)
    axes[0].plot(time_steps, error_p3[:,1], color=colors[6], label='Feature 3 (y)', linewidth=1)
    axes[0].plot(time_steps, error_p4[:,1], color=colors[7], label='Feature 4 (y)', linewidth=1)
    axes[0].set_title('Euclidean Distance Errors')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Error (pixels)')
    axes[0].legend(fontsize='small', ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    # Plot combined error norm (sum of all feature errors)
    total_error = error_p1_norm + error_p2_norm + error_p3_norm + error_p4_norm
    axes[1].plot(time_steps, total_error, color='black', label='Total Error', linewidth=1)
    axes[1].set_title('Total Error (Sum of All Features)')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Error (pixels)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()


# if __name__ == "__main__":
# features = np.load("recorded_markers.npy", allow_pickle=True)
# desired_points = np.array(
#     [
#         [99, 186],
#         [535, 187],
#         [190, 394],
#         [481, 277],
#     ]
# ).flatten()
# load statistics from multiple pickle files

# plot_features(features, desired_points)