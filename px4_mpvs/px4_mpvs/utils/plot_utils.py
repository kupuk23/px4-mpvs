import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting


def plot_stats(statistics):
    plot_features(statistics["recorded_features"], statistics["desired_points"])
    plot_weights(statistics["recorded_wp"], statistics["recorded_ws"])

def plot_weights(w_p, w_s):
    """
    Plot the weights w_p and w_s over time.

    Args:
        w_p (np.ndarray): Weights for the platform.
        w_s (np.ndarray): Weights for the spacecraft.
    """
    fig, ax = plt.subplots()
    ax.plot(w_p, label="w_p", color="blue")
    ax.plot(w_s, label="w_s", color="orange")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Weight Value")
    ax.legend()
    plt.title("Weights Over Time")
    plt.show()

def plot_features(features, desired):
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
    ax.scatter(p1[-1, 0], p1[-1, 1], c="k", marker="o", s=30, linewidths=1, label="Final Features")
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

    plt.title("Feature Points and Desired Points Plot")

    plt.show()


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
