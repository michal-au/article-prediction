from mpl_toolkits.mplot3d import Axes3D  # musi tu zustat
import matplotlib.pyplot as plt
import numpy as np
from code.lib.plot import init_pgf_fig, save_fig


def plot_tree_regions():
    init_pgf_fig(plt, 0.48)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.text(1.5, 1, '$R_1$')
    ax.text(3.5, 1, '$R_2$')
    ax.text(1, 3, '$R_3$')
    ax.text(3, 3, '$R_4$')
    ax.axis([0, 4, 0, 4])

    plt.plot([0, 4], [2, 2], color='black', linewidth=1)
    plt.plot([3, 3], [0, 2], color='black', linewidth=1)
    plt.plot([2, 2], [2, 4], color='black', linewidth=1)
    save_fig(plt, 'cart_regions')


def plot_tree_3d():
    init_pgf_fig(plt, 0.48)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X = np.array([0, 1, 2, 2.0001, 3, 3.0001, 4])
    Y = np.array([0, 1, 2, 2.0001, 3, 3.0001, 4])
    X, Y = np.meshgrid(X, Y)
    Z = ((Y<=2) * (X<=3))*0.5 + ((Y<=2) * (X>3))*0 + ((Y>2) * (X<=2))*1.3 + ((Y>2) * (X>2))*0.8
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, shade=True, cmap='magma')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$h(x)$')
    ax.xaxis.labelpad, ax.yaxis.labelpad, ax.zaxis.labelpad = -14, -14, -14
    ax.xaxis.set_rotate_label(False), ax.yaxis.set_rotate_label(False)
    ax.w_xaxis.set_ticklabels([]), ax.w_yaxis.set_ticklabels([]), ax.w_zaxis.set_ticklabels([])
    ax.xaxis.set_ticks(np.arange(0, 5, 1)), ax.yaxis.set_ticks(np.arange(0, 5, 1)), ax.zaxis.set_ticks(np.arange(0, 1.5, 0.4))
    ax.set_zlim(0, 1.5)
    ax.view_init(elev=15.)
    save_fig(plt, 'cart_3d')


def partitioning():
    init_pgf_fig(plt, 1)

    np.random.seed(seed=14)
    x = np.random.uniform(low=0, high=4, size=(500,))
    np.random.seed(seed=15)
    y = np.random.uniform(low=0, high=4, size=(500,))
    z = x*y

    #### Natrenovat strom stacilo jednorazove, vysledek je do grafu dopsanej rucne
    # x_joined = np.column_stack((x, y))
    # from sklearn import tree
    # clf = tree.DecisionTreeRegressor(max_depth=2)
    # clf = clf.fit(x_joined, z)
    # da se zobrazit na http://webgraphviz.com/
    # print(tree.export_graphviz(clf, out_file=None, feature_names=['x1','x2']))

    c_map = 'Blues'
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.scatter(x, y, c=z, cmap=c_map)
    ax2.scatter(x, y, c=z, cmap=c_map)
    ax2.plot([2.5216, 2.5216], [0,4], color='black')
    ax3.scatter(x, y, c=z, cmap=c_map)
    ax3.plot([2.5216, 2.5216], [0,4], color='black')
    ax3.plot([2.5216, 4], [2.1089, 2.1089], color='black')
    ax4.scatter(x, y, c=z, cmap=c_map)
    ax4.plot([2.5216, 2.5216], [0,4], color='black')
    ax4.plot([0, 2.5216], [1.6366, 1.6366], color='black')
    ax4.plot([2.5216, 4], [2.1089, 2.1089], color='black')
    save_fig(plt, 'cart_partitioning')


if __name__ == '__main__':
    plot_tree_regions()
    plot_tree_3d()
    partitioning()
