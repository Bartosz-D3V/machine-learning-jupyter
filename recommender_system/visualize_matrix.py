def visualize_matrix(plt, y):
    plt.imshow(y)
    plt.colorbar()
    plt.xlabel('Users (indices)')
    plt.ylabel('Movies (indices)')
