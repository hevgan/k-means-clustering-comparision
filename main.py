from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import numbers
import time

class distanceMetric(Enum):
    Chebyshev  = 0
    Euclidean  = 1 
    CosineSimilarity = 2
    Manhattan = 3
    Minkowski_P3 = 4

def randomlyDistributedPoints(plane_size, num_of_points):
    assert isinstance(plane_size, numbers.Number) and isinstance(num_of_points, numbers.Number)
    points = [(np.random.uniform(plane_size), np.random.uniform(plane_size)) for _ in range(num_of_points)]
    x, y = zip(*points)
    return x, y

def initializeCentroids(plane_size, num_of_centroids):
    return randomlyDistributedPoints(plane_size, num_of_centroids)
    
def closestCentroid(unzipped_points, unzipped_centroids, type_of_distance_metric=distanceMetric.Euclidean):
    which_centroids = []
    if isinstance(type_of_distance_metric, distanceMetric):
        points, centroids = tuple(zip(*unzipped_points)), tuple(zip(*unzipped_centroids))
        for point in points:
            best_distance = np.Inf
            best_centroid = None
            for centroid in centroids:
                current_distance = calculateDistance(point, centroid, type_of_distance_metric)
                best_distance, best_centroid  = (current_distance, centroid) if current_distance < best_distance else (best_distance, best_centroid)
            which_centroids.append(best_centroid)
        return which_centroids

    else:
        raise TypeError(f'unknown distance metric of type {type(type_of_distance_metric)}')

def calculateDistance(point_a, point_b, type_of_distance_metric=distanceMetric.Euclidean):
    if type_of_distance_metric == distanceMetric.Chebyshev:
        return max(np.abs(point_a[0]-point_b[0]), np.abs(point_a[1]-point_b[1]))
    if type_of_distance_metric == distanceMetric.Euclidean:
        return np.sqrt((point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2)
    if type_of_distance_metric == distanceMetric.CosineSimilarity:
        return (point_a[0]*point_b[0])+(point_a[1]*point_b[1])/(np.sqrt(point_a[0]**2 + point_a[1]**2) * np.sqrt(point_b[0]**2 + point_b[1]**2))
    if type_of_distance_metric == distanceMetric.Manhattan:
        return np.abs(point_a[0]-point_b[0])+np.abs(point_a[1]-point_b[1])
    if type_of_distance_metric == distanceMetric.Minkowski_P3:
        P = 3
        return (np.abs(point_a[0]-point_b[0])**P + np.abs(point_a[1]-point_b[1])**P)**(1/P)


def moveCentroids(points, centroids):
    ...

def main(plane_size, num_of_points, num_of_centroids):


    fig = plt.figure()

    points_x, points_y = randomlyDistributedPoints(plane_size, num_of_points)
    centroid_x, centroid_y = initializeCentroids(plane_size, num_of_centroids)

    while(True):
       
        best_centroid_list = closestCentroid((points_x, points_y), (centroid_x, centroid_y), distanceMetric.Euclidean)
        best_centroid_set = list(set(best_centroid_list))

        cluster_names = [f'K-{i}' for i in range(len(best_centroid_set))]
        colors = ['orange', 'green', 'blue', 'yellow', 'black']
        color_map = {best_centroid_set[i] : colors[i] for i in range(len(cluster_names))}

        for i, point in enumerate(tuple(zip(*(points_x, points_y)))):
            color = color_map[best_centroid_list[i]]
            label_t = cluster_names[best_centroid_set.index(best_centroid_list[i])]
            plt.scatter(point[0], point[1], c=color)

        plt.scatter(centroid_x,centroid_y, c='red')
        plt.pause(0.1)        



    """
    fig, axs = plt.subplots(3,2, sharex=True, sharey=True)
    fig.suptitle('K-means comparision')
    _, _, _ = closestCentroid(points_x, points_y, distanceMetric.Chebyshev)

    plt.subplot(2, 2, 1)
    plt.title(distanceMetric.Chebyshev.name)
    plt.scatter(points_x,points_y, c ='green')
    plt.scatter(centroid_x,centroid_y, c='red')

    plt.subplot(2, 2, 2)
    plt.title(distanceMetric.Euclidean.name)
    plt.scatter(points_x,points_y, c ='green')
    plt.scatter(centroid_x,centroid_y, c='red')
    
    plt.subplot(2, 2, 3)
    plt.title(distanceMetric.Mahalanobis.name)
    plt.scatter(points_x,points_y, c ='green')
    plt.scatter(centroid_x,centroid_y, c='red')
           
    plt.subplot(2, 2, 4)
    plt.title(distanceMetric.CosineSimilarity.name)
    plt.scatter(points_x,points_y, c ='green')
    plt.scatter(centroid_x,centroid_y, c='red')
    
    plt.show()

    """


if __name__ == '__main__':
    main(plane_size = 1000, num_of_points = 100, num_of_centroids = 5)