from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import numbers

class distanceMetric(Enum):
    Chebyshev  = 0
    Euclidean  = 1 
    Mahalanobis = 2
    CosineSimilarity = 3
    Manhattan = 4
    Minkowski = 5

def randomlyDistributedPoints(plane_size, num_of_points):
    assert isinstance(plane_size, numbers.Number) and isinstance(num_of_points, numbers.Number)
    points = [(np.random.uniform(plane_size), np.random.uniform(plane_size)) for _ in range(num_of_points)]
    x, y = zip(*points)
    return x, y

def initializeCentroids(plane_size, num_of_centroids):
    return randomlyDistributedPoints(plane_size, num_of_centroids)
    
def closestCentroid(unzipped_points, unzipped_centroids, type_of_distance_metric):
    if isinstance(type_of_distance_metric, distanceMetric):
        #TODO zip points and centroids to tuples for iteration and calculation
        points = ...
        centroids = ...
        for point in points:
            for centroid in centroids:
                current_distance = calculateDistance(point, centroid)
                #TODO map current point to a specific centroid according to distance 
    else:
        raise TypeError(f'unknown distance metric {type_of_distance_metric}')

def calculateDistance(point_a, point_b, type_of_distance_metric):
    ...

def moveCentroids(points, centroids):
    ...

def main(plane_size, num_of_points, num_of_centroids):
    fig, axs = plt.subplots(3,2, sharex=True, sharey=True)
    fig.suptitle('K-means comparision', fontsize=16)

    points_x, points_y = randomlyDistributedPoints(plane_size, num_of_points)
    centroid_x, centroid_y = initializeCentroids(plane_size, num_of_centroids)

    """
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
    main(plane_size = 1000, num_of_points = 100, num_of_centroids = 4)