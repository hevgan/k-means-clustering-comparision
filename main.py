from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import numbers
from collections import Counter
from itertools import product

class distanceMetric(Enum):
    Chebyshev  = 0
    Euclidean  = 1 
    Manhattan = 2
    Minkowski_P3 = 3

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
    if type_of_distance_metric == distanceMetric.Manhattan:
        return np.abs(point_a[0]-point_b[0])+np.abs(point_a[1]-point_b[1])
    if type_of_distance_metric == distanceMetric.Minkowski_P3:
        P = distanceMetric.Minkowski_P3.value
        return (np.abs(point_a[0]-point_b[0])**P + np.abs(point_a[1]-point_b[1])**P)**(1/P)

def moveCentroids(points, centroids,best_centroid_list, best_centroid_set,  type_of_distance_metric=distanceMetric.Euclidean):

    length = len(best_centroid_set)
    set_sums_x = [0 for j in range(length)]
    set_sums_y = [0 for j in range(length)]
    element_counts = dict(Counter(best_centroid_list))

    x,y = points
    pos_and_centroid = [ ((x[i], y[i]), best_centroid_list[i])for i in range(len(best_centroid_list)) ]

    for i in range(len(best_centroid_list)):
        curr_x = x[i]
        curr_y = y[i]
        curr_centroid = pos_and_centroid[i][1]
        set_index = best_centroid_set.index(curr_centroid)
        set_sums_x[set_index] += curr_x
        set_sums_y[set_index] += curr_y

    for i in range(len(set_sums_x)):
        set_sums_x[i]/= element_counts[best_centroid_set[i]]
        set_sums_y[i]/= element_counts[best_centroid_set[i]]

    return set_sums_x, set_sums_y

def main(plane_size, num_of_points, num_of_centroids):

    fig = plt.figure()
    plt.xlim(0,plane_size)
    plt.ylim(0,plane_size)
    fig.tight_layout()
    original_points_x, original_points_y = randomlyDistributedPoints(plane_size, num_of_points)
    original_centroid_x, original_centroid_y = initializeCentroids(plane_size, num_of_centroids)
    
    colors = ['b', 'g', 'c', 'm', 'y', 'k']

    for metricType in distanceMetric:
        done = False
        step = 0
        points_x, points_y = original_points_x, original_points_y
        centroid_x, centroid_y = original_centroid_x, original_centroid_y

        while(not done):

            plt.pause(0.1)       
            best_centroid_list = closestCentroid((points_x, points_y), (centroid_x, centroid_y), metricType)
            best_centroid_set = list(set(best_centroid_list))
            previous_centroid_x, previous_centroid_y = centroid_x, centroid_y


            cluster_names = [f'K-{i}' for i in range(len(best_centroid_set))]
            color_map = {best_centroid_set[i] : colors[np.mod(i,len(colors))] for i in range(len(cluster_names))}
            plt.clf()
            for i, point in enumerate(tuple(zip(*(points_x, points_y)))):
                color = color_map[best_centroid_set[best_centroid_set.index(best_centroid_list[i])]]
                plt.scatter(point[0], point[1], c=color)

            plt.scatter(centroid_x,centroid_y, c='red')
            plt.savefig(f'plots/{metricType.name}/step_{step}.png')

            step+=1

            centroid_x, centroid_y = moveCentroids((points_x, points_y), (centroid_x, centroid_y), best_centroid_list, best_centroid_set, metricType)
            
            if(centroid_y == previous_centroid_y and previous_centroid_x == centroid_x):
                done = True 


        print(f"{metricType.name} DONE")


if __name__ == '__main__':
    main(plane_size = 1000, num_of_points = 100, num_of_centroids = 6)
    input()