import numpy as np


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def process_point_cloud_single_patient(vertices,center, n_regions_phi=32, n_regions_theta=32,catesian=False):
    # Convert vertices to spherical coordinates
    spherical_data = np.zeros((len(vertices), 3))
    catesian_data = np.zeros((len(vertices), 3))
    
    for i, point in enumerate(vertices):
        x, y, z = point
        a,b,c = center
        r, theta, phi = cartesian_to_spherical(x-a, y-b, z-c)
        spherical_data[i] = np.array([r, theta, phi])
        catesian_data[i] = np.array([x,y,z])
        
    phi_bins = np.linspace(0, np.pi, n_regions_phi + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_regions_theta + 1)

    selected_points = []
    for i in range(n_regions_phi):
        for j in range(n_regions_theta):
            phi_center = (phi_bins[i] + phi_bins[i + 1]) / 2
            theta_center = (theta_bins[j] + theta_bins[j + 1]) / 2

            # Find the point with the closest phi and theta to the region center
            dist_to_center = np.sqrt((spherical_data[:, 1] - theta_center)**2 + (spherical_data[:, 2] - phi_center)**2)
            closest_point_index = np.argmin(dist_to_center)

            if catesian ==True:
                closest_point = catesian_data[closest_point_index]
            else:
                closest_point = spherical_data[closest_point_index]
           
            selected_points.append(closest_point)
    return np.array(selected_points)

def process_point_cloud_single_patient_withNormals(vPn, n_regions_phi=32, n_regions_theta=32):
    # Convert vertices to spherical coordinates
    vertices = vPn[:,:3]
    spherical_data = np.zeros((len(vertices), 3))
    catesian_data = np.zeros((len(vertices), 3))
    
    for i, point in enumerate(vertices):
        x, y, z = point
        r, theta, phi = cartesian_to_spherical(x, y, z)
        spherical_data[i] = np.array([r, theta, phi])
        catesian_data[i] = np.array([x,y,z])
        
    phi_bins = np.linspace(0, np.pi, n_regions_phi + 1)
    theta_bins = np.linspace(-np.pi, np.pi, n_regions_theta + 1)

    selected_points = []
    for i in range(n_regions_phi):
        for j in range(n_regions_theta):
            phi_center = (phi_bins[i] + phi_bins[i + 1]) / 2
            theta_center = (theta_bins[j] + theta_bins[j + 1]) / 2

            # Find the point with the closest phi and theta to the region center
            dist_to_center = np.sqrt((spherical_data[:, 1] - theta_center)**2 + (spherical_data[:, 2] - phi_center)**2)
            closest_point_index = np.argmin(dist_to_center)
            closest_point = vPn[closest_point_index]
           
            selected_points.append(closest_point)

    return np.array(selected_points)
