# %% 
# Overall Goal: Construct a three-dimensional thread art program. 
# Inputs:
# - A set of hooks, which are points in space that the thread can be attached to.
# - A list of colors, which are—in order—the colors of the thread we can use. 
# - Pairs of (viewing angle, image)
# Output:
# - A list of hooks for each color such that an appropriately colored thread going through each hook in the corresponding list will, when viewed from each angle in the input list, look like the corresponding image. 
# - A 3D model displayed in a matplotlib window that, when viewed from each angle in the input list, looks like the corresponding image.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw
from itertools import product
import time
from IPython.display import clear_output
from tqdm import tqdm
from itertools import combinations
import scipy.ndimage as ndimage
from copy import deepcopy

VOXELS_PER_INCH = 400
RESOLUTION = 1000

# %% Hook Generation and Projection
def generate_hooks_prism(**kwargs):
    """
    Generate hooks along the edges of a rectangular prism, with parameters specified in kwargs.
    params:
        kwargs: dict, keyword arguments.
            min_dist_between_hooks: float, minimum distance between centers of adjacent hooks. 
            side_size1: float, size of the first side of the base of the prism. 
            side_size2: float, size of the second side of the base of the prism. 
            side_size3: float, size of the third side of the base of the prism. 
    """
    min_dist_between_hooks = kwargs["min_dist_between_hooks"] if "min_dist_between_hooks" in kwargs else 0.5 * VOXELS_PER_INCH
    side_size1 = kwargs["side_size1"] if "side_size1" in kwargs else 10 * VOXELS_PER_INCH
    side_size2 = kwargs["side_size2"] if "side_size2" in kwargs else 10 * VOXELS_PER_INCH
    side_size3 = kwargs["side_size3"] if "side_size3" in kwargs else 10 * VOXELS_PER_INCH
    hook_size = kwargs["hook_size"] if "hook_size" in kwargs else 0.03 * VOXELS_PER_INCH # inches? seems close to correct

    hooks = np.empty((0, 3))

    eps = 0 # jank fix TODO
    # Define the 8 vertices of the rectangular prism
    vertices = np.array([
        [eps, eps, eps],
        [side_size1 - eps, eps, eps],
        [side_size1 - eps, side_size2 - eps, eps],
        [eps, side_size2 - eps, eps],
        [eps, eps, side_size3 - eps],
        [side_size1 - eps, eps, side_size3 - eps],
        [side_size1 - eps, side_size2 - eps, side_size3 - eps],
        [eps, side_size2 - eps, side_size3 - eps],
    ])
    eps = hook_size / 2

    # Define the 12 edges as pairs of vertices
    edges = [
        (vertices[0], vertices[1]),
        (vertices[1], vertices[2]),
        (vertices[2], vertices[3]),
        (vertices[3], vertices[0]),
        (vertices[4], vertices[5]),
        (vertices[5], vertices[6]),
        (vertices[6], vertices[7]),
        (vertices[7], vertices[4]),
        (vertices[0], vertices[4]),
        (vertices[1], vertices[5]),
        (vertices[2], vertices[6]),
        (vertices[3], vertices[7]),
    ]

    for edge_start, edge_end in edges:
        # Calculate the number of hooks along the edge
        edge_vector = edge_end - edge_start
        edge_length = np.linalg.norm(edge_vector)
        edge_eps = edge_vector * eps / edge_length
        num_hooks = max(int(np.floor(edge_length / min_dist_between_hooks)) + 1, 2)
        # Generate points along the edge
        for i in range(1, num_hooks - 1):
            point1 = edge_start + ((edge_vector) * i / (num_hooks - 1) - edge_eps) 
            point2 = edge_start + ((edge_vector) * i / (num_hooks - 1) + edge_eps)
            hooks = np.vstack((hooks, point1))
            hooks = np.vstack((hooks, point2))

    # Remove duplicate hooks by rounding and using np.unique, removed this because it reorders the hooks, which we don't want, and adjusted so we shouldn't have duplicate hooks at the corners
    # centers = np.unique(np.round(centers, decimals=8), axis=0)

    return np.array(hooks)

def perspective_projection(points_3d, camera_position, camera_target, up_vector, fov_degrees, aspect_ratio, near, far):
    """
    Project 3D points onto 2D plane given camera parameters
    
    Parameters:
    - points_3d: numpy array of shape (N, 3) containing 3D points
    - camera_position: numpy array [x, y, z] of camera location
    - camera_target: numpy array [x, y, z] point camera is looking at
    - up_vector: numpy array [x, y, z] defining camera's up direction
    - fov_degrees: vertical field of view in degrees
    - aspect_ratio: width/height ratio of viewport
    - near: near clipping plane distance
    - far: far clipping plane distance
    
    Returns:
    - numpy array of shape (N, 2) containing projected 2D points
    """
    # 1. Create view matrix (camera space transform)
    forward = camera_target - camera_position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    
    view_matrix = np.array([
        [right[0], right[1], right[2], -np.dot(right, camera_position)],
        [up[0], up[1], up[2], -np.dot(up, camera_position)],
        [-forward[0], -forward[1], -forward[2], np.dot(forward, camera_position)],
        [0, 0, 0, 1]
    ])
    
    # 2. Create projection matrix
    fov_radians = np.radians(fov_degrees)
    f = 1.0 / np.tan(fov_radians / 2)
    
    projection_matrix = np.array([
        [f/aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ])
    
    # 3. Transform points
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # Apply view transform
    points_view = points_homogeneous @ view_matrix.T
    
    # Apply projection transform
    points_projected = points_view @ projection_matrix.T
    
    # 4. Perform perspective divide
    points_projected = points_projected[:, :3] / points_projected[:, 3:4]
    
    # 5. Convert to screen coordinates (just x and y)
    points_2d = points_projected[:, :2]
    
    return points_2d

def hooks_to_2d_image(hooks, camera_position):
    """
    Convert hooks to a 2D image, as seen from a given angle.
    """
    assert len(hooks) >= 3
    assert hooks.shape[1] == 3
    assert any(camera_position[i] < np.min(hooks[:, i]) or camera_position[i] > np.max(hooks[:, i]) for i in range(3))
    
    camera_target = np.mean(hooks, axis=0) # center of the hooks
    up_vector = np.array([0, 0, 1])
    fov_degrees = 60
    aspect_ratio = 4/3
    near = 0.1
    far = 1000
    projected_hooks = perspective_projection(hooks, camera_position, camera_target, up_vector, fov_degrees, aspect_ratio, near, far)

    # normalize 
    projected_hooks = (projected_hooks - projected_hooks.min(axis=0))
    max_x, max_y = projected_hooks.max(axis=0)
    projected_hooks = projected_hooks / max(max_x, max_y)

    return projected_hooks

def plot_hooks_2d(hooks):
    """
    Plots hooks in 2D.
    """
    x, y = zip(*hooks)
    num_hooks = len(x)
    alphas = np.linspace(1, 0.01, num_hooks)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_hooks))
    plt.scatter(x, y, color=colors, s=[1]*num_hooks, alpha=alphas)
    plt.show()

plot_hooks_2d(hooks_to_2d_image(generate_hooks_prism(), np.array([20, 15, 15]) * VOXELS_PER_INCH))

def pixelate(points_3d, camera_position, plot=False):
    """
    Project and pixelate the hooks with improved visualization
    
    Args:
        points_3d: 3D points to project
        camera_position: Position of the camera
        resolution: Base resolution for the output image
    """
    resolution = RESOLUTION
    # Project points to 2D
    projected_hooks = hooks_to_2d_image(points_3d, camera_position)
    
    # Calculate bounds while maintaining aspect ratio
    max_x = projected_hooks[:, 0].max()
    min_x = projected_hooks[:, 0].min()
    max_y = projected_hooks[:, 1].max()
    min_y = projected_hooks[:, 1].min()
    
    # Calculate dimensions preserving aspect ratio
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = width / height
    
    if aspect_ratio > 1:
        pixel_width = resolution
        pixel_height = int(resolution / aspect_ratio)
    else:
        pixel_height = resolution
        pixel_width = int(resolution * aspect_ratio)
    
    # Create pixel grid
    pixels = np.zeros((pixel_height, pixel_width))
    
    # Normalize coordinates to pixel space
    normalized_hooks = np.copy(projected_hooks)
    normalized_hooks[:, 0] = (normalized_hooks[:, 0] - min_x) / width * (pixel_width - 1)
    normalized_hooks[:, 1] = (normalized_hooks[:, 1] - min_y) / height * (pixel_height - 1)
    
    # Plot points with anti-aliasing
    for hook in normalized_hooks:
        x, y = hook.astype(int)
        if 0 <= x < pixel_width and 0 <= y < pixel_height:
            pixels[y, x] += 1
    
    # Apply Gaussian smoothing for anti-aliasing
    sigma = 0.5  # Adjust this value to control smoothing
    pixels = ndimage.gaussian_filter(pixels, sigma)
    
    # Final normalization and clipping
    pixels = np.clip(pixels, 0.15, 0.85)  # Avoid pure black and white

    # invert the image
    pixels = 1 - pixels
    
    # Create figure with improved aesthetics
    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(pixels, cmap='gray', origin='lower', interpolation='nearest')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    return pixels

hooks = generate_hooks_prism()
camera_pos = np.array([20, 15, 15]) * VOXELS_PER_INCH
pixelate(hooks, camera_pos, plot=True)

# %% Build through_voxels_dict
def through_voxels(p0, p1):
    assert len(p0) == len(p1) == 3
    # 3D version of through_pixels. 

    d = max(int(((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2 + (p0[2]-p1[2])**2) ** 0.5), 1)

    voxels = p0 + (p1-p0) * np.array([np.arange(d+1), np.arange(d+1), np.arange(d+1)]).T / d
    voxels = np.unique(np.round(voxels), axis=0).astype(int)

    return voxels

def build_through_voxels_dict(hooks):
    """
    Build a dictionary of voxels through which a thread can pass.
    """
    d = {}
    combs = list(combinations(range(len(hooks)), 2))
    
    for i, j in tqdm(combs, desc="Building voxel dict"):
        # make sure they're both even or both odd, otherwise skip
        if i % 2 == j % 2: 
          voxels = through_voxels(hooks[i], hooks[j])
          # double check that i < j
          if i > j: i, j = j, i
          d[(i, j)] = voxels

    return d

through_voxels_dict = build_through_voxels_dict(generate_hooks_prism())

# %% Fitness function
def place_image(hooks, position, image):
    pixels_2d = pixelate(hooks, position)

    non_white_indices = np.where(pixels_2d < 0.9)
    non_white_coords = np.column_stack(non_white_indices)
    
    points_center = np.mean(non_white_coords, axis=0)

    output_size = pixels_2d.shape[1], pixels_2d.shape[0]
    tmp_arr = np.array(image)
    x_ratio = output_size[0] / tmp_arr.shape[0]
    y_ratio = output_size[1] / tmp_arr.shape[1]

    scale = min(x_ratio, y_ratio)
    new_size = int(tmp_arr.shape[0] * scale), int(tmp_arr.shape[1] * scale)

    image.resize(new_size[::-1], Image.Resampling.LANCZOS)
    image = np.array(image.convert("L"))
    image = image[::-1]

    image_center = np.array(image.shape[:2]) // 2

    # Calculate translation
    translation = points_center - image_center
    
    # Apply translation to image
    translated_image = np.zeros_like(pixels_2d)
    y_start = max(0, int(translation[0]))
    y_end = min(pixels_2d.shape[0], int(translation[0]) + image.shape[0])
    x_start = max(0, int(translation[1]))
    x_end = min(pixels_2d.shape[1], int(translation[1]) + image.shape[1])

    bounds = (y_start, y_end, x_start, x_end)
    
    if image.ndim == 3:
        translated_image = np.zeros((*pixels_2d.shape, 3))
        translated_image[y_start:y_end, x_start:x_end, :] = image[:y_end-y_start, :x_end-x_start, :]
        image = np.mean(translated_image, axis=2)  # Convert to grayscale
    else:
        translated_image[y_start:y_end, x_start:x_end] = image[:y_end-y_start, :x_end-x_start]
        image = translated_image

    assert image.shape == pixels_2d.shape

    return image, bounds



def get_fitness_function(hooks, positions, images):
    formatted_images = []

    for image, position in zip(images, positions):
        image, bounds = place_image(hooks, position, image)

        formatted_images.append((image, bounds)) # where bounds is the bounds of the actual image. 


    def fitness_function(points_3d):
        loss = 0
        for tup, position in zip(formatted_images, positions):
            image, bounds = tup
            pixels_2d = pixelate(points_3d, position)
            # assert pixels_2d.shape == image.shape
            
            loss += np.abs(pixels_2d[bounds[0]:bounds[1], bounds[2]:bounds[3]] - image[bounds[0]:bounds[1], bounds[2]:bounds[3]]).mean()
        return loss

    return fitness_function

exfxn = get_fitness_function(generate_hooks_prism(), [np.array([20, 15, 15]) * VOXELS_PER_INCH], [Image.open("example_images/nike.png")])

# test
print(exfxn(generate_hooks_prism()))

# %% Optimization
def optimize(hooks, color, positions, images, through_voxels_dict, n_lines=10):
    fitness_function = get_fitness_function(hooks, positions, images)

    # initialize all_through_voxels to just the hooks array and start at hook 0
    all_through_voxels = hooks
    all_hooks = [0]

    even_hooks = list(np.array(range(len(hooks) // 2)) * 2)
    odd_hooks = list(np.array(range(len(hooks) // 2)) * 2 + 1)

    n_successful = 0

    failed = set()
    lines = set()

    for i in tqdm(range(n_lines), desc="Optimizing"):
        prev_hook = all_hooks[-1]
        if prev_hook % 2 == 0:
            current_hook = prev_hook + 1
            possible_hooks = deepcopy(odd_hooks)
        else:
            current_hook = prev_hook - 1
            possible_hooks = deepcopy(even_hooks)
        
        all_hooks.append(current_hook)
        possible_hooks.remove(current_hook)

        best_loss = np.inf
        losses = []

        # randomly select 1/4 of the possible hooks
        possible_hooks = np.random.choice(possible_hooks, len(possible_hooks)//8, replace=False)

        # calculate the loss for each possible hook
        for possible_hook in possible_hooks:
            sortd = tuple(sorted((current_hook, possible_hook)))

            if sortd not in lines and sortd not in failed:

                through_voxels = through_voxels_dict[sortd]
                temp_voxels = np.concatenate([all_through_voxels, through_voxels])

                try:
                    loss = fitness_function(temp_voxels) 
                    n_successful += 1

                    if loss < best_loss:
                        best_loss = loss
                        new_hook = possible_hook
                except:
                    failed.add(sortd)
        
        losses.append(best_loss)
        new_sortd = tuple(sorted((current_hook, new_hook)))
        lines.add(new_sortd)
        # update the current hook
        all_hooks.append(new_hook)
        all_through_voxels = np.concatenate([all_through_voxels, through_voxels_dict[new_sortd]])

    all_through_voxels = np.unique(all_through_voxels, axis=0)

    print("Number of successful lines: ", n_successful)
    print("Number of failed lines: ", len(failed))
    print("Failed lines: ", failed)

    # pixelate the voxels
    pixels = pixelate(all_through_voxels, np.array([20, 15, 15]) * VOXELS_PER_INCH)

    plt.figure(figsize=(10, 10))
    plt.imshow(pixels, cmap='gray', origin='lower', interpolation='nearest')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    return all_hooks, all_through_voxels, losses

# test
output = optimize(generate_hooks_prism(), "black", [np.array([20, 15, 15]) * VOXELS_PER_INCH], [Image.open("example_images/nike.png")], through_voxels_dict, n_lines=20)

# %%
pixels = pixelate(output[1], np.array([20, 15, 15]) * VOXELS_PER_INCH)
hooks = generate_hooks_prism()
image = Image.open("example_images/nike.png")

image, bounds = place_image(hooks, np.array([20, 15, 15]) * VOXELS_PER_INCH, image)

plt.figure(figsize=(10, 10))
plt.imshow(pixels, cmap='gray', origin='lower', interpolation='nearest')
plt.imshow(image, cmap='gray', origin='lower', interpolation='nearest', alpha=0.5)

plt.axis('equal')
plt.tight_layout()
plt.show()

losses = output[2]

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.show()

# %%
def main(hooks, colors, positions, images):
    assert len(colors) == 1 # for now, as depth buffer is not implemented.
    assert len(positions) == len(images) # need to optimize each image from a different angle.
    assert all(len(position) == 3 for position in positions)

    assert len(hooks) >= 3

    n_lines = len(hooks) * 5
    
    # actually optimize here across all angles and images
    through_voxels_dict = build_through_voxels_dict(hooks)

    fitness_function = get_fitness_function(hooks, positions, images)

    optimize(hooks, colors[0], positions, images, through_voxels_dict, n_lines)

if __name__ == "__main__":
    hooks = generate_hooks_prism()
    colors = ["black"]

    positions = [np.array([4, 4, 4]) * VOXELS_PER_INCH, np.array([-4, 4, 4]) * VOXELS_PER_INCH]
    images = [Image.open("example_images/bowie_heroes/bowie_monochrome.jpg"), Image.open("example_images/butterfly/butterfly.png")]

    main(hooks, colors, positions, images)
