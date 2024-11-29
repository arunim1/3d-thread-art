import numpy as np
import time
from scratch import perspective_projection
import matplotlib.pyplot as plt

def benchmark_perspective_projection(sizes):
    """Run perspective projection benchmark for different array sizes."""
    times = []
    
    # Fixed camera parameters
    camera_position = np.array([4.0, 4.0, 4.0])
    camera_target = np.array([0.0, 0.0, 0.0])
    up_vector = np.array([0.0, 0.0, 1.0])
    fov_degrees = 60.0
    aspect_ratio = 1.0
    near = 0.1
    far = 100.0
    
    for size in sizes:
        # Generate random 3D points
        points_3d = np.random.rand(size, 3) * 10.0  # Random points in [0,10) cube
        
        # Time the projection
        start_time = time.time()
        _ = perspective_projection(
            points_3d, 
            camera_position, 
            camera_target, 
            up_vector, 
            fov_degrees, 
            aspect_ratio, 
            near, 
            far
        )
        end_time = time.time()
        
        times.append(end_time - start_time)
        print(f"Size {size}: {times[-1]:.4f} seconds")
    
    return times

def plot_results(sizes, times):
    """Plot the benchmark results."""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-')
    plt.xlabel('Number of Points')
    plt.ylabel('Time (seconds)')
    plt.title('Perspective Projection Performance')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Test with different array sizes
    sizes = [100, 1000, 10000, 100000, 1000000]
    times = benchmark_perspective_projection(sizes)
    plot_results(sizes, times)
