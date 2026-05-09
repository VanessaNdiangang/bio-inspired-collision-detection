# stimulus_generator.py

import os
import cv2
import numpy as np

def generate_looming(n_frames, start_radius, end_radius):
    frames = []
    radii = []
    centre = (32, 32)

    for i in range(n_frames):
        frame = np.ones((64, 64), dtype=np.uint8) * 255
        radius = int(start_radius + (end_radius - start_radius) * (i / (n_frames - 1)))
        cv2.circle(frame, centre, radius, 0, -1)
        frames.append(frame)
        radii.append(radius)

    return frames, radii

def generate_receding(n_frames, start_radius, end_radius):
    frames = []
    centre = (32, 32)
    
    for i in range(n_frames):
        frame = np.ones((64, 64), dtype=np.uint8) * 255
        
        # Start large, shrink to small
        radius = int(start_radius + (end_radius - start_radius) * (i / (n_frames - 1)))
        
        cv2.circle(frame, centre, radius, 0, -1)
        frames.append(frame)
    
    return frames

def generate_lateral(n_frames, radius=8, travel=60):
    frames = []
    start_x = 32 - travel // 2
    end_x = 32 + travel // 2
    
    for i in range(n_frames):
        frame = np.ones((64, 64), dtype=np.uint8) * 255
        
        x = int(start_x + (end_x - start_x) * (i / (n_frames - 1)))
        centre = (x, 32)
        cv2.circle(frame, centre, radius, 0, -1)
        frames.append(frame)

    return frames

def generate_looming_lighting(n_frames, start_radius, end_radius):
    frames = []
    radii = []
    centre = (32, 32)

    for i in range(n_frames):
        frame = np.ones((64, 64), dtype=np.uint8) * 255
        radius = int(start_radius + (end_radius - start_radius) * (i / (n_frames - 1)))
        cv2.circle(frame, centre, radius, 0, -1)

        # Apply sinusoidal brightness modulation
        brightness = 0.5 + (np.sin(i / n_frames * 2 * np.pi) + 1) / 2  # Scale to [0.5, 1.5]
        frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)    

        frames.append(frame)
        radii.append(radius)

    return frames, radii

def generate_looming_camera(n_frames, start_radius, end_radius, shift_per_frame=1):
    frames = []
    radii = []
    centre = (32, 32)
    cumulative_shift = 0

    for i in range(n_frames):
        frame = np.ones((64, 64), dtype=np.uint8) * 255
        radius = int(start_radius + (end_radius - start_radius) * (i / (n_frames - 1)))
        cv2.circle(frame, centre, radius, 0, -1)

        # Apply random lateral shift to simulate camera movement
        cumulative_shift += np.random.randint(-shift_per_frame, shift_per_frame + 1)        
        M = np.float32([[1, 0, cumulative_shift], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (64, 64))
        frames.append(frame)
        radii.append(radius)

    return frames, radii

def save_stimulus(frames, stimulus_type, filename, **kwargs):
    # 1. create folder
    os.makedirs("stimulus_videos", exist_ok=True)
    
    # 2. save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frames[0].shape
    out = cv2.VideoWriter(f"stimulus_videos/{filename}.mp4", fourcc, 20.0, (width, height), isColor=False)
    for frame in frames:
        out.write(frame)
    out.release()
    
    # 3. compute labels based on stimulus_type
    n_frames = len(frames)
    if stimulus_type == 'looming':
        start_radius = kwargs.get('start_radius', 2)
        end_radius = kwargs.get('end_radius', 30)
        radii = [int(start_radius + (end_radius - start_radius) * (i / (n_frames - 1))) for i in range(n_frames)]
        labels = [1 if r >= 16 else 0 for r in radii]
    elif stimulus_type == 'receding':
        labels = [0] * n_frames
    elif stimulus_type == 'lateral':
        labels = [0] * n_frames
    else:
        labels = [0] * n_frames
    
    # 4. save csv
    with open(f"stimulus_videos/{filename}.csv", 'w') as f:
        f.write("Frame,TrueLabel,StimulusType\n")
        for i, label in enumerate(labels):
            f.write(f"{i},{label},{stimulus_type}\n")

if __name__ == "__main__":
    print("Looming:")
    frames, radii = generate_looming(30, 2, 30)
    for frame in frames:
        cv2.imshow("Stimulus", frame)
        cv2.waitKey(50)

    print("Receding:")
    frames = generate_receding(30, 30, 2)
    for frame in frames:
        cv2.imshow("Stimulus", frame)
        cv2.waitKey(50)

    print("Lateral:")
    frames = generate_lateral(30, radius=8)
    for frame in frames:
        cv2.imshow("Stimulus", frame)
        cv2.waitKey(50)

    print("Looming with lighting:")
    frames, radii = generate_looming_lighting(30, 2, 30)
    for frame in frames:
        cv2.imshow("Stimulus", frame)
        cv2.waitKey(50)

    print("Looming with camera motion:")
    frames, radii = generate_looming_camera(30, 2, 30)
    for frame in frames:
        cv2.imshow("Stimulus", frame)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

   # --- Looming variants (slow, medium, fast) ---
    for i, end_r in enumerate([15, 30, 45, 55, 62], 1):
        frames, radii = generate_looming(30, 2, end_r)
        save_stimulus(frames, 'looming', f'looming_{i:02d}',
                      start_radius=2, end_radius=end_r)
        print(f"Saved looming_{i:02d} (end_radius={end_r})")

    # --- Receding variants ---
    for i, start_r in enumerate([15, 30, 45, 55, 62], 1):
        frames = generate_receding(30, start_r, 2)
        save_stimulus(frames, 'receding', f'receding_{i:02d}')
        print(f"Saved receding_{i:02d} (start_radius={start_r})")

    # --- Lateral variants (varying travel distance) ---
    for i, travel in enumerate([20, 30, 40, 50, 60], 1):
        frames = generate_lateral(30, radius=8, travel=travel)
        save_stimulus(frames, 'lateral', f'lateral_{i:02d}')
        print(f"Saved lateral_{i:02d} (travel={travel})")

    # --- Looming with lighting variants (varying brightness frequency) ---
    for i, freq_mult in enumerate([0.5, 1.0, 1.5, 2.0, 3.0], 1):
        frames, radii = generate_looming_lighting(30, 2, 30)
        save_stimulus(frames, 'looming', f'looming_lighting_{i:02d}',
                      start_radius=2, end_radius=30)
        print(f"Saved looming_lighting_{i:02d}")

    # --- Looming with camera motion variants (varying shift intensity) ---
    for i, shift in enumerate([1, 2, 3, 4, 5], 1):
        frames, radii = generate_looming_camera(30, 2, 30,
                                                shift_per_frame=shift)
        save_stimulus(frames, 'looming', f'looming_camera_{i:02d}',
                      start_radius=2, end_radius=30)
        print(f"Saved looming_camera_{i:02d} (shift={shift})")

    print(f"\nAll stimuli saved to stimulus_videos/")