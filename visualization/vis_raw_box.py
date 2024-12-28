import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from envs.box_generator import BoxGenerator


def visualize_boxes(boxes, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))

    max_dim = max(max(box.dimensions) for box in boxes)
    spacing = max_dim * 1.2
    boxes_per_row = int(np.ceil(np.sqrt(len(boxes))))

    for i, (box, color) in enumerate(zip(boxes, colors)):

        row = i // boxes_per_row
        col = i % boxes_per_row
        start_x = col * spacing
        start_y = row * spacing

        l, w, h = box.dimensions

        x = [start_x + x for x in [0, l, l, 0, 0, l, l, 0]]
        y = [start_y + y for y in [0, 0, w, w, 0, 0, w, w]]
        z = [0, 0, 0, 0, h, h, h, h]

        vertices = list(zip(x, y, z))

        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
        ]

        poly3d = art3d.Poly3DCollection(faces, alpha=0.3)
        poly3d.set_facecolor(color)
        ax.add_collection3d(poly3d)

    total_size = spacing * (boxes_per_row + 0.5)
    ax.set_xlim(0, total_size)
    ax.set_ylim(0, total_size)
    ax.set_zlim(0, max_dim * 1.2)

    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    ax.set_zlabel("Height")
    ax.set_title(title)

    ax.view_init(elev=30, azim=45)


def main():
    container_size = (100, 100, 100)
    generator = BoxGenerator(container_size)
    boxes = generator.generate_boxes(10, 20)

    print(f"Number of boxes generated: {len(boxes)}")
    print("\nBox dimensions:")
    total_volume = 0
    for i, box in enumerate(boxes, 1):
        volume = box.get_volume()
        total_volume += volume
        print(f"Box {i}: dimensions = {box.dimensions}, volume = {volume:.2f}")
    print(f"\nTotal volume: {total_volume:.2f}")

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualize_boxes(boxes, "Random Boxes Visualization")
    plt.savefig(
        f"visualization/box_generator_visualization_{timestamp}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


if __name__ == "__main__":
    main()
