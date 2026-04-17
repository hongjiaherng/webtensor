import svgwrite
import os


def create_webtensor_logo(
    filename="webtensor_logo.svg", dark=False, with_text=True, style="solid"
):
    """
    style: "solid" (plain top/right faces) or "grid" (3x3 top/right faces)
    """
    # --- Expanded Canvas Sizing ---
    # Icon is 180x180 to allow a 2px buffer.
    # Text width expanded to 1100px to fit the larger 160px font size.
    canvas_size = (1000, 180) if with_text else (180, 180)
    dwg = svgwrite.Drawing(filename, profile="full", size=canvas_size)

    # --- Official Color Palette ---
    color_rust_orange = "#F05032"
    color_webgpu_blue_dark = "#015a9c"
    color_webgpu_blue_light = "#008df3"
    color_white = "#FFFFFF"

    # --- Color Mapping ---
    top_face_color = color_rust_orange
    right_face_color = color_webgpu_blue_dark
    w_square_color = color_webgpu_blue_light
    main_text_color = color_white if dark else color_webgpu_blue_dark

    # --- Cube Geometry (Mathematically Centered) ---
    # Shifted to 90, 90 to perfectly frame within the 180 height boundary
    cx = 90
    cy = 90
    s = 80

    # The gap between the 3 main faces
    g = 3
    g_inner = g * 2

    # Isometric directional offsets
    off_top = (0, -g)
    off_left = (-g, g / 2)
    off_right = (g, g / 2)

    # --- Base Coordinates & Shifting ---
    p0 = (0, 0)
    p1 = (-s, -s / 2)
    p2 = (0, -s)
    p3 = (s, -s / 2)
    p4 = (-s, s / 2)
    p5 = (0, s)
    p6 = (s, s / 2)

    def shift(points, dx, dy):
        return [(x + cx + dx, y + cy + dy) for x, y in points]

    top_face = shift([p0, p1, p2, p3], *off_top)
    right_face = shift([p0, p3, p6, p5], *off_right)

    # --- Draw Top and Right Faces ---
    if style == "solid":
        dwg.add(dwg.polygon(top_face, fill=top_face_color))
        dwg.add(dwg.polygon(right_face, fill=right_face_color))

    elif style == "grid":
        # 3x3 Grid for Top Face
        top_origin_x = cx + off_top[0]
        top_origin_y = cy - s + off_top[1]
        top_group = dwg.g(
            transform=f"matrix(1, 0.5, -1, 0.5, {top_origin_x}, {top_origin_y})"
        )
        rect_size_3x3 = (s - 2 * g_inner) / 3
        for j in range(3):
            for i in range(3):
                x, y = i * (rect_size_3x3 + g_inner), j * (rect_size_3x3 + g_inner)
                top_group.add(
                    dwg.rect(
                        insert=(x, y),
                        size=(rect_size_3x3, rect_size_3x3),
                        fill=top_face_color,
                    )
                )
        dwg.add(top_group)

        # 3x3 Grid for Right Face
        right_origin_x = cx + off_right[0]
        right_origin_y = cy + off_right[1]
        right_group = dwg.g(
            transform=f"matrix(1, -0.5, 0, 1, {right_origin_x}, {right_origin_y})"
        )
        for j in range(3):
            for i in range(3):
                x, y = i * (rect_size_3x3 + g_inner), j * (rect_size_3x3 + g_inner)
                right_group.add(
                    dwg.rect(
                        insert=(x, y),
                        size=(rect_size_3x3, rect_size_3x3),
                        fill=right_face_color,
                    )
                )
        dwg.add(right_group)

    # --- Left Face Tensor Grid (5x5 Hollow 'W') ---
    tile_n_left = 5
    tile_size_left = s / tile_n_left
    tile_gap_left = 2

    left_origin_x = cx - s + off_left[0]
    left_origin_y = cy - s / 2 + off_left[1]
    left_group = dwg.g(
        transform=f"matrix(1, 0.5, 0, 1, {left_origin_x}, {left_origin_y})"
    )

    W_mask = [
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0],
    ]

    for j in range(tile_n_left):
        for i in range(tile_n_left):
            if W_mask[j][i] == 1:
                x = i * tile_size_left + (tile_gap_left / 2)
                y = j * tile_size_left + (tile_gap_left / 2)
                rect_size_left = tile_size_left - tile_gap_left
                left_group.add(
                    dwg.rect(
                        insert=(x, y),
                        size=(rect_size_left, rect_size_left),
                        fill=w_square_color,
                    )
                )

    dwg.add(left_group)

    # --- Main Text "webtensor" ---
    if with_text:
        # Font size increased to 160px. Baseline adjusted slightly to keep it vertically aligned with the cube.
        dwg.add(
            dwg.text(
                "webtensor",
                insert=(cx + s + 32, cy + 55),
                fill=main_text_color,
                style="font-size:160px; font-family:sans-serif; font-weight:bold; letter-spacing: -4px;",
            )
        )

    dwg.save()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    folders = ["solid", "grid"]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

        create_webtensor_logo(
            os.path.join(folder, "light.svg"), dark=False, with_text=True, style=folder
        )
        create_webtensor_logo(
            os.path.join(folder, "dark.svg"), dark=True, with_text=True, style=folder
        )
        create_webtensor_logo(
            os.path.join(folder, "favicon.svg"), with_text=False, style=folder
        )
