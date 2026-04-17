import svgwrite


def main(filename="webtensor_logo.svg", dark=False, with_text=True):
    # Canvas sizing based on variant
    canvas_size = (740, 200) if with_text else (190, 190)
    dwg = svgwrite.Drawing(filename, profile="full", size=canvas_size)

    # --- Always Transparent Background ---

    # --- Official Color Palette ---
    color_rust_orange = "#F05032"
    color_webgpu_blue_dark = "#015a9c"
    color_webgpu_blue_light = "#008df3"
    color_white = "#FFFFFF"

    # --- 3D Face Mapping ---
    top_face_color = color_rust_orange
    right_face_color = color_webgpu_blue_dark

    # --- Grid Face Mapping ---
    w_square_color = color_webgpu_blue_light

    # Text color switches based on light/dark mode for visibility
    main_text_color = color_white if dark else color_webgpu_blue_dark

    # --- Cube Geometry (Isometric Exploded View) ---
    cx = 100 if with_text else 95
    cy = 95
    s = 80

    # The gap between the 3 main faces
    g = 3

    # Double the gap for the internal 3x3 grids on the Top and Right faces
    g_inner = g * 2

    # Isometric directional offsets to pull faces outward
    off_top = (0, -g)
    off_left = (-g, g / 2)
    off_right = (g, g / 2)

    # --- Top Face Tensor Grid (3x3) ---
    # Origin is the top-most vertex of the cube
    top_origin_x = cx + off_top[0]
    top_origin_y = cy - s + off_top[1]

    top_group = dwg.g(
        transform=f"matrix(1, 0.5, -1, 0.5, {top_origin_x}, {top_origin_y})"
    )

    # Calculate square size for a 3x3 grid (3 squares, 2 gaps)
    rect_size_3x3 = (s - 2 * g_inner) / 3

    for j in range(3):
        for i in range(3):
            x = i * (rect_size_3x3 + g_inner)
            y = j * (rect_size_3x3 + g_inner)

            top_group.add(
                dwg.rect(
                    insert=(x, y),
                    size=(rect_size_3x3, rect_size_3x3),
                    fill=top_face_color,
                )
            )

    dwg.add(top_group)

    # --- Right Face Tensor Grid (3x3) ---
    # Origin is the center vertex of the cube
    right_origin_x = cx + off_right[0]
    right_origin_y = cy + off_right[1]

    right_group = dwg.g(
        transform=f"matrix(1, -0.5, 0, 1, {right_origin_x}, {right_origin_y})"
    )

    for j in range(3):
        for i in range(3):
            x = i * (rect_size_3x3 + g_inner)
            y = j * (rect_size_3x3 + g_inner)

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
    tile_gap_left = 2  # Kept small to preserve the 5x5 W resolution

    # Origin is the left-most vertex of the cube
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
        # Pushed right to account for the exploded gap
        dwg.add(
            dwg.text(
                "webtensor",
                insert=(cx + s + 40, cy + 35),
                fill=main_text_color,
                style="font-size:100px; font-family:sans-serif; font-weight:bold; letter-spacing: -2px;",
            )
        )

    dwg.save()
    print(f"Logo saved as {filename}")


if __name__ == "__main__":
    main("logo_light.svg", dark=False, with_text=True)
    main("logo_dark.svg", dark=True, with_text=True)
    main("icon_light.svg", dark=False, with_text=False)
    main("icon_dark.svg", dark=True, with_text=False)
