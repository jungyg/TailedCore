import numpy as np
import cv2
import os


def generate_colors(n):
    """Generate n distinct colors."""
    colors = []
    for i in range(n):
        # Generate bright and distinct colors
        hue = int(255 * i / n)
        saturation = 255
        value = 255
        colors.append(
            cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][
                0
            ]
        )
    return colors


def draw_x(img, center, size, color, thickness):
    """Draw an 'X' on the image at the specified center, size, and color."""
    cv2.line(
        img,
        (center[0] - size, center[1] - size),
        (center[0] + size, center[1] + size),
        color,
        thickness,
    )
    cv2.line(
        img,
        (center[0] + size, center[1] - size),
        (center[0] - size, center[1] + size),
        color,
        thickness,
    )


def draw_legend(
    legend_img, label_names, colors, box_height=20, box_width=30, font_scale=1.0
):
    """Draw a legend on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # text_color = (255, 255, 255)
    text_color = (0, 0, 0)
    for i, (label_name, color) in enumerate(zip(label_names, colors)):
        bottom_left = (10, i * (box_height + 5) + 10)  # Add some padding
        top_right = (bottom_left[0] + box_width, bottom_left[1] + box_height)
        cv2.rectangle(
            legend_img, bottom_left, top_right, color.astype(int).tolist(), -1
        )  # Ensure color is int
        cv2.putText(
            legend_img,
            label_name,
            (bottom_left[0] + box_width + 5, bottom_left[1] + box_height - 5),
            font,
            font_scale,
            text_color,
        )


def draw_extra_scores(
    img, scores, y_positions, line_length_scale, height, margin, linewidth
):
    """Draw horizontal lines corresponding to extra scores with aligned left points."""
    start_x = 0  # Starting x-coordinate for all lines
    for score, y_position in zip(scores, y_positions):
        end_x = start_x + int(line_length_scale * score)
        center_y = int(np.interp(y_position, (-1, 1), (height - margin - 1, margin)))
        cv2.line(img, (start_x, center_y), (end_x, center_y), (0, 0, 0), linewidth)


def plot_scatter(
    scores,
    labels,
    label_names,
    extra_labels=None,
    extra_scores=None,
    th=None,
    point_size=2,
    linewidth=1,
    width=320,
    height=1280,
    y_lim=(-1, 1),
    margin=60,
    filename="scatter_plot.jpg",
):

    if extra_labels is None:
        extra_labels = np.zeros_like(labels)

    if len(scores) != len(labels) or len(scores) != len(extra_labels):
        raise ValueError("Length of scores, labels, and extra_labels must be the same")

    # img = np.zeros((height, width, 3), dtype=np.uint8)
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    normalized_scores = np.interp(scores, y_lim, (height - margin - 1, margin))
    unique_labels = set(labels)
    colors = generate_colors(len(unique_labels))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    x_center = width // 2

    for score, label, extra_label in zip(normalized_scores, labels, extra_labels):
        color = label_to_color[label]
        center = (x_center, int(score))
        if extra_label == 0:
            cv2.circle(img, center, point_size, color.tolist(), -1)
        else:
            shifted_center = (center[0] + 20, center[1])
            draw_x(img, shifted_center, point_size, color.tolist(), point_size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for y_value in np.linspace(y_lim[0], y_lim[1], 11):
        normalized_y = int(np.interp(y_value, y_lim, (height - margin - 1, margin)))
        text = f"{y_value:.2f}"
        cv2.putText(img, text, (5, normalized_y), font, 1, (0, 0, 0), 1)

    if th is not None:
        normalized_th = int(np.interp(th, y_lim, (height - margin - 1, margin)))
        cv2.line(
            img,
            (0, normalized_th),
            (width, normalized_th),
            (0, 0, 0),
            linewidth,
            lineType=cv2.LINE_AA,
        )

    # Extra scores image
    if extra_scores is not None:
        extra_scores_img = np.full((height, width, 3), 255, dtype=np.uint8)
        max_line_length = width - margin  # Maximum length of the horizontal line
        draw_extra_scores(
            extra_scores_img,
            extra_scores,
            scores,
            max_line_length,
            height,
            margin,
            linewidth,
        )

        img = np.hstack((img, extra_scores_img))

    # Create a separate image for the legend
    legend_height = height
    legend_width = width  # Adjust as needed
    legend_img = np.full((legend_height, legend_width, 3), 255, dtype=np.uint8)
    draw_legend(legend_img, label_names, colors)

    combined_img = np.hstack((img, legend_img))

    # Save the combined plot as a jpg file
    cv2.imwrite(filename, combined_img)


# Example usage
if __name__ == "__main__":
    scores = np.array([0.2, -0.3, 0.5, 0.7, -0.1, 0.3, -0.5, 0.8, 1.0])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1])
    extra_labels = np.array(
        [0, 1, 0, 1, 0, 1, 0, 0, 1]
    )  # Extra labels indicating circle (0) or 'x' (non-zero)
    extra_scores = np.array([0.2, 1, 0.5, 1, 0.4, 1, 0.7, 0.25, 1])
    # extra_scores = None
    label_names = ["Category A", "Category B"]
    th = 0.0  # Example threshold value

    plot_scatter(
        scores,
        labels,
        label_names,
        extra_labels,
        extra_scores,
        th,
        point_size=5,
        linewidth=2,
        margin=80,
        filename="my_colored_scatter_cv2_with_legend.jpg",
    )
