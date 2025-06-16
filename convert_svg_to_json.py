#!/usr/bin/env python3
"""
Converts a folder of SVG floorplans into a sorted JSON format.
This version uses keyword-based heuristics to sort layers and conditionally
applies a spatial sort to layers below a certain complexity threshold.
"""
import os
import json
import argparse
import math
import re
from xml.etree import ElementTree
from tqdm import tqdm
import numpy as np

# --- Configuration ---

# The script will only perform the expensive spatial sort on layers
# with FEWER polylines than this number. Adjust as needed.
SPATIAL_SORT_THRESHOLD = 1500

# Layers are sorted based on this priority dictionary.
SORTING_KEYWORDS = {
    0: ["wall", "墙", "砼", "conc"],
    1: ["colu", "柱", "结构", "梁"],
    2: ["stair", "elev", "core", "梯"],
    3.0: ["door", "门"],
    3.1: ["win", "窗"],
    4: ["fixt", "furn", "eqpm", "sanitary", "家具", "洁具", "设备"],
    5: ["rail", "sill", "栏杆"],
    6: ["hatch", "patt", "填充"],
    7: ["dote", "hidd", "lower", "void"],
    8: ["park", "prkg", "车位"],
}

# Layers with these keywords will be ignored entirely.
EXCLUSION_KEYWORDS = [
    "text",
    "dims",
    "axis",
    "numb",
    "iden",
    "mark",
    "sign",
    "level",
    "area",
    "plot",
    "pltw",
    "edit",
    "文字",
    "尺寸",
    "标注",
    "索引",
    "defpoint",
    "noprint",
    "标高",
    "说明",
    "编号",
    "名称",
    "参照",
    "图框",
]

SCALE_FACTOR = 1.0
CIRCLE_SEGMENTS = 16

# --- Sorting and Parsing Logic ---


def get_layer_priority(layer_name):
    """Assigns a priority score to a layer based on keywords."""
    lowered_name = layer_name.lower()
    for keyword in EXCLUSION_KEYWORDS:
        if keyword in lowered_name:
            return 99  # Exclude
    for priority, keywords in SORTING_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered_name:
                return priority
    return 10  # Default priority


def sort_polylines_spatially(polylines):
    """
    Sorts a list of polylines to create a more continuous path.
    This is an O(n^2) operation and should only be used on small lists.
    """
    if len(polylines) < 2:
        return polylines

    # Use the centroid for a more stable starting point
    start_idx = min(
        range(len(polylines)),
        key=lambda i: (
            np.mean(polylines[i], axis=0)[1],
            np.mean(polylines[i], axis=0)[0],
        ),
    )

    sorted_polylines = [polylines.pop(start_idx)]

    while polylines:
        last_point = sorted_polylines[-1][-1]
        closest_idx = min(
            range(len(polylines)),
            key=lambda i: np.linalg.norm(np.array(polylines[i][0]) - last_point),
        )
        sorted_polylines.append(polylines.pop(closest_idx))

    return sorted_polylines


def parse_path_d(d_string):
    """Parses the 'd' attribute of an SVG path, returning a list of numpy arrays."""
    path_regex = re.compile(r"([MLHVACZ])([^MLHVACZ]*)", re.IGNORECASE)
    polylines, current_poly, last_point = [], [], [0, 0]
    for command, args_str in path_regex.findall(d_string):
        try:
            args = [float(n) for n in re.findall(r"-?\d*\.?\d+", args_str)]
        except ValueError:
            continue
        is_relative, command = command.islower(), command.upper()
        if command == "M":
            if len(args) < 2:
                continue
            if current_poly:
                polylines.append(current_poly)
            current_poly, x, y = [], args[0], args[1]
            if is_relative:
                x += last_point[0]
                y += last_point[1]
            current_poly.append([x, y])
            last_point = [x, y]
            for i in range(2, len(args), 2):
                if i + 1 >= len(args):
                    continue
                x, y = args[i], args[i + 1]
                if is_relative:
                    x += last_point[0]
                    y += last_point[1]
                current_poly.append([x, y])
                last_point = [x, y]
        elif command == "L":
            if not args:
                continue
            for i in range(0, len(args), 2):
                if i + 1 >= len(args):
                    continue
                x, y = args[i], args[i + 1]
                if is_relative:
                    x += last_point[0]
                    y += last_point[1]
                current_poly.append([x, y])
                last_point = [x, y]
        elif command == "H":
            for x in args:
                if is_relative:
                    x += last_point[0]
                y = last_point[1]
                current_poly.append([x, y])
                last_point = [x, y]
        elif command == "V":
            for y in args:
                if is_relative:
                    y += last_point[1]
                x = last_point[0]
                current_poly.append([x, y])
                last_point = [x, y]
        elif command == "Z":
            if current_poly:
                if len(current_poly) > 1 and tuple(current_poly[0]) != tuple(
                    current_poly[-1]
                ):
                    current_poly.append(current_poly[0])
                polylines.append(current_poly)
                current_poly = []
        elif command == "A":
            for i in range(0, len(args), 7):
                if i + 6 >= len(args):
                    continue
                x, y = args[i + 5], args[i + 6]
                if is_relative:
                    x += last_point[0]
                    y += last_point[1]
                current_poly.append([x, y])
                last_point = [x, y]
    if current_poly:
        polylines.append(current_poly)
    return [np.array(p) for p in polylines]


def parse_shape(el):
    """Parses <circle> and <ellipse> elements into a numpy array."""
    points = []
    try:
        if el.tag.endswith("circle"):
            cx, cy, r = float(el.get("cx")), float(el.get("cy")), float(el.get("r"))
            for i in range(CIRCLE_SEGMENTS + 1):
                angle = 2 * math.pi * i / CIRCLE_SEGMENTS
                points.append([cx + r * math.cos(angle), cy + r * math.sin(angle)])
        elif el.tag.endswith("ellipse"):
            cx, cy, rx, ry = (
                float(el.get("cx")),
                float(el.get("cy")),
                float(el.get("rx")),
                float(el.get("ry")),
            )
            for i in range(CIRCLE_SEGMENTS + 1):
                angle = 2 * math.pi * i / CIRCLE_SEGMENTS
                points.append([cx + rx * math.cos(angle), cy + ry * math.sin(angle)])
    except (TypeError, ValueError):
        return []
    return [np.array(points)]


def process_svg_file(svg_path):
    """Extracts and sorts polylines from a single SVG file."""
    try:
        tree = ElementTree.parse(svg_path)
        root = tree.getroot()
    except ElementTree.ParseError:
        print(f"\nWarning: Could not parse {svg_path}. Skipping.")
        return None

    ns = {"svg": "http://www.w3.org/2000/svg"}
    polylines_by_layer = {}

    for g in root.findall(".//svg:g[@id]", ns):
        layer_id = g.get("id")
        if not layer_id:
            continue
        priority = get_layer_priority(layer_id)
        if priority >= 99:
            continue

        layer_polylines = []
        for el in g.findall("./svg:path", ns):
            d = el.get("d")
            if d:
                layer_polylines.extend(parse_path_d(d))
        for el in g.findall("./svg:circle", ns) + g.findall("./svg:ellipse", ns):
            layer_polylines.extend(parse_shape(el))

        if layer_polylines:
            if priority not in polylines_by_layer:
                polylines_by_layer[priority] = []
            polylines_by_layer[priority].extend(layer_polylines)

    final_sorted_polylines = []
    for priority in sorted(polylines_by_layer.keys()):
        polys_for_layer = polylines_by_layer[priority]

        # --- NEW: Conditional Spatial Sort ---
        if len(polys_for_layer) <= SPATIAL_SORT_THRESHOLD:
            # It's a small enough layer, so we can afford the spatial sort.
            sorted_polys = sort_polylines_spatially(polys_for_layer)
        else:
            # This layer is too complex, skip the slow sort to save time.
            sorted_polys = polys_for_layer  # Use the original order for this layer.

        final_sorted_polylines.extend([p.tolist() for p in sorted_polys])

    if SCALE_FACTOR != 1.0:
        for poly in final_sorted_polylines:
            for i, (x, y) in enumerate(poly):
                poly[i] = [x * SCALE_FACTOR, y * SCALE_FACTOR]

    return final_sorted_polylines


def main(args):
    """Main function to run the conversion."""
    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' not found.")
        return

    os.makedirs(args.output_folder, exist_ok=True)

    svg_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(".svg")]
    if not svg_files:
        print(f"No .svg files found in folder '{args.input_folder}'.")
        return

    print(
        f"Converting {len(svg_files)} SVG files from '{args.input_folder}' to '{args.output_folder}'..."
    )

    for svg_file in tqdm(svg_files, desc="Converting"):
        input_path = os.path.join(args.input_folder, svg_file)
        polylines = process_svg_file(input_path)

        if polylines:
            base_name = os.path.splitext(svg_file)[0]
            output_path = os.path.join(args.output_folder, f"{base_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(polylines, f, indent=2)

    print("\nConversion complete.")
    print(
        "Dataset has been automatically sorted using keyword heuristics and conditional spatial sorting."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts and sorts SVG floorplans into JSON format."
    )
    # UPDATED with your requested defaults
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data/svg",
        help="Path to the folder containing your SVG files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/raw",
        help="Path to the output folder for the JSON files.",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
