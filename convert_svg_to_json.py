#!/usr/bin/env python3
"""
Converts a folder of SVG floorplans into the JSON format required for the
Floorplan Copilot project.

This version processes ALL geometry (<path>, <circle>, <ellipse>) found in the
SVG files, without filtering by layer ID.

Each SVG file is converted to a corresponding JSON file. The JSON file will
contain a list of polylines, where each polyline is a list of [x, y] points.

Usage:
    python convert_svg_to_json.py --input_folder /path/to/your/svgs --output_folder data/raw

The script will create the output folder if it doesn't exist.
"""
import os
import json
import argparse
import math
import re
from xml.etree import ElementTree
from tqdm import tqdm

# --- Configuration ---

# Adjust this scaling factor. It converts the SVG coordinate units
# into meters. For example, if the viewBox is 100 units wide and this
# represents 25 meters, the factor would be = 25 / 100 = 0.25.
# By default, coordinates are extracted as-is.
SCALE_FACTOR = 1.0

# Number of segments for approximating circles and ellipses
CIRCLE_SEGMENTS = 16

# --- SVG Parsing Logic ---


def parse_path_d(d_string):
    """
    Parses the 'd' attribute of an SVG path into a list of polylines.
    Supports M, L, H, V, A, and Z commands (absolute and relative).
    """
    # Regex to find commands and their numeric arguments
    path_regex = re.compile(r"([MLHVACZ])([^MLHVACZ]*)", re.IGNORECASE)
    polylines = []
    current_poly = []
    last_point = [0, 0]

    for command, args_str in path_regex.findall(d_string):
        try:
            args = [float(n) for n in re.findall(r"-?\d*\.?\d+", args_str)]
        except ValueError:
            continue  # Skip if arguments are not valid numbers

        is_relative = command.islower()
        command = command.upper()

        if command == "M":  # Moveto
            if current_poly:
                polylines.append(current_poly)
            current_poly = []
            x, y = args[0], args[1]
            if is_relative:
                x += last_point[0]
                y += last_point[1]
            current_poly.append([x, y])
            last_point = [x, y]
            # Handle implicit Lineto commands after M
            for i in range(2, len(args), 2):
                x, y = args[i], args[i + 1]
                if is_relative:
                    x += last_point[0]
                    y += last_point[1]
                current_poly.append([x, y])
                last_point = [x, y]

        elif command == "L":  # Lineto
            for i in range(0, len(args), 2):
                x, y = args[i], args[i + 1]
                if is_relative:
                    x += last_point[0]
                    y += last_point[1]
                current_poly.append([x, y])
                last_point = [x, y]

        elif command == "H":  # Horizontal Lineto
            for x in args:
                if is_relative:
                    x += last_point[0]
                y = last_point[1]
                current_poly.append([x, y])
                last_point = [x, y]

        elif command == "V":  # Vertical Lineto
            for y in args:
                if is_relative:
                    y += last_point[1]
                x = last_point[0]
                current_poly.append([x, y])
                last_point = [x, y]

        elif command == "Z":  # ClosePath
            if current_poly:
                # Close the path by appending the first point
                current_poly.append(current_poly[0])
                polylines.append(current_poly)
                current_poly = []

        elif command == "A":  # Elliptical Arc - we approximate as a line
            # A simple line to the arc's end-point to preserve connectivity
            if len(args) >= 7:
                for i in range(0, len(args), 7):
                    x, y = args[i + 5], args[i + 6]
                    if is_relative:
                        x += last_point[0]
                        y += last_point[1]
                    current_poly.append([x, y])
                    last_point = [x, y]

    if current_poly:
        polylines.append(current_poly)

    return polylines


def parse_circle(el):
    """Parses a <circle> element into a polygonal approximation."""
    try:
        cx = float(el.get("cx"))
        cy = float(el.get("cy"))
        r = float(el.get("r"))
    except (TypeError, ValueError):
        return []

    points = []
    for i in range(CIRCLE_SEGMENTS + 1):
        angle = 2 * math.pi * i / CIRCLE_SEGMENTS
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append([x, y])
    return [points]


def parse_ellipse(el):
    """Parses an <ellipse> element. Ignores rotations for simplicity."""
    try:
        cx = float(el.get("cx"))
        cy = float(el.get("cy"))
        rx = float(el.get("rx"))
        ry = float(el.get("ry"))
    except (TypeError, ValueError):
        return []

    points = []
    for i in range(CIRCLE_SEGMENTS + 1):
        angle = 2 * math.pi * i / CIRCLE_SEGMENTS
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        points.append([x, y])
    return [points]


def process_svg_file(svg_path):
    """Extracts polylines from a single SVG file."""
    try:
        tree = ElementTree.parse(svg_path)
        root = tree.getroot()
    except ElementTree.ParseError:
        print(f"Warning: Could not parse {svg_path}. Skipping.")
        return None

    ns = {"svg": "http://www.w3.org/2000/svg"}
    all_polylines = []

    # Find all geometric elements regardless of their layer
    for el in root.findall(".//svg:path", ns):
        d = el.get("d")
        if d:
            all_polylines.extend(parse_path_d(d))

    for el in root.findall(".//svg:circle", ns):
        all_polylines.extend(parse_circle(el))

    for el in root.findall(".//svg:ellipse", ns):
        all_polylines.extend(parse_ellipse(el))

    # Scale all points if a factor is provided
    if SCALE_FACTOR != 1.0:
        for poly in all_polylines:
            for i, (x, y) in enumerate(poly):
                poly[i] = [x * SCALE_FACTOR, y * SCALE_FACTOR]

    return all_polylines


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
            # Create the output filename
            base_name = os.path.splitext(svg_file)[0]
            output_path = os.path.join(args.output_folder, f"{base_name}.json")

            # Save as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(polylines, f, indent=2)

    print("\nConversion complete.")
    print(f"Check the output folder: '{args.output_folder}'")
    print(
        f"IMPORTANT: Make sure the 'SCALE_FACTOR' ({SCALE_FACTOR}) is set correctly to convert coordinates to meters."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts SVG floorplans to JSON format."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data/svg",
        help="Path to the folder containing the SVG files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/raw",
        help="Path to the output folder for the JSON files (default: 'data/raw').",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
