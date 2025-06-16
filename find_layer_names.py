import os
import argparse
from xml.etree import ElementTree
from tqdm import tqdm
from collections import Counter

def find_unique_layer_ids(svg_folder):
    """
    Scans all SVG files in a folder and returns a set of all unique <g> element IDs.
    """
    if not os.path.isdir(svg_folder):
        print(f"Error: Folder not found at '{svg_folder}'")
        return None, None

    svg_files = [f for f in os.listdir(svg_folder) if f.lower().endswith('.svg')]
    if not svg_files:
        print(f"Error: No .svg files found in '{svg_folder}'")
        return None, None

    print(f"Scanning {len(svg_files)} SVG files...")

    all_layer_ids = Counter()
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    for filename in tqdm(svg_files, desc="Scanning SVGs"):
        filepath = os.path.join(svg_folder, filename)
        try:
            tree = ElementTree.parse(filepath)
            root = tree.getroot()
            # Find all <g> elements that have an 'id' attribute
            layers = root.findall('.//svg:g[@id]', ns)
            for layer in layers:
                layer_id = layer.get('id')
                if layer_id:
                    all_layer_ids[layer_id] += 1
        except ElementTree.ParseError:
            print(f"\nWarning: Could not parse XML in '{filename}'. Skipping.")
            continue

    return all_layer_ids

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Find all unique layer names (group IDs) in a directory of SVG files."
    )
    parser.add_argument(
        "--svg_folder",
        type=str,
        default="data/svg",
        help="Path to the folder containing your SVG files."
    )
    args = parser.parse_args()

    layer_counts = find_unique_layer_ids(args.svg_folder)

    if layer_counts:
        # Sort layers alphabetically for a consistent base order
        sorted_layers = sorted(layer_counts.keys())

        print(f"\n✅ Found {len(sorted_layers)} unique layer IDs across all files.")
        print("-" * 50)
        
        print("\nList of all unique layers found (sorted alphabetically):")
        for layer in sorted_layers:
            print(f"- {layer} (found in {layer_counts[layer]} files)")

        print("\n" + "-" * 50)
        print("\nCopy the list below into the `LAYER_DRAWING_ORDER` variable")
        print("in `convert_svg_to_json.py` and rearrange it into a logical drawing order.")
        print("(e.g., walls first, then stairs, windows, fixtures, etc.)")
        print("-" * 50)
        
        # Print in a Python list format for easy copy-pasting
        print("LAYER_DRAWING_ORDER = [")
        for layer in sorted_layers:
            print(f"    '{layer}',")
        print("]")

if __name__ == "__main__":
    main()