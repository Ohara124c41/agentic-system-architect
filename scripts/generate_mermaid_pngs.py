"""Generate PNG images from Mermaid diagram files."""

import os
from pathlib import Path
from mermaid import Mermaid

def generate_pngs():
    """Generate PNG files for all .mmd files in docs/figures/."""

    # Paths
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "docs" / "figures"
    output_dir = project_root / "outputs" / "figures"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .mmd files
    mmd_files = list(figures_dir.glob("*.mmd"))

    print(f"Found {len(mmd_files)} Mermaid diagram files")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    for mmd_file in sorted(mmd_files):
        print(f"Processing: {mmd_file.name}")

        # Read the mermaid content
        with open(mmd_file, 'r', encoding='utf-8') as f:
            mermaid_code = f.read()

        # Output path
        output_path = output_dir / f"{mmd_file.stem}.png"

        try:
            # Create Mermaid object and render
            diagram = Mermaid(mermaid_code)

            # Save to file
            diagram.to_png(str(output_path))
            print(f"  -> Saved: {output_path.name}")

        except Exception as e:
            print(f"  -> ERROR: {e}")

    print("-" * 50)
    print("Done!")

if __name__ == "__main__":
    generate_pngs()
