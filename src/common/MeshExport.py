from pathlib import Path
import trimesh


def export_to_glb(mesh: trimesh.Scene, output_path: str) -> None:
    """Export a trimesh Scene to GLB format.

    Args:
        mesh: The trimesh Scene to export.
        output_path: Path for the output .glb file.
    """
    # Ensure output directory exists
    output_path = str(output_path)
    parent = Path(output_path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    mesh.export(file_obj=output_path, file_type="glb")