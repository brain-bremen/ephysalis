import h5py
import sys
import argparse


def print_tree(name, obj, indent=0):
    """Recursively print the structure of an HDF5 file as a tree."""
    print("  " * indent + f"- {name}")
    if isinstance(obj, h5py.Group):
        for key, item in obj.items():
            print_tree(key, item, indent + 1)


def main(file_path):
    try:
        with h5py.File(file_path, "r") as hdf_file:
            print(f"Structure of '{file_path}':")
            print_tree("/", hdf_file["/"])
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display the structure of an HDF5 file."
    )
    parser.add_argument("file_path", help="Path to the HDF5 file.")
    args = parser.parse_args()

    main(args.file_path)
