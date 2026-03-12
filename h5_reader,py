import h5py
import argparse

"""
This script prints the internal structure of an HDF5 file (groups, datasets, shapes, and dtypes).
You provide the HDF5 file path with the -p or --path option.
"""

def print_h5_structure(h5_file_path):
    def recursively_print(name, obj):
        indent = '  ' * (name.count('/') - 1)
        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}[Dataset] {name} - shape: {obj.shape}, dtype: {obj.dtype}")

    with h5py.File(h5_file_path, 'r') as f:
        print(f"HDF5 file: {h5_file_path}")
        f.visititems(recursively_print)


def main():
    parser = argparse.ArgumentParser(
        description="Print HDF5 structure (groups/datasets, shapes, dtypes)."
    )
    parser.add_argument("-p", "--path", required=True, help="Path to the HDF5 file")
    #parser.add_argument("-p", "--path", required=True, help="Path to the HDF5 file")
    args = parser.parse_args()

    print_h5_structure(args.path)

if __name__ == "__main__":
    main()