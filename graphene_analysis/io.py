import glob
import os

class UnableToFindPdbFile(Exception):
    pass


def get_pdb_file_name(path, pdb_file_name=None):
    """
    Return the correct pdb file name.

    Arguments:
        path: The path to the directory containing the trajectory.

    Returns:
        pdb file name: The name of the pdb file.
    """

    pdb_files_in_path = glob.glob(os.path.join(path, "*.pdb"))

    if not pdb_files_in_path:
        raise UnableToFindPdbFile(f"No pdb-file found in path {path}.")

    # If name for pdb file is given, we take that file.
    if pdb_file_name:
        return [
            file for file in pdb_files_in_path if pdb_file_name + ".pdb" in file
        ][0]
    # Otherwise, we take the first file given by glob.
    else:
        if len(pdb_files_in_path) > 1:
            pdb_file_name_found = pdb_files_in_path[0].split("/")[-1]
            print(
                f"WARNING: More than one pdb-file found. Proceeding with {pdb_file_name_found}"
            )
        return pdb_files_in_path[0]