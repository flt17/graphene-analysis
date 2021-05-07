import math
import numpy as np
import pandas
import scipy
import scipy.signal
import sys, os
from tqdm.notebook import tqdm
from ovito.io import import_file
from ovito.modifiers import *
from ovito.vis import Viewport, TachyonRenderer


sys.path.append("../")
from graphene_analysis import global_variables
from graphene_analysis import utils


class KeyNotFound(Exception):
    pass


class VariableNotSet(Exception):
    pass


class UnphysicalValue(Exception):
    pass


class GrapheneSystem:
    """
    Gather computed properties from simulations for easy comparison.
    Attributes:
    Methods:
    """

    def __init__(self, name: str):

        """
        Arguments:
          name (str) :  The name of the instance of the class.
          simulations (dictonary) : Dictionary of all simulations performed on the given system
                                    labelled by user-given names.
        """
        self.name = name
        self.simulations = {}

    def add_simulation(self, simulation_name: str, directory_path: str):
        """
        Initialise instance of Simulation class with given name and directory path and add
        it ot the simulation dictionary.
        Arguments:
            simulation_name (str) : Name which will be used in a dictionary to access the
                                computed properties and raw data.
            directory path (str) :  Path to the simulation directory.
        Returns:
        """

        self.simulations[simulation_name] = Simulation(directory_path, self.name)


class Simulation:
    """
    Perform post-processing of a MD simulation.
    Attributes:
    Methods:
    """

    def __init__(self, directory_path: str, system: str = None):
        """
        Arguments:
            directory path (str) :  Path to the simulation directory.
        """
        self.system = system
        self.defect_type = "Pristine"
        self.directory_path = directory_path
        self.time_between_frames = None
        # set system periodicity per default:
        self.set_pbc_dimensions("xyz")

    def set_pbc_dimensions(self, pbc_dimensions: str):
        """
        Set in which direction pbc apply.
        Arguments:
            pbc_dimensions (str) : string of directions in which pbc apply.
        Returns:
        """

        if not global_variables.DIMENSION_DICTIONARY.get(pbc_dimensions):
            raise KeyNotFound(
                f"Specified dimension {pbc_dimensions} is unknown. Possible options are {global_variables.DIMENSION_DICTIONARY.keys()}"
            )
        self.pbc_dimensions = pbc_dimensions

    def set_sampling_times(
        self,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
        time_between_frames: float = None,
    ):

        """
        Set times for analysis of trajectories.
        Arguments:
            start_time (int) : Start time for analysis.
            end_time (int) : End time for analysis.
            frame_frequency (int): Take every nth frame only.
            time_between_frames (float): Time (in fs) between two frames in sampled trajectory, e.g. 100 fs.
        Returns:
        """

        self.start_time = start_time if start_time is not None else self.start_time
        self.frame_frequency = (
            frame_frequency if frame_frequency is not None else self.frame_frequency
        )
        self.time_between_frames = (
            time_between_frames
            if time_between_frames is not None
            else self.time_between_frames
        )

        total_time = (
            self.position_universe.trajectory.n_frames - 1
        ) * self.time_between_frames
        self.end_time = (
            total_time
            if end_time == -1
            else end_time
            if end_time is not None
            else self.end_time
        )

        print(f"SUCCESS: New sampling times.")
        print(f"Start time: \t \t{self.start_time} \t fs")
        print(f"End time: \t \t{self.end_time} \t fs")
        print(f"Time between frames: \t{self.time_between_frames} \t fs")
        print(f"Frame frequency: \t{self.frame_frequency}")

    def _get_sampling_frames(
        self,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
        time_between_frames: float = None,
    ):
        """
        Determine sampling frames from given sampling times.
        Arguments:
            start_time (int) : Start time for analysis.
            end_time (int) : End time for analysis.
            frame_frequency (int): Take every nth frame only.
            time_between_frames (float): Time (in fs) between frames. Usually, this is set at the very beginning.
                            Exception applies only to calculation of friction where this is set in the method.
        Returns:
            start_frame (int) : Start frame for analysis.
            end_frame (int) : End frame for analysis.
            frame_frequency (int): Take every nth frame only.
        """
        time_between_frames = (
            time_between_frames if time_between_frames else self.time_between_frames
        )

        start_time = start_time if start_time is not None else self.start_time
        end_time = end_time if end_time is not None else self.end_time

        frame_frequency = int(
            frame_frequency if frame_frequency is not None else self.frame_frequency
        )

        start_frame = int(start_time / time_between_frames)
        end_frame = int(end_time / time_between_frames)

        return start_frame, end_frame, frame_frequency

    def read_in_simulation_data(
        self,
        trajectory_file_name: str = None,
        topology_file_name: str = None,
        trajectory_format: str = "dcd",
    ):

        """
        Setup all selected simulation data.
        Arguments:
            trajectory_file_name (str) : Name of the trajcetory file.
            topology_file_name (str) : Name of the topology file (currently only pdb). If not given, first file taken.
            trajectory_format (str) : File format of trajectory, default is dcd.
        Returns:
        """
        # setup topology based on only pdb file in directoy
        path_to_topology = utils.get_path_to_file(
            self.directory_path, "pdb", topology_file_name
        )
        self.path_to_topology = path_to_topology

        self.topology = utils.get_ase_atoms_object(path_to_topology)

        # Read in trajectory
        self.position_universe = utils.get_mdanalysis_universe(
            self.directory_path,
            trajectory_file_name,
            topology_file_name,
            trajectory_format,
        )

        self.species_in_system = np.unique(self.position_universe.atoms.names)

    def find_defective_atoms(self):
        """
        Find all defective atoms in the system. This part of the code was highly supported by Michael B Davies.
        Arguments:
        Returns:
        """

        # create the ovito “pipeline”
        pipeline = import_file(os.path.abspath(self.path_to_topology))
        PTM = PolyhedralTemplateMatchingModifier(
            color_by_type=True, output_orientation=True
        )

        # list of all the different types it can calc (see doc)
        ovito_structures = np.array(
            [
                (0, "other"),
                (1, "fcc"),
                (2, "hcp"),
                (3, "bcc"),
                (4, "ico"),
                (5, "sc"),
                (6, "cubic"),
                (7, "hex"),
                (8, "graphene"),
            ]
        )

        # tell it to calculate graphene (not on by default)
        PTM.structures[PolyhedralTemplateMatchingModifier.Type.GRAPHENE].enabled = True

        # append to pipeline
        pipeline.modifiers.append(PTM)

        # run calc
        data = pipeline.compute()

        # summary
        n_struct = ovito_structures[np.unique(data.particles["Structure Type"])]
        count_struct = np.bincount(np.array(data.particles["Structure Type"]))

        self.defective_atoms_ids = np.where(
            np.array(data.particles["Structure Type"]) == 0
        )[0]
        self.pristine_atoms_ids = np.where(
            np.array(data.particles["Structure Type"]) == 8
        )[0]

        # finish by clustering atoms to defects
        self._assign_defective_atoms_to_defects()

    def _assign_defective_atoms_to_defects(self):
        """
        Assign defective atoms to specific defects.
        Arguments:
        Returns:
        """

        # allowed atoms per defect for each type
        allowed_atoms_per_type = {"Divacancy": 12, "Stone-Wales": 8}

        # get type from system:
        allowed_names_per_type = {
            "Divacancy": ["DV", "Divacancy"],
            "Stone-Wales": ["SW", "Stone-Wales"],
        }

        # assign defect type:
        for keys, values in allowed_names_per_type.items():
            if any(allowed_acronym in self.system for allowed_acronym in values):
                self.defect_type = keys

        # if not found assign pristine and print error
        if self.defect_type == "Pristine":
            raise KeyNotFound(
                f"Cannot guess defect type from system name given: {self.system}",
                f"Please rename the system.",
            )

        # check split is possible
        if (
            len(self.defective_atoms_ids) % allowed_atoms_per_type.get(self.defect_type)
            != 0
        ):
            raise UnphysicalValue(
                f"From system name I guessed you are dealing with {self.defect_type} defects.",
                f"Each {self.defect_type} is formed by {allowed_atoms_per_type.get(self.defect_type)} atoms.",
                f"However, I found {len(self.defective_atoms_ids)} defective atoms which cannot be properly divided.",
            )

        # if everything worked we can continue clustering the atoms
        # start by computing vectors between all defective atoms
        vectors_between_atoms = (
            self.topology[self.defective_atoms_ids].positions[np.newaxis, :]
            - self.topology[self.defective_atoms_ids].positions[:, np.newaxis]
        )

        # apply MIC
        vectors_MIC = utils.apply_minimum_image_convention_to_interatomic_vectors(
            vectors_between_atoms, self.topology.cell
        )

        # get distances based on vectors
        distances = np.linalg.norm(vectors_MIC, axis=2)

        # sort distances and only take the first N elements and only once
        indices_assigned = np.unique(
            (
                np.sort(
                    np.argsort(distances)[
                        :, 0 : allowed_atoms_per_type.get(self.defect_type)
                    ]
                )
            ),
            axis=0,
        )

        # use this indices to get atom ids from defective atoms
        self.defective_atoms_ids_clustered = self.defective_atoms_ids[indices_assigned]

    def sample_atomic_height_distribution(
        self, start_time: int = None, end_time: int = None, frame_frequency: int = None
    ):
        """
        Sample the atomic heights of the graphene sheet relativ to the COM.
        Arguments:
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:
        """

        # get information about sampling
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # use local variable for universe
        tmp_universe = self.position_universe

        # define local variable to save heights
        atomic_heights = []

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm(
                (tmp_universe.trajectory[start_frame:end_frame])[
                    :: int(frame_frequency)
                ]
            )
        ):

            # get center of mass, here this corresponds to center of geometry
            center_of_mass = tmp_universe.atoms.center_of_mass()

            # save relative heights to array
            atomic_heights.extend(
                tmp_universe.atoms.positions[:, 2] - center_of_mass[2]
            )

        # save array format as numpy array to class
        self.atomic_height_distribution = np.array(atomic_heights)
