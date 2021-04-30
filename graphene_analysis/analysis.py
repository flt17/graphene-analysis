import math
import numpy as np
import pandas
import scipy
import scipy.signal
import sys
from tqdm.notebook import tqdm


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

        self.simulations[simulation_name] = Simulation(directory_path)


class Simulation:
    """
    Perform post-processing of a MD simulation.
    Attributes:
    Methods:
    """

    def __init__(self, directory_path: str):
        """
        Arguments:
            directory path (str) :  Path to the simulation directory.
        """

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
            self.position_universes[0].trajectory.n_frames - 1
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
        self.topology = utils.get_ase_atoms_object(path_to_topology)

        # Read in trajectory
        self.position_universe = utils.get_mdanalysis_universe(
            self.directory_path,
            trajectory_file_name,
            topology_file_name,
            trajectory_format,
        )

        self.species_in_system = np.unique(self.position_universe.atoms.names)
