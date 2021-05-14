import math
import numpy as np
import scipy
import sys, os
from tqdm.notebook import tqdm
from ovito.io import import_file
from ovito.modifiers import *

import MDAnalysis as mdanalysis
from MDAnalysis.lib.distances import capped_distance
from ase.visualize import view

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

    def find_defective_atoms(
        self, pristine: bool = False, number_of_artificial_defects: int = 36
    ):
        """
        Find all defective atoms in the system. This part of the code was highly supported by Michael B Davies.
        Arguments:
            pristine (boolean): If true, random atoms are assigned as defect centers (minimum distance 10A).
            number_of_artificial_defects (int): In case of pristine graphene how many defect centers will be created.
        Returns:
        """

        # standard proceedure if no 'artificial' defects need to be created
        if not pristine:
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
            PTM.structures[
                PolyhedralTemplateMatchingModifier.Type.GRAPHENE
            ].enabled = True

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

        else:

            random_defect_center_atoms = self.pick_atoms_obeying_distance_criterion(
                number_of_artificial_defects
            )

            # we will now assign these atoms as defective atoms
            self.defective_atoms_ids = random_defect_center_atoms
            self.defective_atoms_ids_clustered = random_defect_center_atoms.reshape(
                -1, 1
            )

            # then we compute atoms around it, due to definition we need to save this differently
            self.find_atoms_around_defects_within_cutoff(cutoff=2.5)

            # now save as defective atom ids and clustered
            self.defective_atoms_ids = np.sort(
                np.concatenate(self.atoms_ids_around_defects_clustered)
            )
            self.defective_atoms_ids_clustered = self.atoms_ids_around_defects_clustered

    def pick_atoms_obeying_distance_criterion(
        self, number_of_atoms_to_be_picked: int, minimum_distance_criterion: float = 15
    ):
        """
        Pick N randoms atoms being separated by minimum distance. So far based on topology file (static pdb)
        Arguments:
            number_of_atoms_to_be_picked (int): Number of atoms selected.
            minimum_distance_criterion (float): Minimum distance separating the atoms in angstroms.
        Returns:
            picked_atom_indices (array) : Indices of picked atoms
        """

        # define array to save indices
        indices_picked_atoms = []

        # intialise variable
        distances_accepted_to_new = 0

        # start looping over number of atoms to be picked
        for atom_number in np.arange(number_of_atoms_to_be_picked):

            # for first just pick random number
            if atom_number == 0:
                indices_picked_atoms.append(
                    np.random.randint(0, self.topology.get_global_number_of_atoms())
                )

            # for all future atoms we have to check whether this atom has already been picked and
            # whether it obeys the minimum distance criterion
            else:
                # now check that distances are larger than cutoff, otherwise pick different atom
                while np.any(distances_accepted_to_new < minimum_distance_criterion):
                    # try random index
                    trial_index = np.random.randint(
                        0, self.topology.get_global_number_of_atoms()
                    )

                    # compute distances to already picked indices
                    distances_accepted_to_new = np.linalg.norm(
                        utils.apply_minimum_image_convention_to_interatomic_vectors(
                            self.topology[indices_picked_atoms].positions
                            - self.topology[trial_index].position,
                            self.topology.cell,
                        ),
                        axis=1,
                    )

                # add accepted index to list
                indices_picked_atoms.append(trial_index)

                # reset distances
                distances_accepted_to_new = 0

        return np.asarray(indices_picked_atoms)

    def _assign_defective_atoms_to_defects(self):
        """
        Assign defective atoms to specific defects.
        Arguments:
        Returns:
        """

        # allowed atoms per defect for each type
        allowed_atoms_per_type = {"Divacancy": 12, "Stone-Wales": 14}

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

        # define local universe of topology
        universe_flat_configuration = mdanalysis.Universe(self.path_to_topology)

        # variable to save all clusters in
        clusters_around_defective_atoms = []

        # now loop over defective atoms ids
        for count, atom_id in enumerate((self.defective_atoms_ids)):

            # we will do a cluster grow around each atom
            # assign atom id as newly added neighbour
            new_neighbors = count

            number_previous_neighbors = 0
            # define variable to check if there are any new additions pro step
            additions = 1

            # allocate array
            all_neighbors_per_defect = []

            # continue as long as new atoms are found
            while additions > 0:

                # find all atoms around new neighbors (or all neighbors) within 3A
                potential_new_neighbors = np.unique(
                    capped_distance(
                        universe_flat_configuration.atoms.positions[
                            self.defective_atoms_ids[new_neighbors]
                        ],
                        universe_flat_configuration.atoms[
                            self.defective_atoms_ids
                        ].positions,
                        2.8,
                        box=universe_flat_configuration.dimensions,
                        return_distances=False,
                    )[:, 1]
                )

                # now compute distances between chosen neighbors and potential neighbors
                pairs, distances = capped_distance(
                    universe_flat_configuration.atoms[
                        self.defective_atoms_ids[potential_new_neighbors]
                    ].positions,
                    universe_flat_configuration.atoms[
                        self.defective_atoms_ids[potential_new_neighbors]
                    ].positions,
                    200,
                    box=universe_flat_configuration.dimensions,
                    return_distances=True,
                )

                # new neighbors are those having at least two neighbors close by
                new_neighbors = potential_new_neighbors[
                    np.where(
                        np.sort(distances.reshape((len(potential_new_neighbors), -1)))[
                            :, 2
                        ]
                        < 2.8
                    )
                ]

                additions = len(new_neighbors) - number_previous_neighbors
                number_previous_neighbors = len(new_neighbors)

            clusters_around_defective_atoms.append(new_neighbors)

        defective_atoms_ids_clustered = self.defective_atoms_ids[
            np.unique(clusters_around_defective_atoms, axis=0)
        ]

        if defective_atoms_ids_clustered.shape[1] != allowed_atoms_per_type.get(
            self.defect_type
        ):
            raise UnphysicalValue(
                f"From system name I guessed you are dealing with {self.defect_type} defects.",
                f"Each {self.defect_type} is formed by {allowed_atoms_per_type.get(self.defect_type)} atoms.",
                f"Something, however, went wrong in clustering the atoms per defect.",
            )

        self.defective_atoms_ids_clustered = defective_atoms_ids_clustered

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

    def sample_lattice_parameter(
        self, supercell_x_replica: int = 60, start_time: int = None, end_time: int = None, frame_frequency: int = None
    ):
        """
        Sample the atomic heights of the graphene sheet relativ to the COM.
        Arguments:
            supercell_x_replica (int): Number of unit cell replica in x direction
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

        # define local variable to save lattice parameter
        lattice_parameter = []

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm(
                (tmp_universe.trajectory[start_frame:end_frame])[
                    :: int(frame_frequency)
                ]
            )
        ):

            # compute lattice parameter per frame
            lattice_parameter.append(frames.dimensions[0]/supercell_x_replica)

        # save array format as numpy array to class
        self.lattice_parameter = np.array(lattice_parameter)

    def find_atoms_around_defects_within_cutoff(
        self,
        cutoff: float = 2.0,
        COM_as_reference: bool = False,
    ):
        """
        Get atom ids of defects in 'close' proximity to the defect center.
        Arguments:
                cutoff (float): cutoff in angstroms within atoms will be assigned to the defect.
                COM_as_reference (bool): Radial cutoff applied from defect COM.
        Returns:
        """

        # To obtain the same number for all defects of the same type we need to look
        # at the flat structure, i.e. the pdb file.
        # Create a temporary Universe based on pdb file only.
        universe_flat_configuration = mdanalysis.Universe(self.path_to_topology)

        # check if COM_as_reference is turned on.
        if COM_as_reference:

            # compute COMs
            COMs_all_defects = self.get_center_of_mass_of_defects()

            # Now loop over all defect centers and find atoms within distance
            self.atoms_ids_around_defects_clustered = np.array(
                [
                    capped_distance(
                        COM_of_defect,
                        universe_flat_configuration.atoms.positions,
                        cutoff,
                        box=universe_flat_configuration.dimensions,
                        return_distances=False,
                    )[:, 1]
                    for COM_of_defect in COMs_all_defects
                ]
            )

        else:
            # Now use feature from MDAnalysis. This will be used to check the adjacent atoms
            scan_neighbors = mdanalysis.lib.NeighborSearch.AtomNeighborSearch(
                universe_flat_configuration.atoms, self.topology.cell.cellpar()
            )

            # Now loop over all defects which were previously assigned and find all atom ids
            # within the cutoff specfied (default 2 angstroms)
            self.atoms_ids_around_defects_clustered = np.array(
                [
                    scan_neighbors.search(
                        universe_flat_configuration.atoms[defective_atoms_per_defect],
                        cutoff,
                        "A",
                    ).indices
                    for defective_atoms_per_defect in self.defective_atoms_ids_clustered
                ]
            )

    def get_center_of_mass_of_defects(
        self, atoms_at_given_frame=None, dimensions_at_given_frame=None
    ):
        """
        Get the center of mass of each defect. This COM is computed on the basis of the atoms
        identified by OVITO which were then assigned to the respective defects.
        This function can either be used on the flat structure or the rippled configuration.
        Arguments:
                atoms_at_given_frame (MDAnalysis Atoms): Contains the positions of all atoms at a given frame.
                        If empty, the topology file will be used instead (flat structure).
                dimensions_at_given_frame (np.array): Dimensions of the cell at a given frame of a trajectory.
        Returns:
                COMs_per_defects (np.array) : Array containing the three dimensional coordinates of the COM for each defect.
        """

        # check if frame empty
        if not atoms_at_given_frame:

            # look at the topology
            # Create a temporary Universe based on pdb file only.
            tmp_universe = mdanalysis.Universe(self.path_to_topology)

            atoms_at_given_frame = tmp_universe.atoms

            dimensions_at_given_frame = tmp_universe.dimensions

        # check whether defective atoms ids were already identified
        if not hasattr(self, "defective_atoms_ids_clustered"):
            raise VariableNotSet(
                f"Cannot find atom ids clustered per defect.",
                f"Please run 'INSTANCE.find_defective_atoms()' before running this routine.",
            )

        # now compute COM for all defects based on previously assigned ids
        return np.array(
            [
                utils.get_center_of_mass_of_atoms_in_accordance_with_MIC(
                    atoms_at_given_frame[atom_ids_per_defect], dimensions_at_given_frame
                )
                for atom_ids_per_defect in self.defective_atoms_ids_clustered
            ]
        )

    def get_orientation_of_defects(self):
        """
        Get the orientation of a defect w.r.t. to cartesian coordinates.
        Arguments:
        Returns:
        """

        # check whether defective atoms ids were already identified
        if not hasattr(self, "defective_atoms_ids_clustered"):
            raise VariableNotSet(
                f"Cannot find atom ids clustered per defect.",
                f"Please run 'INSTANCE.find_defective_atoms()' before running this routine.",
            )

        # Always look at the flat structure to obtain initial orientation.
        # Create a temporary Universe based on pdb file only.
        universe_flat_configuration = mdanalysis.Universe(self.path_to_topology)

        # we need the center of mass for each defect to obtain the orientation
        COMs_per_defect = self.get_center_of_mass_of_defects()

        # first translate to local coordinate system at COMs of the defect
        # and make sure it satisfies pbc
        positions_defective_atoms_relative_to_COMs = np.asarray(
            [
                utils.apply_minimum_image_convention_to_interatomic_vectors(
                    universe_flat_configuration.atoms[
                        defective_atoms_per_defect
                    ].positions
                    - COMs_per_defect[count_defect],
                    self.topology.cell,
                )
                for count_defect, defective_atoms_per_defect in enumerate(
                    self.defective_atoms_ids_clustered
                )
            ]
        )

        # get distances to COM which will be used to compute orientation
        distances_defective_atoms_to_COMs = np.linalg.norm(
            positions_defective_atoms_relative_to_COMs, axis=2
        )

        # now compute orientation dependent on defect type:
        if self.defect_type == "Divacancy":

            # identify indices of 4 second nearest atoms, i.e. edges of octagon.
            indices_2nd_nearest_atoms = np.argsort(distances_defective_atoms_to_COMs)[
                :, 4:8
            ]

            # identify positions of 4 second nearest nearest atoms
            positions_2nd_nearest_atoms = np.array(
                [
                    positions_defective_atoms_relative_to_COMs[defect, indices]
                    for defect, indices in enumerate(indices_2nd_nearest_atoms)
                ]
            )

            # compute distances from first atom of the octagon edges to the rest for each defect
            distances_2nd_nearest_atoms_relative = np.asarray(
                [
                    np.linalg.norm(
                        positions_2nd_nearest_atoms[defect]
                        - positions_2nd_nearest_atoms[defect][0],
                        axis=1,
                    )
                    for defect in np.arange(positions_2nd_nearest_atoms.shape[0])
                ]
            )

            # for these atom groups we want to find the vectors ranging from one end ot the other of the octagon.
            vectors_between_octagon_edges = np.asarray(
                [
                    np.mean(
                        positions_2nd_nearest_atoms[defect][
                            np.argsort(distances_2nd_nearest_atoms_relative[defect])[
                                0:2
                            ]
                        ],
                        axis=0,
                    )
                    - np.mean(
                        positions_2nd_nearest_atoms[defect][
                            np.argsort(distances_2nd_nearest_atoms_relative[defect])[
                                2:4
                            ]
                        ],
                        axis=0,
                    )
                    for defect in np.arange(positions_2nd_nearest_atoms.shape[0])
                ]
            )

            # make sure they all point in the same direction (low y to high y)
            vectors_between_octagon_edges[
                np.where(vectors_between_octagon_edges[:, 1] < 0)[0]
            ] *= -1

            # define normed dot product
            # note, that we define the origin (0 degrees) relative to the y-axis.
            normed_dot_product = np.clip(
                vectors_between_octagon_edges[:, 1]
                / np.linalg.norm(vectors_between_octagon_edges, axis=1),
                -1.0,
                1.0,
            )

            # based on these vectors we can compute the orientation.
            self.orientations_per_defect = np.asarray(
                [
                    np.arccos(normed_dot_product[defect])
                    if vectors_between_octagon_edges[defect, 0] >= 0
                    else -np.arccos(normed_dot_product[defect])
                    for defect in np.arange(len(normed_dot_product))
                ]
            )

        # now for Stone-Wales defect
        elif self.defect_type == "Stone-Wales":

            # identify indices of 2 nearest atoms, i.e. edges of pentagons.
            indices_nearest_atoms = np.argsort(distances_defective_atoms_to_COMs)[
                :, 0:2
            ]

            # identify positions of 2 nearest nearest atoms
            positions_nearest_atoms = np.array(
                [
                    positions_defective_atoms_relative_to_COMs[defect, indices]
                    for defect, indices in enumerate(indices_nearest_atoms)
                ]
            )

            # compute vector from first to second index
            vectors_between_pentagon_edges = np.asarray(
                [
                    (
                        positions_nearest_atoms[defect][1]
                        - positions_nearest_atoms[defect][0]
                    )
                    for defect in np.arange(positions_nearest_atoms.shape[0])
                ]
            )
            # based on these vectors we can compute the orientation.
            # note, that we define the origin (0 degrees) relative to the x-axis (pentagon-axis).
            self.orientations_per_defect = np.arccos(
                np.clip(
                    vectors_between_pentagon_edges[:, 0]
                    / np.linalg.norm(vectors_between_pentagon_edges, axis=1),
                    -1,
                    1,
                )
            )

        # now for pristine graphene
        elif self.defect_type == "Pristine":

            # identify indices of 3 second nearest atoms, i.e. atoms next to artificial vacancy.
            indices_2nd_nearest_atoms = np.argsort(distances_defective_atoms_to_COMs)[
                :, 1:4
            ]

            # identify positions of 3 second nearest nearest atoms
            positions_2nd_nearest_atoms = np.array(
                [
                    positions_defective_atoms_relative_to_COMs[defect, indices]
                    for defect, indices in enumerate(indices_2nd_nearest_atoms)
                ]
            )

            # check how many atoms are below (- y) COM atom and how many below for each defect
            _, count_atoms_below_COM = np.unique(
                np.where(positions_2nd_nearest_atoms[:, :, 1] < 0)[0],
                return_counts=True,
            )

            # now loop over number of arrays and check how often it appears in the counter
            # reference is 1, i.e. only on atom is below the COM. If 2, assign pi.
            self.orientations_per_defect = np.asarray(
                [
                    0 if count_atoms_below_COM[defect] == 1 else np.pi
                    for defect in np.arange(self.defective_atoms_ids_clustered.shape[0])
                ]
            )

    def compute_local_environments_geometry(
        self,
        r2_score_criterion=0.90,
        start_time: int = None,
        end_time: int = None,
        frame_frequency: int = None,
    ):
        """
        Compute local curvature and inclination around defects.
        Arguments:
            r2_score_criterion: Minimum R2 score required to allow analysis.
            start_time (int) : Start time for analysis (optional).
            end_time (int) : End time for analysis (optional).
            frame_frequency (int): Take every nth frame only (optional).
        Returns:
            fitting_success_rate (float): Percentage success rate of the fit of the geometry.
        """

        # get information about sampling
        start_frame, end_frame, frame_frequency = self._get_sampling_frames(
            start_time, end_time, frame_frequency
        )

        # use local variable for universe
        tmp_universe = self.position_universe

        # obtain initial orientation of each defect
        self.get_orientation_of_defects()

        # local variable to see how many fits fail
        count_fit_fail = 0

        # initialise arrays for jacobian and hessian eigenvalues
        jacobians = []
        hessians_eigenvalues = []

        # Loop over trajectory
        for count_frames, frames in enumerate(
            tqdm(
                (tmp_universe.trajectory[start_frame:end_frame])[
                    :: int(frame_frequency)
                ]
            )
        ):

            # obtain centers of mass of defects
            COMs_per_defect = self.get_center_of_mass_of_defects(
                tmp_universe.atoms, tmp_universe.dimensions
            )

            # to be able to fit a auxilary function to the the relative atomic positions
            # we need first to make the environments comparable.
            # To this end, we will 1. translate all defects to a local coordinate system and
            # 2. Rotate the defects around the z-axis so that all local environments are equally orientated.

            # 1. Translation to COM of defects
            # we need to make sure this satisfies the pbc
            positions_local_atoms_relative_to_COMs = np.asarray(
                [
                    utils.apply_minimum_image_convention_to_interatomic_vectors(
                        tmp_universe.atoms[local_atoms_per_defect].positions
                        - COMs_per_defect[count_defect],
                        utils.get_cell_vectors_from_lengths_and_angles(
                            tmp_universe.dimensions
                        ),
                    )
                    for count_defect, local_atoms_per_defect in enumerate(
                        self.atoms_ids_around_defects_clustered
                    )
                ]
            )

            # 2. Rotate around z-axis
            positions_local_atoms_translated_and_rotated = np.asarray(
                [
                    np.matmul(
                        utils.rotation_matrix(
                            np.array([0, 0, 1]), self.orientations_per_defect[defect]
                        ),
                        positions_local_atoms_relative_to_COMs[defect].T,
                    ).T
                    for defect in np.arange(self.orientations_per_defect.shape[0])
                ]
            )

            # now we have to fit a 2D function to the relative positions
            # f_h(x,y) = a + bx + cy + dxy + ex^2 + f y^2

            # compute coefficients and residuum for each defect saved in fitting_data
            fitting_data = np.asarray(
                [
                    scipy.linalg.lstsq(
                        np.c_[
                            np.ones(atomic_positions.shape[0]),
                            atomic_positions[:, :2],
                            np.prod(atomic_positions[:, :2], axis=1),
                            atomic_positions[:, :2] ** 2,
                        ],
                        atomic_positions[:, 2],
                    )[0:2]
                    for atomic_positions in positions_local_atoms_translated_and_rotated
                ]
            )

            # extract residuums for each defect
            residuums = fitting_data[:, 1]

            # use these to compute r2-score
            r2_scores_per_defect = 1 - residuums / (
                positions_local_atoms_translated_and_rotated.shape[1]
                - np.var(positions_local_atoms_translated_and_rotated[:, :, 2], axis=1)
            )

            # now check where r2 scores satisfy criterion
            black_sheep_indices = np.where(r2_scores_per_defect < r2_score_criterion)[0]

            # update counter for fit failures
            count_fit_fail += len(black_sheep_indices)

            # get coefficients for successful fits
            valid_coefficients = np.concatenate(
                np.delete(fitting_data[:, 0], black_sheep_indices)
            ).reshape((-1, 6))

            # now compute Jacobian of analytical at defect center, i.e. [b,c]
            jacobians.extend(valid_coefficients[:, 1:3])

            # define hessian
            hessians = np.dstack(
                [
                    2 * valid_coefficients[:, 4],
                    valid_coefficients[:, 3],
                    valid_coefficients[:, 3],
                    2 * valid_coefficients[:, 5],
                ]
            ).reshape(-1, 2, 2)

            # now compute eigenvalues of each hessian
            hessians_eigenvalues.extend(np.linalg.eig(hessians)[0])

        # wrap up everything

        self.jacobians = np.asarray(jacobians)
        self.hessians_eigenvalues = np.asarray(hessians_eigenvalues)

        # print successrate
        fitting_success_rate = (
            1 - count_fit_fail / (count_fit_fail + self.jacobians.shape[0])
        ) * 100

        print(
            f"Fitted {fitting_success_rate} % of environemnts successfully by applying R2-threshold of {r2_score_criterion}.",
            f"Only {count_fit_fail}/{count_fit_fail+self.jacobians.shape[0]} failed.",
        )

        return fitting_success_rate
