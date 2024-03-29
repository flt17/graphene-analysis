import numpy as np
import os
import pytest
import sys
from ase.visualize import view


# sys.path.append("../")
from graphene_analysis import analysis


class TestSimulationReadInSimulationData:
    def test_sets_up_position_universes_for_pimd(self):
        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data()

        assert len(simulation.position_universe.trajectory) == 18


class TestSimulationFindDefectiveAtoms:
    def test_returns_correct_ids_for_abundance_of_divacancy(self):

        path = "./files/trajectories/divacancy_54/"

        simulation = analysis.Simulation(path, "Divacancy 54")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        assert len(simulation.defective_atoms_ids) % 12 == 0

    def test_returns_correct_ids_for_divacancy(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        assert len(simulation.defective_atoms_ids) % 12 == 0

    def test_returns_error_due_to_wrong_system_name(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Bla Bla 36")
        simulation.read_in_simulation_data()

        with pytest.raises(analysis.KeyNotFound):
            simulation.find_defective_atoms()

    def test_returns_correct_ids_for_pristine(self):

        path = "./files/trajectories/pristine/"

        simulation = analysis.Simulation(path, "Pristine")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms(pristine=True, number_of_artificial_defects=12)

        assert len(simulation.defective_atoms_ids) % 12 == 0


class TestSimulationFindAtomsAroundDefectsWithinCutoff:
    def test_returns_correct_ids_for_divacancy(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        simulation.find_atoms_around_defects_within_cutoff(cutoff=2.0)

        assert simulation.atoms_ids_around_defects_clustered.shape[0] == 36

    def test_returns_correct_ids_for_divacancy_COM(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        simulation.find_atoms_around_defects_within_cutoff(
            cutoff=4.5, COM_as_reference=True
        )

        assert simulation.atoms_ids_around_defects_clustered.shape[0] == 36

    def test_returns_correct_ids_for_stone_wales(self):

        path = "./files/trajectories/stone-wales_12/"

        simulation = analysis.Simulation(path, "Stone-Wales 12")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        simulation.find_atoms_around_defects_within_cutoff(cutoff=2.0)
        assert simulation.atoms_ids_around_defects_clustered.shape[0] == 12

    def test_returns_correct_ids_for_stone_wales_COM(self):

        path = "./files/trajectories/stone-wales_12/"

        simulation = analysis.Simulation(path, "Stone-Wales 12")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        simulation.find_atoms_around_defects_within_cutoff(
            cutoff=4.5, COM_as_reference=True
        )

        assert simulation.atoms_ids_around_defects_clustered.shape[0] == 12

    def test_returns_correct_ids_for_pristine(self):

        path = "./files/trajectories/pristine/"

        simulation = analysis.Simulation(path, "Pristine")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms(pristine=True, number_of_artificial_defects=12)

        simulation.find_atoms_around_defects_within_cutoff(cutoff=2.0)

        assert simulation.atoms_ids_around_defects_clustered.shape[1] == 19

    def test_returns_correct_ids_for_pristine_COM(self):

        path = "./files/trajectories/pristine/"

        simulation = analysis.Simulation(path, "Pristine")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms(pristine=True, number_of_artificial_defects=12)

        simulation.find_atoms_around_defects_within_cutoff(
            cutoff=4.5, COM_as_reference=True
        )

        assert simulation.atoms_ids_around_defects_clustered.shape[1] == 25


class TestSimulationSampleAtomicHeightDistribution:
    def test_returns_sensible_value_for_standard_deviation(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=100
        )

        simulation.sample_atomic_height_distribution()

        std_distribution = np.std(simulation.atomic_height_distribution)

        assert np.isclose(std_distribution, 3.4, atol=0.1)


class TestSimulationGetCenterOfMassOfDefects:
    def test_raises_error_because_atoms_were_not_found(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()

        with pytest.raises(analysis.VariableNotSet):
            simulation.get_center_of_mass_of_defects()

    def test_returns_correct_center_of_masses_for_flat(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        COMs = simulation.get_center_of_mass_of_defects()
        assert COMs.shape == (36, 3)

    def test_returns_correct_center_of_masses_for_random_frame(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        simulation.position_universe.trajectory[-1]

        COMs = simulation.get_center_of_mass_of_defects(
            simulation.position_universe.atoms,
            simulation.position_universe.dimensions,
        )
        assert COMs.shape == (36, 3)


class TestSimulationGetOrientationsOfDefects:
    def test_returns_correct_orientations_for_divacancies(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        simulation.get_orientation_of_defects()

        assert len(simulation.orientations_per_defect) == 36

    def test_returns_correct_orientations_for_stone_wales(self):

        path = "./files/trajectories/stone-wales_12/"

        simulation = analysis.Simulation(path, "Stone-Wales 12")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        simulation.get_orientation_of_defects()

        assert len(simulation.orientations_per_defect) == 12

    def test_returns_correct_orientations_for_pristine(self):

        path = "./files/trajectories/pristine/"

        simulation = analysis.Simulation(path, "Pristine")
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms(pristine=True, number_of_artificial_defects=12)

        simulation.get_orientation_of_defects()

        assert len(simulation.orientations_per_defect) == 12


class TestSimulationComputeLocalEnvironmentsGeometry:
    def test_returns_correct_geometries_for_divacancy(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=100
        )

        simulation.find_defective_atoms()

        simulation.find_atoms_around_defects_within_cutoff()

        success_rate = simulation.compute_local_environments_geometry()
        assert success_rate > 90

    def test_returns_correct_geometries_for_stone_wales(self):

        path = "./files/trajectories/stone-wales_12/"

        simulation = analysis.Simulation(path, "Stone-Wales 12")
        simulation.read_in_simulation_data()
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=100
        )

        simulation.find_defective_atoms()
        simulation.find_atoms_around_defects_within_cutoff()

        success_rate = simulation.compute_local_environments_geometry()
        assert success_rate > 90

    def test_returns_correct_geometries_for_pristine(self):

        path = "./files/trajectories/pristine/"

        simulation = analysis.Simulation(path, "Pristine ")
        simulation.read_in_simulation_data()
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=100
        )

        simulation.find_defective_atoms(pristine=True, number_of_artificial_defects=12)
        simulation.find_atoms_around_defects_within_cutoff()

        success_rate = simulation.compute_local_environments_geometry()

        assert success_rate > 90


class TestComputeHeightAutoCorrelationFunction:
    def test_raises_error_for_too_large_correlation_time(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=100
        )

        with pytest.raises(analysis.UnphysicalValue):        
            simulation.compute_height_autocorrelation_function(
                correlation_time=50000, number_of_blocks=2
            )

class TestComputeHeightCrossCorrelationFunction:
    def test_raises_error_for_too_large_correlation_time(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=100
        )

        with pytest.raises(analysis.UnphysicalValue):        
            simulation.compute_height_crosscorrelation_function(
                correlation_time=50000, number_of_blocks=2, number_of_atoms = 2, correlated_distance = 10
            )

    def test_returns_reasonable_values(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 36")
        simulation.read_in_simulation_data()
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=100
        )
      
        simulation.compute_height_crosscorrelation_function(
                correlation_time=500, number_of_blocks=1, number_of_atoms = 2, correlated_distance = 10
            )

        assert len(simulation.HCCF.keys()) == 2


class TestComputeHeightAutoCorrelationFunctionParallel:
    def test_computes_ACF(self):

        path ="./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path, "Divacancy 2")
        simulation.read_in_simulation_data()
        simulation.set_sampling_times(
            start_time=0, end_time=-1, frame_frequency=1, time_between_frames=100
        )

           
        simulation.compute_height_autocorrelation_function_parallel(
                n_cores=2, correlation_time=500, number_of_blocks=2
            )

        assert (simulation.HACF_para["ct: 500"])[1].shape[1]==2,5


        

