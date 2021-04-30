import numpy as np
import os
import pandas
import pytest
import sys


sys.path.append("../")
from graphene_analysis import analysis


class TestSimulationReadInSimulationData:
    def test_sets_up_position_universes_for_pimd(self):
        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data()

        assert len(simulation.position_universe.trajectory) == 18

class TestSimulationFindDefectiveAtoms:
    def test_returns_correct_ids_for_divacancy(self):

        path = "./files/trajectories/divacancy_36/"

        simulation = analysis.Simulation(path)
        simulation.read_in_simulation_data()
        simulation.find_defective_atoms()

        assert len(simulation.defective_atoms_ids)%12 ==0
