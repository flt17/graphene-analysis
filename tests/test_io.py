import pytest
import sys
from unittest import mock

sys.path.append("../")
from graphene_analysis import io


class TestGetPathToFile:
    def test_returns_path_not_found(self):
        path = "./files"
        suffix = "random_suffix"

        with pytest.raises(io.UnableToFindFile):
            io.get_path_to_file(path, suffix)

    def test_returns_first_file(self):
        path = "./files"
        suffix = "pdb"

        file_name = io.get_path_to_file(path, suffix)

        assert file_name == "./files/graphene_1_divacancy.pdb"

    def test_returns_requested_file(self):
        path = "./files"
        suffix = "pdb"
        prefix = "graphene_1_divacancy"

        file_name = io.get_path_to_file(path, suffix, prefix)

        assert file_name == "./files/graphene_1_divacancy.pdb"


class TestGetAseAtomsObject:
    @mock.patch.object(io, "get_path_to_file")
    def test_returns_ase_AO_from_pdb(self, get_path_to_file_mock):
        path_to_pdb = "./files/graphene_20_monovacancy.pdb"

        ase_AO = io.get_ase_atoms_object(path_to_pdb)

        get_path_to_file_mock.assert_not_called()
        assert ase_AO.get_global_number_of_atoms() == 7180

    def test_returns_ase_AO_from_lammpstrj(self):
        # This file is not readible by ase.
        path_to_pdb = "./files/graphene_1_divacancy.pdb"

        ase_AO = io.get_ase_atoms_object(path_to_pdb)

        assert ase_AO.get_global_number_of_atoms() == 7198
