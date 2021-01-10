
import sys
sys.path.append('../')

from graphene_analysis import io


class TestGetPdfFileName:

	def test_returns_first_file(self):
		path = './test_files'

		file_name = io.get_pdb_file_name(path)

		assert file_name == './test_files/test2.pdb'


	def test_returns_requested_file(self):
		path = './test_files'
		name = 'test1'

		file_name = io.get_pdb_file_name(path, name)

		assert file_name == './test_files/test1.pdb'
