import unittest

import numpy as np
import pandas as pd
from all_in import get_df_subj, sorted_nicely


class TestUtilities(unittest.TestCase):

    def test_get_df_subj(self):
        """ This function tests the function that selects the subject-specific data frames """

        # Create input data frame
        df_subj = pd.DataFrame(index=np.arange(10) + 20)
        df_subj['subj_num'] = np.array([30, 30, 30, 30, 30, 31, 31, 31, 31, 31])
        df_subj['test'] = np.arange(10)

        # Extract subject-specific data frame
        df_subj = get_df_subj(df_subj, 30)

        # Create expected data frame
        expected_df = pd.DataFrame(index=np.arange(5))
        expected_df['subj_num'] = np.array([31, 31, 31, 31, 31])  # expected bc. Python starts counting at 0
        expected_df['test'] = np.arange(5) + 5

        # Test output
        df_subj.equals(expected_df)

    def test_sorted_nicely(self):
        """ This function tests the sorting function used for sorting the file paths """

        file_paths = ['al_data/first_experiment/sub_5/behav/sub-5_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_1/behav/sub-1_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_8/behav/sub-8_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_4/behav/sub-4_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_2/behav/sub-2_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_6/behav/sub-6_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_9/behav/sub-9_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_3/behav/sub-3_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_10/behav/sub-10_task-helicopter_behav.tsv',
                      'al_data/first_experiment/sub_7/behav/sub-7_task-helicopter_behav.tsv']

        # Sort all file names according to participant ID
        file_paths_sorted = sorted_nicely(file_paths)

        # Expected test results
        file_paths_expected = ['al_data/first_experiment/sub_1/behav/sub-1_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_2/behav/sub-2_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_3/behav/sub-3_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_4/behav/sub-4_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_5/behav/sub-5_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_6/behav/sub-6_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_7/behav/sub-7_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_8/behav/sub-8_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_9/behav/sub-9_task-helicopter_behav.tsv',
                               'al_data/first_experiment/sub_10/behav/sub-10_task-helicopter_behav.tsv']

        self.assertTrue(file_paths_sorted == file_paths_expected)
