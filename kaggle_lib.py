from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from kaggle import KaggleApi
import time
import pandas as pd
import os
import webbrowser
import pickle

class Submitter:
    def __init__( self, compete, work_dir, default_submission_id=0):
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()

        self.work_dir = work_dir
        self.compete = compete
        self.default_submission_id = default_submission_id

    def submit(
            self,
            predicted: pd.DataFrame,
            file_name='submission.csv',
            message=None,
            save_model=True,
            model=None,
            model_name=None,
            submit=True,
            submission_id=None,
            submission_name=None,
            open_in_browser=False
       ):

        # Folder and files
        if submission_id is None:
            submission_id = self.default_submission_id
            self.default_submission_id += 1

        submission_folder_name = ' '.join([
            str(submission_id),
            ' -- ',
            submission_name
        ])

        new_folder_path = self.work_dir + f'/{submission_folder_name}'

        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)

        # Save model
        if model_name is None:
            model_name = str(model)

        if save_model:
            with open(f'{new_folder_path}/{model_name}.pickle', 'wb') as pickle_file:
                pickle.dump(model, pickle_file)

        # Submission
        predicted.to_csv(f'{new_folder_path}/{file_name}', index=False)
        if message is None:
            if model is None:
                message = file_name
            else:
                message = str(model)

        with open(new_folder_path + '/message.txt', 'w') as message_file:
            message_file.write(' '.join([str(submission_id), ' -- ', submission_name, '\n', message]))
            message_file.close()

        # Upload
        if submit:
            print('Uploading submission...')
            command = f'kaggle competitions submit -c {self.compete} -f "{new_folder_path}/{file_name}" -m "{message}"'
            print(command)
            output = os.system(command)
            print('Output: ', output)

        # Open in browser
        if open_in_browser:
            webbrowser.open(f'https://www.kaggle.com/c/{self.compete}/submissions', new=2)

    def check_submission(self):
        last_submission = self.kaggle_api.competitions_submissions_list(self.compete)[0]
        print('Date: ', last_submission['date'])
        print('Status: ', last_submission['status'])
        print('Score: ', last_submission['publicScore'])

def search_params(
        model, X, y, random=False,
        param_grid=None, distributions=None,
        fit_args=(), fit_kwargs={}
):
    grid_search = None
    if random:
        grid_search = RandomizedSearchCV(model, distributions)
    else:
        grid_search = GridSearchCV(model, param_grid)

    start_time = time.time()

    grid_search.fit(X, y, *fit_args, **fit_kwargs)

    print(f'Time: {time.time() - start_time} sec')
    print(f'Best estimator: {grid_search.best_estimator_}')

    return grid_search


