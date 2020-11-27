from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from kaggle import KaggleApi
import time
import pandas as pd
import os
import webbrowser
import pickle

class Submission():
    submission_id = 0

    def __init__(self, compete, name, work_dir, description=None):
        self.compete = compete
        self.name = f'{Submission.submission_id} -- {name}'
        self.description = description
        self.new_folder_path = work_dir + f'/{name}'

        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()

        Submission.submission_id += 1

        if not os.path.exists(self.new_folder_path):
            os.mkdir(self.new_folder_path)

    def save_model(self, model, model_file_name=None):
        if model_file_name is None:
            model_file_name = str(model).replace('\\', '')

        with open(f'{self.new_folder_path}/{model_file_name}.pickle', 'wb') as pickle_file:
            pickle.dump(model, pickle_file)

        return self

    def save_desc(self, desc_file_name='description.txt'):
        with open(f'{self.new_folder_path}/{desc_file_name}', 'w') as file:
            file.write(self.description)

        return self

    def save_predictions(self, predictions, index, predictions_file_name='predictions.csv', columns=None):
        pd.DataFrame(dict(zip(columns, [index, predictions])))\
            .to_csv(f'{self.new_folder_path}/{predictions_file_name}', index=False)

        return self

    def open_in_browser(self):
        webbrowser.open(f'https://www.kaggle.com/c/{self.compete}/submissions', new=2)

        return self

    def submit(self, predictions_file_name='predictions.csv'):
        print('Uploading submission...')
        command = f'kaggle competitions submit -c {self.compete} -f "{self.new_folder_path}/{predictions_file_name}" -m "{self.description}"'
        print(command)
        output = os.system(command)
        print('Output: ', output)

        return self

    def check_results(self, timeout=5):
        time.sleep(timeout)
        last_submission = self.kaggle_api.competitions_submissions_list(self.compete)[0]

        print('Description: ', last_submission['description'])
        print('Date: ', last_submission['date'])
        print('Status: ', last_submission['status'])
        print('Score: ', last_submission['publicScore'])

        return self

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
    print(f'Best params: {grid_search.best_params_}')

    return grid_search


