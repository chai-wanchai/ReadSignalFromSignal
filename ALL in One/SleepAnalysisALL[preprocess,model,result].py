import Preprocess as PR
import Model
import  TestRandomTestset as TrainTest
if __name__ == '__main__':
    Model.SelectColumn = ['EOG(L)_min', 'EOG(L)_max', 'EOG(L)_avg', 'EOG(L)_SD', 'EOG(L)_kurtosis',
              'EOG(R)_min', 'EOG(R)_max', 'EOG(R)_avg', 'EOG(R)_SD', 'EOG(R)_kurtosis',
              'EEG(sec)_min', 'EEG(sec)_max', 'EEG(sec)_avg', 'EEG(sec)_SD', 'EEG(sec)_kurtosis',
              'EEG_min', 'EEG_max', 'EEG_avg', 'EEG_SD', 'EEG_kurtosis',
              'ECG_min', 'ECG_max', 'ECG_avg', 'ECG_SD', 'ECG_kurtosis',
              'stage']
    Model.Start()

