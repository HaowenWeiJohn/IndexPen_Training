import os
import sys
import platform
sys.path.insert(1, '/home/hwei/work/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training/HPC_Final_Study2_Complete_Test_Class_Balance/participants_data_analysis')


analysis_session = [(2,1), (2,2), (2,3), (2,4)]

os_type = platform.system()
if os_type=="Windows":
    offset = ''
else:
    offset='Python3 '

for session_info in analysis_session:
    participant_name = 'participant_' + str(session_info[0])
    session_name = 'session_'+ str(session_info[1])

    # transfer current session, prepare lite model, model for next session
    participants_session_transfer_train = "participants_session_transfer_train.py"

    participants_session_transfer_train_fresh_model = "participants_session_transfer_train_fresh_model.py"
    # analysis current session using previous model, raw acc
    participants_session_raw_acc_evaluation = "participants_session_raw_acc_evaluation.py"

    # analysis current session using debouncer algorithm
    participants_session_raw_prediction = "participants_session_raw_prediction.py"
    participants_session_raw_prediction_evaluation = "participants_session_raw_prediction_evaluation.py"
    #
    # transfer train

    os.system(offset+" ".join((participants_session_transfer_train, participant_name, session_name)))
    os.system(offset+" ".join((participants_session_transfer_train_fresh_model, participant_name, session_name)))
    #
    # analysis transfer train best model and fresh model
    os.system(offset+" ".join((participants_session_raw_acc_evaluation, participant_name, session_name)))

    os.system(offset+" ".join((participants_session_raw_prediction, participant_name, session_name)))

    os.system(offset+" ".join((participants_session_raw_prediction_evaluation, participant_name, session_name)))





# plot current final result. Current raw accuracy and current evaluation


