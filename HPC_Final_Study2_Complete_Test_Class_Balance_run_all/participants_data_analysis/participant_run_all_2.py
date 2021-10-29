import os
import sys
import platform

sys.path.insert(1,
                '/home/hwei/work/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training/HPC_Final_Study2_Complete_Test_Class_Balance/participants_data_analysis')


analysis_session = [
    # (9,5),
    # (18,3),
    # (12, 4),
    # (20,4),
    # (21,4)
    # (21, 1)
    (16, 1), (16, 2), (16, 3), (16, 4), (16, 5),
    (18, 1), (18, 2), (18, 3), (18, 4), (18, 5)

]

os_type = platform.system()
print(os_type)
if os_type == "Windows":
    offset = ''
else:
    offset = 'python3 '

for session_info in analysis_session:
    participant_name = 'participant_' + str(session_info[0])
    session_name = 'session_' + str(session_info[1])

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

    os.system(offset + " ".join((participants_session_transfer_train, participant_name, session_name)))
    os.system(offset + " ".join((participants_session_transfer_train_fresh_model, participant_name, session_name)))
    #
    # analysis transfer train best model and fresh model
    os.system(offset + " ".join((participants_session_raw_acc_evaluation, participant_name, session_name)))

    os.system(offset + " ".join((participants_session_raw_prediction, participant_name, session_name)))

    os.system(offset + " ".join((participants_session_raw_prediction_evaluation, participant_name, session_name)))

# plot current final result. Current raw accuracy and current evaluation
