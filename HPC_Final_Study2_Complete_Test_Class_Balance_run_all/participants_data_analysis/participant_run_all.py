import os
import sys
import platform

sys.path.insert(1,
                '/home/hwei/work/HaowenWeiDeepLearning/IndexPenTrainingDir/IndexPen_Training/HPC_Final_Study2_Complete_Test_Class_Balance/participants_data_analysis')


analysis_session = [
    (1 , 1),( 1, 2),( 1, 3),( 1, 4),( 1, 5),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 1), (4, 2), (4, 3), (4, 4), (4,5),
    (5,1), (5,2), (5,3), (5,4), (5, 5)


    # (8, 5),
    # (9,1), (9,2), (9,3), (9,4), (9, 5)

    # # (18,3),
    # (12, 4),
    # # (20,4),
    # # (21,4)
    # # (21, 1)
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

    # os.system(offset + " ".join((participants_session_transfer_train, participant_name, session_name)))
    # os.system(offset + " ".join((participants_session_transfer_train_fresh_model, participant_name, session_name)))
    #
    # analysis transfer train best model and fresh model
    os.system(offset + " ".join((participants_session_raw_acc_evaluation, participant_name, session_name)))

    # os.system(offset + " ".join((participants_session_raw_prediction, participant_name, session_name)))
    #
    # os.system(offset + " ".join((participants_session_raw_prediction_evaluation, participant_name, session_name)))

# plot current final result. Current raw accuracy and current evaluation
