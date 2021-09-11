import os


analysis_session = [(1,4)]

for session_info in analysis_session:
    participant_name = 'participant_' + str(session_info[0])
    session_name = 'session_'+ str(session_info[1])

    # transfer current session, prepare lite model, model for next session
    participants_session_transfer_train = "participants_session_transfer_train.py"

    # analysis current session using previous model, raw acc
    participants_session_raw_acc_evaluation = "participants_session_raw_acc_evaluation.py"

    # analysis current session using debouncer algorithm
    participants_session_raw_prediction = "participants_session_raw_prediction.py"
    participants_session_raw_prediction_evaluation = "participants_session_raw_prediction_evaluation.py"

    os.system(" ".join((participants_session_transfer_train, participant_name, session_name)))

    os.system(" ".join((participants_session_raw_acc_evaluation, participant_name, session_name)))

    os.system(" ".join((participants_session_raw_prediction, participant_name, session_name)))

    os.system(" ".join((participants_session_raw_prediction_evaluation, participant_name, session_name)))





# plot current final result. Current raw accuracy and current evaluation


