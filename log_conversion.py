#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:57:35 2023

@author: ZI\stepan.tikhomirov
"""
import pandas as pd
from scipy.io import loadmat
from io import StringIO
import functools

def read_cleanreversal(file_dir, task_parameters):
    '''
    Read and process reversal task files

    Parameters
    ----------
    file_dir : string (path is independent from system)
        Path to the reversal task file
    task_parameters : dict
        Dictionary of task parameters for different tasks, e.g. correct answers to a task.

    Returns
    -------
    df_new : pandas.core.frame.DataFrame
        Final dataframe which will be converted to tsv
    df_timing_events : pandas.core.frame.DataFrame
        Dataframe containing log data which will be used for plotting

    '''
    df_no_pulses = loadmat(file_dir)
    # Get number of trials
    # ("nt" is the number of the last trial)
    nr_trials = int(df_no_pulses['nt'].flatten())

    # Create dataframe with timing of stimulus, response, and feedback
    # Select time variable in mat file (structure)
    timings_mat = df_no_pulses['T']
    # A structure is read by scipy as a list of arrays with column names or keys as dtype
    timings_columns = timings_mat.dtype
    # Create dictionary of timings
    timings_dict = {n: timings_mat[n][0, 0] for n in timings_columns.names}
    # Get keys of timing vectors (i.e. those with length equal to the number of trials)
    timings_keys = [n for n, v in timings_dict.items() if v.size == nr_trials]
    # Remove onset_trialend timing - not informative (in place method)
    timings_keys.remove("onset_trialend")

    # Create data frame with trial numbers as index
    df_timing = pd.DataFrame(
        np.concatenate([timings_dict[c].reshape((nr_trials, 1))
                       for c in timings_keys], axis=1),
        index=list(range(0, nr_trials)),
        columns=timings_keys)

    if "choice_onset" not in df_timing.columns:
        return None, None
    # Change column names to be consistent with BIDS format
    df_timing = df_timing.rename(
        columns={'trial_onset': 'stim', 'onset_fb': 'feedback', 'choice_onset': 'response'})

    # Change timing of response = 0 to nan
    df_timing.loc[df_timing["response"] == 0, "response"] = np.nan

    # Convert all columns to float type
    df_timing = df_timing.astype(
        {"stim": np.float64, "feedback": np.float64, "response": np.float64})
    # Subtract timing of first volume to make sure the "onset" column indicates timing wrt the first volume
    time_first_vol = timings_dict['time_begin'].flatten()[
        0]  # Get timing of first volume to make sure the "onset" column indicates timing wrt the first volume
    df_timing = (
        df_timing - time_first_vol)  # * convert_time_unit_to_sec # Not necessary to convert, data already appears to be in seconds

    # Wide to long format
    df_timing_long = df_timing.unstack().reset_index()

    # Rename columns
    df_timing_long = df_timing_long.rename(
        columns={'level_0': 'event_type', 'level_1': 'trial_nr', 0: 'onset'})

    # Make sure trial numbers start at 1
    df_timing_long["trial_nr"] += 1

    # Create event types dataframe
    sel_vars = ["A", "C", "R", "S", "rt",
                "p_u",
                "random_lr"]  # , "ISI", "ISI_final", "pres_ISI", "ITI", "pres_ITI", "ITI_w_null"] # Selected variables from .mat file to keep

    df_events = pd.DataFrame(
        np.concatenate([df_no_pulses[c].reshape((1, nr_trials))
                       for c in sel_vars], axis=0).transpose(),
        index=list(range(0, nr_trials)),
        columns=sel_vars)

    # Rename columns
    df_events = df_events.rename(
        columns={'A': 'response', 'C': 'correct', 'R': 'reward', 'S': 'correct_resp', 'rt': 'response_time',
                 'p_u': 'prob_events', 'random_lr': 'random_leftright'})  # 'V':'block_nr',

    # Add block number
    version = df_no_pulses["Task_Version"]
    if version == "A":
        # reversal_version_A["V"]
        df_events["block_nr"] = task_parameters["version_A"]["block_nr"]
    elif version == "B":
        # reversal_version_B["V"]
        df_events["block_nr"] = task_parameters["version_B"]["block_nr"]

    # Change response and state from numeric to string (1 -> Card1; 2 -> Card2)
    df_events["response"] = df_events["response"].replace(
        [1, 2], ["Card1", "Card2"])
    df_events["correct_resp"] = df_events["correct_resp"].replace([1, 2], [
                                                                  "Card1", "Card2"])

    # Change the random_leftright (indicator of on which side card 1 is) from numeric binary to string binary to make it more interpretable; Card 1 is always drawn on the side indicated by random_leftright (see rev_trial.m)
    df_events["random_leftright"] = df_events["random_leftright"].replace([1, 2], [
                                                                          "left", "right"])

    # Trial type
    df_events["trial_type"] = np.select([
        ((df_events["correct_resp"] == df_events["response"])
         & (df_events["prob_events"] == 0)),
        # informative reward     = correct rewarded response
        ((df_events["correct_resp"] == df_events["response"])
         & (df_events["prob_events"] == 1)),
        # misleading punishment  = probabilistic error
        ((df_events["correct_resp"] != df_events["response"])
         & (df_events["prob_events"] == 0)),
        # misleading reward      = probabilistic win
        ((df_events["correct_resp"] != df_events["response"])
         & (df_events["prob_events"] == 1)),
        # informative punishment = incorrect response
    ], [
        "informative reward", "misleading punishment", "misleading reward", "informative punishment"
    ])

    # Create trial_nr column out of index
    df_events["trial_nr"] = df_events.index + 1

    # Merge events with timing dataframe
    df_timing_events = pd.merge(df_timing_long, df_events, on=["trial_nr"])
    # Add value column and fill with shown stimulus (location of both Card 1 and 2) and response
    df_timing_events["reversal_display"] = np.nan
    stim = np.select([
        (df_events["random_leftright"] ==
         "left"), (df_events["random_leftright"] == "right")
    ], [
        "Card2_right_Card1_left", "Card1_right_Card2_left"]
    )
    df_timing_events.loc[df_timing_events["event_type"]
                         == "stim", "reversal_display"] = stim
    df_timing_events.loc[df_timing_events["event_type"] ==
                         "response", "reversal_display"] = df_events["response"]

    # Remove first NaN and append NaN
    # Compute duration
    df_timing_events["duration"] = list(df_timing_events["onset"].diff().values[1:]) + [
        np.nan]
    # Reorder columns; drop response column
    df_timing_events = df_timing_events[
        ["onset", "duration", "trial_nr", "trial_type", "event_type", "reversal_display", "response", "correct", "response_time", "correct_resp"]]  # .sort_values(by=["event_type", "trial_nr"], key=lambda x: x.map({'stim': 0, 'response': 1, 'feedback': 2} ))

    # Custom sorting: create categorical event_type column and sort dataframe based on that
    cat_event_type_order = pd.CategoricalDtype(
        ["stim", "response", "feedback"],
        ordered=True
    )
    # Change to categorical
    df_timing_events['event_type'] = df_timing_events['event_type'].astype(
        cat_event_type_order)

    # Sort values
    df_timing_events = df_timing_events.sort_values(
        by=['trial_nr', 'event_type'])

    # Change categorical back to string (necessary to fill n/a values later)
    df_timing_events['event_type'] = df_timing_events['event_type'].astype(
        'string')
    df_timing_events = df_timing_events[df_timing_events["event_type"] != "feedback"]
    df_new = df_timing_events.copy()
    df_new = df_new[~df_new['event_type'].isin(['response'])]
    df_new.drop(columns=["trial_nr"], inplace=True)
    df_new.reset_index(drop=True, inplace=True)
    df_new = df_new.fillna("n/a")
    return df_new, df_timing_events


def read_clean_other_tasks(file_dir):
    '''
    Read and clean faces, nback, reward task files

    Parameters
    ----------
    file_dir : string (path is independent from system)
        Path to the log file

    Returns
    -------
    bids_df : pandas.core.frame.DataFrame
        Final dataframe which will be converted to tsv
    data_string : str
        Original string containing the whole log file 

    '''
    #open the file in python
    with open(file_dir, 'r') as file:
        data_string = file.read()
        # Separate raw dataframe from already reduced dataframe, if the latter is present
        idx_start_Trial = [m.start()
                           for m in re.finditer('Trial', data_string)]
        idx_start_EventType = [m.start()
                               for m in re.finditer('Event Type', data_string)]

        # If there are multiple "Event Type" columns, extract the relevant data based on their positions.
        if len(idx_start_EventType) > 1:
            # Separate the raw dataframe from any already reduced dataframe, if present.
            data_string = pd.read_csv(
                StringIO(data_string[idx_start_Trial[0]:idx_start_EventType[1]]), sep='\t')
        else:
            # If there is only one "Event Type" column, extract the dataframe starting from the first "Trial."
            data_string = pd.read_csv(
                StringIO(data_string[idx_start_Trial[0]:]), sep='\t')
    #clean the data. remove pulse, quit, instruct etc.
    time_first_vol = data_string.loc[data_string["Event Type"] == "Pulse", "Time"].values[0]
    data_string["Time"] -= time_first_vol
    data_string = data_string[data_string["Event Type"] != "Quit"]
    data_string = data_string[data_string["Event Type"] != "Pulse"]
    data_string = data_string[data_string["Code"] != "instruct"]
    data_string = data_string[data_string["Code"] != "ende"]
    data_string = data_string[data_string["Code"] != "black screen"]
    data_string = data_string[data_string["Code"] != "fixation"]
    data_string = data_string[data_string["Code"] != "perf"]
    #converting time into seconds
    data_string['onset'] = data_string['Time'].astype(float) / 10000
    data_string['duration'] = data_string['Duration'].astype(float) / 10000

    # Select 'Event Type' column
    data_string['event_type'] = data_string['Code']

    # Create a new DataFrame with the desired columns
    bids_df = data_string[['onset', 'duration', 'event_type']].fillna(
        0).reset_index(drop=True)
    start_time = bids_df["onset"].min()
    bids_df["onset"] = bids_df["onset"] - start_time
    
    return bids_df, data_string


def read_clean_selfref(file_dir):
    '''
    Rean and clean selfref task lig files

    Parameters
    ----------
    file_dir : string (path is independent from system)
        Path to the log file

    Returns
    -------
    bids_df : pandas.core.frame.DataFrame
        Final dataframe which will be converted into tsv file.
    data_string : TYPE
        DESCRIPTION.

    '''
    try:
        # Attempt to read the file using UTF-8 encoding.
        with open(file_dir, 'r', encoding='utf-8') as file:
            data_string = file.read()
    except:
        # If UTF-8 encoding fails, try ISO-8859-1 encoding.
        with open(file_dir, 'r', encoding='iso-8859-1') as file:
            data_string = file.read()
    #replace german letters
    data_string = data_string.replace('Ü', 'UE')
    data_string = data_string.replace('ü', 'ue')
    data_string = data_string.replace('Ä', 'AE')
    data_string = data_string.replace('ä', 'ae')
    data_string = data_string.replace('Ö', 'OE')
    data_string = data_string.replace('ö', 'oe')

    idx_start_Trial = [m.start() for m in re.finditer('Trial', data_string)]
    idx_start_EventType = [m.start()
                           for m in re.finditer('Event Type', data_string)]

    # If there are multiple "Event Type" columns, extract the relevant data based on their positions.
    if len(idx_start_EventType) > 1:
        # Separate the raw dataframe from any already reduced dataframe, if present.
        data_string = pd.read_csv(
            StringIO(data_string[idx_start_Trial[0]:idx_start_EventType[1]]), sep='\t')
    else:
        # If there is only one "Event Type" column, extract the dataframe starting from the first "Trial."
        data_string = pd.read_csv(StringIO(data_string[idx_start_Trial[0]:]), sep='\t')
    ##clean the data. remove pulse, quit, instruct etc.
    time_first_vol = data_string.loc[data_string["Event Type"] == "Pulse", "Time"].values[0]
    data_string["Time"] -= time_first_vol
    data_string = data_string[data_string["Event Type"] != "Quit"]
    data_string = data_string[data_string["Event Type"] != "Pulse"]

    data_string = data_string[data_string["Code"] != "instruct"]
    data_string = data_string[data_string["Code"] != "ende"]
    data_string = data_string[data_string["Code"] != "black screen"]
    data_string = data_string[data_string["Code"] != "fixation"]
    #convert the time into seconds
    data_string['onset'] = data_string['Time'].astype(float) / 10000
    data_string['duration'] = data_string['Duration'].astype(float) / 10000

    # Select 'Event Type' column
    data_string['event_type'] = data_string['Code']
    
    # Create a new DataFrame with the desired columns
    bids_df = data_string[['onset', 'duration', 'event_type']].fillna(
        0).reset_index(drop=True)
    # Remove rows at index 0 and 1 (they contain number 2 in the event_type, though there is no stimulus)
    bids_df = bids_df.drop([0, 1]).reset_index(drop=True)  
    start_time = bids_df["onset"].min()
    bids_df["onset"] = bids_df["onset"] - start_time
    return bids_df, data_string


def clean_read_data(subject, session, task, table, plotFlag=True):
    '''
    Combine reading, cleaning, processing and plotting functions

    Parameters
    ----------
   subject : string
       Subject number
   session : string 
       Session number
   task : string 
       Task name
   table: pandas.core.frame.DataFrame 
       Table which contains paths of each available event file
   plotFlag : bool
       Object which is True by default and responsible for plotting

    Returns
    -------
    bids_df_new : pandas.core.frame.DataFrame
        Final dataframe which is converted to tsv file.

    '''
    output_file_name = None
    # Filter the table based on subject, session, and task
    filtered_data = table[(table['subject'] == subject) & (
        table['session'] == session) & (table['task'] == task)]
    # initialize path variable
    log_path = ""
    if not filtered_data.empty:
        #extract log and bold path if they are found
        bold_paths = filtered_data.iloc[0]['func']
        log_paths = filtered_data.iloc[0]['event']
        if log_paths == []:
            return None, None, None
        #There are three reversal files. To ensure that we process the correct file, we should create a filter condition (ends with "WS.mat")
        if task == "reversal":
            if log_paths:
                for path in log_paths:
                    if path.endswith("WS.mat"):
                        log_path = path
                        break
            if bold_paths:
                bold_path = bold_paths[0]
        # the same with nback. we should take .xls file 
        elif task == "nback":
            if log_paths:
                for path in log_paths:
                    if path.endswith(".xls"):
                        log_path = path
                        break
            if bold_paths:
                bold_path = bold_paths[0]
        #In case of reward we should take .log file
        elif task == "reward":
            if log_paths:
                for path in log_paths:
                    if path.endswith(".log"):
                        log_path = path
                        break
            if bold_paths:
                bold_path = bold_paths[0]
        else:
            if log_paths:
                log_path = log_paths[0]
            if bold_paths:
                bold_path = bold_paths[0]
    else:
        return None, None, None
    bids_df_new = None
    #start reading, cleaning and plotting data 
    if task == "selfref":
        # firstly, open the file and clean it 
        bids_df, df_raw = read_clean_selfref(log_path)
        if bids_df is not None:
            #if bids_df is not none we can start to process this data
            bids_df = calculate_response_times(bids_df)
            #take the correct file name using bold file
            output_file_name = os.path.split(bold_path)[-1]
            output_file_name = output_file_name.replace('bold.nii.gz', '')
            output_file_path = os.path.join(
                rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.tsv")
            bids_df_new, bids_df = process_selfref(bids_df, answer_file_df)
            bids_df_new.to_csv(output_file_path, sep='\t', index=False)
            if plotFlag:
                fig = plotting_selfref(bids_df)
                output_file_path = os.path.join(
                    derivatives_dir, "plots", "sub-" + subject, output_file_name + "events.html")
                pio.write_html(fig, output_file_path, auto_open=False)
                dict_selfref = json_selfref()
                output_file_path1 = os.path.join(
                    rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.json")
                with open(output_file_path1, 'w') as json_data:
                    json.dump(dict_selfref, json_data)
    elif task == "reversal":
        bids_df_new, bids_df = read_cleanreversal(
            log_path, dict_reversal_answers)
        if bids_df_new is not None and bids_df is not None:
            output_file_name = os.path.split(bold_path)[-1]
            output_file_name = output_file_name.replace('bold.nii.gz', '')
            output_file_path = os.path.join(
                rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.tsv")
            bids_df_new.to_csv(output_file_path, sep='\t', index=False)
            if plotFlag:
                fig = plotting_reversal(bids_df)
                output_file_path = os.path.join(
                    derivatives_dir, "plots", "sub-" + subject, output_file_name + "events.html")
                pio.write_html(fig, output_file_path, auto_open=False)
    elif task == "reward":
        bids_df, df_raw = read_clean_other_tasks(log_path)
        if bids_df is not None:
            output_file_name = os.path.split(bold_path)[-1]
            output_file_name = output_file_name.replace('bold.nii.gz', '')
            output_file_path = os.path.join(
                rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.tsv")
            bids_df = calculate_response_times(bids_df)
            bids_df_new = process_reward(bids_df)
            bids_df_new.to_csv(output_file_path, sep='\t', index=False)
            if plotFlag:
                fig = plotting_reward(bids_df_new)
                output_file_path = os.path.join(
                    derivatives_dir, "plots", "sub-" + subject, output_file_name + "events.html")
                pio.write_html(fig, output_file_path, auto_open=False)
                dict_reward = json_reward()
                output_file_path1 = os.path.join(
                    rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.json")
                with open(output_file_path1, 'w') as json_data:
                    json.dump(dict_reward, json_data)
    elif task == "faces":
        bids_df, df_raw = read_clean_other_tasks(log_path)
        if bids_df is not None:
            output_file_name = os.path.split(bold_path)[-1]
            output_file_name = output_file_name.replace('bold.nii.gz', '')
            output_file_path = os.path.join(
                rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.tsv")
            bids_df = calculate_response_times(bids_df)
            bids_df_new, bids_df = process_faces(bids_df, dict_faces_answers)
            bids_df_new.to_csv(output_file_path, sep='\t', index=False)
            if plotFlag:
                fig = plotting_faces(bids_df)
                output_file_path = os.path.join(
                    derivatives_dir, "plots", "sub-" + subject, output_file_name + "events.html")
                pio.write_html(fig, output_file_path, auto_open=False)
                dict_faces = json_faces()
                output_file_path1 = os.path.join(
                    rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.json")
                with open(output_file_path1, 'w') as json_data:
                    json.dump(dict_faces, json_data)
    elif task == "nback":
        bids_df, df_raw = read_clean_other_tasks(log_path)
        if bids_df is not None:
            output_file_name = os.path.split(bold_path)[-1]
            output_file_name = output_file_name.replace('bold.nii.gz', '')
            output_file_path = os.path.join(
                rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.tsv")
            bids_df = calculate_response_times(bids_df)
            bids_df_new, bids_df = process_nback(bids_df)
            bids_df_new.to_csv(output_file_path, sep='\t', index=False)
            if plotFlag:
                fig = plotting_nback(bids_df)
                output_file_path = os.path.join(
                    derivatives_dir, "plots", "sub-" + subject, output_file_name + "events.html")
                pio.write_html(fig, output_file_path, auto_open=False)
                dict_nback = json_nback()
                output_file_path1 = os.path.join(
                    rawdata_dir, "sub-" + subject, 'ses-' + session, "func", output_file_name + "events.json")
                with open(output_file_path1, 'w') as json_data:
                    json.dump(dict_nback, json_data)
    return bids_df_new, output_file_name, log_path


def calculate_response_times(bids_df):
    '''
    Calculate response time (response time - stimulus time) for faces, selfref, nback and reward tasks

    Parameters
    ----------
    bids_df : pandas.core.frame.DataFrame
        Cleaned dataframe (not final dataframe).

    Returns
    -------
    bids_df : pandas.core.frame.DataFrame
        The same datframe with the response time column.

    '''
    def remove_second_consecutive_1(df):
        '''
        Remove the second consecutive response (as we need the first response)

        Parameters
        ----------
        df : pandas.core.frame.DataFrame
            Cleaned dataframe (not final dataframe).

        Returns
        -------
        df : pandas.core.frame.DataFrame
            Cleaned dataframe.

        '''
        remove_doubleresp = []
        for i in range(1, len(df)):
            if df.at[i, 'event_type'] == "1" and df.at[i - 1, 'event_type'] == "1":
                remove_doubleresp.append(i)
        df = df.drop(remove_doubleresp).reset_index(drop=True)
        return df
    
    bids_df = remove_second_consecutive_1(bids_df)
    indices_to_remove = []
    i = 0
    # Loop through the DataFrame
    while i < len(bids_df):
        if bids_df.at[i, 'event_type'] in ["vCSp", "lCSp", "wCSp", "no UCS", "CSm"]:
            if i + 1 < len(bids_df) and str(bids_df.at[i + 1, 'event_type']).isdigit():
                indices_to_remove.append(i + 1)
            i += 1
        else:
            i += 1

    # Remove the rows using the collected indices
    bids_df.drop(indices_to_remove, inplace=True)

    # Reset the index
    bids_df.reset_index(drop=True, inplace=True)
    response_indices = bids_df[(bids_df['event_type'] == "1") | (bids_df['event_type'] == "2") | (bids_df['event_type'] == "3")
                               | (bids_df['event_type'] == "4")].index

    last_stimulus = {}  # Keep track of the last stimulus for each type

    for index in response_indices:
        response_event = bids_df.loc[index]
        response_type = response_event['event_type']
        closest_stimulus = None
        min_time_diff = float('inf')
        indices_to_remove = []
        i = 0
        # Loop through the DataFrame
        while i < len(bids_df):
            if bids_df.at[i, 'event_type'] in ["vCSp", "lCSp", "wCSp"]:
                if i + 1 < len(bids_df) and bids_df.at[i + 1, 'event_type'].isdigit():
                    indices_to_remove.append(i + 1)
                i += 1
            else:
                i += 1
        
        # Remove the rows using the collected indices
        bids_df.drop(indices_to_remove, inplace=True)
        # Reset the index
        bids_df.reset_index(drop=True, inplace=True) 
        # Find the closest preceding non-"1" or non-"2" event of the same type
        for i in range(index - 1, -1, -1):
            if bids_df.loc[i, 'event_type'] == response_type:
                continue  # Skip consecutive "1" or "2" events of the same type
            if bids_df.loc[i, 'event_type'] not in ["1", "2", "3", "4"]:
                time_diff = response_event['onset'] - bids_df.loc[i, 'onset']
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_stimulus = bids_df.loc[i]

        # Calculate the response time and update the DataFrame in the same row
        if closest_stimulus is not None:
            response_time = response_event['onset'] - closest_stimulus['onset']
            bids_df.at[index-1, 'response_time'] = response_time
            # Update the last stimulus for this type
            last_stimulus[response_type] = closest_stimulus
    return bids_df


def process_faces(df, correct_responses):
    '''
    Process dataframes for faces task

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Cleaned dataframe with response time column.
    correct_responses : dict
        Correct answers for faces task

    Returns
    -------
    df_new : pandas.core.frame.DataFrame
        Final dataframe which will be converted to tsv
    df : pandas.core.frame.DataFrame
        Dataframe which will be used for plotting.
    '''
    conditions = []
    current_condition = None
    for event_type in df['event_type']:
        if event_type == 'MatchForms':
            current_condition = 'MatchForms'
        elif event_type == 'MatchFaces':
            current_condition = 'MatchFaces'
        conditions.append(current_condition)
    df['trial_type'] = conditions
    df = df[~df['event_type'].isin(['MatchForms', 'MatchFaces'])]
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    # Initialize 'Correctness' column in the original DataFrame
    df['correct'] = None

    correctness = []
    for index, row in df.iterrows():
        if row['event_type'] in ['1', '2']:
            response = int(row['event_type'])
            previous_row = df.iloc[index - 1]
            stimulus = previous_row['event_type']
            if stimulus in correct_responses and correct_responses[stimulus] == response:
                correctness.append("Correct")
            else:
                correctness.append("Incorrect")

            # Set the response in the 'responses' column of the stimulus row
            df.at[index - 1, 'responses'] = response

    # Assign the correctness values to the 'Correctness' column
    df.loc[df['event_type'].isin(['1', '2']), 'correct'] = correctness

    # Create a new DataFrame with 'Correctness' shifted one step up
    df_new = df.copy()
    df_new['correct'] = df_new['correct'].shift(-1)

    # Remove responses (1, 2) from the 'event_type' column in df_new
    df_new = df_new[~df_new['event_type'].isin(['1', '2'])]
    df_new = df_new.rename(columns={'event_type': 'faces_display'})
    for index, row in df_new.iterrows():
        if row['correct'] == "Correct":
            df_new.at[index, 'correct'] = "1"
        elif row['correct'] == "Incorrect":
            df_new.at[index, 'correct'] = "0"
    df_new = df_new.fillna("n/a")
    return df_new, df


def process_nback(df):
    '''
    Process dataframes for nback task 

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Cleaned dataframe with response time column.

    Returns
    -------
    df_new : pandas.core.frame.DataFrame
        Final dataframe which will be converted to tsv
    df : pandas.core.frame.DataFrame
        Dataframe which will be used for plotting.
    '''
    stimuli = []
    responses = []
    correctness = []
    # To keep track of the three previous stimuli (it is important for 2back condition)
    previous_stimuli = [None, None, None]
    #creating condition (trial_type) column
    conditions = []
    current_condition = None
    for event_type in df['event_type']:
        if event_type == '0back':
            current_condition = '0back'
        elif event_type == '2back':
            current_condition = '2back'
        conditions.append(current_condition)
    df['trial_type'] = conditions
    df = df[~df['event_type'].isin(['0back', '2back'])]
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    
    for i in range(len(df)):
        event_type = df['event_type'][i]
        if event_type in ['num1', 'num2', 'num3', 'num4']:
            #define the stimuli
            stimuli.append(event_type)
            responses.append(None)
            previous_stimuli[2] = previous_stimuli[1]
            previous_stimuli[1] = previous_stimuli[0]
            previous_stimuli[0] = event_type
        elif event_type in ['1', '2', '3', '4']:
            #define the responses
            stimuli.append(None)
            responses.append(int(event_type))
        else:
            stimuli.append(None)
            responses.append(None)

        if df['trial_type'][i] == '0back':
            if previous_stimuli[0] is not None and responses[i] is not None:
                # taking the number from the stimulus title (num1 - 1)
                #after that compare the response with the previous stimulus number (0back)
                if responses[i] == int(previous_stimuli[0].replace('num', '')):
                    correctness.append('Correct')
                else:
                    correctness.append('Incorrect')
            else:
                correctness.append(None)
        elif df['trial_type'][i] == '2back':
            #the same but 2 back
            if responses[i] is not None and all(x is not None for x in previous_stimuli):
                if responses[i] == int(previous_stimuli[2].replace('num', '')):
                    correctness.append('Correct')
                else:
                    correctness.append('Incorrect')
            else:
                correctness.append(None)
        else:
            correctness.append(None)
    for index, row in df.iterrows():
        if row['event_type'] in ['1', '2', '3', '4']:
            response = int(row['event_type'])
            df.at[index - 1, 'responses'] = response
    df['correct'] = correctness
    df_new = df.copy()
    df_new['correct'] = df_new['correct'].shift(-1)
    # The last element gets shifted out, so fill the last row with None
    df_new.at[df.index[-1], 'correct'] = None
    df_new = df_new.rename(columns={'event_type': 'nback_display'})
    df_new = df_new[~df_new['nback_display'].isin(['1', '2', '3', '4'])]
    for index, row in df_new.iterrows():
        if row['correct'] == "Correct":
            df_new.at[index, 'correct'] = "1"
        elif row['correct'] == "Incorrect":
            df_new.at[index, 'correct'] = "0"
    df_new = df_new.fillna("n/a")
    return df_new, df


def process_selfref(df, answer_file_df):
    '''
    Process dataframes for selfref task

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Cleaned dataframe with response time column.
    answer_file_df : pandas.core.frame.DataFrame
        Correct answers for selfref task 
        

    Returns
    -------
    df_new : pandas.core.frame.DataFrame
        Final dataframe which will be converted to tsv
    df : pandas.core.frame.DataFrame
        Dataframe which will be used for plotting.

    '''
    # the same method to create condition column
    conditions = []
    current_condition = None

    for event_type in df['event_type']:
        if event_type == 'Selbst':
            current_condition = 'Self'
        elif event_type == 'Merkel':
            current_condition = 'Merkel'
        elif event_type == "Silben":
            current_condition = "Syllables"
        conditions.append(current_condition)

    df['trial_type'] = conditions
    df = df[~df['event_type'].isin(["Selbst", 'Merkel', "Silben"])]
    df.reset_index(drop=True, inplace=True)
    #correct responses as dict
    stimuli_to_responses = dict(
        zip(answer_file_df['word'].str.lower(), answer_file_df['Correct']))
    correctness = []
    #checking the correctness only for syllables condition
    for index, row in df.iterrows():
        if row['trial_type'] == "Syllables" and row['event_type'] in ['1', '2']:
            if index > 0:
                previous_row = df.loc[index - 1]
                if previous_row['trial_type'] == "Syllables" and "Silben" in previous_row['event_type']:
                    # Extract and lowercase the word after the hyphen
                    stimulus = previous_row['event_type'].split(
                        ' - ')[1].lower()

                    if stimulus in stimuli_to_responses:
                        response = int(row['event_type'])
                        correct_response = stimuli_to_responses[stimulus]

                        if response == correct_response:
                            correctness.append("Correct")
                        else:
                            correctness.append("Incorrect")
                else:
                    correctness.append("None")
            if row['event_type'] in ['1', '2']:
                response = int(row['event_type'])
                df.at[index - 1, 'responses'] = response
        else:
            # For rows that are not 'Syllables' or not responses
            correctness.append("None")

    df['correct'] = correctness
    df = df[df['trial_type'] != None]
    df_new = df.copy()
    df_new['correct'] = df_new['correct'].shift(-1)
    # The last element gets shifted out, so fill the last row with None
    df_new.at[df.index[-1], 'correct'] = "None"
    df_new = df_new.rename(columns={'event_type': 'selfref_display'})
    df_new = df_new[~df_new['selfref_display'].isin(['1', '2'])]
    for index, row in df_new.iterrows():
        if row['correct'] == "Correct":
            df_new.at[index, 'correct'] = "1"
        elif row['correct'] == "Incorrect":
            df_new.at[index, 'correct'] = "0"
    df_new = df_new.fillna("n/a")
    return df_new, df


def process_reward(df1):
    '''
    Process dataframes for reward task

    Parameters
    ----------
    df1 : pandas.core.frame.DataFrame
        Cleaned dataframe with response time column.

    Returns
    -------
    df_new : pandas.core.frame.DataFrame
        Final dataframe which will be converted into tsv.

    '''
    conditions = []
    current_condition = None
    balance = 0  # Initialize the balance
    balance_initialized = False  # Flag to track if balance has been initialized

    for event_type in df1['event_type']:
        if event_type == 'wCSp':
            current_condition = 'wCSp'
        elif event_type == 'vCSp':
            current_condition = 'vCSp'
        elif event_type == "CSm":
            current_condition = "CSm"
        elif event_type == "lCSp":
            current_condition = "lCSp"
        conditions.append(current_condition)

    df1['trial_type'] = conditions

    results = []
    # these values are taken from the ask code
    time_window_beg_trial = 0.3
    on_time_ratio = 0.95
    slow_ratio = 1.05
    df1["correct"] = None
    on_time = None
    for i in range(len(df1)):
        event_type = df1.at[i, "event_type"]
        if event_type in ["vCSp", "lCSp", "wCSp"]:
            current_condition = event_type

        if event_type == "1":
            response_time = df1.at[i - 1, "response_time"]
            #on time if the response time was less than the time window
            on_time = response_time <= time_window_beg_trial if response_time is not None else None
            correct = "fast" if on_time else "slow"
            results.append(correct)

            if not balance_initialized:
                # Initialize balance only once
                balance = 0
                balance_initialized = True

        if event_type == "UCS" and (i + 1 >= len(df1) or df1.at[i + 1, "event_type"] != "1"):
            #if there is no response
            results.append("n/a")

        if event_type == "Feedback":
            if results:
                # multiply time window and on_time_ratio if the response is fast and if it is slow that multiply it by slow_ratio
                time_window_beg_trial *= on_time_ratio if results[-1] == "fast" else slow_ratio
               # if ICSp and slow response than subsract 2 from the balance
               # if wCSp and the response is fast than add 2 to balance
            trial_type = df1.at[i, "trial_type"]
            if trial_type == "lCSp" and not on_time:
                balance -= 2
            elif trial_type == "wCSp" and on_time:
                balance += 2
            # Set the balance for the current trial
            df1.at[i, "balance"] = balance
            #delete UCS (flash) from the event_type column and create flash column
    for index, row in df1.iterrows():
        if row['event_type'] in ["UCS", "no UCS"]:
            flash = row['event_type']
            df1.at[index - 1, 'flash'] = flash

    df1['response_time'] = df1['response_time'].shift(-1)
    
    # Add the results to the DataFrame
    df1.loc[df1["event_type"] == "Feedback", "correct"] = results

    # Remove unnecessary rows
    df_new = df1[~df1['event_type'].isin(['1', "Balance", "UCS", "no UCS"])]

    df_new = df_new.rename(columns={"event_type": "reward_display"})
    df_new = df_new.fillna("n/a")
    return df_new


# correct answers for faces task
faces_answer_string = """ 
               file				ecode		itime	resp ;
               "Form_1.jpg"	Form_1    	5000	1 ;
               "Form_2.jpg"	Form_2    	5000	2 ;
               "Form_3.jpg"	Form_3    	5000	2 ;
               "Form_4.jpg"	Form_4    	5000	1 ;
               "Form_5.jpg"	Form_5    	5000	2 ;
               "Form_6.jpg"	Form_6    	5000	1 ;
               "boy4a.jpg"		boy4a    	5000	2 ;
               "girl2a.jpg"	girl2a    	5000	1 ;
               "boy2a.jpg"		boy2a    	5000	2 ;
               "girl4a.jpg"	girl4a    	5000	1 ;
               "boy6a.jpg"		boy6a    	5000	1 ;
               "girl3a.jpg"	girl3a    	5000	2 ;
               "Form_6.jpg"	Form_6    	5000	1 ;
               "Form_5.jpg"	Form_5    	5000	2 ;
               "Form_4.jpg"	Form_4    	5000	1 ;
               "Form_3.jpg"	Form_3    	5000	2 ;
               "Form_2.jpg"	Form_2    	5000	2 ;
               "Form_1.jpg"	Form_1    	5000	1 ;
               "girl1a.jpg"	girl1a    	5000	2 ;
               "boy5a.jpg"		boy5a    	5000	2 ;
               "girl5a.jpg"	girl5a    	5000	2 ;
               "boy3a.jpg"		boy3a    	5000	1 ;
               "girl6a.jpg"	girl6a    	5000	1 ;
               "boy1a.jpg"		boy1a    	5000	1 ;
               "Form_1.jpg"	Form_1    	5000	1 ;
               "Form_2.jpg"	Form_2    	5000	2 ;
               "Form_3.jpg"	Form_3    	5000	2 ;
               "Form_4.jpg"	Form_4    	5000	1 ;
               "Form_5.jpg"	Form_5    	5000	2 ;
               "Form_6.jpg"	Form_6    	5000	1 ;
               "boy4a.jpg"		boy4a    	5000	2 ;
               "girl2a.jpg"	girl2a    	5000	1 ;
               "boy2a.jpg"		boy2a    	5000	2 ;
               "girl4a.jpg"	girl4a    	5000	1 ;
               "boy6a.jpg"		boy6a    	5000	1 ;
               "girl3a.jpg"	girl3a    	5000	2 ;
               "Form_6.jpg"	Form_6    	5000	1 ;
               "Form_5.jpg"	Form_5    	5000	2 ;
               "Form_4.jpg"	Form_4    	5000	1 ;
               "Form_3.jpg"	Form_3    	5000	2 ;
               "Form_2.jpg"	Form_2    	5000	2 ;
               "Form_1.jpg"	Form_1    	5000	1 ;
               "girl1a.jpg"	girl1a    	5000	2 ;
               "boy5a.jpg"		boy5a    	5000	2 ;
               "girl5a.jpg"	girl5a    	5000	2 ;
               "boy3a.jpg"		boy3a    	5000	1 ;
               "girl6a.jpg"	girl6a    	5000	1 ;
               "boy1a.jpg"		boy1a    	5000	1 ;
               """
faces_answer_string = faces_answer_string.replace("\t\t\t", "\t").replace("\t\t", "\t").replace(";",
                                                                                                "")  # Replace triple and double tabs with single tab and remove semicolon

df_faces_answers = pd.read_csv(StringIO(faces_answer_string), sep='\t')
df_faces_answers.columns = [column_name.strip() for column_name in
                            df_faces_answers.columns]  # Strip white spaces from column names
df_faces_answers = df_faces_answers.loc[:, [
    "ecode", "resp"]].drop_duplicates()  # .to_dict()
dict_faces_answers = dict([(key.strip(), value) for key, value in zip(df_faces_answers['ecode'],
                                                                      df_faces_answers[
    'resp'])])


# reversal correct answers
dict_reversal_answers = {}

# First do version A
dict_reversal_answers["version_A"] = {}
dict_reversal_answers["version_A"]["S"] = list([
    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
     0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
     0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7,
     0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
     0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
     0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
     0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1]])  # Flattened array to copy it here, now reshape to original form

dict_reversal_answers["version_A"]["block_nr"] = list([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7])  # Flattened array to copy it here, now reshape to original form

dict_reversal_answers["version_A"]["probability_of_misleading_feedback"] = list(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0]])

dict_reversal_answers["version_A"]["correct_card"] = list([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2]])  # Flattened array to copy it here, now reshape to original form

dict_reversal_answers["version_A"]["task_structure"] = list([[0., 0., 0.],
                                                             [30., 0.9, 1.],
                                                             [20., 0.1, 2.],
                                                             [10., 0.3, 3.],
                                                             [25., 0.7, 4.],
                                                             [20., 0.3, 5.],
                                                             [25., 0.8, 6.],
                                                             [30., 0.1, 7.]])

# Then do version B
dict_reversal_answers["version_B"] = {}
dict_reversal_answers["version_B"]["S"] = list(
    [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
      0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
      0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3,
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7,
      0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
      0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
      0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
      0.3, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
      0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1]])  # Flattened array to copy it here, now reshape to original form

dict_reversal_answers["version_B"]["block_nr"] = list([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7])  # Flattened array to copy it here, now reshape to original form

dict_reversal_answers["version_B"]["probability_of_misleading_feedback"] = list(
    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
      1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
      1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 1, 0, 0, 0]])

dict_reversal_answers["version_B"]["correct_card"] = list([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2]])  # Flattened array to copy it here, now reshape to original form

dict_reversal_answers["version_B"]["task_structure"] = list([[0., 0., 0.],
                                                             [30., 0.9, 1.],
                                                             [20., 0.1, 2.],
                                                             [10., 0.3, 3.],
                                                             [25., 0.7, 4.],
                                                             [20., 0.3, 5.],
                                                             [25., 0.8, 6.],
                                                             [30., 0.1, 7.]])


# selfref answers
selfRef_answer_string = 'word\tcorrect response (1: no, 2: yes)\nspeziell\t2\nsimpel\t2\nbeherrscht\t2\nbeharrlich\t1\nehrgeizig\t1\nmoralisch\t1\nredselig\t1\nvorsichtig\t1\nfeurig\t2\nzerstreut\t2\nsachlich\t2\nerschöpft\t2\nreserviert\t1\nfromm\t1\nskeptisch\t2\nbullig\t2\nalbern\t2\nlehrhaft\t2\nanalytisch\t1\nseltsam\t2\nstumm\t1\nmarkant\t2\nnaiv\t2\nautonom\t1\ngenügsam\t1\nforsch\t1\nlangsam\t2\nangepasst\t1\nstolz\t1\neigenwillig\t1\nzierlich\t2\nruhig\t2\nzeitlos\t2\ndirekt\t2\nverträumt\t2\nstill\t1\nverwegen\t1\nverlegen\t1\nneutral\t2\ngehorsam\t1\nbestimmend\t1\nverspielt\t2\nlenkbar\t2\naufgewühlt\t1\nsittsam\t2\nvornehm\t2\nfrech\t1\nharmlos\t2\nsehnsüchtig\t1\nwahllos\t2\neifrig\t2\nsentimental\t1\nlaut\t1\nminimal\t1\nkontrolliert\t1\nbesorgt\t2\nschrill\t1\nschüchtern\t2\nirritiert\t1\nnachgiebig\t1\nabwesend\t1\nverletzlich\t1\nscheu\t1\nstoisch\t2\nsteif\t1\nempfindlich\t1\nungestüm\t1\nstrebsam\t2\nunnahbar\t1\nkritisch\t2\nautark\t2\nnormal\t2\nraffiniert\t1\nironisch\t1\nschweigsam\t2\nbedächtig\t1\ndefensiv\t1\nheroisch\t1\nsparsam\t2\nzaghaft\t2\nwählerisch\t1\nmassiv\t2\nmystisch\t2\nkritisch\t2\nschlicht\t1\nlistig\t2\nparadox\t1\nkomisch\t2\nwortkarg\t2\nsorglos\t2\n'

# Turn into pandas dataframe
answer_file_df = pd.read_csv(StringIO(selfRef_answer_string),
                             sep='\t')

# Make sure all keys are integers
answer_file_df = answer_file_df.astype(
    {"correct response (1: no, 2: yes)": "int64"})

# Substitute any non-ASCII appropriate German characters
substitutions = [('ã–', 'ae'), ('ã„', 'ae'), ('ãœ', 'ae'), ('ä', 'ae'), ('ü', 'ue'),
                 ('ö', 'oe')]  # Each tuple (a,b) specifies a character a to be replaced by a character b
answer_file_df["word"] = list(
    map(functools.partial(multisub, substitutions), list(answer_file_df["word"].values)))
answer_file_df = answer_file_df.rename(
    columns={'correct response (1: no, 2: yes)': "Correct"})


def multisub(subs, str):
    """Simultaneously perform all substitutions specified in subs on the string str.

        Parameters
        ----------
        subs : list of tuples
            List of substitutions to make in the string, e.g. subs = [('no', 'yes')], you want to replace 'no' with 'yes'.
        str : string
            String of characters on which to perform substitutions.

        Returns
        -------
        string
            The string with substitutions made.
    """

    pattern = '|'.join('(%s)' % re.escape(p) for p, s in
                       subs)
    substs = [s for p, s in subs]
    def replace(m): return substs[m.lastindex - 1]
    return re.sub(pattern, replace, str)


def plotting_faces(df):
    '''
    Plot faces events

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe for plotting containing stimuli and responses.
    '''
    # finding incorrect response in the event_type column
    df.loc[df['correct'] == 'Incorrect', 'event_type'] = 'incorrect'
    z = []  # Collect all the 'z' values
    for event_type in df['event_type']:
        event_type_str = str(event_type)  # Convert to string
        if event_type_str.isnumeric():
            z.append(1)
        elif event_type_str == "incorrect":
            z.append(0.5)
        else:
            z.append(0)

    # Create the heatmap
    fig = go.Figure(go.Heatmap(
        z=z,
        x=df['onset'],
        y=df['trial_type'],
        colorscale=[[0, "lightgrey"], [0.5, "indianred"], [1, "#8cc751"]],
        showscale=True,
        colorbar=dict(tickvals=[0, 0.5, 1], ticktext=["Stimulus", "Incorrect Response", "Response"],
                      tickfont=dict(size=14))))

    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),  # Sort the y-axis
        xaxis_title="Time",
        yaxis_title="trial_type")
    return fig


def plotting_nback(df):
    '''
    Plot nback events

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe for plotting containing stimuli and responses.

    '''
    df.loc[df['correct'] == 'Incorrect', 'event_type'] = 'incorrect'
    z = []
    for event_type in df['event_type']:
        event_type_str = str(event_type)  # Convert to string
        if event_type_str.isnumeric():
            z.append(1)
        elif event_type == "incorrect":
            z.append(0.5)
        else:
            z.append(0)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=df['onset'],
        y=df['trial_type'],
        colorscale=[[0, "lightgrey"], [0.5, "indianred"], [1, "#8cc751"]],
        showscale=True,
        colorbar=dict(tickvals=[0, 0.5, 1], ticktext=["Stimulus", "Incorrect Response", "Response"],
                      tickfont=dict(size=14))))

    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),  # Sort the y-axis
        xaxis_title="Time",
        yaxis_title="trial_type")
    return fig


def plotting_selfref(df):
    '''
    Plot selfref events

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe for plotting containing stimuli and responses.

    '''
    df.loc[df['correct'] == 'Incorrect', 'event_type'] = 'incorrect'
    z = []
    for index, row in df.iterrows():
        trial_type = row['trial_type']
        event_type = row['event_type']
        if trial_type == 'Self' or trial_type == 'Merkel':
            if event_type in ["1", "2"]:
                z.append(0.5)
            else:
                z.append(0)
        elif trial_type == "Syllables":
            if event_type in ["1", "2"]:
                z.append(1)
            elif event_type == "incorrect":
                z.append(0.75)
            else:
                z.append(0)
        else:
            z.append(0)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=df['onset'],
        y=df['trial_type'],
        colorscale=[[0, "lightgrey"], [0.5, "lightskyblue"],
                    [0.75, "indianred"], [1, "#8cc751"]],
        showscale=True,
        colorbar=dict(tickvals=[0, 0.5, 0.75, 1], ticktext=["Stimulus", "Yes or No", "Incorrect Response", "Correct response"],
                      tickfont=dict(size=14))))

    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),  # Sort the y-axis
        xaxis_title="Time",
        yaxis_title="trial_type")
    return fig


def plotting_reward(df):
    '''
    Plot faces events

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe for plotting containing stimuli and responses.

    '''
    z = []
    for event, correct in zip(df["reward_display"], df["correct"]):
        if event == "Feedback" and correct == "slow":
            z.append(1)
        if event == "Feedback" and correct == "fast":
            z.append(0.5)
        if event in ["vCSp", "lCSp", "wCSp", "CSm"]:
            z.append(0)
            #2 subplots for balance and event plots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Balance", "Trials"),
        row_heights=[0.3, 0.7])
    fig.add_trace(go.Scatter(x=df["onset"], y=df["balance"],
                  mode='lines', name='balance', connectgaps=True), row=1, col=1)

    fig.update_traces(mode="lines+markers")

    fig.update_yaxes(row=1, col=1, showgrid=True, gridwidth=vars.grid_width, gridcolor=vars.grid_color,
                     zeroline=True, zerolinewidth=vars.grid_width, zerolinecolor=vars.grid_color)
    
    fig.add_trace(go.Heatmap(
        z=z,
        x=df['onset'],
        y=df['trial_type'],
        colorscale=[[0, "lightgrey"], [0.5, "#8cc751"], [1, "indianred"]],
        showscale=True,
        colorbar=dict(tickvals=[0, 0.5, 1], ticktext=["Conditioned Stimulus", "Fast response", "Slow response"],
                      tickfont=dict(size=14))), row=2, col=1)

    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),  # Sort the y-axis
        xaxis_title="Time",
        yaxis_title="trial_type")
    return fig


def plotting_reversal(df):
    '''
    Plot faces events

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe for plotting containing stimuli and responses.
    '''
    z = []
    for event, correct in zip(df["event_type"], df["correct"]):
        if event == "stim":
            z.append(0)
        elif event == "response":
            if correct == 1:
                z.append(1)
            elif correct == 0:
                z.append(0.5)
            else:
                z.append(0)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=df['onset'],
        y=df['trial_type'],
        colorscale=[[0, "lightgrey"], [0.5, "indianred"], [1, "#8cc751"]],
        showscale=True,
        colorbar=dict(tickvals=[0, 0.5, 1], ticktext=["Stimulus", "Incorrect response", "Correct response"],
                      tickfont=dict(size=14))))

    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),  # Sort the y-axis
        xaxis_title="Time",
        yaxis_title="trial_type")
    return fig


def json_faces():
    '''
    Create json description file 

    Returns
    -------
    dict_face : dict
        Dictionary containing descriptive information.

    '''
    dict_face = {"onset": {
        "LongName": "Trial start time",
        "Description": "Trial start time in seconds. Relative to the start of the task, not the start of the scan"
    },
        "duration": {
        "LongName": "Stimuli duration",
        "Description": "Stimuli presentation time in seconds"
    },
        "faces_display": {
        "LongName": "Face or form displayed",
        "Description": "Face or form displayed during the trial"
    },
        "response_time": {
        "LongName": "Response time",
        "Description": "Trial response time in seconds"
    },
        "trial_type": {
        "LongName": "Face or Form condition",
    },
        "correct": {
        "LongName": "Correct response marker",
        "Description": "Did the participant respond correct number?",
        "Levels": {
            "1": "Yes",
            "0": "No"
        }
    },
        "responses": {
        "LongName": "Responses",
        "Description": "Responses of participants"
    }
    }
    return dict_face


def json_nback():
    '''
    Create json description file 

    Returns
    -------
    dict_nback : dict
        Dictionary containing descriptive information.

    '''
    dict_nback = {"onset": {
        "LongName": "Trial start time",
        "Description": "Trial start time in seconds. Relative to the start of the task, not the start of the scan"
    },
        "duration": {
        "LongName": "Stimuli duration",
        "Description": "Stimuli presentation time in seconds"
    },
        "nback_display": {
        "LongName": "Number displayed ",
        "Description": "Number (1-4) displayed during the trial"
    },
        "response_time": {
        "LongName": "Response time",
        "Description": "Trial response time in seconds"
    },
        "trial_type": {
        "LongName": "n-back condition",
        "Description": "n-back condition displayed"
    },
        "correct": {
        "LongName": "Correct response marker",
        "Description": "Did the participant respond with the correct number?",
        "Levels": {
            "1": "Yes",
            "0": "No"
        }
    },
        "responses": {
        "LongName": "Participant response",
        "Description": "Participant response number (1-4)",
        "Levels": {
            "1": "Number 1",
            "2": "Number 2",
            "3": "Number 3",
            "4": "Number 4",
        }
    }
    }
    return dict_nback


def json_selfref():
    '''
    Create json description file 

    Returns
    -------
    dict_selfref : dict
        Dictionary containing descriptive information.

    '''
    dict_selfref = {"onset": {
        "LongName": "Trial start time",
        "Description": "Trial start time in seconds. Relative to the start of the task, not the start of the scan"
    },
        "duration": {
        "LongName": "Stimuli duration",
        "Description": "Stimuli presentation time in seconds"
    },
        "selfref_display": {
        "LongName": "Number displayed ",
        "Description": "Word displayed during the trial"
    },
        "response_time": {
        "LongName": "Response time",
        "Description": "Trial response time in seconds"
    },
        "trial_type": {
        "LongName": "selfref condition",
        "Description": "selfref condition displayed"
    },
        "correct": {
        "LongName": "Correct response marker",
        "Description": "Did the participant respond with the correct number?",
        "Levels": {
            "1": "Yes",
            "0": "No"
        }
    },
        "responses": {
        "LongName": "Participant response",
        "Description": "Participant response number",
        "Levels": {"if trial_type == Self and Merkel and responses == 1": "response is yes",
                   "if trial_type == Self and Merkel and responses == 0": "response is no"
                   }
    }
    }
    return dict_selfref


def json_reward():
    '''
    Create json description file 

    Returns
    -------
    dict_reward : dict
        Dictionary containing descriptive information.

    '''
    dict_reward = {"onset": {
        "LongName": "Trial start time",
        "Description": "Trial start time in seconds. Relative to the start of the task, not the start of the scan"
    },
        "duration": {
        "LongName": "Stimuli duration",
        "Description": "Stimuli presentation time in seconds"
    },
        "reward_display": {
        "LongName": "Stimulus displayed ",
        "Description": "Arrow (CS) displayed during the trial"
    },
        "response_time": {
        "LongName": "Response time",
        "Description": "Trial response time in seconds"
    },
        "trial_type": {
        "LongName": "Reward condition",
        "Description": "Reward condition displayed"
    },
        "correct": {
        "LongName": "Correct response marker",
        "Description": "Did the participant respond on time?",
        "Levels": {
            "Fast": "Within the time window",
            "Slow": "Longer then time window"
        }
    },
        "balance": {
        "LongName": "Balance",
        "Description": "Participant balance after each trial"
    },
        "flash": {
        "LongName": "Flash (unconditiones stimulus)",
        "Description": "Flash or no flash after conditioned stimulus"}
    }
    return dict_reward


def generate_table_html_face(bids_df_final, log_path):
    '''
    Generates html summary table 

    Parameters
    ----------
    bids_df_final : pandas.core.frame.DataFrame
        Final dataframe containing neccessary information.
    log_path : str
        Path to the log file.

    Returns
    -------
    table_html : str
        String which cam be opened as HTML file containing table.

    '''
    # defining the style of the table
    table_styles = (
        'width: 100%;'
        'border-collapse: collapse;'
        'border: 1px solid black;'
        'border-spacing: 5px;'
        'row-gap: 1ch;'
        'column-gap: 10%;')
    
    # the style of cells
    cell_styles = "text-align: center; padding: 5px"
    # Create the HTML table
    table_html = f'<table style="{table_styles}">'
    table_html += '<tr class="header-row"><th style="background-color: lightgrey;">Filepath</th><th style="background-color: lightgrey;">Trial type</th><th style="background-color: lightgrey;">Number of Stimuli</th><th style="background-color: lightgrey;">Valid responses</th><th style="background-color: lightgrey;">Correct Responses %</th><th style="background-color: lightgrey;">Response time (mean)</th></tr>'
    # Separate data based on trial_type
    trial_types = ["MatchForms", "MatchFaces"]
    # initialize empty lists  
    match_forms = []
    match_faces = []
    correct_forms = []
    correct_faces = []
    valid_forms = []
    valid_faces = []
    response_face = []
    response_form = []
    for trial_type, correct, response_time in zip(bids_df_final["trial_type"], bids_df_final["correct"], bids_df_final["response_time"]):
        if trial_type == "MatchForms" and correct == "1":
            #calculating the number of the correct answers
            correct_forms.append(1)
        elif trial_type == "MatchFaces" and correct == "1":
            #calculating the number of the correct answers
            correct_faces.append(1)
        if trial_type == "MatchForms":
            #calculatiing the number of stimuli
            match_forms.append(1)
        elif trial_type == "MatchFaces":
            match_faces.append(1)
        if trial_type == "MatchForms" and response_time != "n/a":
            #calculating the number of valid responses
            valid_forms.append(1)
        elif trial_type == "MatchFaces" and response_time != "n/a":
            valid_faces.append(1)
        if trial_type == "MatchForms":
            #lists with response times
            response_form.append(response_time)
        elif trial_type == "MatchFaces":
            response_face.append(response_time)
    # Calculate the number of stimuli and correctness for each trial type
    num_stimuli = [len(match_forms), len(match_faces)]
    #the percentage of correctness 
    correctness = [(len(correct_forms) / len(valid_forms)) * 100, (len(correct_faces) / len(valid_faces))*100]
    num_valid = [len(valid_forms), len(valid_faces)]
    response_form = [value for value in response_form if value != "n/a"]
    response_face = [value for value in response_face if value != "n/a"]
    #response mean calculating
    response_mean = [round(sum(response_form)/len(valid_forms), 2), round(sum(response_face)/len(valid_faces), 2)]
    # Add data to the table with separate rows for each trial type
    for i in range(2):  # Assuming there are two trial types
        cell_style = f'style="{cell_styles}"'
        if i == 0:  # Only for the first row
            table_html += f'<tr><td {cell_style} rowspan="2">{log_path}</td><td {cell_style}>{trial_types[i]}</td><td {cell_style}>{num_stimuli[i]:.2f}</td><td {cell_style}>{num_valid[i]}</td><td {cell_style}>{correctness[i]:.2f}</td><td {cell_style}>{response_mean[i]}</td></tr>'
        else:
            table_html += f'<tr><td {cell_style}>{trial_types[i]}</td><td {cell_style}>{num_stimuli[i]:.2f}</td><td {cell_style}>{num_valid[i]}</td><td {cell_style}>{correctness[i]:.2f}</td><td {cell_style}>{response_mean[i]}</td></tr>'
    table_html += '</table>'
    return table_html


def generate_table_html_selfref(bids_df_final, log_path):
    '''
    Generates html summary table 

    Parameters
    ----------
    bids_df_final : pandas.core.frame.DataFrame
        Final dataframe containing neccessary information.
    log_path : str
        Path to the log file.

    Returns
    -------
    table_html : str
        String which cam be opened as HTML file containing table.

    '''
    # defining the style of the table
    table_styles = (
        'width: 100%;'
        'border-collapse: collapse;'
        'border: 1px solid black;'
        'border-spacing: 5px;'
        'row-gap: 1ch;'
        'column-gap: 10%;')
    
    # the style of cells
    cell_styles = "text-align: center; padding: 5px"
    # Create the HTML table
    table_html = f'<table style="{table_styles}">'
    table_html += '<tr class="header-row"><th style="background-color: lightgrey;">Filepath</th><th style="background-color: lightgrey;">Trial type</th><th style="background-color: lightgrey;">Number of Stimuli</th><th style="background-color: lightgrey;">Valid responses</th><th style="background-color: lightgrey;">Correct Responses %</th><th style="background-color: lightgrey;">Response time (mean)</th></tr>'    
    # Separate data based on trial_type
    trial_types = ["Self", "Merkel", "Syllables"]
    # similar calculations
    self_count = []
    syllables_count = []
    merkel_count = []
    correct_syllables = []
    valid_self = []
    valid_merkel = []
    valid_syllables = []
    response_self = []
    response_merkel = []
    response_syllables = []
    for trial_type, correct, response_time in zip(bids_df_final["trial_type"], bids_df_final["correct"], bids_df_final["response_time"]):
        if trial_type == "Self":
            self_count.append(1)
        elif trial_type == "Merkel":
            merkel_count.append(1)
        elif trial_type == "Syllables":
            syllables_count.append(1)
        if trial_type == "Syllables" and correct == "1":
            correct_syllables.append(1)
        if trial_type == "Self" and response_time != "n/a":
            valid_self.append(1)
        elif trial_type == "Merkel" and response_time != "n/a":
            valid_merkel.append(1)
        elif trial_type == "Syllables" and response_time != "n/a":
            valid_syllables.append(1)
        if trial_type == "Self":
            response_self.append(response_time)
        elif trial_type == "Merkel":
            response_merkel.append(response_time)
        elif trial_type == "Syllables":
            response_syllables.append(response_time)
    # Calculate the number of stimuli and correctness for each trial type
    num_stimuli = [len(self_count), len(merkel_count), len(syllables_count)]
    correctness = [0, 0, (len(correct_syllables) / len(valid_syllables))*100]
    num_valid = [len(valid_self), len(valid_merkel), len(valid_syllables)]
    
    response_self = [value for value in response_self if value != "n/a"]
    response_merkel = [value for value in response_merkel if value != "n/a"]
    response_syllables = [value for value in response_syllables if value != "n/a"]
    response_mean = [round(sum(response_self)/len(valid_self), 2), round(sum(response_merkel)/len(valid_merkel), 2), round(sum(response_syllables)/len(valid_syllables), 2)]
    # Add data to the table with separate rows for each trial type
    cell_style = f'style="{cell_styles}"'
    table_html += f'<tr><td {cell_style} rowspan="3">{log_path}</td>'
    for i in range(3):  # Assuming there are two trial types
        table_html += f'<td {cell_style}>{trial_types[i]}</td><td {cell_style}>{num_stimuli[i]:.2f}</td><td {cell_style}>{num_valid[i]}</td><td {cell_style}>{correctness[i]:.2f}</td><td {cell_style}>{response_mean[i]}</td></tr>'
    table_html += '</table>'
    return table_html


def generate_table_html_reversal(bids_df_final, log_path):
    '''
    Generates html summary table 

    Parameters
    ----------
    bids_df_final : pandas.core.frame.DataFrame
        Final dataframe containing neccessary information.
    log_path : str
        Path to the log file.

    Returns
    -------
    table_html : str
        String which cam be opened as HTML file containing table.

    '''
    # defining the style of the table
    table_styles = (
        'width: 100%;'
        'border-collapse: collapse;'
        'border: 1px solid black;'
        'border-spacing: 5px;'
        'row-gap: 1ch;'
        'column-gap: 10%;')
    # the style of cells
    cell_styles = "text-align: center; padding: 5px"
    # Create the HTML table
    table_html = f'<table style="{table_styles}">'
    table_html += '<tr class="header-row"><th style="background-color: lightgrey;">Filepath</th><th style="background-color: lightgrey;">Trial type</th><th style="background-color: lightgrey;">Number of Stimuli</th><th style="background-color: lightgrey;">Valid responses</th><th style="background-color: lightgrey;">Correct Responses %</th><th style="background-color: lightgrey;">Response time (mean)</th></tr>'    
    # Separate data based on trial_type
    trial_types = ["informative reward", "misleading punishment", "misleading reward", "informative punishment"]
    # similar calculations. 
    infor_reward_count = []
    mislead_pun_count = []
    mislead_reward_count = []
    infor_pun_count = []
    correct_infor_reward = []
    correct_mislead_pun = []
    correct_mislead_reward = []
    correct_infor_pun = []
    valid_infor_reward = []
    valid_mislead_pun = []
    valid_mislead_reward = []
    valid_infor_pun = []
    response_infor_reward = []
    response_mislead_pun = []
    response_mislead_reward = []
    response_infor_pun = []
    for trial_type, correct, response_time in zip(bids_df_final["trial_type"], bids_df_final["correct"], bids_df_final["response_time"]):
        if trial_type == "informative reward":
            infor_reward_count.append(1)
        elif trial_type == "misleading punishment":
            mislead_pun_count.append(1)
        elif trial_type == "misleading reward":
            mislead_reward_count.append(1)
        elif trial_type == "informative punishment":
            infor_pun_count.append(1)
        if trial_type == "informative reward" and correct == 1:
            correct_infor_reward.append(1)
        elif trial_type == "misleading punishment" and correct == 1:
            correct_mislead_pun.append(1)
        elif trial_type == "misleading reward" and correct == 1:
            correct_mislead_reward.append(1)
        elif trial_type == "informative punishment" and correct == 1:
            correct_infor_pun.append(1)
        if trial_type == "informative reward" and response_time != "n/a":
            valid_infor_reward.append(1)
        elif trial_type == "misleading punishment" and response_time != "n/a":
            valid_mislead_pun.append(1)
        elif trial_type == "misleading reward" and response_time != "n/a":
            valid_mislead_reward.append(1)
        elif trial_type == "informative punishment" and response_time != "n/a":
            valid_infor_pun.append(1)
        if trial_type == "informative reward":
            response_infor_reward.append(response_time)
        elif trial_type == "misleading punishment":
            response_mislead_pun.append(response_time)
        elif trial_type == "misleading reward":
            response_mislead_reward.append(response_time)
        elif trial_type == "informative punishment":
            response_infor_pun.append(response_time)
    # Calculate the number of stimuli and correctness for each trial type
    num_stimuli = [len(infor_reward_count), len(mislead_pun_count), len(mislead_reward_count), len(infor_pun_count)]
    correctness = [(len(correct_infor_reward) / len(valid_infor_reward))*100,
                   (len(correct_mislead_pun) / len(valid_mislead_pun))*100,
                   (len(correct_mislead_reward) / len(valid_mislead_reward))*100,
                   (len(correct_infor_pun) / len(valid_infor_pun))*100]
    num_valid = [len(valid_infor_reward), len(valid_mislead_pun), len(valid_mislead_reward), len(valid_infor_pun)]
    
    response_infor_reward = [value for value in response_infor_reward if value != "n/a"]
    response_mislead_pun = [value for value in response_mislead_pun if value != "n/a"]
    response_mislead_reward = [value for value in response_mislead_reward if value != "n/a"]
    response_infor_pun = [value for value in response_infor_pun if value != "n/a"]
    
    response_mean = [round(sum(response_infor_reward)/len(valid_infor_reward), 2),
                     round(sum(response_mislead_pun)/len(valid_mislead_pun), 2), 
                     round(sum(response_mislead_reward)/len(valid_mislead_reward), 2),
                     round(sum(response_infor_pun)/len(valid_infor_pun), 2)]
    # Add data to the table with separate rows for each trial type
    cell_style = f'style="{cell_styles}"'
    table_html += f'<tr><td {cell_style} rowspan="4">{log_path}</td>'
    for i in range(4):
        table_html += f'<td {cell_style}>{trial_types[i]}</td><td {cell_style}>{num_stimuli[i]:.2f}</td><td {cell_style}>{num_valid[i]}</td><td {cell_style}>{correctness[i]:.2f}</td><td {cell_style}>{response_mean[i]}</td></tr>'
    table_html += '</table>'
    return table_html


def generate_table_html_nback(bids_df_final, log_path):
    # defining the style of the table
    table_styles = (
        'width: 100%;'
        'border-collapse: collapse;'
        'border: 1px solid black;'
        'border-spacing: 5px;'
        'row-gap: 1ch;'
        'column-gap: 10%;')
    
    # the style of cells
    cell_styles = "text-align: center; padding: 5px"
    # Create the HTML table
    table_html = f'<table style="{table_styles}">'
    table_html += '<tr class="header-row"><th style="background-color: lightgrey;">Filepath</th><th style="background-color: lightgrey;">Trial type</th><th style="background-color: lightgrey;">Number of Stimuli</th><th style="background-color: lightgrey;">Valid responses</th><th style="background-color: lightgrey;">Correct Responses %</th><th style="background-color: lightgrey;">Response time (mean)</th></tr>'    
    # Separate data based on trial_type
    trial_types = ["0back", "2back"]
    # similar calculations
    oback_count = []
    twoback_count = []
    correct_oback = []
    correct_2back = []
    valid_oback = []
    valid_2back = []
    response_oback = []
    response_2back = []
    for trial_type, correct, response_time in zip(bids_df_final["trial_type"], bids_df_final["correct"], bids_df_final["response_time"]):
        if trial_type == "0back" and correct == "1":
            correct_oback.append(1)
        elif trial_type == "2back" and correct == "1":
            correct_2back.append(1)
        if trial_type == "0back":
            oback_count.append(1)
        elif trial_type == "2back":
            twoback_count.append(1)
        if trial_type == "0back" and response_time != "n/a":
            valid_oback.append(1)
        elif trial_type == "2back" and response_time != "n/a":
            valid_2back.append(1)
        if trial_type == "0back":
            response_oback.append(response_time)
        elif trial_type == "2back":
            response_2back.append(response_time)
    # Calculate the number of stimuli and correctness for each trial type
    num_stimuli = [len(oback_count), len(twoback_count)]
    correctness = [(len(correct_oback) / len(valid_oback)) * 100, (len(correct_2back) / len(valid_2back))*100]
    num_valid = [len(valid_oback), len(valid_2back)]
    response_oback = [value for value in response_oback if value != "n/a"]
    response_2back = [value for value in response_2back if value != "n/a"]
    response_mean = [round(sum(response_oback)/len(valid_oback), 2), round(sum(response_2back)/len(valid_2back), 2)]
    # Add data to the table with separate rows for each trial type
    for i in range(2):  # Assuming there are two trial types
        cell_style = f'style="{cell_styles}"'
        if i == 0:  # Only for the first row
            table_html += f'<tr><td {cell_style} rowspan="2">{log_path}</td><td {cell_style}>{trial_types[i]}</td><td {cell_style}>{num_stimuli[i]:.2f}</td><td {cell_style}>{num_valid[i]}</td><td {cell_style}>{correctness[i]:.2f}</td><td {cell_style}>{response_mean[i]}</td></tr>'
        else:
            table_html += f'<tr><td {cell_style}>{trial_types[i]}</td><td {cell_style}>{num_stimuli[i]:.2f}</td><td {cell_style}>{num_valid[i]}</td><td {cell_style}>{correctness[i]:.2f}</td><td {cell_style}>{response_mean[i]}</td></tr>'    
    table_html += '</table>'
    return table_html


def generate_table_html_reward(bids_df_final, log_path):
    '''
    Generates html summary table 

    Parameters
    ----------
    bids_df_final : pandas.core.frame.DataFrame
        Final dataframe containing neccessary information.
    log_path : str
        Path to the log file.

    Returns
    -------
    table_html : str
        String which cam be opened as HTML file containing table.

    '''
    # defining the style of the table
    table_styles = (
        'width: 100%;'
        'border-collapse: collapse;'
        'border: 1px solid black;'
        'border-spacing: 5px;'
        'row-gap: 1ch;'
        'column-gap: 10%;')
    # the style of cells
    cell_styles = "text-align: center; padding: 5px"
    
    # Create the HTML table
    table_html = f'<table style="{table_styles}">'
    table_html += '<tr class="header-row"><th style="background-color: lightgrey;">Filepath</th><th style="background-color: lightgrey;">Trial type</th><th style="background-color: lightgrey;">Number of Stimuli</th><th style="background-color: lightgrey;">Valid responses</th><th style="background-color: lightgrey;">Fast responses (within the time window) %</th><th style="background-color: lightgrey;">Response time (mean)</th></tr>'    
    # Separate data based on trial_type
    trial_types = ["wCSp", "lCSp", "vCSp", "CSm"]
    # similar calculations, however we do not calculate correctness, number of valid responses and response time for CSm
    wCSp_count = []
    ICSp_count = []
    vCSp_count = []
    CSm_count = []
    correct_wCSp = []
    correct_ICSp = []
    correct_vCSp = []
    valid_wCSp = []
    valid_ICSp = []
    valid_vCSp = []
    response_wCSp = []
    response_ICSp = []
    response_vCSp = []
    for event_type, trial_type, correct, response_time in zip(bids_df_final["reward_display"],bids_df_final["trial_type"], bids_df_final["correct"], bids_df_final["response_time"]):
        if event_type == "wCSp":
            wCSp_count.append(1)
        elif event_type == "lCSp":
            ICSp_count.append(1)
        elif event_type == "vCSp":
            vCSp_count.append(1)
        elif event_type == "CSm":
            CSm_count.append(1)
        if trial_type == "wCSp" and correct == "fast":
            correct_wCSp.append(1)
        elif trial_type == "lCSp" and correct == "fast":
            correct_ICSp.append(1)
        elif trial_type == "vCSp" and correct == "fast":
            correct_vCSp.append(1)
        if trial_type == "wCSp" and response_time != "n/a":
            valid_wCSp.append(1)
        elif trial_type == "lCSp" and response_time != "n/a":
            valid_ICSp.append(1)
        elif trial_type == "vCSp" and response_time != "n/a":
            valid_vCSp.append(1)
        if trial_type == "wCSp":
            response_wCSp.append(response_time)
        elif trial_type == "lCSp":
            response_ICSp.append(response_time)
        elif trial_type == "vCSp":
            response_vCSp.append(response_time)
    # Calculate the number of stimuli and correctness for each trial type
    num_stimuli = [len(wCSp_count), len(ICSp_count), len(vCSp_count), len(CSm_count)]
    correctness = [round((len(correct_wCSp) / len(valid_wCSp))*100, 2),
                   round((len(correct_ICSp) / len(valid_ICSp))*100, 2),
                   0,
                   "n/a"]
    num_valid = [len(valid_wCSp), len(valid_ICSp), len(valid_vCSp), "n/a"]
    
    response_wCSp = [value for value in response_wCSp if value != "n/a"]
    response_ICSp = [value for value in response_ICSp if value != "n/a"]
    response_vCSp = [value for value in response_vCSp if value != "n/a"]
    
    response_mean = [round(sum(response_wCSp)/len(valid_wCSp), 2),
                     round(sum(response_ICSp)/len(valid_ICSp), 2), 
                     round(sum(response_vCSp)/len(valid_vCSp), 2),
                     "n/a"]
    # Add data to the table with separate rows for each trial type
    cell_style = f'style="{cell_styles}"'
    table_html += f'<tr><td {cell_style} rowspan="4">{log_path}</td>'
    for i in range(4):  
            table_html += f'<td {cell_style}>{trial_types[i]}</td><td {cell_style}>{num_stimuli[i]}</td><td {cell_style}>{num_valid[i]}</td><td {cell_style}>{correctness[i]}</td><td {cell_style}>{response_mean[i]}</td></tr>'
    table_html += '</table>'
    return table_html


def generate_section_html(subject, session, task, bids_df_final, log_path):
    """ Create html section with image and table. Save it to the plots directory
    Parameters
    ----------
    subject : string
        Subject number
    session : string
        Session number
    task : string
        Task name
    resulting_dataframe : pandas.core.frame.DataFrame
        Dataframe containing cleaned and aligned puls, resp and trigger values.

    Returns
    -------
    None.
    """
    # link for the section
    anchor_link = f"{subject.replace(' ', '_')}_{session.replace(' ', '_')}_{task}"
    # output dir for saving html file
    output_dir = plots_dir + "group/"

    # Create individual HTML files for each figure and table
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>subject: {subject} - session: {session} - task: {task}</title>
    </head>
    <body>
        <h1>subject: {subject} - session: {session} - task: {task}</h1>
    """

    # HTML content for the figure
    image_filename = output_file_name + "events.html"
    image_path = os.path.join(plots_dir, "sub-" + subject, image_filename)

    if not os.path.exists(image_path):
        print(f"Image not found for: {subject} - {session} - {task}")
        return

    html_content += f'<iframe src="{image_path}" width="1200" height="700" frameborder="0" loading="lazy"></iframe>'

    # HTML content for the table
    if task == "faces":
        table_html = generate_table_html_face(bids_df_final, log_path)
    elif task == "selfref":
        table_html = generate_table_html_selfref(bids_df_final, log_path)
    elif task == "reversal":
        table_html = generate_table_html_reversal(bids_df_final, log_path)
    elif task == "nback":
        table_html = generate_table_html_nback(bids_df_final, log_path)
    elif task == "reward":
        table_html = generate_table_html_reward(bids_df_final, log_path)
    html_content += table_html

    # Close the individual HTML file
    html_content += """
    </body>
    </html>
    """

    # Define the output HTML file path
    output_file_path = os.path.join(
        output_dir, image_filename)

    with open(output_file_path, "w") as html_file:
        html_file.write(html_content)


def delete_function(log_path, table):
    '''
    Removes unneccesary log file path from the table 

    Parameters
    ----------
    log_path : string 
        Log file path.
    table : pandas.core.frame.DataFrame
        Table contaaining all paths to physio and event files.

    Returns
    -------
    table : pandas.core.frame.DataFrame
        Table contaaining all paths to physio and event files.

    '''
    for paths in table["event"]:
        if log_path in paths:
           paths.remove(log_path)
    return table


def delete_files_rawdata(folder_path, log_path):
    """Delete unnecessary event files from the rawdata
    Parameters
    ----------
    folder_path: string
    Path to the folder which contains the unnecessary files (rawdata/sub/ses/func)
    log_path : string
    Log file path.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(folder_path, file)
            if file_path == log_path:
                os.remove(file_path)

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title> Figures </title>
</head>
<body>
    <h1>Table of Contents</h1>
    <ul>
"""

# the table of contents with links to section headers
for subject in subjects_list:
    for session in sessions_list:
        html_content += f'<li><strong>Subject: {subject} - session: {session}</strong></li>\n<ul>\n'
        for task in tasks:
            bids_df_new, output_file_name, log_path = clean_read_data(subject, session, task, table)
            if output_file_name is not None:
                task_link = output_file_name + "events"
                html_file_path = os.path.join(
                    plots_dir + "group/", f"{task_link}.html")
            

                if os.path.exists(html_file_path):
                    html_content += f'<li><a href="{task_link}.html">{task}</a></li>\n'
                else:
                    print(f"HTML file not found: {html_file_path}")
        html_content += "</ul>\n"

# Close the HTML tags for the table of contents
html_content += """
    </ul>
"""

# Close the HTML document
html_content += """
</body>
</html>
"""
# Save the HTML content to a file
with open(contents_log_dir, "w") as html_file:
    html_file.write(html_content)


contents_log_dir = "/data/Stepan_Tikhomirov/SuperDop/derivatives/MNI-physio/plots/group/log_figures.html"
plots_dir = '/data/Stepan_Tikhomirov/SuperDop/derivatives/MNI-physio/plots/'
bids_df_reward_new, output_file_name, log_path = clean_read_data("ZI004", "01", "faces", table)
generate_section_html("ZI004", "01", "faces", bids_df_reward_new, log_path)



for subject in subjects_list:
    for session in sessions_list:
        for task in tasks:
            bids_df, output_file_name, log_path = clean_read_data(subject, session, task, table, plotFlag=True)
            if bids_df is not None:
                try:
                    generate_section_html(subject, session, task, bids_df, log_path)
                except FileNotFoundError:
                    # Handle missing file gracefully by skipping and printing a message
                    print(f"File not found: sub-{subject}_ses-{session}_task-{task}_run-01_physio.tsv")
                    continue
