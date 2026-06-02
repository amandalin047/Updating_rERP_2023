import mne
from mne.preprocessing import ICA
import numpy as np
import scipy
from collections import OrderedDict

# NOTE: filter frequencies are hard-coded in this function, not passed as arguments!!
RANDOM_STATE = 42 # Fix this throughout the script -- always!!

def delete_time_segments(raw, *,
                         time_threshold: float=3.0,
                         after_event_code_buffer: float=1.5,
                         before_event_code_buffer: float=0.5,
                         verbose: bool=False):
    annot = raw.annotations
    diffs = np.diff(annot.onset)
    ends = [int(i) for i in np.where(diffs > time_threshold)[0]]
    starts = [int(j)+1 for j in np.where(diffs > time_threshold)[0]]
    
    afterEventcodeBuffers = [OrderedDict({"onset": annot.onset[i]+after_event_code_buffer,
                                        "duration": 0.0,
                                        "description": "bad (afterEventcodeBuffer)"})
                                        for i in ends]
    beforeEventcodeBuffers = [OrderedDict({"onset": annot.onset[i]-before_event_code_buffer,
                                          "duration": 0.0,
                                          "description": "(beforeEventcodeBuffer)"})
                                        for i in starts]
    concat_annot_list = list(annot) + afterEventcodeBuffers + beforeEventcodeBuffers
    concat_annot_list.sort(key=lambda x: x["onset"])
    concat_annot = mne.Annotations(onset=[item["onset"] for item in concat_annot_list],
                                duration=np.diff(np.array([item["onset"]
                                                           for item in concat_annot_list] + [raw.times[-1]+1])),  # duration is the time until the next annotation, except for the last one which is until the end of the recording
                                description=[item["description"] for item in concat_annot_list])
    raw.set_annotations(concat_annot)
    
    data_del_eeg_only = raw.get_data(picks="eeg", reject_by_annotation="omit", verbose=verbose)
    return raw, data_del_eeg_only


def custom_ref(eeg_data, eeg_ch_names: list, ref_channels: dict):
    ch_to_idx = {ch: idx for idx, ch in enumerate(eeg_ch_names)}

    vals = - np.array(list(ref_channels.values())) 
    cols_to_subtract = [ch_to_idx[ch] for ch in ref_channels]
    
    n_eeg_channels, _ = eeg_data.shape
    A = np.eye(n_eeg_channels)
    A[:, cols_to_subtract] += np.tile(vals, (n_eeg_channels, 1))
    
    # use picks in raw.apply_function(... picks="eeg"), no need to manually delete eog rows
    #data = raw.get_data()
    #n_channels, _ = data.shape
    #eog_rows = np.where(np.array(raw.get_channel_types()) == "eog")[0]
    #rows = np.arange(n_channels)
    #rows = np.delete(rows, eog_rows)
    #A = np.eye(n_channels)
    #A[np.ix_(rows, cols_to_subtract)] += np.tile(vals, (n_channels - len(rows), 1))
    
    reref_eeg_data = A @ eeg_data
    return reref_eeg_data


def preprocess_for_ica(raw, *,
                       standard_montage: str | None=None,
                       eog_channels: list | None=None,
                       ref_channels: list | dict | None=None,
                       return_raw_reref_only: bool=False,
                       time_threshold: float=3.0,
                       after_event_code_buffer: float=1.5,
                       before_event_code_buffer: float=0.5,
                       rank_tol: float=1e-4,
                       verbose: bool=False):
    """ref_channels: if list, subtracts the mean of the channels in the list, e.g.,
                              ref_channels = ['M1', 'M2'] will subtract (M1 + M2) / 2 from all EEG channels
                     if dict, specify what fraction of the channel to subtract, e.g.,
                              ref_channels = {'TP10': 0.5} will subtract 0.5 * TP10 from all EEG channels"""
    # set built-in montage 
    raw = raw.copy() # important: do NOT mutate the input raw object; make a copy
    ch_names = raw.ch_names
    
    if eog_channels is None:
        eog_channels = [ch for ch in raw.ch_names if (ch.upper() in {"VEO", "HEO", "VEOG", "HEOG"})
                                                  or ("EOG" in ch.upper())]
        if verbose:
            print(f"eog_channels not provided. setting channel types based on channel names ('eog' if 'EO' in channel_name else 'eeg')")
    raw.set_channel_types({ch: "eog" if ch in eog_channels else "eeg" for ch in ch_names})
    
    m_orig = raw.get_montage()
    pos = np.stack(list(m_orig.get_positions()["ch_pos"].values()), axis=0)
    if np.any(np.any(np.isnan(pos), axis=1)):
        if verbose:
            print(f"Input raw file has missing channel positions; setting channel montage.")
        if standard_montage is None:
            print(f"standard_montage is not provided; defaulting to MNE-shipped standard_1020 and renaming all channel names to upper case.")
            standard_montage = "standard_1020"
        m = mne.channels.make_standard_montage(standard_montage)
        rename_ch = {ch: ch.upper() for ch in m.ch_names}
        m = m.rename_channels(rename_ch, on_missing="raise", verbose=verbose)
        raw.set_montage(m, on_missing="raise")

    raw = raw.set_annotations(raw.annotations.copy())
    if verbose:  
        boundaries = [item for item in raw.annotations if item["description"] == "boundary"]
        print(f"Found {len(boundaries)} boundary event(s) in the raw dataset:")
        print(boundaries)

    # set rerefernce
    if ref_channels is None:
        ref_channels = []
        print(f"\nWARNING: ref_channels not provided. Defaulting to an empty list. MNE will not attempt any re-referencing of the data!\n")
    elif isinstance(ref_channels, list):
        raw = raw.copy().set_eeg_reference(ref_channels=ref_channels, ch_type="eeg")

        #raw, ref_data = mne.set_eeg_reference(raw, ref_channels=ref_channels, ch_type="eeg", copy=False)   # modifies in-place
                                                                                         # mne.set_eeg_reference returns a tuple (re-referenced instance, ref_data)
                                                                                         # where ref_data is the data that was subtracted, which can be useful for debugging
                                                                                         # otherwise, raw.set_eeg_reference can also be used                                                                                                                                                
        # e.g.,
        # print(ref_data.shape) <- for debugging
        # or compare with manual calulation... etc... <- for debugging
    
    elif isinstance(ref_channels, dict):
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[], include=[])
        eeg_ch_names = [raw.ch_names[p] for p in eeg_picks]
        raw.apply_function(custom_ref, picks=eeg_picks, channel_wise=False,
                                       eeg_ch_names=eeg_ch_names, ref_channels=ref_channels)
    if return_raw_reref_only:
        return raw
    raw_reref = raw.copy()
    
    # detrend data (remove dc offset)
    data = raw.get_data()
    data_detrended = scipy.signal.detrend(data, axis=-1, type="constant")
    raw = mne.io.RawArray(data=data_detrended, info=raw_reref.info.copy()) # pass a copy of info to prevent potential weird shared-state MNE behavior (per ChatGPT's suggestion)
    raw = raw.set_annotations(raw_reref.annotations.copy())                # same with annotations

    # strong bandpass filter (highpass especially important for ica)
    bandpass_params_for_ica = mne.filter.create_filter(
        data=raw.get_data(),
        sfreq=raw.info["sfreq"],
        l_freq=1.0, h_freq=30.0,  
        method="iir",
        iir_params={"order": 8, "ftype": "butter"},
        verbose=True
    )
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=30.0, method="iir",
                                    iir_params=bandpass_params_for_ica, verbose=verbose)
    
    raw_for_ica_copy = raw_for_ica.copy()
    raw_for_ica, data_del_eeg_only = delete_time_segments(raw_for_ica_copy,
                                                 time_threshold=time_threshold,
                                                 after_event_code_buffer=after_event_code_buffer,
                                                 before_event_code_buffer=before_event_code_buffer,
                                                 verbose=verbose)
    if verbose:
        print("Printing the first 20 rows of annotations with idle time segments marked as bad...")
        print(raw_for_ica.annotations.to_data_frame().head(20))

    n_eeg_channels, n_times = data_del_eeg_only.shape
    rank = np.linalg.matrix_rank(data_del_eeg_only, tol=rank_tol)
    assert n_eeg_channels == (len(raw_for_ica.ch_names) - len(eog_channels))

    if verbose:
        print(f"\nWARNING: Data matrix shape: (n_eeg_channels={n_eeg_channels}, n_times={n_times}); rank = {rank}!\n")
        if rank < n_eeg_channels:
            print("Data matrix is rank-deficient! Set ICA n_components <= rank!")
    
    # do not drop eog channels here (might need eog channels for find_bads_eog)
    # use ica.fit(....picks="eeg")
    return raw_reref, raw_for_ica, rank


def detrend_and_iir_bandpass(raw_to_filter, *,
                             l_freq: float=0.1, h_freq: float=30.0,
                             order: int=2,
                             ftype: str="butter",
                             verbose: bool=False):
    # detrend data (remove dc offset)
    raw_to_filter_copy = raw_to_filter.copy()
    data = raw_to_filter_copy.get_data()
    data_detrended = scipy.signal.detrend(data, axis=-1, type="constant")

    raw_filtered = mne.io.RawArray(data=data_detrended, info=raw_to_filter_copy.info.copy()) 
    raw_filtered = raw_filtered.set_annotations(raw_to_filter_copy.annotations.copy())
    
    iir_params=dict(order=order, ftype=ftype)
    raw_filtered = raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method="iir",
                                       iir_params=iir_params, verbose=verbose)
    return raw_filtered


# pass raw_reref as raw_to_clean (the raw object the ica solution is applied on must have the same sensor space as that which the ica algorithm was fitted on)
def perform_ica(*, raw_to_clean, raw_for_ica,
                n_components: int,
                noise_cov: mne.Covariance | None=None,
                method: str="infomax",
                fit_params: dict | None=None,
                max_iter: int | str="auto",
                manual_inspection: bool=False,
                corr_threshold: float=0.85,
                eog_like_channels: list | None=None, 
                l_freq: float=0.1, h_freq: float=30.0,
                order: int=2, ftype: str="butter",
                verbose: bool=False):
    """eog_like_channels is mandatory if the dataset contains no EOG channels;
    if data contains both HEOG (horizontal) and VEOG (vertical), consider only passing VEOG"""
    
    raw_clean = raw_to_clean.copy()
    ica = ICA(n_components=n_components,
              noise_cov=noise_cov,
              method=method,
              fit_params=fit_params,
              max_iter=max_iter,
              random_state=RANDOM_STATE,
              verbose=verbose)
    ica.fit(raw_for_ica,
            picks="eeg",                 # IMPORTANT: fit on eeg channels only
            reject_by_annotation=True,
            verbose=verbose)             # IMPORTANT: omit 'bad segments' (i.e., the segments to delete) from the data before fitting
    assert ica.current_fit == "raw"

    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw_for_ica.copy(),
                                                ch_name=eog_like_channels,
                                                threshold=corr_threshold,
                                                measure="correlation",
                                                l_freq=None, h_freq=None, # raw_for_ica already filtered; don't filter it here again
                                                reject_by_annotation=True,
                                                verbose=verbose)
    # eog_scores can be used for debugging; leaving it here
    if len(eog_indices) > 0 and not manual_inspection:
        ica.exclude = eog_indices
        if verbose:
            print(f"\n >> Most likely blink IC(s): {[int(i) for i in eog_indices]}")
            print(f"Excluding the following ICs: {eog_indices}")
        ica.apply(raw_clean)
    elif not manual_inspection:
        print(f"No blink components identified based on corr_threshold = {corr_threshold}. ICA not applied.")

    raw_clean_filt = detrend_and_iir_bandpass(raw_clean,
                                              l_freq=l_freq, h_freq=h_freq,
                                              order=order, ftype=ftype,
                                              verbose=verbose)
    if manual_inspection:
        if verbose:
            print(f"Blink-like component indices identified via correlation (might be empty if no component crosses corr_threshold = {corr_threshold}): {eog_indices}.")
            print(f"Manaul inspection mode: ICA not applied. Estimating sources (from 0.1 - 30 Hz IIR order 2 bandpass filtered raw data) given the ICA unmixing matrix. Please also check ica.plot_properties() for better judgement.")
        sources = ica.get_sources(raw_clean_filt)
        return sources, eog_indices, ica
    
    return raw_clean_filt, eog_indices, ica


if __name__ == "__main__":
    DATA_DIR = "/Users/jowanglin/regression-based_ERP/data/eeg/crystal"

    NOISE_COV = None                      # Noise covariance used for pre-whitening. If None (default), channels are scaled to unit variance (“z-standardized”) as a group by channel type prior to the whitening by PCA.

    METHOD = "infomax"                    # MNE accepts 'fastica' | 'infomax' | 'picard'; using "picard" to match EEGLAB default -> need to pip install python-picard
    FIT_PARAMS={"extended": True,         # EEGLAB default is infomax extended
                "weights": None,          # The initialized unmixing matrix. Defaults to None, which means the identity matrix is used.
                "l_rate": None,           # This quantity indicates the relative size of the change in weights. Defaults to 0.01 / log(n_features ** 2).
                "block": None,            # The block size of randomly chosen data segments. Defaults to floor(sqrt(n_times / 3.)).           
                "w_change": 1e-12,        # The change at which to stop iteration. Defaults to 1e-12.
                "anneal_deg": 60.0,       # The angle (in degrees) at which the learning rate will be reduced. Defaults to 60.0.
                "anneal_step": 0.9,       # The factor by which the learning rate will be reduced. Defaults to 0.9.
                "n_subgauss": 1,          # The number of subgaussian components. Only considered for extended Infomax. Defaults to 1.
                "kurt_size": 6000,        # The window size for kurtosis estimation. Only considered for extended Infomax. Defaults to 6000.
                "blowup": 10000}          # The maximum difference allowed between two successive estimations of the unmixing matrix. Defaults to 10000.
    # for picard
    #FIT_PARAMS={"tol": 1e-7,
            #"ortho": False, # If True, uses Picard-O. Otherwise, uses the standard Picard.
            #"fastica_it": None} # If an int, perform fastica_it iterations of FastICA before running Picard. It might help starting from a better point.
    MAX_ITER=1000 
    #allow_ref_meg -> irrelevant
    CORR_THRESHOLD = 0.85
    VERBOSE=True 

    num = 1
    raw_file_name = f"subj{str(num).zfill(3)}_reref_filt.set"
    raw = mne.io.read_raw_eeglab(f"{DATA_DIR}/{raw_file_name}", preload=True, verbose=False)

    raw_reref, raw_for_ica, rank = preprocess_for_ica(raw,
                                                  standard_montage="standard_1020",
                                                  eog_channels=["HEO", "VEO"],
                                                  ref_channels=["M1", "M2"],
                                                  time_threshold=3.0,
                                                  after_event_code_buffer=1.5,
                                                  before_event_code_buffer=0.5,
                                                  rank_tol=1e-4,
                                                  verbose=True)
    print("\n==========================================  ICA  ==========================================\n")
    raw_clean, _ = perform_ica(raw_to_clean=raw_reref,
                           raw_for_ica=raw_for_ica,
                           n_components=rank,
                           noise_cov=NOISE_COV,
                           method=METHOD,
                           fit_params=FIT_PARAMS,
                           max_iter=MAX_ITER,
                           manual_inspection=False,
                           corr_threshold=CORR_THRESHOLD,
                           eog_like_channels=["VEO"], 
                           verbose=True)
