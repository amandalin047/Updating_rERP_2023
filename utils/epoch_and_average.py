import numpy as np
import pandas as pd
import mne
from collections import OrderedDict
from collections.abc import Iterable

def revise_annot(df_annot: pd.DataFrame, *,
                 fixation: str, non_final: str,
                 codes_after_fixation: list|tuple|set|None=None,
                 codes_after_fixation_are_final: bool=False,
                 **conditions) -> mne.Annotations:   
    """IMPORTANT!!!
       df_annot: please get the data frame via pd.DataFrame(raw.annotations);
                 do NOT use raw.annotations.to_data_frame() -- I don't know why this causes downstream annotations setting to be empty, i.e.,
                 if annot_revised is the returned revised mne.Annotations object, then doing
                 raw.set_annotations(annot_revised) will lead to empty annoations.
       conditions should be: event_description=event_codes, where event_codes is an iterable, e.g.,
       high_constraint=(240, 241, 242, 243), low_constraint=(244, 245, 246, 247)"""
    
    if "orig_time" not in df_annot.columns or "extras" not in df_annot.columns:
        print(f"""ABORT: either 'orig_time' or 'extras' was not found in df_annot.columns, which is consistent with if the
        annotations data frame was converted via mne.Annotations.to_data_frame(), which will lead to downstream errors.
        Please ensure that df_annot is converted directly via Pandas DataFrame constructor, i.e.,
        df_annot = Pandas.DataFrame(mne.Annotations).""")
        return
    
    cond_dict = OrderedDict({key: [str(v) for v in val]
                                          for key, val in conditions.items()}) # in case integer event codes are passed
                                                                               # and ensures key (condition description) order
    cond_descs = list(cond_dict.keys())
    all_cond_codes = np.array(list(cond_dict.values()))
    
    desc = list(df_annot["description"])
    desc = [d.strip() for d in desc]  # remove leading or trailing white spaces to be safe
    desc_revised = desc.copy()

    if codes_after_fixation is None:
        codes_after_fixation = set([desc[i] for i in range(len(desc)) if desc[i-1] == fixation])
    #else:
        #assert set(codes_after_fixation) == set([desc[i] for i in range(len(desc)) if desc[i-1] == fixation])

    for i, d in enumerate(desc):
        if d == fixation:
            step = 1
        elif d == non_final:
            desc_revised[i] = "w" + str(step) 
            step += 1
        elif d in codes_after_fixation:
            if desc[i-1] == fixation:
                desc_revised[i] = "w" + str(step) 
                step += 1
            elif desc[i-1] == non_final:
                if not codes_after_fixation_are_final:
                    desc_revised[i] = "w" + str(step) 
            else:
                pass
        elif d in all_cond_codes.flatten():  # use flatten, as flatten always returns a copy; need all_cond_codes to stay 2D
            cond_idx = np.where(all_cond_codes == d)[0][0]
            cond = cond_descs[cond_idx]
            desc_revised[i-step: i] = [s + f"/{cond}" for s in desc_revised[i-step: i]]
            step = 0
    
    annot_revised = mne.Annotations(onset=df_annot["onset"], duration=df_annot["duration"], description=desc_revised)
    return annot_revised

def eeglab_logic_bin_epoch(raw: mne.io.Raw,
                     fixation: str,
                     non_final: str,
                     codes_after_fixation: list|tuple|set|None,
                     codes_after_fixation_are_final: bool,
                     position_range: tuple,
                     conditions_list: list,
                     tmin: float, tmax: float,
                     baseline: tuple | None,
                     reject_peak_to_peak: dict | None,  # rejection (maximum peak-to-peak) is based on a signal difference calculated for each channel separately 
                                                        # applying baseline correction does not affect the rejection procedure as the difference will be preserved.
                                                        # peak-to-peak unit is in volts (V) for eeg  and eog channels
                     resample: float | None=None,
                     preload: bool=True,
                     verbose: bool=False):
    """condtitions_list should be a list of dicts, e.g.,
       [{'high_constraint': (240, 241, 242, 243), 'low_constraint': (244, 245, 246, 247)},
        {'emo_word': (240, 241, 244, 245), 'neu_word': (242, 243, 246, 247)}]"""
    
    def get_time_locks_descs(position_range: tuple,
                             cond_descs: Iterable):
        word_positions = [f"w{i}" for i in range(position_range[0], position_range[1]+1)]
        if cond_descs:
            time_locks_descs = [f"{w}/{c}" for w in word_positions for c in cond_descs]
        else:
            time_locks_descs = word_positions
        return time_locks_descs
    
    df_annot = pd.DataFrame(raw.annotations)
    
    epochs_list = []
    for conditions in conditions_list:
        annot_revised = revise_annot(df_annot.copy(),
                                 fixation=fixation,
                                 non_final=non_final,
                                 codes_after_fixation=codes_after_fixation,
                                 codes_after_fixation_are_final=codes_after_fixation_are_final,
                                 **conditions)
        raw_revised = raw.copy()
        raw_revised.set_annotations(annot_revised)

        events, event_id = mne.events_from_annotations(raw_revised)
        time_locks_descs = get_time_locks_descs(position_range, conditions)
        time_locks = {k: v for k, v in event_id.items() if k in time_locks_descs}

        epochs = mne.Epochs(raw_revised,
                            events=events, event_id=time_locks,
                            tmin=tmin, tmax=tmax, baseline=baseline,  
                            on_missing="raise",
                            reject=reject_peak_to_peak,   # rejct_tmin and reject_tmax both default to None
                                                          # so the entire epoch is considered
                            detrend=None,                 # no detrending (already detrended externally using scipy before filtering)
                            reject_by_annotation=False,   # no 'bads' annotations anyway
                            event_repeated="error",
                            preload=preload,
                            verbose=verbose)
        if resample is not None:
            epochs = epochs.resample(sfreq=resample)
        epochs_list.append(epochs)
    return epochs_list
