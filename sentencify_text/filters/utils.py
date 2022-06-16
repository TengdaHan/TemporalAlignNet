import random
import numpy as np
from langdetect import DetectorFactory, detect_langs
DetectorFactory.seed = 0


def filter_language(cap_list):
    """randomly select 5 (or all) sentences, check avg language score"""
    assert isinstance(cap_list, list)
    assert all(isinstance(cap, str) for cap in cap_list)
    try:
        cap_subset = random.sample(cap_list, 5)
    except:
        cap_subset = cap_list
    cap_subset = [str(i) for i in cap_subset]
    cap_subset = [cap for cap in cap_subset if len(cap.split()) >= 4]
    probs = []
    for cap in cap_subset:
        try:
            lang = detect_langs(cap)
        except:
            # possible reason: all numbers, symbols
            continue
        lang_dict = {i.lang:i.prob for i in lang}
        try:
            probs.append(lang_dict['en'])
        except:
            probs.append(0)
    if len(probs) == 0:
        avg_prob = 0.0
    else:
        avg_prob = np.mean(probs)
    return avg_prob > 0.9


def filter_length(cap_list):
    """if words per caption is too short, or too few captions"""
    cap_list = [str(i) for i in cap_list]
    num_word = [len(i.split(' ')) for i in cap_list]
    num_cap = len(cap_list)
    return (num_cap > 10) and (np.mean(num_word) > 5)


def merge_linebreaks(cap_list, start_list, end_list):
    assert len(cap_list) == len(start_list) == len(end_list)

    # remove caption glitch (time < 0.2 seconds)
    duration = np.array(end_list) - np.array(start_list)
    passed = duration > 0.2
    cap_list = np.array(cap_list)[passed].tolist()
    start_list = np.array(start_list)[passed].tolist()
    end_list = np.array(end_list)[passed].tolist()
    num_caps = len(cap_list)

    # check caption length and linebreak
    caps_tmp, starts_tmp, ends_tmp = [], [], []
    for idx in range(num_caps):
        cap, start, end = cap_list[idx], start_list[idx], end_list[idx]
        cap = str(cap).strip()
        if cap == '':  # empty string
            continue
        if '[' in cap and ']' in cap:  # e.g. [MUSIC]
            continue
        if '\n' in cap: # if the second row is repeated next, remove the second row
            if (idx + 1 < num_caps) \
                and cap_list[idx+1].strip().split('\n')[0].strip() \
                    == cap.split('\n')[-1].strip():
                new_cap = ' '.join(cap.split('\n')[0:-1])
            else:
                new_cap = cap.replace('\n', ' ')
        else:
            new_cap = cap
        caps_tmp.append(new_cap)
        starts_tmp.append(start)
        ends_tmp.append(end)

    # a second-round dedup is necessary: some text repeats 3 times
    duplicate_flag = []
    for cap, cap_ in zip(caps_tmp[:-1], caps_tmp[1:]):
        if len(cap_) >= len(cap) and cap_.startswith(cap):
            duplicate_flag.append(1.0)
        else:
            duplicate_flag.append(0.0)

    if sum(duplicate_flag) > 0:
        caps_tmp_, starts_tmp_, ends_tmp_ = [], [], []
        num_caps_tmp = len(caps_tmp)

        for idx in range(num_caps_tmp - 1):
            cap, start, end = caps_tmp[idx], starts_tmp[idx], ends_tmp[idx]
            cap = str(cap).strip()

            if duplicate_flag[idx] == 1:
                if idx > 0 and duplicate_flag[idx - 1] == 1:
                    continue
                else:
                    starts_tmp_.append(start)
            elif duplicate_flag[idx] == 0:
                if idx > 0 and duplicate_flag[idx - 1] == 1:
                    ends_tmp_.append(end)
                    caps_tmp_.append(cap)
                else:
                    starts_tmp_.append(start)
                    ends_tmp_.append(end)
                    caps_tmp_.append(cap)

        if duplicate_flag[-1] == 0:
            starts_tmp_.append(starts_tmp[-1])
        ends_tmp_.append(ends_tmp[-1])
        caps_tmp_.append(caps_tmp[-1])

        assert len(caps_tmp_) == len(starts_tmp_) == len(ends_tmp_)

        starts_tmp = starts_tmp_
        ends_tmp = ends_tmp_
        caps_tmp = caps_tmp_

    # if time has overlap, take avg
    is_overlap = np.array(starts_tmp[1::]) - np.array(ends_tmp[0:-1]) < 0
    if is_overlap.sum() > 0:
        avg_timestamp = np.array([starts_tmp[1::], ends_tmp[0:-1]]).mean(0)
        starts_tmp = np.array(starts_tmp)
        starts_tmp[1::][is_overlap] = avg_timestamp[is_overlap]
        ends_tmp = np.array(ends_tmp)
        ends_tmp[0:-1][is_overlap] = avg_timestamp[is_overlap]

        # assert no overlap
        assert (np.array(starts_tmp[1::]) - np.array(ends_tmp[0:-1]) < 0).sum() == 0
        starts_tmp = starts_tmp.tolist()
        ends_tmp = ends_tmp.tolist()

    return caps_tmp, starts_tmp, ends_tmp