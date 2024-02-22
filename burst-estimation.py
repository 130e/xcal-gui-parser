# %%
# import sys
# sys.path.append("localpkg")
import utilities as ut

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime

# %%
fname = "/mnt/c/Users/simme/Documents/Tools/xcal-gui-parser/pdsch-1.csv"
df = pd.read_csv(fname, index_col=0)
print(
    "Nu",
    df["Numerology"].unique(),
    "Layer",
    pd.unique(df["Num Rx"]),
    "CA",
    np.max(df["Num PDSCH Status"]),
)
print("RB in one CA:")
tmp = sorted(pd.unique(df["Num Rbs"]))
print(np.min(tmp), np.max(tmp))
# tmp = [x for x in df["Num PDSCH Status"] if not pd.isna(x)]
# ut.plot_ecdf(tmp)

"""
umber of slot per frame(10ms)
- 15 kHz -> 10
- 30kHz -> 20
- 120kHz -> 80
"""
# config value
df.columns.values

# calculate PDSCH
"""
Only consider one type of CA
Sum alloc RB in one slot.
collect timestamp.
"""


def count_pdsch_rb(df, num_slot):
    sfn_cycle = 0
    prev_sfn = 0
    SFN_CYCLE_PIVOT = 512
    SFN_LEN = 10  # ms

    slot_len = SFN_LEN / num_slot  # ms

    cur_ts = 0
    carrier_n = 0
    rb_cnt = 0
    tb_cnt = 0
    prev_slot = 0
    prev_sfn = 0

    timestamps = []
    total_rb = []
    total_tb = []

    # sfn ranges 0->1023; slot ranges 0->slot_num

    for i, r in df.iterrows():
        slot = r["Slot"]
        rb_cnt += r["Num Rbs"]
        tb_cnt += r["TB Size"]
        # read new slot
        if carrier_n == 0:
            # should be int if not malformed
            carrier_n = r["Num PDSCH Status"]
            # update ts
            sfn = r["Frame"]
            if prev_sfn - sfn > SFN_CYCLE_PIVOT:
                # new sfn_cycle
                sfn_cycle += 1
            cur_ts = slot * slot_len + (sfn_cycle * 1024 + sfn) * SFN_LEN
            # sfn_leap = sfn - prev_sfn
            # cur_ts += (slot - prev_slot + sfn_leap * num_slot) * slot_len
            # prev_slot = slot
            prev_sfn = sfn
        # if last CA in slot
        if carrier_n <= 1:
            timestamps.append(cur_ts)
            total_rb.append(rb_cnt)
            rb_cnt = 0
            total_tb.append(tb_cnt)
            tb_cnt = 0

        carrier_n -= 1
        # cum += r["RB Alloc"]
        # if pd.isna(r["Num Layer"]):
        #     layer = r["Num_Layer"]
        # else:
        #     layer = r["Num Layer"]
    # offset
    st_v = timestamps[0]
    timestamps = [x - st_v for x in timestamps]
    # last line
    return timestamps, total_rb, total_tb


# x,y = rb_alloc(df.iloc[200:], 20)
slot_per_frame = 80
ts_all, rbs_all, tbs_all = count_pdsch_rb(df, slot_per_frame)

print("RB in a slot (min max):")
tmp = sorted(pd.unique(rbs_all))
print(np.min(tmp), np.max(tmp))

plt.plot(ts_all, rbs_all)
plt.show()

# %%

# %%
"""Find a period of time where num of CA is mostly stable"""


def get_range(x, st, end):
    stidx, endidx = 0, 0
    for i, e in enumerate(x):
        if stidx == 0:
            if e >= st:
                stidx = i
        else:
            if e >= end:
                endidx = i
                break
    return stidx, endidx


st, end = get_range(ts_all, 0, 3000)
ts = ts_all[st:end]
rbs = rbs_all[st:end]
tbs = tbs_all[st:end]

# rbs_t = []
# for rb in rbs:
#     if rb > 66:
#         rbs_t.append(rb%66)
#     else:
#         rbs_t.append(rb)

plt.plot(ts, rbs)
plt.show()
plt.plot(ts, tbs)
plt.show()

# %%
"""
Try out different threshold, 
which is the best for spliting full burst
The threshold is SISO threshold
"""


# ordered_y = sorted(rbs)
# N = len(ordered_y)
# max_rb = max(rbs)
# print(max_rb, min(rbs), "avg", np.average(rbs))
# low = ordered_y[round(N*0.05)]
# high = ordered_y[round(N*0.8)]
# MAX_RB = 66
def plot_ecdf(a, title="", xlabel="X", ylabel="CDF(%)"):
    x, y = ut.ecdf(a)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.0) * 100
    plt.plot(x, y, drawstyle="steps-post")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.title(title)


MAX_RB = np.max(rbs)
print("MAX RB", MAX_RB)

low = MAX_RB * 0.2
high = MAX_RB * 0.3
print(low, high)

plot_ecdf(rbs, "CDF. RB allocated in slot", "RB")
plt.axvline(x=low, color="tab:red")
plt.axvline(x=high, color="tab:green")

# rbtmp = [min(x, MAX_RB) for x in rbs]
# ut.plot_ecdf(rbtmp)

# %%
"""
Calculate burst period and value, using only rbs
Comeback and calculate TB
"""
burst_duration = []
gap_duration = []
burst_start_ts = 0
burst_end_ts = 0

burst_rbs = []
burst_tbs = []
noburst_rbs = []
noburst_tbs = []

slot_per_frame = 20
slot_len = 10 / slot_per_frame

buffer_sizes = []
tbs_cum = 0

burst_state = False

x = ts
for i in range(len(x) - 1):
    rb = rbs[i]
    tb = tbs[i]
    if burst_state:
        tbs_cum += tb
        if rb <= low:
            if x[i + 1] - x[i] >= slot_len:
                # burst end
                burst_state = False
                burst_end_ts = x[i]
                burst_duration.append(burst_end_ts - burst_start_ts + slot_len)
                buffer_sizes.append(tbs_cum)
                tbs_cum = 0
        burst_rbs.append(rb)
        burst_tbs.append(tb)
    else:
        if rb >= high:
            burst_state = True
            burst_rbs.append(rb)
            burst_tbs.append(tb)
            burst_start_ts = x[i]
            tbs_cum = tb
            gap_duration.append(burst_start_ts - burst_end_ts + slot_len)
        else:
            noburst_rbs.append(rb)
            noburst_tbs.append(tb)

# %%
"""Burst stats calculation"""
print(
    "Nu",
    df["Numerology"].unique(),
    "Layer",
    pd.unique(df["Num Rx"]),
    "CA",
    np.max(df["Num PDSCH Status"]),
)
print("low", low, "high", high)
gap_duration = [x for x in gap_duration if x < 1000]
print(
    "Average duration (ms)\n burst:",
    np.mean(burst_duration),
    "\n no burst:",
    np.mean(gap_duration),
)
print()

total_rbs = sum(burst_rbs) + sum(noburst_rbs)
print("Percentage of RB that are allocated in burst: \n", sum(burst_rbs) / total_rbs)
print()

print(
    "Percentage of allocated slots that are scheduled in burst: \n",
    len(burst_rbs) / (len(burst_rbs) + len(noburst_rbs)),
)
print()

total_tbs = sum(burst_tbs) + sum(noburst_tbs)
print("Percentage of TBS that are allocated in burst: \n", sum(burst_tbs) / total_tbs)
print()

bs = sorted(buffer_sizes)[10:-10]
print("Buffer size:", np.mean(bs), np.max(bs), np.min(bs))
ut.plot_ecdf(bs)

# %%
ut.plot_ecdf(buffer_sizes)

# %%
print("RB Alloc during burst VS not burst")
ut.draw_multi_ecdf({"burst": burst_rbs, "noburst": noburst_rbs}, xlabel="rbs")

# %%
print("Burst duration VS gap duration")
ut.draw_multi_ecdf({"burst": burst_duration, "gap": gap_duration}, xlabel="ms")

# %%
ut.plot_ecdf(gap_duration)
