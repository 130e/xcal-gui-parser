# jupyter nbconvert --to script Utility.ipynb
import subprocess
import time
import json
import math
import pandas as pd
import numpy as np
import statistics as stat
import csv
import code
from datetime import datetime
import glob, os
import re
import functools
import operator
import io
import datetime
import random

from enum import Enum

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as plticker
from matplotlib.pyplot import cm
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D

sec = 'seconds'

def read_nsa_status_log(fname, m, d, y=2023):
    #test_date = pd.Timestamp(datestr)
    for_pd = io.StringIO()
    with open(fname) as f:
        for line in f:
            new_line = re.sub(r',', '|', line.rstrip(), count=6)
            print (new_line, file=for_pd)
    for_pd.seek(0)
    df = pd.read_csv(for_pd, sep='|', names=['Time', 'Chipset Time', 'UE-NET', 'Channel', 'Tech', 'Message', 'Info'])
    df[sec] = pd.to_datetime(df['Time'])
    # print(df.loc[0, 'Time'].to_pydatetime().timestamp())
    df[sec] = df[sec].map(lambda x: x.replace(month=m, day=d, year=y).tz_localize('America/Los_Angeles').to_pydatetime().timestamp())
    # print(df.loc[0, 'Time'])
    #df['Time'] = df['Time'].map(lambda x: (x.replace(month=m, day=d).to_pydatetime() - datetime.datetime(1970,1,1)).total_seconds())
    return df

# df_nsa = read_nsa_status_log('/mnt/c/Users/simme/Documents/Logs-XCAL5/202302/1', 2, 13)
# Add 1 Day if you have to  

def df_sec_range(df, start, end, reset_ts=False):
    d = df[df[sec]>start]
    d = d[d[sec]<end]
    d = d.reset_index(drop=True)
    if reset_ts:
        d[sec] -= d.loc[0, sec]
    return d

# should we round the value?
def read_iperf_log(fname):
    with open(fname, 'r') as f:
        # sender/recv
        if 'sender' in fname:
            text = ''.join(f.readlines()[:-11])
            data = json.loads(text)
        else:
            data = json.load(f)
        jitter = []
        seconds = []
        thput = []
        start_time = data['start']['timestamp']['timesecs']
        df = None
        # UDP
        if data['start']['test_start']['protocol'] == 'UDP':
            loss = []
            for interval in data['intervals']:
                seconds.append(round(interval['sum']['start'], 1))
                jitter.append(round(interval['sum']['jitter_ms'], 1))
                thput.append(round(interval['sum']['bits_per_second']))
                loss.append(interval['sum']['lost_percent'])
            df = pd.DataFrame({'seconds':seconds, 'throughput':thput, 'jitter':jitter, 'loss_rate':loss})
        # TCP
        elif data['start']['test_start']['protocol'] == 'TCP':
            for interval in data['intervals']:
                seconds.append(round(interval['sum']['start'], 1))
                thput.append(round(interval['sum']['bits_per_second']))
            df = pd.DataFrame({'seconds':seconds, 'throughput':thput})
        return df, start_time

def read_ping_log(fname):
    ts = []
    latency = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        start_time = float(lines[1].split(' ')[0].strip('[]'))
        for line in lines[1:]:
            items = line.split(' ')
            if len(items) != 9:
                break
            ts.append(float(items[0].strip('[]')))
            latency.append(float(items[-2].replace('time=', '')))
        df = pd.DataFrame({'seconds':ts, 'latency':latency})
        return df, start_time
    

# CDF
def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


def plot_ecdf(a, title=None, xlabel=None, ylabel=None):
    x, y = ecdf(a)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle='steps-post')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.title(title)
    plt.show()

"""
pair with ecdf
t = val_diff(df[sec])
t = [x for x in t if x<0.25]
plot_ecdf(t, 'Sampling gap excluding rare values', 'gap/s')
"""
def val_diff(data):
    r = []
    p = 0
    for e in data:
        if e-p<0:
            print(e, p)
            break
        r.append(e - p)
        p = e
    return r[1:]

def stacking_plot(dfs, labels, title='', figsize=(6.4,4.8)):
    xidx = sec
    n = len(labels)
    if n!=len(dfs):
        print('Data & label mismatch')
        return
    fig, ax = plt.subplots(n, sharex=True, figsize=figsize)
    fig.suptitle(title)
    colors = list(mcolors.TABLEAU_COLORS)
    for i,label in enumerate(labels):
        df = dfs[i].dropna(subset=[label])
        c = colors[i]
        ax[i].scatter(df[xidx], df[label], color=c, label=label, s=1)
        ax[i].tick_params(axis='y', labelcolor=c)
        ax[i].set_ylabel(label, color=c)
    ax[-1].set_xlabel(xidx, color=c)
    return ax

def read_iperf_server(fname):
    with open(fname, 'r') as f:
        # text = ''.join(f.readlines()[:-11])
        data = json.load(f)
        seconds, thput, retrans, snd_cwnd, rtt = [], [], [], [], []
        start_time = data['start']['timestamp']['timesecs']
        for interval in data['intervals']:
            sample = interval['streams'][0]
            seconds.append(round(sample['start'], 1))
            thput.append(round(sample['bits_per_second']))
            # retrans.append(sample['retransmits'])
            snd_cwnd.append(sample['snd_cwnd'])
            rtt.append(sample['rtt'])
        df = pd.DataFrame({'seconds':seconds, 'throughput':thput, \
                           'snd_cwnd':snd_cwnd, 'rtt':rtt})
        return df, start_time
    
# smoothing
def smoothing(df, wnd=1):
    res = list(df['throughput'])
    n = len(df)
    if wnd<1:
        return
    for i in range(wnd, n-wnd):
        res[i] = sum(df.iloc[i-wnd:i+wnd+1]['throughput'])/(2*wnd)
    return res

# Estimation related
def to_ms(t):
    return 1000 * round(t, 3)

def tpsum(df, categories):
    df = df.dropna(subset=categories, how='all')[categories+[sec]]
    df = df.fillna(0)
    df['Total']= df[categories].sum(axis=1)
    df = df[df['Total'] != 0]
    return df

def sliding_wnd_max(x, y, interval=15, half_wnd=15):
    # x is sec index, y is data index, assuming x,y are list
    x = [ to_ms(ix) for ix in x]
    maxIdx = len(x)
    result = []
    ts = []
    # every 30ms by default
    t = interval
    lt,rt = 0, t+half_wnd
    lidx,ridx = 0,0
    while t < x[-1]:
        sample = []
        while lidx < maxIdx:
            if x[lidx] > lt:
                break
            lidx += 1
        ridx = lidx
        while ridx < maxIdx:
            if x[ridx] < rt:
                sample.append(y[ridx])
            else:
                break
            ridx += 1
        # max
        if len(sample) > 0:
            result.append(np.max(sample))
            ts.append(t)
        t += interval
        lt = t - half_wnd
        rt = t + half_wnd
    return ts, result

# parsing HO / udp reorder
class rrc:
    def __init__(self) -> None:
        self.cmd = None
        self.keyToUse = None
        self.cellGroupId = None
        self.absoluteFrequencySSB = None
        self.SubcarrierSpacing = None
        self.carrierBandwidth = None
        self.frequencyBandList = None
        self.SplitBearerConfig_PrimaryPath = None
    message = None
    info = None
    ts = None
    rrc_type = None
    direction = None

class handoverInfo:
    def __init__(self) -> None:
        self.configInfo = None # ho cmd
        self.configAckInfo = None # ho ack
        self.hoInfoList = [] # rach messages
    hoType = None
    reorder = None
    reorderDiff = None
    gap = None
    interval = None
    

class reorderEvent:
    def __init__(self, start_ts, start_id, ord_id) -> None:
        self.start_ts = start_ts
        self.start_id = start_id
        self.end_ts = start_ts
        self.end_id = start_id
        self.ord_id = ord_id
        self.reorderGap = ord_id - start_id
        self.reorderNum = 1
    
    def summary(self):
        self.reorderGap = self.ord_id - self.end_id
        self.reorderNum = self.end_id - self.start_id + 1

class LINK:
    LTE=0
    MMW=1

def getHoType(ho):
    if ho.configInfo.frequencyBandList < 260:
        return LINK.LTE
    else:
        return LINK.MMW

class handoverParam:
    def __init__(self, linkId) -> None:
        self.dstLink = linkId
        self.markId = linkId+1
    reorderNum = 0
    reorderGap = 0
    lossPkt = 0
    delayGapms = 0


def parse_rrc(ts, msg, text):
    # ignore useless rrc
    if pd.isna(text):
        return None
    info = rrc()
    info.ts = ts
    info.message = msg
    # parse info
    items = text.split('/')
    info.cmd = items[0]
    if 'measurementReport' in info.message:
        info.rrc_type = 'Measurement'
        info.info = text
    elif 'cellGroupId' in info.cmd:
        info.rrc_type = 'reconfig'
    elif 'Split Bearer Config' in info.cmd:
        info.rrc_type = 'SplitBearConfig'
    elif 'LTE systemInformation' in info.message: # ho to lte
        info.rrc_type = 'HO'
        info.info = text
    # only ho and other message left
    elif 'rrcConnectionReconfiguration' in info.message or 'RACH' in info.message:
        info.rrc_type = 'HO'
        info.info = text
    else:
        info.rrc_type = 'Other'

    text = next((x for x in items if 'SubcarrierSpacing' in x), None)
    if text:
        info.SubcarrierSpacing  = text.split(':')[-1].strip()
    text = next((x for x in items if 'absoluteFrequencySSB' in x), None)
    if text:
        info.absoluteFrequencySSB = int(text.split(':')[-1].strip())
    text = next((x for x in items if 'frequencyBandList' in x), None)
    if text:
        info.frequencyBandList = int(text.split(':')[-1].strip())
    return info


def process_ho_csv(fname, m, d):
    df_nsa = read_nsa_status_log(fname, m, d)
    rrc_events = []
    handovers = []
    cur_ho = None
    for i,r in df_nsa.iterrows():
        info = parse_rrc(r[sec], r['Message'], r['Info'])
        if info:
            rrc_events.append(info)
            if info.rrc_type == 'HO':
                if 'rrcConnectionReconfiguration' in info.message:
                    if 'Request' in info.cmd:
                        cur_ho = handoverInfo()
                        handovers.append(cur_ho)
                        cur_ho.configInfo = info
                        cur_ho.hoType = getHoType(cur_ho)
                    elif 'Success' in info.cmd:
                        cur_ho.configAckInfo = info
                    else:
                        print('Error: Unkown rrc')
                elif 'RACH' in info.message:
                    cur_ho.hoInfoList.append(info)
                elif 'LTE systemInformation' in info.message:
                    # if cur_ho != None:
                    #     if cur_ho.configAckInfo == None:
                    #         print('No prev compelete ho', cur_ho.configInfo.info)
                    if cur_ho == None or cur_ho.configAckInfo == None:
                        continue
                    if info.ts - cur_ho.configAckInfo.ts > 1.0:
                        cur_ho = handoverInfo()
                        handovers.append(cur_ho)
                        cur_ho.configInfo = info
                        cur_ho.hoType = LINK.LTE
                else:
                    print('Error: Unkown rrc')
    return rrc_events, handovers

# threshold is the closeness of reord to ho
def process_reord_oldver(fname, threshold=1):
    df_fld = pd.read_csv(fname, names=['id', 'length', 'sender_ts', 'receiver_ts'])
    df_fld[sec] = df_fld['receiver_ts']
    df_fld[sec] /= 1e9
    reorder_events = []
    ord_id = 0
    reord = reorderEvent(-1, -1, 0)
    for i in range(len(df_fld)):
        cur_id = df_fld.loc[i, 'id']
        if ord_id > cur_id:
            # reordered state
            if reord.start_id > cur_id:
                print('Unexpected:', i)
            reord_ts = df_fld.loc[i, sec]
            if threshold < (reord_ts - reord.start_ts):
                # new event
                reord = reorderEvent(reord_ts, cur_id, ord_id)
                reorder_events.append(reord)
            else:
                reord.end_ts = reord_ts
                reord.end_id = cur_id
        else:
            ord_id = cur_id
    return reorder_events

# new version
def process_reord(fname):
    df_fld = pd.read_csv(fname, names=['id', 'length', 'sender_ts', 'receiver_ts'])
    df_fld[sec] = df_fld['receiver_ts']
    df_fld[sec] /= 1e9
    reorder_events = []
    ord_id = 0
    reord = reorderEvent(-1, -1, 0)
    for i in range(len(df_fld)):
        cur_id = df_fld.loc[i, 'id']
        if ord_id > cur_id:
            # reordered state
            if reord.start_id > cur_id:
                print('Unexpected:', i)
            reord_ts = df_fld.loc[i, sec]
            if cur_id > reord.ord_id:
                # sum up old reorder event
                reord.summary()
                # new event
                reord = reorderEvent(reord_ts, cur_id, ord_id)
                reorder_events.append(reord)
                if reord.reorderNum == None:
                    print('Issue')
            else:
                reord.end_ts = reord_ts
                reord.end_id = cur_id
        else:
            ord_id = cur_id
    return reorder_events

# correlate ho with rrc
def correlate_rrc_reord(handover, reorder, threshold=2):
    for r in reorder:
        tmp = threshold
        corHoIdx = None
        for i,ho in enumerate(handover):
            if np.abs(r.start_ts - ho.configAckInfo.ts) < np.abs(tmp):
                tmp = r.start_ts - ho.configAckInfo.ts
                corHoIdx = i
        if np.abs(tmp) < threshold:
            if handover[corHoIdx].reorder != None:
                # compare which reord is closest
                if handover[corHoIdx].reorderDiff < 0 and tmp > 0:
                    pass
                elif np.abs(handover[corHoIdx].reorderDiff) < np.abs(tmp):
                    continue
            handover[corHoIdx].reorder = r
            handover[corHoIdx].reorderDiff = tmp

# write input trace
INQ, OUTQ = 1,0
RF_MARK, MMW_MARK = 1,2

def ms(x):
    return x * 1000

class Event:
    INIT=0
    CMD=1
    HO=2
    HANDLEDUP=3
    HANDLERWND=4
    INIT_SCHED=5
    HANDLESOL=6

def write_trace_line(f, text):
    for i in range(len(text)-1):
        f.write(str(text[i]) + ',')
    f.write(str(text[-1]) + '\n')

def set_dedup(f, t, duration):
    # time event qid enable
    text = [t, Event.HANDLEDUP, INQ, 1]
    write_trace_line(f, text)
    text = [t+duration, Event.HANDLEDUP, INQ, 0]
    write_trace_line(f, text)

def trace_init(f, start_s, duration_s, sv_fname):
    nfq_end_ms = (start_s + duration_s) * 1000 + 3000
    text = [0, Event.INIT, OUTQ, RF_MARK, nfq_end_ms]
    write_trace_line(f, text)
    text = [0, Event.INIT, INQ, RF_MARK, nfq_end_ms]
    write_trace_line(f, text)
    # text = [0, Event.CMD, 'rm server-test.log']
    # write_trace_line(f, text)
    # text = [0, Event.CMD, './scripts/load-tc-rules.sh mod veth1 75 10000 '
    #         '$((200*1000000)) $((256*1024)) 5 &']
    # write_trace_line(f, text)
    # text = [0, Event.CMD, './scripts/load-tc-rules.sh mod veth2 33 10000 '
    #         '$((1200*1000000)) $((1024*1024)) 3 &']
    # write_trace_line(f, text)
    text = [start_s*1000, Event.CMD, 
            'iperf3 -c 10.0.0.2 -t {} -i 0.1 -J --logfile {} &'.format(duration_s, sv_fname)]
    write_trace_line(f, text)


def print_fig_cfg():
    text = """textsize=24
params = {
'axes.labelsize': textsize,
'font.size': textsize,
'legend.fontsize': textsize-5,
'xtick.labelsize': textsize,
'ytick.labelsize': textsize,
'lines.linewidth':2.5,
'text.usetex': False,
'figure.figsize': [9, 4]
}
mpl.rcParams.update(params)
# for e in mpl.rcParams.keys():
#     if 'line' in e:
#         print(e)"""
    print(text)

def fig_cfg():
    textsize=24
    params = {
    'axes.labelsize': textsize,
    'font.size': textsize,
    'legend.fontsize': textsize-4,
    'xtick.labelsize': textsize,
    'ytick.labelsize': textsize,
    'lines.linewidth':2.5,
    'text.usetex': False,
    'figure.figsize': [9, 4]
    }
    mpl.rcParams.update(params)
    # for e in mpl.rcParams.keys():
    #     if 'line' in e:
    #         print(e)

def draw_multi_ecdf(data_dict, xlabel, ylabel='CDF (%)', fname=None):
    lines_cycle = iter(['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3))])
    for key in data_dict.keys():        
        x, y = ecdf(data_dict[key])
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0.)
        if key == "sw":
            key = "M2HO"
        plt.plot(x, y, label=key, linestyle=next(lines_cycle))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if len(data_dict) != 1:
        plt.legend()
    plt.tight_layout()
    if fname != None:
        plt.savefig('{}.pdf'.format(fname), dpi=600, transparent=True)
    plt.show()

# CW processing functions
from scipy.signal import find_peaks

def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[n:] - cumsum[:-n]) / float(n)

def find_nearest(array, value):
    if len(array) == 0:
        return
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# return maximum K value sorted
def maxN(data, k):
    ar = np.array(data)
    ind = np.argpartition(ar, -k)[-k:]
    return np.sort(ar[ind])

def maxN_idx(data, k):
    ar = np.array(data)
    ind = np.argpartition(ar, -k)[-k:]
    return ind

def read_trace(fname):
    prevHoType = LINK.LTE
    hoStats = {LINK.LTE:{LINK.LTE:[], LINK.MMW:[]}, LINK.MMW:{LINK.LTE:[], LINK.MMW:[]}}
    with open(fname, 'r') as f:
        for line in f:
            items = line.split(',')
            if int(items[1]) == Event.HO:
                ts = round(int(items[0])/1000, 1)
                hoType = int(items[3]) - 1
                hoStats[prevHoType][hoType].append(ts-1)
                prevHoType = hoType
    return hoStats