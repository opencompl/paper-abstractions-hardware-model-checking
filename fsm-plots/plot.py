#!/usr/bin/env python3

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from statistics import geometric_mean

from typing import Callable


resultsDir = "../fsm-raw-data/"

def setGlobalDefaults():
    ## Use TrueType fonts instead of Type 3 fonts
    #
    # Type 3 fonts embed bitmaps and are not allowed in camera-ready submissions
    # for many conferences. TrueType fonts look better and are accepted.
    # This follows: https://www.conference-publishing.com/Help.php
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    ## Enable tight_layout by default
    #
    # This ensures the plot has always sufficient space for legends, ...
    # Without this sometimes parts of the figure would be cut off.
    matplotlib.rcParams['figure.autolayout'] = True

    ## Legend defaults
    matplotlib.rcParams['legend.frameon'] = False
    
    # Hide the right and top spines
    #
    # This reduces the number of lines in the plot. Lines typically catch
    # a readers attention and distract the reader from the actual content.
    # By removing unnecessary spines, we help the reader to focus on
    # the figures in the graph.
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False

matplotlib.rcParams['figure.figsize'] = 10,4
plt.tight_layout()

# Color palette
light_gray = "#cacaca"
dark_gray = "#827b7b"
light_blue = "#a6cee3"
dark_blue = "#1f78b4"
light_green = "#b2df8a"
dark_green = "#33a02c"
light_red = "#fb9a99"
dark_red = "#e31a1c"
black = "#000000"
white = "#ffffff"

colors = [light_gray, dark_gray, light_blue, dark_blue, light_green, dark_green, light_red, dark_red, white, black]

matplotlib.rcParams.update({"font.size": 20})

def save(figure, name):
    # Do not emit a creation date, creator name, or producer. This will make the
    # content of the pdfs we generate more deterministic.
    metadata = {'CreationDate': None, 'Creator': None, 'Producer': None}

    figure.savefig(name, metadata=metadata)

    # Close figure to avoid warning about too many open figures.
    plt.close(figure)
    
    print(f'written to {name}')

# helper for str_from_float.
# format float in scientific with at most *digits* digits.
#
# precision of the mantissa will be reduced as necessary,
# as much as possible to get it within *digits*, but this
# can't be guaranteed for very large numbers.
def get_scientific(x: float, digits: int):
    # get scientific without leading zeros or + in exp
    def get(x: float, prec: int) -> str:
      result = f'{x:.{prec}e}'
      result = result.replace('e+', 'e')
      while 'e0' in result:
        result = result.replace('e0', 'e')
      while 'e-0' in result:
        result = result.replace('e-0', 'e-')
      return result

    result = get(x, digits)
    len_after_e = len(result.split('e')[1])
    prec = max(0, digits - len_after_e - 2)
    return get(x, prec)

# format float with at most *digits* digits.
# if the number is too small or too big,
# it will be formatted in scientific notation,
# optionally a suffix can be passed for the unit.
#
# note: this displays different numbers with different
# precision depending on their length, as much as can fit.
def str_from_float(x: float, digits: int = 3, suffix: str = '') -> str:
  result = f'{x:.{digits}f}'
  before_decimal = result.split('.')[0]
  if len(before_decimal) == digits:
    return before_decimal
  if len(before_decimal) > digits:
    # we can't even fit the integral part
    return get_scientific(x, digits)

  result = result[:digits + 1] # plus 1 for the decimal point
  if float(result) == 0:
    # we can't even get one significant figure
    return get_scientific(x, digits)

  return result[:digits + 1]

# Attach a text label above each bar in *rects*, displaying its height
def autolabel(ax, rects, label_from_height: Callable[[float], str] =str_from_float, xoffset=0, yoffset=1, **kwargs):
    # kwargs is directly passed to ax.annotate and overrides defaults below
    assert 'xytext' not in kwargs, "use xoffset and yoffset instead of xytext"
    default_kwargs = dict(
        xytext=(xoffset, yoffset),
        fontsize="smaller",
        rotation=0,
        ha='center',
        va='bottom',
        textcoords='offset points')

    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            label_from_height(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            **(default_kwargs | kwargs),
        )

# utility to print times as 1h4m, 1d15h, 143.2ms, 10.3s etc.
def str_from_ms(ms):
  def maybe_val_with_unit(val, unit):
    return f'{val}{unit}' if val != 0 else ''

  if ms < 1000:
    return f'{ms:.3g}ms'

  s = ms / 1000
  ms = 0
  if s < 60:
    return f'{s:.3g}s'

  m = int(s // 60)
  s -= 60*m
  if m < 60:
    return f'{m}m{maybe_val_with_unit(math.floor(s), "s")}'

  h = int(m // 60)
  m -= 60*h;
  if h < 24:
    return f'{h}h{maybe_val_with_unit(m, "m")}'

  d = int(h // 24)
  h -= 24*d
  return f'{d}d{maybe_val_with_unit(h, "h")}'

def autolabel_ms(ax, rects, **kwargs):
  autolabel(ax, rects, label_from_height=str_from_ms, **kwargs)


def calculate_speedup(tool1, tool2, property, df):
    tool1 = np.array(df[property+'-'+tool1])
    tool2 = np.array(df[property+'-'+tool2])
    return tool2 / tool1

def plot_speedup(benchmark, p):
  df = pd.read_csv(resultsDir+benchmark+'.csv')
  speedup_avr = calculate_speedup('fsm', 'avr', p, df)
  speedup_circtbmc = calculate_speedup('fsm', 'circtbmc', p, df)
  fig, ax = plt.subplots()
  width = 0.8
  ax.bar(np.array(df['states']) + width / 2, speedup_avr, width, label='Speedup vs. AVR', color=light_green)
  ax.bar(np.array(df['states']) - width / 2, speedup_circtbmc, width, label='Speedup vs. edamame', color=dark_green)
  ax.set_xticks(df['states'])
  ax.axhline(y=1, color=dark_red, linestyle='--', linewidth=1, label='Speedup = 1')
  ax.set_xlabel("# states")
  ax.set_ylabel("Speedup [x]", rotation="horizontal", horizontalalignment="left", y=1)
  ax.legend(loc="center right", ncols=1, frameon=False, bbox_to_anchor=(1.2, 0.5))
  save(fig, benchmark+"_"+p + "_speedup.pdf")

def plot_comparison(benchmark, p):

  df = pd.read_csv(resultsDir+benchmark+'.csv')
  fig, ax = plt.subplots()
  ax.plot(np.array(df['states']), df[p+'-fsm'], label='fsm', color=light_red)
  ax.plot(np.array(df['states']), df[p+'-avr'], label='avr', color=dark_green)
  ax.plot(np.array(df['states']), df[p+'-circtbmc'], label='edamame', color=light_green)
  ax.set_xticks(df['states'])
  ax.set_xlabel("# states")
  ax.set_ylabel("Time [ms]", rotation="horizontal", horizontalalignment="left", y=1)
  ax.legend(loc="center right", ncols=1, frameon=False, bbox_to_anchor=(1.2, 0.5))
  save(fig, benchmark+"_"+p + ".pdf")

def getTimeList(benchmark, tool, num_bounds, num_reps):
  timeList = []
  with open(f"{resultsDir}/{benchmark}/{tool}.txt") as file:
    for j in range(num_bounds):
      bound = file.readline()
      total = 0
      for i in range(num_reps):
        total += float(file.readline().strip())
      avg = total/num_reps
      timeList.append(avg)
  return timeList

def getBounds(benchmark, tool, num_bounds, num_reps):
  boundList = []
  with open(f"{resultsDir}/{benchmark}/{tool}.txt") as file:
    for j in range(num_bounds):
      bound = file.readline()
      total = 0
      for i in range(num_reps):
        total += float(file.readline().strip())
      avg = total/num_reps
      boundList.append(int(bound.strip()))
  return boundList


def main():
    parser = argparse.ArgumentParser(
        prog='plot',
        description='Plot the figures for this paper',
    )
    # parser.add_argument('names', nargs='+', choices=['all', 'hls', 'err', 'linear', 'opentitan'])
    # parser.add_argument('props', nargs='+', choices=['all', 'safety', 'liveness'])
    # args = parser.parse_args()

    setGlobalDefaults()

    # will add more options as it becomes clearer what we plot and in what form
    # plotAll = 'all' in args.names

    benchmarks = ['err', 'linear', 'hls', 'opentitan', 'pulp']
    tools = ['avr', 'ric3', 'fsm']
    toolsLabels = ['AVR', 'rIC3', 'EDAMAME-FSM']
    properties = ['safety-sat', 'safety-unsat']

    dfErr = pd.read_csv(resultsDir+'err.csv')
    dfLin = pd.read_csv(resultsDir+'linear.csv')
    dfOpenTitan = pd.read_csv(resultsDir+'opentitan.csv')
    dfHls = pd.read_csv(resultsDir+'hls.csv')
    dfPulp = pd.read_csv(resultsDir+'pulp.csv')
    
    # linear

    lineStyles = ['dashed', 'dotted', 'solid']

    for prop in properties: 
      fig, ax = plt.subplots()
      for cnt, tool in enumerate(tools): 
        ax.plot(dfLin['states'], dfLin[tool+"_"+prop], label=toolsLabels[cnt], color=colors[(cnt+1)*2], linewidth = 3, linestyle = lineStyles[cnt % len(lineStyles)])
      ax.set_xticks(dfLin['states'])
      ax.set_xlabel("#states")
      ax.set_ylabel("Time [s]\n (log)", rotation="horizontal", horizontalalignment="left", y=1.1)
      ax.set_yscale('log')

      if 'unsat' not in prop:
        plt.axhline(y=300,linewidth=1, color='red', ls='--')
      plt.tight_layout()
      if 'unsat' not in prop:
        ax.text(np.floor(ax.get_xlim()[1]*0.88), 300 - 200, f'timeout', va='center', color='red')
      if 'unsat' in prop:
        ax.set_ylim((pow(10,-3),pow(10,2)))
      else: 
        ax.set_ylim((pow(10,-3),pow(10,3)))
      ratios = []
      for i in range(len(dfLin)):
          ratios.append(dfLin['fsm_'+prop][i] / dfLin['ric3_'+prop][i])
      print(f"FSM/RIC3 average ratio: {geometric_mean(ratios)}")
      print(f"{(1-geometric_mean(ratios)) * 100}% faster geomean on linear fsms with "+prop)
      ax.legend(bbox_to_anchor=(0.5,  1),  ncols=3, frameon=False, loc='center')
      save(fig, prop+"_linear.pdf")

    # error

    for prop in properties: 
      fig, ax = plt.subplots()
      for cnt, tool in enumerate(tools): 
        ax.plot(dfErr["states"], dfErr[tool+"_"+prop], label=toolsLabels[cnt], color=colors[(cnt+1)*2], linewidth = 3, linestyle = lineStyles[cnt % len(lineStyles)])
      ax.set_xticks(dfErr["states"])
      ax.set_xlabel("#states")
      ax.set_ylabel("Time [s]\n (log)", rotation="horizontal", horizontalalignment="left", y=1.1)
      ax.set_yscale('log')
      ax.legend(bbox_to_anchor=(0.5,  1.05),  ncols=3, frameon=False, loc='center')
      print(prop)
      if 'unsat' not in prop:
        plt.axhline(y=300,linewidth=1, color='red', ls='--')
      plt.tight_layout()
      ax.set_yscale('log')
      if 'unsat' not in prop:
        ax.text(np.floor(ax.get_xlim()[1]*0.88), 300 - 200, f'timeout', va='center', color='red')
      if 'unsat' in prop:
        ax.set_ylim((pow(10,-3),pow(10,2)))
      else: 
        ax.set_ylim((pow(10,-3),pow(10,3)))
      ratios = []
      for i in range(len(dfErr)):
          ratios.append(dfErr['fsm_'+prop][i] / dfErr['ric3_'+prop][i])
      print(f"FSM/RIC3 average ratio: {geometric_mean(ratios)}")
      print(f"{(1-geometric_mean(ratios)) * 100}% faster geomean on error fsms with "+prop)
      save(fig, prop+"_err.pdf")

    # real world
    for prop in properties: 
      fig, ax = plt.subplots()

      propNames = []
      propFsm = []
      propAVR = []
      propRIC3 = []
      print (dfOpenTitan)
      for idx, entry in dfOpenTitan.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsm_'+prop])
        propAVR.append(entry['avr_'+prop])
        propRIC3.append(entry['ric3_'+prop])
      for idx, entry in dfPulp.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsm_'+prop])
        propAVR.append(entry['avr_'+prop])
        propRIC3.append(entry['ric3_'+prop])
      for idx, entry in dfHls.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsm_'+prop])
        propAVR.append(entry['avr_'+prop])
        propRIC3.append(entry['ric3_'+prop])

      df = pd.DataFrame({'states':propNames, 'EDAMAME-FSM':propFsm, 'AVR':propAVR, 'rIC3':propRIC3})

      x = np.arange(len(propNames))  # the label locations

      ax.bar(x-0.2, propAVR, width=0.2, color=colors[2], align='center', label=toolsLabels[0], hatch = '//')
      ax.bar(x, propRIC3, width=0.2, color=colors[4], align='center', label=toolsLabels[1], hatch ='..')
      ax.bar(x+0.2, propFsm, width=0.2, color=colors[6], align='center', label=toolsLabels[2])


      # Add some text for labels, title and custom x-axis tick labels, etc.
      ax.set_xlabel('#states')
      ax.set_ylabel("Time [s]\n (log)", rotation="horizontal", horizontalalignment="left", y=1.1)
      ax.set_xticks(x, propNames)
      ax.set_yscale('log')
      ax.set_ylim((pow(10,-3),pow(10,0)))
      ax.legend(bbox_to_anchor=(0.5,  1.05),  ncols=3, frameon=False, loc='center')


      ratios = []
      for i in range(len(df)):
          ratios.append(df['EDAMAME-FSM'][i] / df['rIC3'][i])
      print(f"FSM/RIC3 average ratio: {geometric_mean(ratios)}")
      print(f"{(1-geometric_mean(ratios)) * 100}% faster geomean on realworlds with "+prop)

      save(fig, prop+"_rw.pdf")





if __name__ == "__main__":
    main()
