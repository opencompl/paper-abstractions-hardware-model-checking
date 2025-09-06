#!/usr/bin/env python3

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from statistics import geometric_mean

from typing import Callable


resultsDir = "../chc-raw-data/"
fsmResultsDir = "../fsm-raw-data/"

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
  ax.set_xlabel("#states")
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
  ax.set_xlabel("#states")
  ax.set_ylabel("Time [s]", rotation="horizontal", horizontalalignment="left", y=1)
  ax.legend(loc="center right", ncols=1, frameon=False, bbox_to_anchor=(1.2, 0.5))
  save(fig, benchmark+"_"+p + ".pdf")


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

    properties = ['safety-sat']

    for prop in properties:

      df_err = pd.read_csv(fsmResultsDir+'err.csv')
      df_hls = pd.read_csv(fsmResultsDir+'hls.csv')
      df_lin = pd.read_csv(fsmResultsDir+'linear.csv')
      df_ot = pd.read_csv(fsmResultsDir+'opentitan.csv')
      df_pulp = pd.read_csv(fsmResultsDir+'pulp.csv')
      df_hls = pd.read_csv(fsmResultsDir+'hls.csv')


      chc_df_err = pd.read_csv(resultsDir+'err.csv')
      chc_df_hls = pd.read_csv(resultsDir+'hls.csv')
      chc_df_lin = pd.read_csv(resultsDir+'linear.csv')
      chc_df_ot = pd.read_csv(resultsDir+'opentitan.csv')
      chc_df_pulp = pd.read_csv(resultsDir+'pulp.csv')
      chc_df_hls = pd.read_csv(resultsDir+'hls.csv')

      propNames = []
      propFsm = []
      propCHC = []

      for idx, entry in chc_df_ot.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsmt'])
        propCHC.append(entry['chc'])
      for idx, entry in chc_df_pulp.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsmt'])
        propCHC.append(entry['chc'])
      for idx, entry in chc_df_hls.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsmt'])
        propCHC.append(entry['chc'])

      chc_df_rw = pd.DataFrame({'states':propNames, 'fsmt':propFsm, 'chc':propCHC})

      propNames = []
      propFsm = []
      propCHC = []

      for idx, entry in df_ot.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsm_'+prop])
      for idx, entry in df_pulp.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsm_'+prop])
      for idx, entry in df_hls.iterrows(): 
        propNames.append(int(entry["states"]))
        propFsm.append(entry['fsm_'+prop])

      df_rw = pd.DataFrame({'states':propNames, 'fsmt':propFsm})

      # CHC Comparison Plots
      fig, ax = plt.subplots()
      ax.plot(np.array(chc_df_err['states']), chc_df_err['chc'], label='btor2chc', color=dark_blue, linewidth = 3)
      ax.plot(np.array(chc_df_err['states']), chc_df_err['fsmt'], label='EDAMAME-FSM', color=light_red, linewidth = 3)
      ax.set_xticks(chc_df_err['states'])
      ax.set_xlabel("#states")
      ax.set_ylabel("Time [s]\n (log)", rotation="horizontal", horizontalalignment="left", y=1.1)
      ax.set_yscale('log')
      if 'unsat' in prop:
        ax.set_ylim((pow(10,-2),pow(10,3)))
      else: 
        ax.set_ylim((pow(10,-2),pow(10,3)))
      ax.legend(bbox_to_anchor=(0.5, 1.05), ncols=2, frameon=False, loc='center')
      save(fig,prop+"_chc_err.pdf")

      fig, ax = plt.subplots()
      ax.plot(np.array(chc_df_lin['states']), chc_df_lin['chc'], label='btor2chc', color=dark_blue, linewidth = 3)
      ax.plot(np.array(chc_df_lin['states']), chc_df_lin['fsmt'], label='EDAMAME-FSM', color=light_red, linewidth = 3)
      ax.set_xticks(chc_df_lin['states'])
      ax.set_xlabel("#states")
      ax.set_ylabel("Time [s]\n (log)", rotation="horizontal", horizontalalignment="left", y=1.1)
      ax.set_yscale('log')
      if 'unsat' in prop:
        ax.set_ylim((pow(10,-3),pow(10,3)))
      else: 
        ax.set_ylim((pow(10,-3),pow(10,3)))
      ax.legend(bbox_to_anchor=(0.5, 1.05),  ncols=2, frameon=False, loc='center')
      save(fig,prop+"_chc_lin.pdf")

      fig, ax = plt.subplots()
      width = 0.25
      x = np.arange(len(chc_df_rw['states']))
      ax.bar(x + width/2, chc_df_rw['chc'], width, label='btor2chc', color=dark_blue, linewidth = 3)
      ax.bar(x - width/2, chc_df_rw['fsmt'], width, label='EDAMAME-FSM', color=light_red, linewidth = 3)
      ax.set_xticks(x)
      ax.set_xticklabels(df_rw['states'])
      ax.set_xlabel("#states")
      ax.set_ylabel("Time [s]\n (log)", rotation="horizontal", horizontalalignment="left", y=1.1)
      ax.set_yscale('log')
      if 'unsat' in prop:
        ax.set_ylim((pow(10,-3),pow(10,0)))
      else: 
        ax.set_ylim((pow(10,-3),pow(10,0)))
      ax.legend(bbox_to_anchor=(0.5, 1.05), ncols=2, frameon=False, loc='center')
      
      save(fig,prop+"_chc_realworlds.pdf")

      ratios = []
      for i in range(len(chc_df_rw)):
        if i != 2:  
          ratios.append(chc_df_rw['fsmt'][i] / chc_df_rw['chc'][i])
      print(f"FSM/BTOR2CHC: average ratio: {geometric_mean(ratios)}")
      print(f"{(1-geometric_mean(ratios)) * 100}% faster geomean on realworlds fsms with "+prop)

      ratios = []
      for i in range(len(chc_df_lin)):
          ratios.append(chc_df_lin['fsmt'][i] / chc_df_lin['chc'][i])
      print(f"FSM/BTOR2CHC: average ratio: {geometric_mean(ratios)}")
      print(f"{(1-geometric_mean(ratios)) * 100}% faster geomean on linear fsms with "+prop)

      ratios = []
      for i in range(len(chc_df_err)):
          ratios.append(chc_df_err['fsmt'][i] / chc_df_err['chc'][i])
      print(f"FSM/BTOR2CHC: average ratio: {geometric_mean(ratios)}")
      print(f"{(1-geometric_mean(ratios)) * 100}% faster geomean on error fsms with "+prop)



    # if 'all' in args.names or 'hls' in args.names:
    #   if 'all' in args.props or 'safety' in args.props:
    #     plot_comparison('hls', 'safety')
    #     plot_speedup('hls', 'safety')
    #   if 'all' in args.props or 'liveness' in args.props:
    #     plot_comparison('hls', 'liveness')
    #     plot_speedup('hls', 'liveness')

    # if 'all' in args.names or 'err' in args.names:
    #   if 'all' in args.props or 'safety' in args.props:
    #     plot_comparison('err', 'safety')
    #     plot_speedup('err', 'safety')
    #   if 'all' in args.props or 'liveness' in args.props:
    #     plot_comparison('err', 'liveness')
    #     plot_speedup('err', 'liveness')

    # if 'all' in args.names or 'linear' in args.names:
    #   if 'all' in args.props or 'safety' in args.props:
    #     plot_comparison('linear', 'safety')
    #     plot_speedup('linear', 'safety')
    #   if 'all' in args.props or 'liveness' in args.props:
    #     plot_comparison('linear', 'liveness')
    #     plot_speedup('linear', 'liveness')

    # if 'all' in args.names or 'opentitan' in args.names:
    #   if 'all' in args.props or 'safety' in args.props:
    #     plot_comparison('opentitan', 'safety')
    #     plot_speedup('opentitan', 'safety')
    #   if 'all' in args.props or 'liveness' in args.props:
    #     plot_comparison('opentitan', 'liveness')
    #     plot_speedup('opentitan', 'liveness')


if __name__ == "__main__":
    main()
