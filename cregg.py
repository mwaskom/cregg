"""General functions and classes to support PsychoPy experiments."""
from __future__ import division

import os
import sys
import time
import json
import argparse
import subprocess
from glob import glob
from string import letters
from math import floor
from subprocess import call
import numpy as np
import pandas as pd
from numpy.random import RandomState
from psychopy import core, event, visual
import psychopy.monitors.calibTools as calib
from psychopy import logging


class Params(object):
    """Stores all of the parameters needed during the experiment.

    Some parameters are set upon initialization from the file 'params.py',
    others can be set from the command line.

    """
    def __init__(self, exp_name, p_file='params'):
        """Initializer for the params object.

        Parameters
        ----------
        exp_name: string, name of the dict we want from the param file

        p_file: string, the name of a parameter file

        """
        self.exp_name = exp_name
        im = __import__(p_file)
        self.param_module = im
        param_dict = getattr(im, exp_name)
        for key, val in param_dict.iteritems():
            setattr(self, key, val)

        self.time = time.asctime()
        self.git_hash = git_hash()

    def set_by_cmdline(self, arglist):
        """Get runtime parameters off the commandline."""
        # Create the parser, set default args
        parser = argparse.ArgumentParser()
        parser.add_argument("-subject", default="test")
        parser.add_argument("-run", type=int, default=1)
        parser.add_argument("-fmri", action="store_true")
        parser.add_argument("-debug", action="store_true")

        # Add additional arguments by experiment
        try:
            parser = self.param_module.add_cmdline_params(parser)
        except AttributeError:
            pass

        # Parse the arguments
        args = parser.parse_args(arglist)

        # Add command line args to the class dict
        self.__dict__.update(args.__dict__)

        if self.debug:
            self.full_screen = False

        if hasattr(self, "dummy_trs") and not self.fmri:
            self.dummy_trs = 1

    def to_text_header(self, fid):
        """Save the parameters to a text file."""
        for key, val in self.__dict__.items():
            if not key.startswith("_"):
                fid.write("# %s : %s \n" % (key, val))

    def to_json(self, fname):
        """Save the parameters to a .json"""
        data = dict([(k, v) for k, v in self.__dict__.items()
                     if not k.startswith("_")])
        del data["param_module"]

        if not fname.endswith(".json"):
            fname += ".json"
        archive_old_version(fname)
        with open(fname, "w") as fid:
            json.dump(data, fid, sort_keys=True, indent=4)


class DataLog(object):
    """Holds info about file that gets updated throughout experiment."""
    def __init__(self, p, columns):
        """Set things up."""
        self.p = p
        self.columns = columns

        # Figure out the name and clear out old files
        fname_base = p.log_base % dict(subject=p.subject, run=p.run)
        self.fname = fname_base + ".csv"
        archive_old_version(self.fname)

        # Save the parameters to
        p.to_json(fname_base)

        # Write the column header
        column_string = ",".join(map(str, columns)) + "\n"
        with open(self.fname, "w") as fid:
            fid.write(column_string)

    def add_data(self, data_dict):
        """Add a line of data based on a dictionary and stored columns."""
        data_list = [str(data_dict.get(col, None)) for col in self.columns]
        data_str = ",".join(data_list) + "\n"
        with open(self.fname, "a") as fid:
            fid.write(data_str)


class WindowInfo(object):
    """Container for monitor information."""
    def __init__(self, params, monitor):
        """Extracts monitor information from params file and monitors.py."""
        try:
            mod = __import__("monitors")
        except ImportError:
            sys.exit("Could not import monitors.py in this directory.")

        try:
            minfo = getattr(mod, params.monitor_name.replace("-", "_"))
        except IndexError:
            sys.exit("Monitor name '%s' not found in monitors.py")

        size = minfo["size"] if params.full_screen else (800, 600)
        info = dict(units=params.monitor_units,
                    fullscr=params.full_screen,
                    allowGUI=not params.full_screen,
                    color=params.window_color,
                    size=size,
                    monitor=monitor)

        self.name = params.monitor_name
        self.__dict__.update(info)
        self.window_kwargs = info


class WaitText(object):
    """A class for showing text on the screen until a key is pressed. """
    def __init__(self, win, text="Press a key to continue",
                 advance_keys=None, quit_keys=None, **kwargs):
        """Set the text stimulus information."""
        self.win = win
        if advance_keys is None:
            advance_keys = ["space"]
        self.advance_keys = advance_keys
        if quit_keys is None:
            quit_keys = ["escape", "q"]
        self.quit_keys = quit_keys
        self.listen_keys = quit_keys + advance_keys
        self.text = visual.TextStim(win, text=text, **kwargs)

    def draw(self, duration=np.inf):
        """Dislpay text until a key is pressed or until duration elapses."""
        clock = core.Clock()
        t = 0
        # Keep going for the duration
        while t < duration:
            t = clock.getTime()
            self.text.draw()
            self.win.flip()
            for key in event.getKeys(keyList=self.listen_keys):
                if key in self.quit_keys:
                    core.quit()
                elif key in self.advance_keys:
                    return


class PresentationLoop(object):
    """Context manager for the main loop of an experiment."""
    def __init__(self, win, log=None, exit_func=None, fileobj=None):
        self.win = win
        if log is not None:
            self.log = log
        if exit_func is not None:
            self.exit_func = exit_func
        if fileobj is not None:
            self.fileobj = fileobj

    def __enter__(self):

        pass

    def __exit__(self, type, value, traceback):

        self.win.close()
        if hasattr(self, "fileobj"):
            self.fileobj.close()
        if hasattr(self, "exit_func"):
            self.exit_func(self.log)


def archive_old_version(fname):
    """Move a data file to an numbered archive version, if exists."""
    if not os.path.exists(fname):
        return

    base, ext = os.path.splitext(fname)
    n_existing = len(glob(base + "_*" + ext))
    n_current = n_existing + 1
    new_fname = "%s_%d%s" % (base, n_current, ext)
    os.rename(fname, new_fname)


def git_hash():
    """Get the commit hash for the HEAD commmit."""
    out = subprocess.Popen("git rev-parse HEAD",
                           stdout=subprocess.PIPE, shell=True)
    hash = out.communicate()[0].strip()
    return hash


def max_brightness(monitor):
    """Maximize the brightness on a laptop."""
    try:
        call(["brightness", "1"])
    except OSError:
        print "Could not modify screen brightness"


def check_quit(quit_keys=["q", "escape"]):
    """Check if we got a quit key signal and exit if so."""
    keys = event.getKeys(keyList=quit_keys)
    for key in keys:
        if key in quit_keys:
            print "Subject quit execution"
            core.quit()
    event.clearEvents()


def wait_check_quit(wait_time, quit_keys=None):
    """Wait a given time, checking for a quit every second."""
    if quit_keys is None:
        quit_keys = ["q", "escape"]
    for sec in range(int(floor(wait_time))):
        core.wait(1)
        check_quit(quit_keys)
    remaining = wait_time - floor(wait_time)
    if remaining:
        core.wait(remaining)
    event.clearEvents()


def wait_for_trigger(win, params):
    """Hold presentation until we hear trigger keys."""
    event.clearEvents()
    visual.TextStim(win, text="Get ready!").draw()
    win.flip()

    # Here's where we expect pulses
    wait = True
    while wait:
        trigger_keys = ["5", "t"]
        listen_keys = trigger_keys + list(params.quit_keys)
        for key in event.getKeys(keyList=listen_keys):
            if key in params.quit_keys:
                core.quit()
            elif key:
                wait = False
    event.clearEvents()


def precise_wait(win, clock, end_time, stim):
    """Wait with precision controlled by screen refreshes."""
    end_time -= 1 / win.refresh_rate
    now = clock.getTime()
    while now < end_time:
        stim.draw()
        win.flip()
        now = clock.getTime()


def wait_and_listen(listen_for):
    """Do nothing until a specific key is pressed."""
    should_wait = True
    while should_wait:
        for key in event.getKeys():
            if key == listen_for:
                should_wait = False


def draw_all(in_list):
    """Draw every PsychoPy object in a list."""
    for stim in in_list:
        stim.draw()


def categorical(p, size=None):
    """Returns random samples from a categorial distribution."""
    if sum(p) != 1:
        raise ValueError("Values in probability vector must sum to 1")

    if size is None:
        iter_size = 1
    else:
        iter_size = size

    sample = []
    for i in xrange(iter_size):
        multi_sample = np.random.multinomial(1, p)
        sample.append(np.argwhere(multi_sample))

    sample = np.squeeze(sample)
    if size is None:
        sample = np.asscalar(sample)
    return sample


def flip(p=0.5):
    """Shorthand for a bernoulli sample."""
    return np.random.binomial(1, p)


def load_design_csv(params):
    """Load a design file with a quasi-random

    This assumes that designs are named with letters, and then
    randomized across runs for each subject. It uses the subject
    name hash trick to have consisent randomization for each subject ID.

    """
    state = subject_specific_state(params.subject)
    choices = list(letters[:params.n_designs])
    params.sched_id = state.permutation(choices)[params.run - 1]
    design_file = params.design_template % params.sched_id
    design = pd.read_csv(design_file, index_col="trial")
    return design


def subject_specific_state(subject):
    """Obtain a numpy random state that is consistent for a subject ID."""
    state = RandomState(abs(hash(subject)))
    return state


def launch_window(params):
    """Open up a presentation window."""
    calib.monitorFolder = "./calib"
    mon = calib.Monitor(params.monitor_name)
    m = WindowInfo(params, mon)
    win = visual.Window(**m.window_kwargs)
    win.setRecordFrameIntervals(True)
    logging.console.setLevel(logging.CRITICAL)
    return win
