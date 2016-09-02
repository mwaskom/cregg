"""General functions and classes to support PsychoPy experiments."""
from __future__ import division

import os
import sys
import time
import json
import warnings
import argparse
import subprocess
from glob import glob
from string import letters
from math import floor
from subprocess import call
from pprint import pformat
import numpy as np
import pandas as pd
from scipy import stats
from numpy.random import RandomState
from psychopy import core, event, visual, sound
from psychopy.monitors import Monitor
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

        timestamp = time.localtime()
        self.timestamp = time.asctime(timestamp)
        self.date = time.strftime("%Y-%m-%d", timestamp)
        self.time = time.strftime("%H-%M-%S", timestamp)
        self.git_hash = git_hash()

    def __repr__(self):

        return pformat(self.__dict__)

    def __getitem__(self, key):

        return getattr(self, key)

    def get(self, key, default=None):

        return getattr(self, key, default)

    def set_by_cmdline(self, arglist):
        """Get runtime parameters off the commandline."""
        # Create the parser, set default args
        parser = argparse.ArgumentParser()
        parser.add_argument("-subject", default="test")
        parser.add_argument("-cbid")
        parser.add_argument("-run", type=int, default=1)
        parser.add_argument("-fmri", action="store_true")
        parser.add_argument("-debug", action="store_true")
        parser.add_argument("-nolog", action="store_true")

        # Add additional arguments by experiment
        try:
            func_name = self.exp_name + "_cmdline"
            arg_func = getattr(self.param_module, func_name)
            arg_func(parser)
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

        if self.fmri and hasattr(self, "fmri_monitor_name"):
            self.monitor_name = self.fmri_monitor_name

        if self.fmri and hasattr(self, "fmri_screen_number"):
            self.screen_number = self.fmri_screen_number

        if self.fmri and hasattr(self, "fmri_resp_keys"):
            self.resp_keys = self.fmri_resp_keys

    def to_text_header(self, fid):
        """Save the parameters to a text file."""
        for key, val in self.__dict__.items():
            if not key.startswith("_"):
                fid.write("# {} : {} \n".format(key, val))

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

        if not p.nolog:
            self.init_log(p)

    def init_log(self, p):

        # Figure out the name and clear out old files
        kws = dict(subject=p.subject, date=p.date, time=p.time, run=p.run)
        fname_base = p.log_base.format(**kws)
        self.fname = fname_base + ".csv"
        archive_old_version(self.fname)

        # Save the parameters to json with a similar base filename
        p.to_json(fname_base)

        # Write the column header
        column_string = ",".join(map(str, self.columns)) + "\n"
        with open(self.fname, "w") as fid:
            fid.write(column_string)

    def add_data(self, data_dict):
        """Add a line of data based on a dictionary and stored columns."""
        data_list = [str(data_dict.get(col, None)) for col in self.columns]
        data_str = ",".join(data_list) + "\n"
        if not self.p.nolog:
            with open(self.fname, "a") as fid:
                fid.write(data_str)


class WindowInfo(object):
    """Container for monitor information."""
    def __init__(self, params):
        """Extracts monitor information from params file and monitors.py."""
        try:
            mod = __import__("monitors")
        except ImportError:
            sys.exit("Could not import monitors.py in this directory.")

        try:
            minfo = getattr(mod, params.monitor_name.replace("-", "_"))
        except IndexError:
            sys.exit("Monitor not found in monitors.py")

        size = minfo["size"] if params.full_screen else (800, 600)

        monitor = Monitor(name=minfo["name"],
                          width=minfo["width"],
                          distance=minfo["distance"])
        monitor.setSizePix(minfo["size"])

        try:
            if "gamma" in minfo:
                monitor.setGamma(minfo["gamma"])
            if "gamma_grid" in minfo:
                monitor.setGammaGrid(minfo["gamma_grid"])
        except AttributeError:
            warnings.warn("Could not set monitor gamma table.")

        info = dict(units=params.monitor_units,
                    fullscr=params.full_screen,
                    allowGUI=not params.full_screen,
                    color=params.window_color,
                    screen=params.screen_number,
                    size=size,
                    monitor=monitor)

        if hasattr(params, "blend_mode"):
            info["blendMode"] = params.blend_mode
            if params.blend_mode == "add":
                info["useFBO"] = True

        if "refresh_hz" in minfo:
            self.refresh_hz = minfo["refresh_hz"]

        self.name = params.monitor_name
        self.__dict__.update(info)
        self.window_kwargs = info


class WaitText(object):
    """A class for showing text on the screen until a key is pressed. """
    def __init__(self, win, lines=("Press a key to continue"),
                 advance_keys=None, quit_keys=None, height=.5, **kwargs):
        """Set the text stimulus information."""
        self.win = win
        if advance_keys is None:
            advance_keys = ["space"]
        self.advance_keys = advance_keys
        if quit_keys is None:
            quit_keys = ["escape", "q"]
        self.quit_keys = quit_keys
        self.listen_keys = quit_keys + advance_keys
        kwargs["height"] = height

        n = len(lines)
        heights = (np.arange(n)[::-1] - (n / 2 - .5)) * height
        texts = []
        for line, y in zip(lines, heights):
            text = visual.TextStim(win, line, pos=(0, y), **kwargs)
            texts.append(text)
        self.texts = texts

    def draw(self, duration=np.inf, sleep_time=.2):
        """Dislpay text until a key is pressed or until duration elapses."""
        clock = core.Clock()
        for text in self.texts:
            text.draw()
        self.win.flip()
        t = 0
        while t < duration:
            t = clock.getTime()
            for key in event.getKeys(keyList=self.listen_keys):
                if key in self.quit_keys:
                    core.quit()
                elif key in self.advance_keys:
                    return
            time.sleep(sleep_time)


class Fixation(object):
    """Simple fixation point with color as a property."""
    def __init__(self, win, p, color="white"):

        color = p.get("fix_iti_color", color)
        self.win = win
        self.dot = visual.Circle(win, interpolate=True,
                                 fillColor=color,
                                 lineColor=color,
                                 size=p.fix_size)

        self._color = color

    @property
    def color(self):

        return self._color

    @color.setter  # pylint: disable-msg=E0102r
    def color(self, color):

        if color is None:
            color = self.win.color

        self._color = color
        self.dot.setFillColor(color)
        self.dot.setLineColor(color)

    def draw(self):

        self.dot.draw()


class ProgressBar(object):
    """Progress bar to show how far one is in an experiment."""
    def __init__(self, win, p):

        self.p = p

        self.width = width = p.get("prog_bar_width", 5)
        self.height = height = p.get("prog_bar_height", .25)
        self.position = position = p.get("prog_bar_position", -3)

        color = p.get("prog_bar_color", "white")
        linewidth = p.get("prog_bar_linewidth", 2)

        self.full_verts = np.array([(0, 0), (0, 1),
                                    (1, 1), (1, 0)], np.float)

        frame_verts = self.full_verts.copy()
        frame_verts[:, 0] *= width
        frame_verts[:, 1] *= height
        frame_verts[:, 0] -= width / 2
        frame_verts[:, 1] += position

        self.frame = visual.ShapeStim(win,
                                      fillColor=None,
                                      lineColor=color,
                                      lineWidth=linewidth,
                                      vertices=frame_verts)

        self.bar = visual.ShapeStim(win,
                                    fillColor=color,
                                    lineColor=color,
                                    lineWidth=linewidth)

        self._prop_completed = 0

    @property
    def prop_completed(self):
        return self._prop_completed

    @prop_completed.setter
    def prop_completed(self, prop):
        self.update_bar(prop)

    def update_bar(self, prop):

        self._prop_completed = prop
        bar_verts = self.full_verts.copy()
        bar_verts[:, 0] *= self.width * prop
        bar_verts[:, 1] *= self.height
        bar_verts[:, 0] -= self.width / 2
        bar_verts[:, 1] += self.position
        self.bar.vertices = bar_verts
        self.bar.setVertices(bar_verts)

    def draw(self):

        self.bar.draw()
        self.frame.draw()


class PresentationLoop(object):
    """Context manager for the main loop of an experiment."""
    def __init__(self, win, p=None, log=None, fix=None,
                 exit_func=None, feedback_func=None,
                 fileobj=None, tracker=None):

        self.p = p
        self.win = win
        self.fix = fix
        self.log = log
        self.exit_func = exit_func
        self.feedback_func = feedback_func
        self.fileobj = fileobj
        self.tracker = tracker

    def __enter__(self):

        if self.p.fmri:
            wait_for_trigger(self.win, self.p)
            self.fix.draw()
            self.win.flip()
            wait_check_quit(self.p.equilibrium_trs * self.p.tr)

    def __exit__(self, type, value, traceback):

        if self.fileobj is not None:
            self.fileobj.close()
        if self.tracker is not None:
            self.tracker.shutdown()
        if self.feedback_func is not None:
            self.feedback_func(self.win, self.p, self.log)
        if self.exit_func is not None:
            self.exit_func(self.log)
        self.win.close()


class AuditoryFeedback(object):

    def __init__(self, correct="ding", wrong="signon", noresp="click"):

        sound_dir = os.path.join(os.path.dirname(__file__), "sounds")
        sound_name_dict = dict(correct=correct, wrong=wrong, noresp=noresp)
        sound_dict = {}
        for event, sound_type in sound_name_dict.items():
            if sound is None:
                sound_dict[event] = None
            else:
                fname = os.path.join(sound_dir, sound_type + ".wav")
                sound_obj = sound.Sound(fname)
                sound_dict[event] = sound_obj
        self.sound_dict = sound_dict

    def __call__(self, event):

        sound_obj = self.sound_dict[event]
        if sound_obj is not None:
            sound_obj.play()


def make_common_visual_objects(win, p):
    """Return a dictionary with visual objects that are generally useful."""

    # Fixation point
    fix = Fixation(win, p)

    # Progress bar to show during behavioral breaks
    progress = ProgressBar(win, p)

    stims = dict(fix=fix, progress=progress)

    quit_keys = p.get("quit_keys", ["q", "escape"])
    wait_keys = p.get("wait_keys", ["space"])
    finish_keys = p.get("finish_keys", ["return"])

    # Instructions
    if hasattr(p, "instruct_text"):
        instruct = WaitText(win, p.instruct_text,
                            advance_keys=wait_keys,
                            quit_keys=quit_keys)
        stims["instruct"] = instruct

    # Text that allows subjects to take a break between blocks
    if hasattr(p, "break_text"):
        take_break = WaitText(win, p.break_text,
                              advance_keys=wait_keys,
                              quit_keys=quit_keys)
        stims["break"] = take_break

    # Text that alerts subjects to the end of an experimental run
    if hasattr(p, "finish_text"):
        finish_run = WaitText(win, p.finish_text,
                              advance_keys=finish_keys,
                              quit_keys=quit_keys)
        stims["finish"] = finish_run

    return stims


def archive_old_version(fname):
    """Move a data file to an numbered archive version, if exists."""
    if not os.path.exists(fname):
        return

    base, ext = os.path.splitext(fname)
    n_existing = len(glob(base + "_*" + ext))
    n_current = n_existing + 1
    new_fname = "{}_{:d}{}".format(base, n_current, ext)
    os.rename(fname, new_fname)


def git_hash():
    """Get the commit hash for the HEAD commmit."""
    out = subprocess.Popen("git rev-parse HEAD",
                           stdout=subprocess.PIPE,
                           shell=True)
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
    if event.getKeys(keyList=quit_keys):
        print "Subject quit execution"
        core.quit()
    event.clearEvents()


def wait_check_quit(wait_time, quit_keys=["q", "escape"], check_every=1):
    """Wait a given time, checking for a quit periodically."""
    checks = int(floor(wait_time / check_every))
    for _ in range(checks):
        core.wait(check_every)
        check_quit(quit_keys)

    remaining = wait_time - checks * check_every
    if remaining:
        core.wait(remaining)

    event.clearEvents()


def wait_check_pause_quit(win, wait_time, quit_keys=["q", "escape"],
                          pause_keys=["space"], check_every=1):
    """Wait while checking for pause or quit input."""
    raise NotImplementedError("This isn't quite finished yet")
    checks = int(floor(wait_time / check_every))
    for _ in range(checks):

        core.wait(check_every)

        if event.getKeys(keyList=quit_keys):
            print "Subject quit execution"
            core.quit()

        if event.getKeys(keyList=pause_keys):

            pause_start = time.time()
            visual.TextStim(win, text="Experiment paused").draw()
            win.flip()
            paused = True
            while paused:
                if event.getKeys(keyList=pause_keys):
                    paused = False
                core.sleep(check_every)
            pause_end = time.time()

    pause_duration = pause_end - pause_start
    remaining = wait_time - checks * check_every
    if remaining:
        core.wait(remaining)

    return pause_duration


def wait_for_trigger(win, params):
    """Hold presentation until we hear trigger keys."""
    event.clearEvents()
    visual.TextStim(win, text="Get ready!").draw()
    win.flip()

    # Here's where we expect pulses
    wait = True
    while wait:
        listen_keys = list(params.trigger_keys) + list(params.quit_keys)
        for key in event.getKeys(keyList=listen_keys):
            if key in params.quit_keys:
                core.quit()
            elif key:
                wait = False
    event.clearEvents()


def precise_wait(win, clock, end_time, stim):
    """Wait with precision controlled by screen refreshes."""
    now = clock.getTime()
    wait_flips = np.floor((end_time - now) * win.refresh_hz)
    for _ in xrange(int(wait_flips)):
        stim.draw()
        win.flip()
    now = clock.getTime()
    return now


def wait_and_listen(listen_for, sleep_time=None):
    """Do nothing until a specific key is pressed."""
    should_wait = True
    while should_wait:
        if sleep_time is not None:
            core.wait(sleep_time)
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


def flexible_values(val, size=1, random_state=None):
    """Flexibly determine a number of values.

    Input format can be:
        - A numeric value, which will be used exactly.
        - A list of possible values, which will be randomly chosen from.
        - A tuple of (dist, arg0[, arg1, ...]), which will be used to generate
          random observations from a scipy random variable.

    """
    if random_state is None:
        random_state = RandomState()

    if np.isscalar(val):
        out = np.ones(size, np.array(val).dtype) * val
        if size == 1:
            out = out.item()
    elif isinstance(val, list):
        out = random_state.choice(val, size=size)
    elif isinstance(val, tuple):
        rv = getattr(stats, val[0])(*val[1:])
        out = rv.rvs(size=size, random_state=random_state)
    else:
        raise TypeError("`val` must be scalar, set, or tuple")

    return out


def load_design_csv(params):
    """Load a design file with a quasi-random

    This assumes that designs are named with letters, and then
    randomized across runs for each subject. It uses the subject
    name hash trick to have consisent randomization for each subject ID.

    """
    state = subject_specific_state(params.subject)
    choices = list(letters[:params.n_designs])
    params.sched_id = state.permutation(choices)[params.run - 1]
    design_file = params.design_template.format(params.sched_id)
    design = pd.read_csv(design_file, index_col="trial")
    return design


def subject_specific_state(subject, cbid=None):
    """Obtain a numpy random state that is consistent for a subject ID."""
    subject = subject if cbid is None else cbid
    state = RandomState(sum(map(ord, subject)))
    return state


def launch_window(params, test_refresh=True, test_tol=.5):
    """Open up a presentation window and measure the refresh rate."""
    # Get the monitor parameters
    m = WindowInfo(params)
    stated_refresh_hz = getattr(m, "refresh_hz", None)

    # Initialize the Psychopy window object
    win = visual.Window(**m.window_kwargs)

    # Record the refresh rate we are currently achieving
    if test_refresh or stated_refresh_hz is None:
        win.setRecordFrameIntervals(True)
        logging.console.setLevel(logging.CRITICAL)
        flip_time, _, _ = visual.getMsPerFrame(win)
        observed_refresh_hz = 1000 / flip_time

    # Possibly test the refresh rate against what we expect
    if test_refresh and stated_refresh_hz is not None:
        refresh_error = np.abs(stated_refresh_hz - observed_refresh_hz)
        if refresh_error > test_tol:
            msg = ("Observed refresh rate differs from expected by {:.3f} Hz"
                   .format(refresh_error))
            raise RuntimeError(msg)

    # Set the refresh rate to use in the experiment
    if stated_refresh_hz is None:
        msg = "Monitor configuration does not have refresh rate information"
        warnings.warn(msg)
        win.refresh_hz = observed_refresh_hz
    else:
        win.refresh_hz = stated_refresh_hz

    return win
