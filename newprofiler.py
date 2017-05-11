# Python Profiler v3
# Copyright (c) 2015-2017 David R Walker

# TODO:
#   [x] Record only functions in StackLines
#   [ ] Handle per-line hotspots as separate structure (not nested) - ?
#   [ ] Handle timeline as separate structure
#   [x] Use unique stack IDs to dedupe stack tuples
#   [ ] Merge profile data method
#   [ ] add custom metadata values to profile data (e.g. url, op, user id) for filtering / grouping
#   [ ] filter/merge profile data by metadata
#   [x] Expose randomize parameter for stochastic sampling
#   [x] Add rate control (remove interval)
#         - is this more or less misleading if we don't adjust for profiler overhead to achieve rate?
#            - not adjusting for drift might be handy for estimating profiler performance/overheads
#   [x] Finish linux platform driver (get thread CPU times seems to be unfinished!!)
#   [ ] Windows platform driver
#   [ ] Tidy up platform drivers and make a nice platform choosing function
#   [ ] Convert into proper Python module + split into submodules
#   [ ] Basic (temp) dump function (flat) - replace with proper collated version from stack tree
#   [ ] Filter out long tail option (collate items with low ticks as 'Other') to remove noise
#   [ ] Post process to build stack/call graph (have exporters work from this graph instead of raw data) - ?
#   [ ] Record process ID in addition to thread?
#   [ ] Option to merge processes
#   [ ] Option to merge threads
#   [ ] Test performance / optimize on various platforms
#   [ ] Serialize (+append?) to file (lock file?)
#   [ ] Load from file
#   [ ] HTML5 exporter with drill-down
#   [ ] Import/exporter framework
#   [ ] Export to standard profiler formats (e.g. python, callgrind, firefox ThreadProfile json)
#   [ ] Make Python 3 compatible
#   [ ] Decorator to wrap a function with profiler
#   [ ] Function to watch a function in profiler? (e.g. store code object in dict and check)
#   [ ] Option to filter out standard (and custom) libraries? (path prefixes?)
#   [ ] Figure out how to play nicely with time.sleep(), etc. - do we need to patch it?
#           - EINTR / silent signal interrupts
#           - breaks sleep/timeout behaviour in programs - provide optional monkey patches?
#           - or just accept that signals break waits, and is fixed eventually by PEP475
#               ('serious' code should be handling EINTR anyway?)
#   [ ] Figure out how to avoid having to patch thread, wherever possible
#         - maybe spawn a test thread on module import to detect if thread IDs match ?
#   [x] Make interval private on profiler (or don't store)
#   [x] Move all running time stats etc. into _profile_data - already done


import os
import time
import random
from contextlib import contextmanager

# - Scheduler ------------------------------------------------------------------

# Base class for repeated periodic function call
class IntervalScheduler(object):

    default_rate = 1

    def __init__(self, interval_func, interval=0.01, stochastic=False, func_args=(), func_kwargs={}):
        self.interval = interval
        self._random = None
        if stochastic:
            # Our own Random to avoid side effects on shared PRNG
            self._random = random.Random()
        self._running = False
        self._interval_func = interval_func
        self._func_args = func_args
        self._func_kwargs = func_kwargs
        self._init()

    def start(self):
        if not self.is_running():
            self._start()
            self._running = True

    def stop(self):
        if self.is_running():
            self._stop()
            self._running = False

    def is_running(self):
        return self._running

    def get_next_interval(self):
        if self._random:
            return (2.0 * self._random.random() * self.interval)
        else:
            return self.interval

    def tick(self, frame):
        self._interval_func(*self._func_args, _interrupted_frame=frame, **self._func_kwargs)

    # Sub-classes should override the following methods to implement a scheduler
    # that will call self.tick() every self.interval seconds.
    # If the scheduler interupts a Python frame, it should pass the frame that was
    # interrupted to tick(), otherwise it should pass in None.
    def _init(self):
        pass

    def _start(self):
        raise NotImplementedError()

    def _stop(self):
        raise NotImplementedError()

# Uses a separate sleeping thread, which wakes periodically and calls self.tick()
class ThreadIntervalScheduler(IntervalScheduler):

    default_rate = 100

    def _init(self):
        import threading
        self._thread = None
        self._stopping = False
        self._event = threading.Event()

    def _start(self):
        import threading
        self._event.clear()
        def thread_func():
            while not self._event.is_set():
                self._event.wait(timeout=self.get_next_interval())
                self.tick(None)
        self._thread = threading.Thread(target=thread_func, name='profiler')
        self._thread.daemon = True
        self._thread.start()

    def _stop(self):
        self._event.set()
        self._thread.join()
        self._stopping = False


import signal


# Signals the main thread every interval, which calls the tick() method when 
# the timer event is triggered.
# Note that signal handlers are blocked during system calls, library calls, etc.
# in the main thread.
# We compensate for this by keeping track of real, user cpu, and system cpu
# usage between ticks on each thread.
# We prefer ITIMER_REAL, because that will be triggered immediately upon
# returning from a long-blocking system call, so we can add the ticks to the
# most appropriate function.
# However, if the main thread is blocked for a significant period, this will
# reduce the accuracy of samples in other threads, because only the main
# thread handles signals. In such situations, the ThreadIntervalScheduler might
# be more accurate.
# We don't specify an interval and reschedule the next tick ourselves. This
# allows us to dynamically change the sample interval to avoid aliasing, and
# prevents the signal interrupting itself, which can lead to stack errors,
# some strange behaviour when threads are being join()ed, and polluting the
# profile data with stack data from the profiler.
class SignalIntervalScheduler(IntervalScheduler):

    default_rate = 1000
    timer = signal.ITIMER_REAL
    signal = signal.SIGALRM

    def _start(self):
        def signal_handler(signum, frame):
            self.tick(frame)
            if self._run:
                signal.setitimer(self.timer, self.get_next_interval(), 0)
        signal.signal(self.signal, signal_handler)
        signal.siginterrupt(self.signal, False)
        self._run = True
        signal.setitimer(self.timer, self.get_next_interval(), 0)

    def _stop(self):
        self._run = False
        signal.setitimer(self.timer, 0, 0)

# - Platform-specific stuff ----------------------------------------------------

import thread
import threading

class ThreadPlatform(object):

    def __init__(self):
        self.name = ''
        self.lock = threading.Lock()
        self._registered_threads = {}
        self._original_start_new_thread = thread.start_new_thread
        self.platform_init()

    def _patch_thread(self):
        assert threading.current_thread().name == 'MainThread'
        with self.lock:
            self._registered_threads[threading.current_thread().ident] = self.get_current_thread_id()
        def start_new_thread_wrapper(func, args, kwargs={}):
            def thread_func(func, args, kwargs):
                system_tid = self.get_current_thread_id()
                with self.lock:
                    self._registered_threads[threading.current_thread().ident] = system_tid
                return func(*args, **kwargs)
            return self._original_start_new_thread(thread_func, (func, args, kwargs))
        thread.start_new_thread = start_new_thread_wrapper
        threading._start_new_thread = start_new_thread_wrapper

    def _unpatch_thread(self):
        with self.lock:
            self._registered_threads = {}
        thread.start_new_thread = _original_start_new_thread
        threading._start_new_thread = _original_start_new_thread

    def _get_patched_thread_id(self, python_ident):
        #with self.lock:
            return self._registered_threads.get(python_ident)

    def platform_init(self):
        raise NotImplementedError()

    def get_thread_id_from_python_ident(self, python_ident):
        raise NotImplementedError()

    def get_current_thread_id(self):
        raise NotImplementedError()

    def get_thread_cpu_time(self, thread_id=None):
        raise NotImplementedError()


# Single-threaded CPU times using os.times(),
# which actually gives CPU times for the whole
# process.
# Will give bad results if there are actually
# other threads running!
class SingleThreadedPlatform(ThreadPlatform):

    def platform_init(self):
        pass
    
    def get_thread_id_from_python_ident(self):
        return 0

    def get_current_thread_id(self):
        return 0

    def get_thread_cpu_time(self, thread_id=None):
        time_info = os.times()
        return time_info[0] + time_info[1]


class MacPThreadPlatform(ThreadPlatform):

    def platform_init(self):
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library('libc'))
        self._mach_thread_self = libc.mach_thread_self
        self._mach_thread_self.restype = ctypes.c_uint
        # TODO: check these field definitions
        class time_value_t(ctypes.Structure):
            _fields_ = [
                ("seconds", ctypes.c_int),
                ("microseconds",ctypes.c_int)
            ]
        class thread_basic_info(ctypes.Structure):
            _fields_ = [
                ("user_time", time_value_t),
                ("system_time",time_value_t),
                ("cpu_usage",ctypes.c_int),
                ("policy",ctypes.c_int),
                ("run_state",ctypes.c_int),
                ("flags",ctypes.c_int),
                ("suspend_count",ctypes.c_int),
                ("sleep_time",ctypes.c_int)
            ]
        thread_info = libc.thread_info
        thread_info.restype = ctypes.c_int
        thread_info.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(thread_basic_info),
            ctypes.POINTER(ctypes.c_uint)
        ]
        self._thread_info = thread_info
        self._THREAD_BASIC_INFO = 3
        self._out_info = thread_basic_info()
        self._count = ctypes.c_uint(ctypes.sizeof(self._out_info) / ctypes.sizeof(ctypes.c_uint))
        self._patch_thread()

    def get_thread_id_from_python_ident(self, python_ident):
        return self._get_patched_thread_id(python_ident)

    def get_current_thread_id(self):
        return self._mach_thread_self()

    def get_thread_cpu_time(self, python_ident=None):
        import ctypes
        # TODO: Optimize with shared structs, sizes, to minimize allocs per tick
        if python_ident is None:
            thread_id = self.get_current_thread_id()
        else:
            thread_id = self.get_thread_id_from_python_ident(python_ident)
        out_info = self._out_info
        result = self._thread_info(
            thread_id,
            self._THREAD_BASIC_INFO,
            ctypes.byref(out_info),
            ctypes.byref(self._count),
        )
        if result != 0:
            return 0.0
        user_time = out_info.user_time.seconds + out_info.user_time.microseconds / 1000000.0
        system_time = out_info.system_time.seconds + out_info.system_time.microseconds / 1000000.0
        return user_time + system_time
    

class LinuxPThreadPlatform(ThreadPlatform):

    def platform_init(self):
        import ctypes
        import ctypes.util

        pthread = ctypes.CDLL(ctypes.util.find_library('pthread'))
        libc = ctypes.CDLL(ctypes.util.find_library('c'))

        pthread_t = ctypes.c_ulong
        clockid_t = ctypes.c_long
        time_t = ctypes.c_long

        NANOSEC = 1.0 / 1e9

        CLOCK_THREAD_CPUTIME_ID = 3 # from linux/time.h

        class timespec(ctypes.Structure):
            _fields_ = [
                ('tv_sec', time_t),
                ('tv_nsec', ctypes.c_long),
            ]

        # wrap pthread_self()
        pthread_self = pthread.pthread_self
        pthread.argtypes = []
        pthread_self.restype = pthread_t

        # wrap pthread_getcpuclockid()
        pthread_getcpuclockid = pthread.pthread_getcpuclockid
        pthread_getcpuclockid.argtypes = [pthread_t, ctypes.POINTER(clockid_t)]
        pthread_getcpuclockid.restype = clockid_t

        # wrap clock_gettime()
        clock_gettime = libc.clock_gettime
        clock_gettime.argtypes = [clockid_t, ctypes.POINTER(timespec)]
        clock_gettime.restype = ctypes.c_int

        def get_current_thread_id():
            return pthread_self()

        def get_thread_cpu_time(thread_id=None):
            if thread_id is None:
                thread_id = pthread_self()

            # First, get the thread's CPU clock ID
            clock_id = clockid_t()
            error = pthread_getcpuclockid(thread_id, ctypes.pointer(clock_id))
            if error:
                return None

            # Now get time from clock...
            result = timespec()
            error = clock_gettime(clock_id, ctypes.pointer(result))
            if error:
                return None

            cpu_time = result.tv_sec + result.tv_nsec * NANOSEC
            
            return cpu_time

        self._get_current_thread_id = get_current_thread_id
        self._get_thread_cpu_time = get_thread_cpu_time

    def get_current_thread_id(self):
        return self._get_current_thread_id()

    def get_thread_cpu_time(thread_id=None):
        return self._get_thread_cpu_time(thread_id)
   

import sys
if sys.platform == 'darwin':
    thread_platform = MacPThreadPlatform()
elif sys.platform == 'linux':
    thread_platform = LinuxPThreadPlatform()
# TODO: Windows support
else:
    try:
        import thread
    except ImportError:
        pass
    else:
        import warnings
        warnings.warn('Multi-threaded CPU times not supported on this platform!')
    thread_platform = SingleThreadedPlatform()


# - Sample data ----------------------------------------------------------------

import collections

StackLine = collections.namedtuple('StackLine', ['type', 'name', 'file', 'line', 'data'])

def stack_line_from_frame(frame, stype='func', data=None):
    code = frame.f_code
    return StackLine(stype, code.co_name, code.co_filename, code.co_firstlineno, data)


class SampleData(object):

    __slots__ = ['rtime', 'cputime', 'ticks']

    def __init__(self):
        self.rtime = 0.0        # Real / wall-clock time
        self.cputime = 0.0      # User CPU time (single thread)
        self.ticks = 0          # Actual number of samples

    def __str__(self):
        return 'SampleData<r=%.3f, cpu=%.3f, t=%d>' % (
            self.rtime,
            self.cputime,
            self.ticks
        )

    def __repr__(self):
        return str(self)

class RawProfileData(object):

    def __init__(self):
        self.stack_line_id_map = {}     # Maps StackLines to IDs
        self.stack_tuple_id_map = {}    # Map tuples of StackLine IDs to IDs
        self.stack_data = {}            # Maps stack ID tuples to SampleData
        self.time_running = 0.0         # Total amount of time sampling has been active
        self.total_ticks = 0            # Total number of samples we've taken

    def add_sample_data(self, stack_list, rtime, cputime, ticks):
        sm = self.stack_line_id_map
        sd = self.stack_line_id_map.setdefault
        stack_tuple = tuple(
            sd(stack_line, len(sm))
            for stack_line in stack_list
        )

        stack_tuple_id = self.stack_tuple_id_map.setdefault(
            stack_tuple,
            len(self.stack_tuple_id_map),
        )

        if stack_tuple_id in self.stack_data:
            sample_data = self.stack_data[stack_tuple_id]
        else:
            sample_data = self.stack_data[stack_tuple_id] = SampleData()

        sample_data.rtime += rtime
        sample_data.cputime += cputime
        sample_data.ticks += ticks
        self.total_ticks += ticks

    def dump(self, sort='rtime'):
        assert sort in SampleData.__slots__
        # Quick util function to dump raw data in a vaguely-useful format
        # TODO: replace with proper text exporter with sort parameters, etc.
        print '%s:\n\n    %d samples taken in %.3fs:\n' % (
            self.__class__.__name__,
            self.total_ticks,
            self.time_running,
        )
        print '    Ordered by: %s\n' % sort
        # Invert stack -> ID map
        stack_line_map = dict([
            (v, k)
            for k, v
            in self.stack_line_id_map.items()
        ])
        stack_map = dict([
            (v, k)
            for k, v
            in self.stack_tuple_id_map.items()
        ])
        lines = [
            (getattr(sample_data, sort), stack_id, sample_data)
            for stack_id, sample_data
            in self.stack_data.items()
        ]
        lines.sort()
        lines.reverse()
        print '      ticks    rtime  cputime filename:lineno(function)'
        for _, stack_id, sample_data in lines:
            stack = stack_map[stack_id]
            stack_line = stack_line_map[stack[0]]
            print '    %7d % 8.3f % 8.3f %s:%d(%s) : %r' % (
                sample_data.ticks,
                sample_data.rtime,
                sample_data.cputime,
                os.path.basename(stack_line.file),
                stack_line.line,
                stack_line.name,
                stack,
            )
        print


class ThreadClock(object):

    __slots__ = ['rtime', 'cputime']

    def __init__(self):
        self.rtime = 0.0
        self.cputime = 0.0


class Profiler(object):

    _scheduler_map = {
        'signal':SignalIntervalScheduler,
        'thread':ThreadIntervalScheduler
    }


    def __init__(
        self,
        scheduler_type='signal',        # Which scheduler to use
        collect_stacks=True,            # Collect full call-tree data?
        rate=None,
        stochastic=False,
    ):
        self.collect_stacks = collect_stacks
        assert (
            scheduler_type in self._scheduler_map
            or isinstance(scheduler_type, IntervalScheduler)
        ), 'Unknown scheduler type'
        self.scheduler_type = scheduler_type
        if isinstance(scheduler_type, str):
            scheduler_type = self._scheduler_map[scheduler_type]
        if rate is None:
            rate = scheduler_type.default_rate
        self._scheduler = scheduler_type(
            self.sample,
            interval=1.0/rate,
            stochastic=stochastic,
        )
        self.reset()

    def reset(self):
        self._profile_data = RawProfileData()
        self._thread_clocks = {}        # Maps from thread ID to ThreadClock
        self._last_tick = 0
        self.total_samples = 0
        self.sampling_time = 0.0
        self._empty_stack = [StackLine(None, 'null', '', 0, None)]
        self._start_time = 0.0

    def sample(self, _interrupted_frame=None):
        sample_time = time.time()
        current_frames = sys._current_frames()
        current_thread = thread.get_ident()
        for thread_ident, frame in current_frames.items():
            if thread_ident == current_thread:
                frame = _interrupted_frame
            if frame is not None:
                # 1.7 %
                stack = [stack_line_from_frame(frame)]
                if self.collect_stacks:
                    frame = frame.f_back
                    while frame is not None:
                        stack.append(stack_line_from_frame(frame))
                        frame = frame.f_back

                stack.append(StackLine('thread', str(thread_ident), '', 0, None))  # todo: include thread name?
                # todo: include PID?
                # todo: include custom metadata/labels?

                # 2.0 %
                if thread_ident in self._thread_clocks:
                    thread_clock = self._thread_clocks[thread_ident]
                    cputime = thread_platform.get_thread_cpu_time(thread_ident)
                else:
                    thread_clock = self._thread_clocks[thread_ident] = ThreadClock()
                    cputime = thread_platform.get_thread_cpu_time(thread_ident)
                # ~5.5%
                self._profile_data.add_sample_data(
                    stack,
                    sample_time - self.last_tick,
                    cputime - thread_clock.cputime,
                    1
                )
                thread_clock.cputime = cputime
            else:
                self._profile_data.add_sample_data(
                    self._empty_stack, sample_time - self.last_tick, 0.0, 1
                )
        self.last_tick = sample_time
        self.total_samples += 1
        self.sampling_time += time.time() - sample_time


    def start(self):
        import threading
        # reset thread clocks...
        self._thread_clocks = {}
        for thread in threading.enumerate():
            thread_clock = ThreadClock()
            self._thread_clocks[thread.ident] = thread_clock
            cputime = thread_platform.get_thread_cpu_time(thread.ident)
            thread_clock.cputime = cputime
        self._start_time = self.last_tick = time.time()
        self._scheduler.start()

    @contextmanager
    def activated(self):
        try:
            self.start()
            yield self
        finally:
            self.stop()

    def stop(self):
        self._scheduler.stop()
        self._profile_data.time_running += time.time() - self._start_time
        self._start_time = 0.0



def busy(rate=100):
    import time
    profiler = Profiler(rate=rate)
    with profiler.activated():
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return profiler
