# Python Profiler v3
# Copyright (c) 2015-2018 David R Walker

# TODO:
#   [x] Record only functions in StackLines
#   [ ] Option to track individual lines within call stacks - ?
#   [ ] Handle per-line hotspots as separate structure (not nested) - ?
#   [ ] Handle timeline as separate structure [ (timestamp, sample_data), ... ]
#   [x] Use unique stack IDs to dedupe stack tuples
#   [x] Merge profile data method
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
#   [ ] HTML5 exporter with drill-down / (or could export json/xml and rely on existing viewers)
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
#           - or encourage running profiled code in separate thread
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

    def get_thread_cpu_time(self, thread_id=None):
        return self._get_thread_cpu_time(thread_id)
   

import sys
if sys.platform.startswith('darwin'):
    thread_platform = MacPThreadPlatform()
elif sys.platform.startswith('linux'):
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

def stack_line_from_frame(frame, stype='call', data=None):
    code = frame.f_code
    if stype == 'line':
        return StackLine(stype, code.co_name, code.co_filename, frame.f_lineno, data)
    return StackLine(stype, code.co_name, code.co_filename, code.co_firstlineno, data)


class SampleData(object):

    __slots__ = ['rtime', 'cputime', 'ticks']

    def __getstate__(self):
        return (self.rtime, self.cputime, self.ticks)

    def __setstate__(self, data):
        self.rtime, self.cputime, self.ticks = data

    def __init__(self, rtime=0.0, cputime=0.0, ticks=0):
        self.rtime = rtime      # Real / wall-clock time
        self.cputime = cputime  # User CPU time (single thread)
        self.ticks = ticks      # Actual number of samples

    def __str__(self):
        return 'SampleData<r=%.3f, cpu=%.3f, t=%d>' % (
            self.rtime,
            self.cputime,
            self.ticks
        )

    def __repr__(self):
        return str(self)

    def merge(self, other_data):
        self.rtime += other_data.rtime
        self.cputime += other_data.cputime
        self.ticks += other_data.ticks


class RawProfileData(object):

    def __init__(self):
        self.stack_line_id_map = {}     # Maps StackLines to IDs
        self.stack_tuple_id_map = {}    # Map tuples of StackLine IDs to IDs
        self.stack_data = {}            # Maps stack tuple IDs to SampleData
        self.time_running = 0.0         # Total amount of time sampling has been active
        self.total_ticks = 0            # Total number of samples we've taken

    def merge(self, other_data):
        assert isinstance(other_data, RawProfileData)

        # Merge stack lines, reusing our IDs where possible
        trans_line = {}
        for stack_line, old_id in other_data.stack_line_id_map.items():
            trans_line[old_id] = self.stack_line_id_map.setdefault(
                stack_line,
                len(self.stack_line_id_map),
            )

        # Merge stack tuples, translating stack line IDs
        trans_tuple = {}
        for old_stack_tuple, old_tuple_id in other_data.stack_tuple_id_map.items():
            new_tuple = tuple([
                trans_line[old_line_id]
                for old_line_id
                in old_stack_tuple
            ])
            trans_tuple[old_tuple_id] = self.stack_tuple_id_map.setdefault(
                new_tuple,
                len(self.stack_tuple_id_map),
            )
        
        # Merge sample data, translating stack tuple IDs to use new IDs
        for other_stack_tuple_id, other_sample_data in other_data.stack_data.items():
            new_tuple_id = trans_tuple[other_stack_tuple_id]
            if new_tuple_id in self.stack_data:
                sample_data = self.stack_data[new_tuple_id]
            else:
                self.stack_data[new_tuple_id] = sample_data = SampleData()
            sample_data.merge(other_sample_data)

        # Merge totals
        self.time_running += other_data.time_running
        self.total_ticks += other_data.total_ticks

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

    def __getstate__(self):
        return (self.rtime, self.cputime)

    def __setstate__(self, data):
        self.rtime, self.cputime = data


class Profiler(object):

    _scheduler_map = {
        'signal':SignalIntervalScheduler,
        'thread':ThreadIntervalScheduler
    }


    def __init__(
        self,
        scheduler_type='signal',        # Which scheduler to use
        collect_lines=True,             # Collect current line data?
        collect_calls=True,             # Collect call data?
        collect_stacks=True,            # Collect full call-tree data? (implies collect_calls)
        rate=None,
        stochastic=False,
    ):
        self.collect_lines = collect_lines
        self.collect_calls = collect_calls or collect_stacks
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

        # TODO: find cheaper way to get thread names
        # Maybe merge all unnamed (Thread-N) threads together?
        thread_names = dict([(t.ident, t.getName()) for t in threading.enumerate()])

        for thread_ident, frame in current_frames.items():
            if thread_ident == current_thread:
                frame = _interrupted_frame
            if frame is not None:
                stack = []
                if self.collect_lines:
                    stack.append(stack_line_from_frame(frame, stype='line'))
                if self.collect_calls:
                    stack.append(stack_line_from_frame(frame))
                    if self.collect_stacks:
                        frame = frame.f_back
                        while frame is not None:
                            stack.append(stack_line_from_frame(frame))
                            frame = frame.f_back

                stack.append(StackLine('thread', thread_names.get(thread_ident, 'other'), '', 0, None))
                # todo: include custom metadata/labels?

                if thread_ident in self._thread_clocks:
                    thread_clock = self._thread_clocks[thread_ident]
                    cputime = thread_platform.get_thread_cpu_time(thread_ident)
                else:
                    thread_clock = self._thread_clocks[thread_ident] = ThreadClock()
                    cputime = thread_platform.get_thread_cpu_time(thread_ident)
                self._profile_data.add_sample_data(
                    stack,
                    sample_time - self.last_tick,
                    cputime - thread_clock.cputime,
                    1
                )
                thread_clock.cputime = cputime

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

    def get_profile_data(self):
        return self._profile_data


class CallGraphNode(object):

    def __init__(self):
        self.stackline = None
        self.sample_data = None
        self.aggregate_sample_data = None
        self.parent = None
        self.children = {}  # Maps stackline ID to child node

    def get_child_by_id(self, id):
        return self.children.get(id)

    def add_child(self, child, child_id):
        assert isinstance(child, CallGraphNode)
        if child_id in self.children:
            raise ValueError('Child ID already taken', child_id)
        self.children[child_id] = child
        child.parent = self

    def set_sample_data(self, sample_data):
        assert self.sample_data is None

        self.sample_data = sample_data

        # Update aggregate data
        ancestor = self.parent
        while ancestor is not None:
            if ancestor.aggregate_sample_data is None:
                ancestor.aggregate_sample_data = SampleData()
            ancestor.aggregate_sample_data.merge(self.sample_data)
            ancestor = ancestor.parent

    def dump(self, depth=0):
        import linecache
        if self.aggregate_sample_data is not None:
            agg_ticks = ' (%r)' % self.aggregate_sample_data
        else:
            agg_ticks = ''
        print '%s%s %r%s' % (
            '  ' * depth,
            '<%s> %s:%d(%s)' % (
                self.stackline.type,
                os.path.basename(self.stackline.file),
                self.stackline.line,
                self.stackline.name
            ),
            self.sample_data,
            agg_ticks
        ), '' if self.stackline.type != 'line' else linecache.getline(self.stackline.file, self.stackline.line).rstrip()
        for child in self.children.values():
            child.dump(depth=depth + 1)


class CallGraph(object):

    # Context-sensitive call graph (DAG)
    def __init__(self, profile_data):
        self.profile_data = profile_data
        self.root = self._build_graph_from_profile_data(profile_data)

    def get_profile_data(self):
        return self.profile_data

    def _build_graph_from_profile_data(self, profile_data):
        stack_lines = {}
        for line, line_id in profile_data.stack_line_id_map.items():
            assert line_id not in stack_lines, 'Ambiguous stack line ID'
            stack_lines[line_id] = line

        stacks = {}
        for stack, stack_id in profile_data.stack_tuple_id_map.items():
            assert stack_id not in stacks, 'Ambiguous stack tuple ID'
            stacks[stack_id] = stack

        root = CallGraphNode()
        root.stackline = StackLine('root', 'Root', '', 0, None)
        root.sample_data = None

        for stack_id, sample_data in profile_data.stack_data.items():
            stack_tuple = stacks[stack_id]
            node = root
            for stack_line_id in reversed(stack_tuple):
                child = node.get_child_by_id(stack_line_id)
                if child is None:
                    child = CallGraphNode()
                    child.stackline = stack_lines[stack_line_id]
                    node.add_child(child, stack_line_id)
                node = child
            node.set_sample_data(sample_data)

        return root


class Exporter(object):

    def __init__(self, call_graph, file):
        assert isinstance(call_graph, CallGraph)
        self.call_graph = call_graph
        if type(file) is str:
            self.file = open(file, 'w')
        else:
            self.file = file
        self.init_exporter()

    def init_exporter(self):
        pass

    def export(self):
        raise NotImplementedError()


class HTMLExporter(Exporter):

    html_header = """
        <html>
            <head>
                <style>
                    table {
                        width: 100%%;
                        border: 1px solid black;
                    }
                    td {
                        border: 1px solid black;
                    }
                </style>
            </head>
            <body>
                <table>
                    <thead>
                        <tr>
                            <th>&nbsp;</th>
                            <th colspan="3">Tree</th>
                            <th colspan="3">Node</th>
                            <th>&nbsp;</th>
                        </tr>
                        <tr>
                            <th>Call Graph</th>
                            <th>rtime</th>
                            <th>cputime</th>
                            <th>ticks</th>
                            <th>rtime</th>
                            <th>cputime</th>
                            <th>ticks</th>
                            <th>Code</th>
                        </tr>
                    </thead>
                    <tbody>
                        <h1>Context-Sensitive Call Graph</h1>
                        <p>Captured %d samples in %.3f s.</p>
    """

    row_template = """
        <tr id="%(row_id)s">
            <td>%(node)s</td>
            %(aggregate_sample_data)s
            %(node_sample_data)s
            <td><pre><code>%(code)s</code></pre></td>
        </tr>
    """

    sample_data_template = """
        <td>%(rtime).3f</td>
        <td>%(cputime).3f</td>
        <td>%(ticks)d</td>
    """

    empty_sample_data = """
        <td colspan="3"></td>
    """

    html_footer = """
                    </tbody>
                </table>
            </body>
        </html>
    """

    def init_exporter(self):
        self._count = 0

    def export(self):
        profile_data = self.call_graph.get_profile_data()
        self.file.write(
            self.html_header % (
                profile_data.total_ticks,
                profile_data.time_running,
            )
        )
        
        self._export_node(self.call_graph.root)

        self.file.write(
            self.html_footer
        )

    def _get_uid(self):
        self._count += 1
        return self._count

    def _export_node(self, node, _level=0):
        import linecache

        assert isinstance(node, CallGraphNode), node

        row_data = {
            'total_rtime': 0.0,
            'total_cputime': 0.0,
            'total_ticks': 0,
            'rtime': 0.0,
            'cputime': 0.0,
            'ticks': 0,
        }

        row_data['row_id'] = 'node_%d' % self._get_uid()

        row_data['node'] = '%s&lt;%s&gt; %s:%d(%s)' % (
            '&nbsp;&nbsp;' * _level,
            node.stackline.type,
            os.path.basename(node.stackline.file),
            node.stackline.line,
            node.stackline.name
        )

        if node.aggregate_sample_data is not None:
            row_data['aggregate_sample_data'] = self.sample_data_template % {
                'rtime': node.aggregate_sample_data.rtime,
                'cputime': node.aggregate_sample_data.cputime,
                'ticks': node.aggregate_sample_data.ticks,
            }
        else:
            row_data['aggregate_sample_data'] = self.empty_sample_data

        if node.sample_data is not None:
            row_data['node_sample_data'] = self.sample_data_template % {
                'rtime': node.sample_data.rtime,
                'cputime': node.sample_data.cputime,
                'ticks': node.sample_data.ticks,
            }
        else:
            row_data['node_sample_data'] = self.empty_sample_data
        if node.stackline.type == 'line':
            row_data['code'] = linecache.getline(
                node.stackline.file,
                node.stackline.line,
            ).rstrip()
        else:
            row_data['code'] = '' 

        self.file.write(self.row_template % row_data)

        # TODO: apply sort criterion here
        for child in node.children.values():
            self._export_node(child, _level=_level + 1)


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
