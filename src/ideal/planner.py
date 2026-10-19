"""A planner based on ideal membership.

Pipeline: Fast Downward's translator (in src/translate), stnf.py, sas2singular.py, Singular, and
plan extraction from the membership certificate. Every plan is checked by simulation.

Usage (from the repository root):
  ./ideal-downward.py domain.pddl problem.pddl
  ./ideal-downward.py output.sas

Exit codes, following Fast Downward where possible:
   0  plan found
  11  unsolvability proved
  12  no plan found although the monomial of the initial state lies in the ideal (possible only if
      the task is not undirected)
  20  the translator ran out of memory
  21  the translator ran out of time
  22  Singular or the planner ran out of memory
  23  Singular or the planner ran out of time
   1  other errors, for example an unsupported task
"""

import argparse
import collections
import math
import os
import resource
import signal
import subprocess
import sys
import time

from sas import applicable, apply, read_task, satisfies, write_task
from sas2singular import binomials, ring_offsets, singular_script
from stnf import stnf_task

PLAN_FOUND, UNSOLVABLE, NO_PLAN = 0, 11, 12
TRANSLATE_OUT_OF_MEMORY, TRANSLATE_OUT_OF_TIME = 20, 21
OUT_OF_MEMORY, OUT_OF_TIME = 22, 23
TRANSLATOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "translate")
MIN_MEMORY = 64 * 2**20     # bytes that a subprocess gets at least


class Limits:
    """An overall CPU time and memory budget for the planner and the programs that it calls.

    As with the --overall-*-limit options of Fast Downward's driver, every subprocess gets the CPU
    time that remains. It also gets only the memory that the planner itself does not use, so that
    the planner and Singular together stay within the memory limit.
    """

    def __init__(self, time_limit=None, memory_limit=None):
        self.time_limit = time_limit
        self.memory_limit = None if memory_limit is None else memory_limit * 2**20
        if self.memory_limit is not None:
            _lower_limit(resource.RLIMIT_AS, self.memory_limit)
        if self.time_limit is not None:
            _lower_limit(resource.RLIMIT_CPU, math.ceil(self.time_limit))

    def remaining_time(self):
        used = sum(getattr(resource.getrusage(who), field)
                   for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
                   for field in ("ru_utime", "ru_stime"))
        return self.time_limit - used

    def remaining_memory(self):
        return self.memory_limit - _virtual_memory()

    def check(self):
        """None if a subprocess can still be started, otherwise the exit code for the exhausted limit."""
        if self.time_limit is not None and self.remaining_time() < 1:
            return OUT_OF_TIME
        if self.memory_limit is not None and self.remaining_memory() < MIN_MEMORY:
            return OUT_OF_MEMORY
        return None

    def restrict_self(self):
        """Lower the CPU time limit of the planner itself to the time that remains overall."""
        if self.time_limit is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            _lower_limit(resource.RLIMIT_CPU,
                         max(1, math.ceil(usage.ru_utime + usage.ru_stime + self.remaining_time())))

    def for_subprocess(self):
        """A function that sets the remaining budget as limits of a subprocess before it starts."""
        time_left = None if self.time_limit is None else math.ceil(self.remaining_time())
        memory_left = None if self.memory_limit is None else self.remaining_memory()

        def set_limits():
            if time_left is not None:
                _lower_limit(resource.RLIMIT_CPU, time_left)
            if memory_left is not None:
                _lower_limit(resource.RLIMIT_AS, memory_left)
        return set_limits


def _lower_limit(kind, value):
    soft, hard = resource.getrlimit(kind)
    if soft == resource.RLIM_INFINITY or value < soft:
        resource.setrlimit(kind, (value, hard))


def _virtual_memory():
    """The virtual memory of this process in bytes, which is what RLIMIT_AS limits (0 without /proc)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def translate(domain, problem, sas_file, limits):
    """Run Fast Downward's translator. Returns None or an exit code."""
    python_path = os.pathsep.join(filter(None, [TRANSLATOR_DIR, os.environ.get("PYTHONPATH")]))
    process = subprocess.run(
        [sys.executable, "-m", "fast_downward.translate", os.path.abspath(domain),
         os.path.abspath(problem), "--sas-file", os.path.abspath(sas_file)],
        env=dict(os.environ, PYTHONPATH=python_path), stdout=subprocess.DEVNULL,
        preexec_fn=limits.for_subprocess())
    if process.returncode in (0, TRANSLATE_OUT_OF_MEMORY, TRANSLATE_OUT_OF_TIME):
        return process.returncode or None
    if process.returncode == -signal.SIGXCPU:
        return TRANSLATE_OUT_OF_TIME
    if process.returncode == -signal.SIGKILL:
        return TRANSLATE_OUT_OF_MEMORY
    raise RuntimeError(f"the translator failed with exit code {process.returncode}")


def run_singular(singular, script, output, limits):
    """Run Singular. Returns an exit code, or None and the answer, the certificate, and statistics."""
    with open(output, "w") as out:
        process = subprocess.run([singular, "-q", "--ticks-per-sec=1000", script], stdout=out,
                                 preexec_fn=limits.for_subprocess())
    if process.returncode == -signal.SIGXCPU:
        return OUT_OF_TIME, None, None, None
    # Singular stops with "halt 14" when it cannot get more memory. SIGKILL comes from outside,
    # for example from Slurm when the task exceeds its memory.
    if process.returncode in (14, -signal.SIGKILL):
        return OUT_OF_MEMORY, None, None, None
    if process.returncode != 0:
        raise RuntimeError(f"Singular failed with exit code {process.returncode}, see {output}")
    member, terms, stats = None, [], {}
    for line in open(output):
        words = line.split()
        if not words:
            continue
        if words[0] == "MEMBER":
            member = words[1] == "1"
        elif words[0] == "TERM":
            terms.append((int(words[1]), [int(e) for e in words[2].split(",")]))
        elif words[0] in ("STD_MS", "LIFT_MS", "BASIS_SIZE"):
            stats[words[0]] = int(words[1])
    if member is None:
        raise RuntimeError(f"Singular gave no answer, see {output}")
    return None, member, terms, stats


def certificate_graph(task, gens, terms):
    """The graph H of the certificate (see the paper), restricted to state monomials."""
    offsets, n = ring_offsets(task)
    fact = [None] * n
    for var, v in enumerate(task.variables):
        for val in range(len(v.values)):
            fact[offsets[var] + val] = (var, val)

    def state(exponents):
        values = [None] * len(task.variables)
        for i, e in enumerate(exponents):
            if e:
                var, val = fact[i]
                if e > 1 or values[var] is not None:
                    return None
                values[var] = val
        return None if None in values else tuple(values)

    graph = collections.defaultdict(set)
    for j, exponents in terms:
        if j > len(gens):          # a term of the coefficient of the goal monomial
            continue
        pre, eff = gens[j - 1]
        u, w = list(exponents), list(exponents)
        for var, val in pre:
            u[offsets[var] + val] += 1
        for var, val in eff:
            w[offsets[var] + val] += 1
        u, w = state(u), state(w)
        if u is not None and w is not None:
            graph[u].add(w)
            graph[w].add(u)
    return graph


def bfs(graph, start, is_target):
    """Breadth-first search in a graph given as adjacency sets. Returns the states of a shortest
    path from start to a target, or None."""
    parent = {start: None}
    queue = collections.deque([start])
    while queue:
        u = queue.popleft()
        if is_target(u):
            states = [u]
            while parent[u] is not None:
                u = parent[u]
                states.append(u)
            return states[::-1]
        for w in graph[u]:
            if w not in parent:
                parent[w] = u
                queue.append(w)
    return None


def operator_between(task, u, w):
    """An operator that leads from state u to state w, or None if there is none.

    A step of a path in the certificate graph applies the operator of its edge forwards or
    backwards. In an undirected task, a backward step is undone by an inverse of the operator.
    """
    for op in task.operators:
        if applicable(op, u) and apply(op, u) == w:
            return op
    return None


def plan(args, log):
    limits = Limits(args.time_limit, args.memory_limit)
    os.makedirs(args.workdir, exist_ok=True)
    path = {ext: os.path.join(args.workdir, "output" + ext)
            for ext in (".sas", ".sas_stnf", ".sing", ".singular_out")}
    if len(args.inputs) == 2:
        status = translate(*args.inputs, path[".sas"], limits)
        if status is not None:
            log(f"the translator ran out of {'memory' if status == TRANSLATE_OUT_OF_MEMORY else 'time'}")
            return status
        sas_file = path[".sas"]
    else:
        sas_file = args.inputs[0]
    task = read_task(sas_file)
    log(f"task: {len(task.variables)} variables, {len(task.operators)} operators")

    stnf = stnf_task(read_task(sas_file), args.max_operators, use_mutexes=not args.no_mutexes)
    write_task(stnf, path[".sas_stnf"])
    gens = binomials(stnf)
    with open(path[".sing"], "w") as f:
        f.write(singular_script(stnf, gens))
    _, n = ring_offsets(stnf)
    log(f"STNF: {len(stnf.operators)} operators, {len(gens)} binomials in {n} ring variables")

    status = limits.check()
    if status is None:
        status, member, terms, stats = run_singular(args.singular, path[".sing"], path[".singular_out"], limits)
    if status is not None:
        log(f"Singular ran out of {'memory' if status == OUT_OF_MEMORY else 'time'}")
        return status
    limits.restrict_self()
    log(f"Singular: Groebner basis with {stats.get('BASIS_SIZE')} elements in {stats.get('STD_MS')} ms")
    if not member:
        log("result: the monomial of the initial state is not in the ideal, so the task is unsolvable")
        return UNSOLVABLE
    log(f"result: the monomial of the initial state is in the ideal "
        f"(certificate with {len(terms)} terms in {stats.get('LIFT_MS')} ms)")

    graph = certificate_graph(stnf, gens, terms)
    states = bfs(graph, tuple(task.init), lambda u: satisfies(u, task.goal))
    if states is None:
        raise RuntimeError("the certificate does not connect the initial state to a goal state")
    steps = []
    for u, w in zip(states, states[1:]):
        op = operator_between(task, u, w)
        if op is None:
            log(f"no plan: a backward step on the path with {len(states) - 1} steps has no inverse "
                f"operator, so the task is not undirected")
            return NO_PLAN
        steps.append(op)

    state = tuple(task.init)
    for op in steps:
        assert applicable(op, state), f"plan is not applicable: {op.name}"
        state = apply(op, state)
    assert satisfies(state, task.goal), "plan does not reach the goal"
    cost = sum(op.cost for op in steps) if task.metric else len(steps)
    with open(args.plan_file, "w") as f:
        f.writelines(f"({op.name})\n" for op in steps)
        f.write(f"; cost = {cost} ({'general' if task.metric else 'unit'} cost)\n")
    log(f"plan with {len(steps)} steps (path with {len(states) - 1} steps), cost {cost}, "
        f"written to {args.plan_file}")
    return PLAN_FOUND


def main():
    parser = argparse.ArgumentParser(description="A planner based on ideal membership.")
    parser.add_argument("inputs", nargs="+", help="domain.pddl problem.pddl, or a SAS file")
    parser.add_argument("--singular", default=os.environ.get("SINGULAR", "Singular"),
                        help="Singular executable (default: $SINGULAR or Singular)")
    parser.add_argument("--workdir", default=".", help="directory for intermediate files")
    parser.add_argument("--plan-file", default="sas_plan")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="CPU time limit in seconds for the planner and all programs it calls")
    parser.add_argument("--memory-limit", type=int, default=None,
                        help="memory limit in MiB for the planner and all programs it calls")
    parser.add_argument("--max-operators", type=int, default=1_000_000,
                        help="maximum number of operators after the STNF transformation")
    parser.add_argument("--no-mutexes", action="store_true",
                        help="STNF without pruning the copies that violate mutexes")
    args = parser.parse_args()
    if len(args.inputs) > 2:
        parser.error("expected domain and problem files, or one SAS file")

    start = time.time()

    def log(message):
        print(f"[{time.time() - start:8.2f}s] {message}", flush=True)

    def out_of_time(signum, frame):
        # sys.exit() is not safe in signal handlers, so we exit directly, like Fast Downward's translator.
        print("the planner ran out of time", flush=True)
        os._exit(OUT_OF_TIME)

    signal.signal(signal.SIGXCPU, out_of_time)
    try:
        return plan(args, log)
    except MemoryError:
        log("the planner ran out of memory")
        return OUT_OF_MEMORY


if __name__ == "__main__":
    sys.exit(main())
