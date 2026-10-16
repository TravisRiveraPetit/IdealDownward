"""A planner based on ideal membership.

Pipeline: Fast Downward's translator (in src/translate), stnf.py, sas2singular.py, Singular, and
plan extraction from the membership certificate. Every plan is checked by simulation.

Usage (from the repository root):
  ./ideal-downward.py domain.pddl problem.pddl
  ./ideal-downward.py output.sas

Exit codes: 0 plan found, 11 unsolvability proved, 12 no plan found although the initial state is
connected to a goal state (possible only if the task is not reversible), 1 error.
"""

import argparse
import collections
import os
import subprocess
import sys
import time

from sas import applicable, apply, read_task, satisfies, write_task
from sas2singular import binomials, ring_offsets, singular_script
from stnf import stnf_task

PLAN_FOUND, UNSOLVABLE, NO_PLAN = 0, 11, 12
TRANSLATOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "translate")


def translate(domain, problem, sas_file):
    python_path = os.pathsep.join(filter(None, [TRANSLATOR_DIR, os.environ.get("PYTHONPATH")]))
    subprocess.run([sys.executable, "-m", "fast_downward.translate", os.path.abspath(domain),
                    os.path.abspath(problem), "--sas-file", os.path.abspath(sas_file)],
                   env=dict(os.environ, PYTHONPATH=python_path), check=True, stdout=subprocess.DEVNULL)


def run_singular(singular, script, output):
    with open(output, "w") as out:
        subprocess.run([singular, "-q", "--ticks-per-sec=1000", script], stdout=out, check=True)
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
    return member, terms, stats


def certificate_graph(task, gens, terms):
    """The graph H of Proposition 10, restricted to state monomials."""
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


def bfs(start, successors, is_target, limit=None):
    """Breadth-first search. Returns the states of a path to a target and the labels of its edges."""
    parent = {start: None}
    queue = collections.deque([start])
    while queue:
        u = queue.popleft()
        if is_target(u):
            states, labels = [u], []
            while parent[u] is not None:
                u, label = parent[u]
                states.append(u)
                labels.append(label)
            return states[::-1], labels[::-1]
        for w, label in successors(u):
            if w not in parent:
                parent[w] = (u, label)
                queue.append(w)
        if limit is not None and len(parent) > limit:
            return None
    return None


def forward_steps(task, u, w, limit):
    """Operators that lead from u to w: a single operator if one exists, otherwise found by search.

    A search is needed only for a backward step whose operator cannot be undone by a single operator.
    """
    for op in task.operators:
        if applicable(op, u) and apply(op, u) == w:
            return [op]

    def successors(s):
        return [(apply(op, s), op) for op in task.operators if applicable(op, s)]

    found = bfs(u, successors, lambda s: s == w, limit)
    return None if found is None else found[1]


def main():
    parser = argparse.ArgumentParser(description="A planner based on ideal membership.")
    parser.add_argument("inputs", nargs="+", help="domain.pddl problem.pddl, or a SAS file")
    parser.add_argument("--singular", default=os.environ.get("SINGULAR", "Singular"),
                        help="Singular executable (default: $SINGULAR or Singular)")
    parser.add_argument("--workdir", default=".", help="directory for intermediate files")
    parser.add_argument("--plan-file", default="sas_plan")
    parser.add_argument("--max-operators", type=int, default=1_000_000,
                        help="maximum number of operators after the STNF transformation")
    parser.add_argument("--no-mutexes", action="store_true",
                        help="STNF without pruning the copies that violate mutexes")
    parser.add_argument("--search-limit", type=int, default=100_000,
                        help="maximum number of states per search for a backward step")
    args = parser.parse_args()
    if len(args.inputs) > 2:
        parser.error("expected domain and problem files, or one SAS file")

    start = time.time()

    def log(message):
        print(f"[{time.time() - start:8.2f}s] {message}", flush=True)

    os.makedirs(args.workdir, exist_ok=True)
    path = {ext: os.path.join(args.workdir, "output" + ext)
            for ext in (".sas", ".sas_stnf", ".sing", ".singular_out")}
    if len(args.inputs) == 2:
        translate(*args.inputs, path[".sas"])
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

    member, terms, stats = run_singular(args.singular, path[".sing"], path[".singular_out"])
    log(f"Singular: Groebner basis with {stats.get('BASIS_SIZE')} elements in {stats.get('STD_MS')} ms")
    if not member:
        log("result: the initial state is not connected to a goal state, so the task is unsolvable")
        return UNSOLVABLE
    log(f"result: the initial state is connected to a goal state "
        f"(certificate with {len(terms)} terms in {stats.get('LIFT_MS')} ms)")

    graph = certificate_graph(stnf, gens, terms)
    found = bfs(tuple(task.init), lambda u: ((w, None) for w in graph[u]), lambda u: satisfies(u, task.goal))
    if found is None:
        raise RuntimeError("the certificate does not connect the initial state to a goal state")
    states = found[0]
    plan = []
    for u, w in zip(states, states[1:]):
        steps = forward_steps(task, u, w, args.search_limit)
        if steps is None:
            log(f"no plan: a backward step on the path with {len(states) - 1} steps cannot be replaced "
                f"by forward steps (the task is not reversible, or the search limit is too small)")
            return NO_PLAN
        plan += steps

    state = tuple(task.init)
    for op in plan:
        assert applicable(op, state), f"plan is not applicable: {op.name}"
        state = apply(op, state)
    assert satisfies(state, task.goal), "plan does not reach the goal"
    cost = sum(op.cost for op in plan) if task.metric else len(plan)
    with open(args.plan_file, "w") as f:
        f.writelines(f"({op.name})\n" for op in plan)
        f.write(f"; cost = {cost} ({'general' if task.metric else 'unit'} cost)\n")
    log(f"plan with {len(plan)} steps (path with {len(states) - 1} steps), cost {cost}, written to {args.plan_file}")
    return PLAN_FOUND


if __name__ == "__main__":
    sys.exit(main())
