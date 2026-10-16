"""Transform a SAS+ task into simple transition normal form (STNF), following rules R1-R3 of the paper.

In the SAS+ format, a prevail condition is a precondition on a variable that the operator does not
change. Rule R1 reads it as the same fact in precondition and effect, so prevail conditions stay as
they are. Rule R2 splits every operator by the current values of the variables that it changes
without a precondition on them, and rule R3 removes the copies that change nothing. In the output,
every effect has a precondition on its variable, and the copies keep the name of their operator.

By default, rule R2 also skips copies whose precondition contains two facts of a mutex group of
the task. Such copies are never applicable in a reachable state, so every transition from a
reachable state is kept, and solvability does not change. Only transitions between unreachable
states disappear. Without this pruning, for example, put-down(a) in Blocksworld gets copies for
states in which the hand holds a and is empty at the same time, and these copies have no inverse.

Usage: python3 stnf.py output.sas output.sas_stnf
"""

import argparse
import collections
import itertools
import sys

from sas import Operator, check_supported, read_task, write_task


def mutex_groups_of_facts(task):
    groups = collections.defaultdict(list)
    for index, group in enumerate(task.mutexes):
        for fact in group:
            groups[tuple(fact)].append(index)
    return groups


def violates_mutex(facts, groups):
    seen = set()
    for fact in facts:
        for index in groups.get(fact, ()):
            if index in seen:
                return True
            seen.add(index)
    return False


def stnf_operators(task, max_operators=None, use_mutexes=True):
    groups = mutex_groups_of_facts(task) if use_mutexes else {}
    result = []
    for op in task.operators:
        constrained = [(var, pre, post) for _, var, pre, post in op.effects if pre != -1]
        free = [(var, post) for _, var, pre, post in op.effects if pre == -1]
        domains = [range(len(task.variables[var].values)) for var, _ in free]
        for current in itertools.product(*domains):                   # R2
            pre_post = constrained + [(var, val, post) for (var, post), val in zip(free, current)]
            if groups and violates_mutex(op.prevail + [(var, pre) for var, pre, _ in pre_post], groups):
                continue
            prevail = list(op.prevail)
            effects = []
            for var, pre, post in pre_post:
                if pre == post:
                    prevail.append((var, pre))      # this copy does not change var
                else:
                    effects.append(([], var, pre, post))
            if effects:                                                 # R3
                result.append(Operator(op.name, sorted(prevail), sorted(effects, key=lambda e: e[1]), op.cost))
        if max_operators is not None and len(result) > max_operators:
            raise ValueError(f"STNF has more than {max_operators} operators")
    return result


def stnf_task(task, max_operators=None, use_mutexes=True):
    check_supported(task)
    task.operators = stnf_operators(task, max_operators, use_mutexes)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sas_file")
    parser.add_argument("stnf_file")
    parser.add_argument("--max-operators", type=int, default=None)
    parser.add_argument("--no-mutexes", action="store_true", help="keep copies that violate mutexes")
    args = parser.parse_args()
    task = read_task(args.sas_file)
    before = len(task.operators)
    stnf_task(task, args.max_operators, use_mutexes=not args.no_mutexes)
    write_task(task, args.stnf_file)
    print(f"STNF: {before} operators -> {len(task.operators)} operators", file=sys.stderr)


if __name__ == "__main__":
    main()
