"""Transform a SAS+ task into simple transition normal form (STNF), following rules R1-R3 of the paper.

In the SAS+ format, a prevail condition is a precondition on a variable that the operator does not
change. Rule R1 reads it as the same fact in precondition and effect, so prevail conditions stay as
they are. Rule R2 splits every operator by the current values of the variables that it changes
without a precondition on them, and rule R3 removes the copies that change nothing. In the output,
every effect has a precondition on its variable, and the copies keep the name of their operator.

Usage: python3 stnf.py output.sas output.sas_stnf
"""

import argparse
import itertools
import sys

from sas import Operator, check_supported, read_task, write_task


def stnf_operators(task, max_operators=None):
    result = []
    for op in task.operators:
        constrained = [(var, pre, post) for _, var, pre, post in op.effects if pre != -1]
        free = [(var, post) for _, var, pre, post in op.effects if pre == -1]
        domains = [range(len(task.variables[var].values)) for var, _ in free]
        for current in itertools.product(*domains):                   # R2
            prevail = list(op.prevail)
            effects = []
            for var, pre, post in constrained + [(var, val, post) for (var, post), val in zip(free, current)]:
                if pre == post:
                    prevail.append((var, pre))      # this copy does not change var
                else:
                    effects.append(([], var, pre, post))
            if effects:                                                 # R3
                result.append(Operator(op.name, sorted(prevail), sorted(effects, key=lambda e: e[1]), op.cost))
        if max_operators is not None and len(result) > max_operators:
            raise ValueError(f"STNF has more than {max_operators} operators")
    return result


def stnf_task(task, max_operators=None):
    check_supported(task)
    task.operators = stnf_operators(task, max_operators)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sas_file")
    parser.add_argument("stnf_file")
    parser.add_argument("--max-operators", type=int, default=None)
    args = parser.parse_args()
    task = read_task(args.sas_file)
    before = len(task.operators)
    stnf_task(task, args.max_operators)
    write_task(task, args.stnf_file)
    print(f"STNF: {before} operators -> {len(task.operators)} operators", file=sys.stderr)


if __name__ == "__main__":
    main()
