"""Check whether a task is invertible: every operator of its STNF is undone by a single operator.

An operator o' undoes an STNF operator o if, in every state in which o is applicable, o' is
applicable after o and restores the state. With prevail conditions counted on both sides, this
holds if and only if pre(o') is contained in eff(o), eff(o') is contained in pre(o), and o'
mentions every variable that o changes (for variables with at least two values). So o' changes
back exactly the variables that o changes. Invertible tasks are reversible, and for reversible
tasks the ideal test of the planner is exact.

Usage: python3 invertibility.py output.sas
"""

import argparse
import collections

from sas import read_task
from stnf import stnf_task


def pre_and_eff(op):
    pre, eff = dict(op.prevail), dict(op.prevail)
    for _, var, p, post in op.effects:
        pre[var] = p
        eff[var] = post
    return pre, eff


def operators_without_inverse(task):
    """The operators of an STNF task that no operator of the task undoes."""
    operators = []
    by_change = collections.defaultdict(list)
    for op in task.operators:
        pre, eff = pre_and_eff(op)
        change = frozenset((var, pre[var], eff[var]) for var in pre if pre[var] != eff[var])
        operators.append((op, eff, change))
        by_change[change].append(pre)
    missing = []
    for op, eff, change in operators:
        reverse = frozenset((var, post, pre) for var, pre, post in change)
        if not any(all(eff.get(var) == val for var, val in other_pre.items()) for other_pre in by_change[reverse]):
            missing.append(op)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sas_file")
    parser.add_argument("--max-operators", type=int, default=None)
    parser.add_argument("--no-mutexes", action="store_true", help="STNF without mutex pruning")
    args = parser.parse_args()
    task = stnf_task(read_task(args.sas_file), args.max_operators, use_mutexes=not args.no_mutexes)
    missing = operators_without_inverse(task)
    if not missing:
        print(f"invertible: all {len(task.operators)} STNF operators have an inverse")
    else:
        names = sorted({op.name for op in missing})
        print(f"not invertible: {len(missing)} of {len(task.operators)} STNF operators have no inverse, "
              f"from {len(names)} operators, for example: {', '.join(names[:5])}")


if __name__ == "__main__":
    main()
