"""Reading, writing, and simulating SAS+ tasks in Fast Downward's output format (version 3)."""

from dataclasses import dataclass


@dataclass
class Variable:
    name: str
    axiom_layer: int
    values: list


@dataclass
class Operator:
    name: str
    prevail: list   # (var, val): precondition on a variable that the operator does not change
    effects: list   # (conditions, var, pre, post) with conditions a list of (var, val); pre = -1: no precondition
    cost: int


@dataclass
class Task:
    metric: int
    variables: list
    mutexes: list   # each a list of (var, val)
    init: list
    goal: list      # (var, val)
    operators: list
    axioms: list    # raw lines of each axiom rule


class _Reader:
    def __init__(self, text):
        self.lines = text.splitlines()
        self.pos = 0

    def line(self):
        self.pos += 1
        return self.lines[self.pos - 1]

    def expect(self, word):
        found = self.line().strip()
        if found != word:
            raise ValueError(f"line {self.pos}: expected {word!r}, found {found!r}")

    def int(self):
        return int(self.line())

    def ints(self):
        return [int(x) for x in self.line().split()]


def read_task(path):
    r = _Reader(open(path).read())
    r.expect("begin_version")
    version = r.int()
    r.expect("end_version")
    if version != 3:
        raise ValueError(f"unsupported SAS file version {version}")
    r.expect("begin_metric")
    metric = r.int()
    r.expect("end_metric")

    variables = []
    for _ in range(r.int()):
        r.expect("begin_variable")
        name = r.line()
        axiom_layer = r.int()
        values = [r.line() for _ in range(r.int())]
        r.expect("end_variable")
        variables.append(Variable(name, axiom_layer, values))

    mutexes = []
    for _ in range(r.int()):
        r.expect("begin_mutex_group")
        mutexes.append([tuple(r.ints()) for _ in range(r.int())])
        r.expect("end_mutex_group")

    r.expect("begin_state")
    init = [r.int() for _ in variables]
    r.expect("end_state")
    r.expect("begin_goal")
    goal = [tuple(r.ints()) for _ in range(r.int())]
    r.expect("end_goal")

    operators = []
    for _ in range(r.int()):
        r.expect("begin_operator")
        name = r.line()
        prevail = [tuple(r.ints()) for _ in range(r.int())]
        effects = []
        for _ in range(r.int()):
            numbers = r.ints()
            k = numbers[0]
            conditions = [(numbers[1 + 2 * i], numbers[2 + 2 * i]) for i in range(k)]
            var, pre, post = numbers[1 + 2 * k:]
            effects.append((conditions, var, pre, post))
        cost = r.int()
        r.expect("end_operator")
        operators.append(Operator(name, prevail, effects, cost))

    axioms = []
    for _ in range(r.int()):
        r.expect("begin_rule")
        rule = []
        while (line := r.line()).strip() != "end_rule":
            rule.append(line)
        axioms.append(rule)
    return Task(metric, variables, mutexes, init, goal, operators, axioms)


def write_task(task, path):
    out = ["begin_version", "3", "end_version", "begin_metric", str(task.metric), "end_metric",
           str(len(task.variables))]
    for v in task.variables:
        out += ["begin_variable", v.name, str(v.axiom_layer), str(len(v.values)), *v.values, "end_variable"]
    out.append(str(len(task.mutexes)))
    for group in task.mutexes:
        out += ["begin_mutex_group", str(len(group)), *(f"{var} {val}" for var, val in group), "end_mutex_group"]
    out += ["begin_state", *map(str, task.init), "end_state"]
    out += ["begin_goal", str(len(task.goal)), *(f"{var} {val}" for var, val in task.goal), "end_goal"]
    out.append(str(len(task.operators)))
    for op in task.operators:
        out += ["begin_operator", op.name, str(len(op.prevail)), *(f"{var} {val}" for var, val in op.prevail),
                str(len(op.effects))]
        for conditions, var, pre, post in op.effects:
            out.append(" ".join(map(str, [len(conditions), *(x for c in conditions for x in c), var, pre, post])))
        out += [str(op.cost), "end_operator"]
    out.append(str(len(task.axioms)))
    for rule in task.axioms:
        out += ["begin_rule", *rule, "end_rule"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def check_supported(task):
    """The ideal encoding covers SAS+ tasks without axioms and conditional effects."""
    if task.axioms or any(v.axiom_layer != -1 for v in task.variables):
        raise ValueError("tasks with axioms (derived predicates) are not supported")
    if any(conditions for op in task.operators for conditions, _, _, _ in op.effects):
        raise ValueError("tasks with conditional effects are not supported")


def applicable(op, state):
    return (all(state[var] == val for var, val in op.prevail)
            and all(pre == -1 or state[var] == pre for _, var, pre, _ in op.effects))


def apply(op, state):
    successor = list(state)
    for _, var, _, post in op.effects:
        successor[var] = post
    return tuple(successor)


def satisfies(state, partial):
    return all(state[var] == val for var, val in partial)
