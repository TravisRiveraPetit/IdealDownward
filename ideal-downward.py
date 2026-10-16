#! /usr/bin/env python3
"""IdealDownward: planning as ideal membership. See README.md and src/ideal/planner.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "ideal"))

from planner import main

sys.exit(main())
