"""SmartSave Nigeria: a household financial planning and education
affordability simulator.

Everything in this package is explainable arithmetic and seeded random
simulation. There is no machine learning, AI, or automated "advice" here:
every number can be traced back to a formula documented in the module that
computes it.

Module map
----------
validation  -- checks raw user inputs and explains problems in plain language
budget      -- the Household data model and monthly budget arithmetic
strategies  -- the three budgeting decision rules + affordability classification
scenarios   -- configurable economic scenarios and random shock generation
simulator   -- month-by-month cash-flow engine (deterministic or Monte Carlo)
metrics     -- summary statistics over simulation outputs
charts      -- Matplotlib figures used by the app and the experiment scripts
"""

__version__ = "1.0.0"
