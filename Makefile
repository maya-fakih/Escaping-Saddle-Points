# ---------------------------------------------------------------------------
#   make install                 create the venv in ../venv and install numpy + matplotlib
#   make run visual  ALG=gd      open the 3D simulator (you can switch method in the window)
#   make run headless ALG=all    no window, print a results table at the end
#   make test                    run the tests on this branch
#
# Every setting can be changed on the command line, e.g.
#   make run headless ALG=pgd GAMMA=0.01 SEED=4 K=500
#   make run headless GAMMA="1 0.1 0.01 0.001"     (one table per gamma)
#   make run visual ALG=gdnc GAMMA=0.5 X0=0.3 Y0=0 SPEED=20
#   make run headless FUNC=star X0=0 Y0=0       (the harder star saddle)
# ---------------------------------------------------------------------------

VENV ?= ../venv
PY   := $(VENV)/bin/python

# settings (defaults from the assignment; GAMMA must be > 0)
ALG    ?= all
FUNC   ?= assignment
GAMMA  ?= 1
X0     ?= 0.05
Y0     ?= 1e-8
K      ?= 3000
SEED   ?= 0
G_TRIG ?= 1e-3
R      ?= 1e-2
T      ?= 20
SPEED  ?= 5

ARGS := --alg $(ALG) --func $(FUNC) --gamma $(GAMMA) --x0 $(X0) --y0 $(Y0) --k $(K) --seed $(SEED) --g-trig $(G_TRIG) \
        --r $(R) --t $(T) --speed $(SPEED)

# "make run headless" -> headless; "make run" or "make run visual" -> visual
MODE := $(if $(filter headless,$(MAKECMDGOALS)),headless,visual)

.PHONY: install run visual headless test clean

install:
	mkdir -p $(VENV)
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

run:
	$(PY) main.py $(MODE) $(ARGS)

# these two are just words for "make run ..."; they do nothing on their own
visual headless:
	@:

test:
	$(PY) -m unittest discover tests -v

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ tests/__pycache__ frame_*.png
