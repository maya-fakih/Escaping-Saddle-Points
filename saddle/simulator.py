"""
3D simulator: watch an Algorithm walk down the surface.

Window layout (figure coordinates go from 0 to 1, left to right and bottom to top):

  left 3/4  (x from 0.00 to 0.74)  the surface f(x, y), the path (black) and the
                                   current point (red). Drag with the mouse to rotate,
                                   right-drag (or scroll) to zoom.
  right 1/4 (x from 0.76 to 0.99)  the dashboard, top to bottom:
        - Algorithm picker   (radio buttons, one per method on this branch)
        - gamma, x0, y0 boxes + Apply (restart with these values; gamma must be > 0)
          and "Start at (0,0)" (restart exactly on the saddle)
        - Play/Pause, Step   (Play runs at the slider speed, Step does one update)
        - Reset              (back to the start point)
        - steps/s slider     (speed while playing)
        - numbers at the current point: k, z, f, f*, gradient, Hessian
          ("n/a" if the method does not use it) and the work Nf / Ng / NH

Play pauses by itself the first time the point reaches a real minimum (the
success test in saddle/metrics.py). The method itself does not stop there (the
assignment runs all K updates); press Play or Step to keep going.

Usage from code:
    # build(name, gamma, z0) -> a fresh Algorithm
    sim = Simulator(build, ["gd", "pgd"], "gd", gamma=1.0, z0=(0.05, 1e-8))
    sim.show()
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

from saddle.metrics import at_minimum

PANEL_LEFT, PANEL_WIDTH = 0.76, 0.22     # the dashboard column


class Simulator:
    def __init__(self, build, names, first, gamma, z0, speed=5):
        """build(name, gamma, z0) must return a new Algorithm;
        names = the choices in the picker; gamma, z0 = the values to start with."""
        self.build = build
        self.name, self.gamma, self.z0 = first, gamma, tuple(z0)
        self.algo = build(first, gamma, self.z0)
        self.playing = False
        self.reached = None            # k where the point first reached a minimum
        self.message = ""              # e.g. "gamma must be > 0"
        self.fig = plt.figure(figsize=(15, 8.5))

        # ---- left 3/4: the 3D world ----
        self.ax = self.fig.add_axes([0.0, 0.03, 0.74, 0.90], projection="3d")
        self._draw_surface()
        self.fig.text(0.01, 0.01, "drag: rotate   |   right-drag / scroll: zoom",
                      fontsize=9, color="gray")

        # ---- right 1/4: the dashboard, top to bottom; each row starts below the last ----
        L, W = PANEL_LEFT, PANEL_WIDTH
        y = 0.96
        h_radio = 0.034 * len(names) + 0.02            # grows with the number of methods
        self.fig.text(L, y, "Algorithm", fontsize=10, weight="bold", va="bottom")
        y -= h_radio + 0.008                            # small gap under the title
        self.radio = RadioButtons(self.fig.add_axes([L, y, W, h_radio]),
                                  names, active=names.index(first))

        # gamma / x0 / y0: three boxes in one row, names written above them
        y -= 0.065
        self.boxes = {}
        for i, (key, value) in enumerate([("gamma", gamma), ("x0", z0[0]), ("y0", z0[1])]):
            x = L + i * 0.075
            self.fig.text(x, y + 0.037, key, fontsize=9)
            self.boxes[key] = TextBox(self.fig.add_axes([x, y, 0.07, 0.033]), "",
                                      initial=f"{value:g}")
        y -= 0.05
        self.btn_apply = Button(self.fig.add_axes([L, y, 0.105, 0.04]), "Apply")
        self.btn_origin = Button(self.fig.add_axes([L + 0.115, y, 0.105, 0.04]), "Start at (0,0)")

        y -= 0.06
        self.btn_play = Button(self.fig.add_axes([L, y, 0.105, 0.045]), "Play")
        self.btn_step = Button(self.fig.add_axes([L + 0.115, y, 0.105, 0.045]), "Step")
        y -= 0.05
        self.btn_reset = Button(self.fig.add_axes([L, y, W, 0.04]), "Reset")
        y -= 0.045
        self.slider = Slider(self.fig.add_axes([L + 0.05, y, 0.13, 0.025]),
                             "steps/s", 1, 60, valinit=speed, valstep=1)
        self.info = self.fig.text(L, y - 0.02, "", va="top", family="monospace", fontsize=8.5)

        self.btn_apply.on_clicked(self.apply)
        self.btn_origin.on_clicked(self.start_at_origin)
        self.radio.on_clicked(self.choose)
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_step.on_clicked(lambda _: self.advance())
        self.btn_reset.on_clicked(self.reset)
        self.slider.on_changed(self.set_speed)

        # the timer calls advance() "speed" times per second while playing
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / speed))
        self.timer.add_callback(self.advance)

        self.redraw()

    # ------------------------------------------------------------------
    def _draw_surface(self):
        """(Re)draw the surface for the current function. Keeps the camera angle."""
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.cla()
        self.ax.computed_zorder = False      # draw the path ON TOP of the surface, always
        self.ax.view_init(elev=elev, azim=azim)
        self._surface()

    def _surface(self):
        """Draw f once. The box shows the saddle, both minima and the start point.
        It is scaled with the problem: for the assignment the minima sit at
        y = +-sqrt(gamma), so for a small gamma (S3) a fixed box would make the
        whole saddle look flat."""
        p, z0 = self.algo.problem, self.algo.z0
        r = p.scale                                   # distance from saddle to a minimum
        xmax = max(r, 1.2 * abs(z0[0]))
        ymax = max(1.6 * r, 1.2 * abs(z0[1]))
        if p.label == "star saddle":                  # minima on the diagonals: square box
            xmax = ymax = max(xmax, ymax)
        X, Y = np.meshgrid(np.linspace(-xmax, xmax, 80), np.linspace(-ymax, ymax, 80))
        Z = p.peek_f((X, Y))
        if p.zcap is not None:
            Z = np.where(Z > p.zcap, np.nan, Z)       # cut off the very high parts
        self.ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.45, linewidth=0)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("f")
        # the path (black line) and the current point (red dot); updated in redraw()
        (self.path_line,) = self.ax.plot([], [], [], "k.-", lw=1.8, ms=4)
        (self.dot,) = self.ax.plot([], [], [], "o", color="red", ms=10)

    # ------------------------------------------------------------------
    def choose(self, name):
        """Picker clicked: swap in a fresh run of that method from the same start."""
        if self.playing:
            self.toggle_play()
        self.name = name
        self.restart()

    def apply(self, _event=None):
        """Apply clicked: read gamma, x0, y0 from the boxes and restart."""
        try:
            gamma = float(self.boxes["gamma"].text)
            z0 = (float(self.boxes["x0"].text), float(self.boxes["y0"].text))
        except ValueError:
            self.message = "gamma, x0, y0 must be numbers"
            return self.redraw()
        if gamma <= 0:
            self.message = "gamma must be > 0"
            return self.redraw()
        surface_changed = gamma != self.gamma or z0 != self.z0
        self.gamma, self.z0 = gamma, z0
        self.restart(surface_changed)

    def start_at_origin(self, _event=None):
        """Restart exactly on the saddle (0, 0) with the current gamma."""
        self.boxes["x0"].set_val("0")
        self.boxes["y0"].set_val("0")
        self.apply()

    def restart(self, redraw_surface=False):
        """A fresh run of the chosen method with the current gamma and start."""
        if self.playing:
            self.toggle_play()
        self.algo = self.build(self.name, self.gamma, self.z0)
        self.reached, self.message = None, ""
        if redraw_surface:
            self._draw_surface()
        self.redraw()

    def advance(self):
        """One update of the algorithm, then refresh the picture."""
        self.algo.step()
        if self.reached is None and at_minimum(self.algo.problem, self.algo.z):
            self.reached = self.algo.k
            if self.playing:
                self.toggle_play()             # pause once, so you see where it arrived
        if self.algo.done and self.playing:
            self.toggle_play()
        self.redraw()

    def toggle_play(self, _event=None):
        self.playing = not self.playing
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def set_speed(self, value):
        self.timer.interval = int(1000 / value)

    def reset(self, _event=None):
        if self.playing:
            self.toggle_play()
        self.algo.reset()
        self.reached, self.message = None, ""
        self.redraw()

    # ------------------------------------------------------------------
    def redraw(self):
        a, p = self.algo, self.algo.problem
        P = np.array(a.path)
        self.path_line.set_data_3d(P[:, 0], P[:, 1], p.peek_f((P[:, 0], P[:, 1])))
        self.dot.set_data_3d([a.z[0]], [a.z[1]], [p.peek_f(a.z)])
        self.ax.set_title(f"{a.name}   {p.label}   start = ({a.z0[0]:g}, {a.z0[1]:g})")
        self.info.set_text(self.dashboard_text())
        self.fig.canvas.draw_idle()

    def dashboard_text(self):
        """The numbers at the current point (computed with peek_*, so not counted as work)."""
        a, p = self.algo, self.algo.problem
        z, g = a.z, p.peek_grad(a.z)
        lines = [
            f"k      = {a.k} / {a.max_iters}" + ("  done" if a.done else ""),
            f"z      = ({z[0]: .3e}, {z[1]: .3e})",
            f"f(z)   = {p.peek_f(z): .6e}",
            f"f*     = {p.fstar: .6e}",
            f"grad   = [{g[0]: .2e}, {g[1]: .2e}]",
            f"|grad| = {np.linalg.norm(g):.3e}",
        ]
        if a.uses_hessian:
            H = p.peek_hess(z)
            lines += [f"Hessian= [{H[0, 0]: .2e} {H[0, 1]: .2e}]",
                      f"         [{H[1, 0]: .2e} {H[1, 1]: .2e}]"]
        else:
            lines += ["Hessian= n/a (method does not use it)"]
        nf, ng, nh = a.counts
        lines += [f"work   = Nf {nf}  Ng {ng}  NH {nh}"]
        if self.reached is not None:
            lines += [f"reached a minimum at k={self.reached}"]
        if a.failed_at is not None:
            lines += [f"line search failed at k={a.failed_at}"]
        if self.message:
            lines += [f"!! {self.message}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def show(self):
        plt.show()

    def save(self, path):
        """Save the window as an image (works without a screen)."""
        self.fig.savefig(path, dpi=90)
