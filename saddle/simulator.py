"""
3D simulator: watch an Algorithm walk down the surface.

Window layout (figure coordinates go from 0 to 1, left to right and bottom to top):

  left 3/4  (x from 0.00 to 0.74)  the surface f(x, y), the path (black) and the
                                   current point (red). Drag with the mouse to rotate,
                                   right-drag (or scroll) to zoom.
  right 1/4 (x from 0.76 to 0.99)  the dashboard, top to bottom:
        - Algorithm picker   (radio buttons, one per method on this branch)
        - Play/Pause, Step   (Play runs at the slider speed, Step does one update)
        - Reset              (back to the start point)
        - steps/s slider     (speed while playing)
        - numbers at the current point: k, z, f, f*, gradient, Hessian
          ("n/a" if the method does not use it) and the work Nf / Ng / NH

Usage from code:
    sim = Simulator(build, ["gd", "pgd"], "gd")   # build(name) -> a fresh Algorithm
    sim.show()
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

PANEL_LEFT, PANEL_WIDTH = 0.76, 0.22     # the dashboard column


class Simulator:
    def __init__(self, build, names, first, speed=5):
        """build(name) must return a new Algorithm; names = the choices in the picker."""
        self.build = build
        self.algo = build(first)
        self.playing = False
        self.fig = plt.figure(figsize=(15, 8.5))

        # ---- left 3/4: the 3D world ----
        self.ax = self.fig.add_axes([0.0, 0.03, 0.74, 0.90], projection="3d")
        self.ax.computed_zorder = False      # draw the path ON TOP of the surface, always
        self._draw_surface()
        self.fig.text(0.01, 0.01, "drag: rotate   |   right-drag / scroll: zoom",
                      fontsize=9, color="gray")

        # ---- right 1/4: the dashboard (top to bottom, no two boxes overlap) ----
        top = 0.95
        h_radio = 0.045 * len(names) + 0.03            # grows with the number of methods
        self.fig.text(PANEL_LEFT, top, "Algorithm", fontsize=10, weight="bold", va="bottom")
        self.radio = RadioButtons(
            self.fig.add_axes([PANEL_LEFT, top - h_radio, PANEL_WIDTH, h_radio]),
            names, active=names.index(first))
        y = top - h_radio - 0.08                        # next row starts below the picker
        self.btn_play = Button(self.fig.add_axes([PANEL_LEFT, y, 0.105, 0.055]), "Play")
        self.btn_step = Button(self.fig.add_axes([PANEL_LEFT + 0.115, y, 0.105, 0.055]), "Step")
        y -= 0.065
        self.btn_reset = Button(self.fig.add_axes([PANEL_LEFT, y, PANEL_WIDTH, 0.05]), "Reset")
        y -= 0.06
        self.slider = Slider(self.fig.add_axes([PANEL_LEFT + 0.05, y, 0.13, 0.03]),
                             "steps/s", 1, 60, valinit=speed, valstep=1)
        self.info = self.fig.text(PANEL_LEFT, y - 0.03, "", va="top",
                                  family="monospace", fontsize=9)

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
        """Draw f once. The box shows the saddle, both minima and the start point."""
        p, z0 = self.algo.problem, self.algo.z0
        xmax = max(1.0, 1.2 * abs(z0[0]))
        ymax = max(1.5, 1.5 * np.sqrt(p.gamma), 1.2 * abs(z0[1]))
        X, Y = np.meshgrid(np.linspace(-xmax, xmax, 80), np.linspace(-ymax, ymax, 80))
        self.ax.plot_surface(X, Y, p.peek_f((X, Y)), cmap="viridis", alpha=0.45, linewidth=0)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("f")
        # the path (black line) and the current point (red dot); updated in redraw()
        (self.path_line,) = self.ax.plot([], [], [], "k.-", lw=1.8, ms=4)
        (self.dot,) = self.ax.plot([], [], [], "o", color="red", ms=10)

    # ------------------------------------------------------------------
    def choose(self, name):
        """Picker clicked: swap in a fresh run of that method from the same start."""
        if self.playing:
            self.toggle_play()
        self.algo = self.build(name)
        self.redraw()

    def advance(self):
        """One update of the algorithm, then refresh the picture."""
        self.algo.step()
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
        self.redraw()

    # ------------------------------------------------------------------
    def redraw(self):
        a, p = self.algo, self.algo.problem
        P = np.array(a.path)
        self.path_line.set_data_3d(P[:, 0], P[:, 1], p.peek_f((P[:, 0], P[:, 1])))
        self.dot.set_data_3d([a.z[0]], [a.z[1]], [p.peek_f(a.z)])
        self.ax.set_title(f"{a.name}   gamma = {p.gamma}   start = ({a.z0[0]:g}, {a.z0[1]:g})")
        self.info.set_text(self.dashboard_text())
        self.fig.canvas.draw_idle()

    def dashboard_text(self):
        """The numbers at the current point (computed with peek_*, so not counted as work)."""
        a, p = self.algo, self.algo.problem
        z, g = a.z, p.peek_grad(a.z)
        lines = [
            f"k      = {a.k} / {a.max_iters}" + ("  done" if a.done else ""),
            f"z      = ({z[0]: .4e},",
            f"          {z[1]: .4e})",
            "",
            f"f(z)   = {p.peek_f(z): .6e}",
            f"f*     = {p.fstar: .6e}",
            "",
            f"grad f = [{g[0]: .3e},",
            f"          {g[1]: .3e}]",
            f"|grad| = {np.linalg.norm(g):.3e}",
            "",
        ]
        if a.uses_hessian:
            H = p.peek_hess(z)
            lines += ["Hessian =",
                      f"  [{H[0, 0]: .3e} {H[0, 1]: .3e}]",
                      f"  [{H[1, 0]: .3e} {H[1, 1]: .3e}]"]
        else:
            lines += ["Hessian = n/a", "  (method does not use it)"]
        nf, ng, nh = a.counts
        lines += ["", "work so far:", f"  Nf={nf}  Ng={ng}  NH={nh}"]
        if a.failed_at is not None:
            lines += ["", f"line search failed at k={a.failed_at}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def show(self):
        plt.show()

    def save(self, path):
        """Save the window as an image (works without a screen)."""
        self.fig.savefig(path, dpi=90)
