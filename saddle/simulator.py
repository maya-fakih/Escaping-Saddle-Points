"""
3D simulator: watch any Algorithm walk down the surface.

Window layout:
  left 3/4   the surface f(x, y), the current point (red) and its path (black)
  right 1/4  a small dashboard: Play/Pause, Step, Reset, a speed slider
             (steps per second), and the numbers at the current point:
             f, gradient, Hessian ("n/a" if the method does not use it), work.

Usage from code:
    sim = Simulator(GD(Benchmark(1.0), (0.05, 1e-8)))
    sim.show()
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


class Simulator:
    def __init__(self, algorithm, speed=5, title=None):
        self.algo = algorithm
        self.playing = False
        self.fig = plt.figure(figsize=(14, 7.5))
        self.fig.suptitle(title or f"{algorithm.name}   (gamma = {algorithm.problem.gamma})")

        # ---- left 3/4: the 3D world ----
        self.ax = self.fig.add_axes([0.0, 0.02, 0.74, 0.92], projection="3d")
        self._draw_surface()

        # ---- right 1/4: the dashboard ----
        self.btn_play = Button(self.fig.add_axes([0.77, 0.86, 0.10, 0.06]), "Play")
        self.btn_step = Button(self.fig.add_axes([0.88, 0.86, 0.10, 0.06]), "Step")
        self.btn_reset = Button(self.fig.add_axes([0.77, 0.78, 0.21, 0.05]), "Reset")
        self.slider = Slider(self.fig.add_axes([0.80, 0.71, 0.15, 0.03]),
                             "steps/s", 1, 60, valinit=speed, valstep=1)
        self.info = self.fig.text(0.77, 0.66, "", va="top", family="monospace", fontsize=10)

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
        """Draw f once. The box is wide enough to show the saddle and both minima."""
        p = self.algo.problem
        ymax = max(1.5, 1.5 * np.sqrt(p.gamma))
        xmax = max(1.0, 1.2 * abs(self.algo.z0[0]))
        X, Y = np.meshgrid(np.linspace(-xmax, xmax, 80), np.linspace(-ymax, ymax, 80))
        Z = p.peek_f((X, Y))
        self.ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.55, linewidth=0)
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("f")
        # the path (black line) and the current point (red dot); updated in redraw()
        (self.path_line,) = self.ax.plot([], [], [], "k.-", lw=1.5, ms=3, zorder=10)
        (self.dot,) = self.ax.plot([], [], [], "o", color="red", ms=9, zorder=11)

    # ------------------------------------------------------------------
    def advance(self):
        """One step of the algorithm, then refresh the picture."""
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
        F = p.peek_f((P[:, 0], P[:, 1]))
        self.path_line.set_data_3d(P[:, 0], P[:, 1], F)
        self.dot.set_data_3d([a.z[0]], [a.z[1]], [p.peek_f(a.z)])
        self.info.set_text(self.dashboard_text())
        self.fig.canvas.draw_idle()

    def dashboard_text(self):
        """The numbers at the current point (not counted as work)."""
        a, p = self.algo, self.algo.problem
        z, g = a.z, p.peek_grad(a.z)
        lines = [
            f"k      = {a.k}" + ("  (done)" if a.done else ""),
            f"z      = ({z[0]: .4e},",
            f"          {z[1]: .4e})",
            "",
            f"f(z)   = {p.peek_f(z): .6e}",
            f"f*     = {p.fstar: .6e}",
            "",
            "grad f = " + f"[{g[0]: .3e},",
            "          " + f" {g[1]: .3e}]",
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
        """Save the current window as an image (useful without a screen)."""
        self.fig.savefig(path, dpi=90)
