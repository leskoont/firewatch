from firewatch.smoothing import AlertGate, Cooldown, HitWindow


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_hit_window_n_of_m():
    w = HitWindow(window=5, min_hits=3)
    assert [w.update(h) for h in [1, 0, 1, 0, 1]] == [False, False, False, False, True]
    # oldest hit drops out of the window
    assert w.update(False) is False
    assert w.hits == 2


def test_cooldown():
    clock = FakeClock()
    cd = Cooldown(60, clock)
    assert cd.ready()
    cd.trigger()
    clock.t += 59
    assert not cd.ready()
    clock.t += 1
    assert cd.ready()


def test_gate_ignores_flicker():
    gate = AlertGate(["fire", "smoke"], window=6, min_hits=4, cooldown_s=60, clock=FakeClock())
    frames = [["fire"], [], ["fire"], [], ["fire"], [], ["fire"], []]
    assert all(gate.update(f) == [] for f in frames)


def test_gate_alerts_once_then_waits_for_cooldown():
    clock = FakeClock()
    gate = AlertGate(["fire", "smoke"], window=4, min_hits=3, cooldown_s=60, clock=clock)
    results = [gate.update(["fire"]) for _ in range(6)]
    assert results == [[], [], ["fire"], [], [], []]

    clock.t += 61
    assert gate.update(["fire", "smoke"]) == ["fire"]  # smoke has 1 hit so far


def test_cooldown_is_per_class():
    clock = FakeClock()
    gate = AlertGate(["fire", "smoke"], window=2, min_hits=1, cooldown_s=60, clock=clock)
    assert gate.update(["fire"]) == ["fire"]
    clock.t += 10
    # smoke that starts during the fire cooldown is still reported
    assert gate.update(["fire", "smoke"]) == ["smoke"]
    clock.t += 51
    assert gate.update(["fire", "smoke"]) == ["fire"]


def test_gate_reports_all_confirmed_classes():
    gate = AlertGate(["fire", "smoke"], window=3, min_hits=2, cooldown_s=0, clock=FakeClock())
    gate.update(["fire", "smoke"])
    assert sorted(gate.update(["smoke", "fire"])) == ["fire", "smoke"]


def test_gate_ignores_unknown_labels():
    gate = AlertGate(["fire"], window=2, min_hits=1, cooldown_s=0, clock=FakeClock())
    assert gate.update(["smoke", "person"]) == []
