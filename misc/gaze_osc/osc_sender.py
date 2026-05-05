from pythonosc.udp_client import SimpleUDPClient


class OSCSender:
    def __init__(self, cfg):
        self._client = SimpleUDPClient(cfg.osc_ip, cfg.osc_port)

    def send_gaze(self, u: float, v: float):
        """/gaze [u, v, 0.0]  — Float32[3], u/v in [0.0, 1.0]"""
        self._client.send_message("/gaze", [float(u), float(v), 0.0])

    def send_blink(self):
        """/gaze/blink 1.0  — Float32. Sent on blink or landmark loss."""
        self._client.send_message("/gaze/blink", 1.0)
