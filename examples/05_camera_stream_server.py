"""Serve a realtime MuJoCo camera stream in a browser.

This is useful on a server where GUI windows are unavailable.

Run:
    cd /home/mkls/xiao_run/lerobot_smolvla_mujoco_demo
    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    MUJOCO_GL=egl python examples/05_camera_stream_server.py --camera front --host 0.0.0.0 --port 8000

Open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2

from lerobot_smolvla_mujoco_demo import CubeGraspEnv, scripted_grasp_action


CAMERAS = ("front", "top_oblique")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve realtime MuJoCo camera frames over HTTP.")
    parser.add_argument("--camera", choices=CAMERAS, default="front")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manual", action="store_true", help="Stream without scripted grasp motion.")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    server = CameraStreamServer((args.host, args.port), CameraStreamHandler, args)
    print(f"camera: {args.camera}")
    print(f"stream: http://{args.host}:{args.port}")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class CameraStreamServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], args: Any):
        super().__init__(server_address, handler_class)
        self.args = args


class CameraStreamHandler(BaseHTTPRequestHandler):
    server: CameraStreamServer

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_index()
            return
        if self.path == "/stream.mjpg":
            self._send_stream()
            return
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_index(self) -> None:
        args = self.server.args
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MuJoCo Camera - {args.camera}</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
    header {{ padding: 10px 14px; background: #202020; }}
    img {{ display: block; width: min(100vw, {args.width}px); height: auto; }}
  </style>
</head>
<body>
  <header>MuJoCo camera: {args.camera}</header>
  <img src="/stream.mjpg" alt="MuJoCo camera stream">
</body>
</html>
"""
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self) -> None:
        args = self.server.args
        env = CubeGraspEnv(width=args.width, height=args.height, camera=args.camera, seed=args.seed, max_steps=args.steps)
        env.reset(seed=args.seed)
        step = 0

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        frame_period = 1.0 / max(args.fps, 1.0)
        try:
            while True:
                start = time.perf_counter()
                action = env.action_space.sample() * 0.0 if args.manual else scripted_grasp_action(env, step)
                _, _, terminated, truncated, _ = env.step(action)
                rgb = env.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ok:
                    continue

                payload = jpg.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")

                step += 1
                if terminated or truncated or step >= args.steps:
                    step = 0
                    env.reset(seed=args.seed)

                sleep_time = frame_period - (time.perf_counter() - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            env.close()


if __name__ == "__main__":
    main()
