"""Read 3Dconnexion SpaceMouse motion data with hidapi.

This script opens the first supported SpaceMouse device found on Ubuntu and
prints the current 6-DoF state in a loop.

Usage:
  python scripts/env/state_machine/spacemouse.py

If you use IsaacLab's launcher:
	./isaaclab.sh -p /workspace/isaaclab/source/ASBDualArm/scripts/env/state_machine/spacemouse.py
"""

from __future__ import annotations

import argparse
import time


def _device_open_help() -> str:
	return (
		"Failed to open the SpaceMouse device. "
		"If you are running inside Docker, pass the HID device through with a compose device mount, "
		"for example: devices: [\"/dev/hidraw<#>:/dev/hidraw<#>\"]. "
		"On a local Ubuntu host, you may also need to grant access with: sudo chmod 666 /dev/hidraw<#>. "
		"To identify the right device, run: ls -l /dev/hidraw* and check /sys/class/hidraw/hidraw<#>/device/uevent."
	)


def _to_int16(low_byte: int, high_byte: int) -> int:
	value = low_byte | (high_byte << 8)
	if value >= 32768:
		value = -(65536 - value)
	return value


def _scale_axis(raw_value: int, scale: float = 350.0) -> float:
	return max(min(raw_value / scale, 1.0), -1.0)


def _find_hidraw_device():
	try:
		import hid
	except ImportError as exc:
		raise SystemExit("hidapi is not available in this Python environment.") from exc

	device_names = {
		"SpaceMouse Compact",
		"SpaceMouse Wireless",
		"SpaceMouse Pro",
		"SpaceNavigator",
		"SpaceExplorer",
		"SpaceMouse Pro Wireless",
	}
	device_ids = {
		(0x046D, 0xC626),
		(0x046D, 0xC627),
		(0x046D, 0xC62B),
		(0x256F, 0xC635),
		(0x256F, 0xC62E),
		(0x256F, 0xC632),
		(0x256F, 0xC638),
		(0x256F, 0xC641),
		(0x256F, 0xC652),
		(0x256F, 0xC633),
	}

	for device_info in hid.enumerate():
		product_name = device_info.get("product_string", "")
		vendor_id = device_info.get("vendor_id")
		product_id = device_info.get("product_id")
		if (vendor_id, product_id) in device_ids or product_name in device_names:
			return hid, device_info

	raise SystemExit("No supported 3Dconnexion device was found by hidapi.")


def _read_with_hidapi(poll_interval: float) -> None:
	hid, device_info = _find_hidraw_device()
	path = device_info.get("path")
	if isinstance(path, str):
		path_bytes = path.encode("utf-8")
	else:
		path_bytes = path

	device = hid.device()
	device.open_path(path_bytes)
	device.set_nonblocking(True)

	print(f"Using hidapi fallback on {device_info.get('product_string', 'unknown device')}")

	delta_pos = [0.0, 0.0, 0.0]
	delta_rot = [0.0, 0.0, 0.0]

	try:
		while True:
			data = device.read(64)
			if not data:
				time.sleep(poll_interval)
				continue

			if data[0] == 1:
				delta_pos[1] = _scale_axis(_to_int16(data[1], data[2]))
				delta_pos[0] = _scale_axis(_to_int16(data[3], data[4]))
				delta_pos[2] = -_scale_axis(_to_int16(data[5], data[6]))
			elif data[0] == 2:
				delta_rot[1] = _scale_axis(_to_int16(data[1], data[2]), scale=350.0)
				delta_rot[0] = _scale_axis(_to_int16(data[3], data[4]), scale=350.0)
				delta_rot[2] = -_scale_axis(_to_int16(data[5], data[6]), scale=350.0)
			elif data[0] == 3:
				buttons = []
				if len(data) > 1:
					button_byte = data[1]
					buttons = [
						"LEFT" if button_byte & 0x01 else None,
						"RIGHT" if button_byte & 0x02 else None,
					]
					buttons = [button for button in buttons if button is not None]
				print(f"buttons={buttons}")
				continue

			print(
				f"pos=({delta_pos[0]:+.4f}, {delta_pos[1]:+.4f}, {delta_pos[2]:+.4f}) "
				f"rot=({delta_rot[0]:+.4f}, {delta_rot[1]:+.4f}, {delta_rot[2]:+.4f})"
			)
			time.sleep(poll_interval)
	except KeyboardInterrupt:
		print("\nStopped.")
	finally:
		device.close()


def main() -> None:
	parser = argparse.ArgumentParser(description="Read 3Dconnexion SpaceMouse input with pyspacemouse.")
	parser.add_argument(
		"--poll-interval",
		type=float,
		default=0.01,
		help="Sleep time between read attempts in seconds.",
	)
	args = parser.parse_args()

	print("Opening SpaceMouse... press Ctrl+C to stop.")
	_read_with_hidapi(args.poll_interval)


if __name__ == "__main__":
	main()
