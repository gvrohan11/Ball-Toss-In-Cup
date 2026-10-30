# Run the throw on the robot computer (not the laptop)

**Why:** every command from the laptop crosses venue Wi-Fi + WebRTC. On Oct 29 that was
15–400 ms of random delay, which moves the release point around. On the robot's own Linux box
it's a few ms. Same code, same `.env`.

**Ask Viam staff first (9:30):** "Can we SSH into sfhack11's computer? What's the IP / user /
password?" The event lists SSH as a required skill, so it's expected — but it's their machine.

## Option A: SSH (preferred, no config change)
```bash
ssh <user>@<robot-ip>                 # from staff
python3 --version                     # need 3.10+
git clone https://github.com/gvrohan11/Ball-Toss-In-Cup.git && cd Ball-Toss-In-Cup
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-robot.txt # light: no YOLO / torch (the box ran out of disk for YOLO on Oct 29)
```
Copy your `.env` from the laptop (same keys, same VIAM_ADDRESS):
```bash
scp .env <user>@<robot-ip>:~/Ball-Toss-In-Cup/.env      # run this ON THE LAPTOP
```

## Option B: Viam CLI shell (only if there's no SSH)
`viam machines part shell --part=041f1920-0d72-4162-9bc3-0aecd7767204` gives a terminal on the
robot, and `viam machines part cp` copies files. It needs the `ViamShellDanger` fragment added
to the machine config — that's a config change, so **ask staff before adding it**.

## Check it worked
```bash
python check_connection.py            # read-only
python scripts/latency_check.py       # compare with the same command on the laptop
```
Expect the robot's "jitter" number to be a few ms vs tens-hundreds on the laptop.

## Then run as usual (on the robot)
```bash
python pick_and_throw.py --aim-cup --cradle                               # dry run
python pick_and_throw.py --aim-cup --cradle --go --i-checked-with-staff   # live, E-stop!
```
Cup finding without YOLO uses the rim detector (it's what found the cup for every drop on Oct 29).
The Atlas dashboard and `throw_fit.py` still run fine from the laptop.

Keep a laptop terminal SSH'd in for the demo; the robot computer has no screen in the demo.
