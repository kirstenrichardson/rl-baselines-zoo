import os
import subprocess
import datetime
import time

seeds = [64,42,96,33,11]

time_start = datetime.datetime.now()

for seed in seeds:
	command = "python train.py --algo ppo2 --env HalfCheetahBulletEnv-v0 -n 2000000 --tensorboard-log /tmp/stable-baselines/ --verbose 0 --seed " + str(seed)
	subprocess.call(command, shell=True)
	print("Finished training run with seed {}".format(seed))

time_end = datetime.datetime.now()
time_spent = time_end - time_start
time_spent_mins = time_spent.seconds / 60

print("Finished training")
print("Training took {} minutes".format(time_spent_mins))
