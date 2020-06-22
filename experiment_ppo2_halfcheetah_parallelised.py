import os
import subprocess
import datetime
import time
import multiprocessing as mp

seeds = [64,42,96,33,11]

time_start = datetime.datetime.now()

def training_run(seed):
	print(seed)
	command = "python train.py --algo ppo2 --env HalfCheetahBulletEnv-v0 -n 2000000 --tensorboard-log /tmp/stable-baselines/ --verbose 0 --seed " + str(seed)
	subprocess.call(command, shell=True)
	print("Finished training run with seed {}".format(seed))

pool = mp.Pool(5) #or mp.cpu_count()-1
pool.map_async(training_run, seeds).get()
pool.close()
 
time_end = datetime.datetime.now()
time_spent = time_end - time_start
time_spent_mins = time_spent.seconds / 60

print("Finished training")
print("Training took {} minutes".format(time_spent_mins))
