
import subprocess
import threading
import unittest
import time

rc = {
	"aquasim_folder_path": ".",
	"script_name": "MinoSatohLab_01.aqu",
}

def thread_aquasim(env):
	cmd = "aquasimc.exe MinoSatohLab_01.aqu"
	env.ph = subprocess.Popen(cmd.split(), 
		shell = False, 
		stdin = subprocess.PIPE,
		stdout = subprocess.DEVNULL,
		universal_newlines = True)
	env.ph.wait()
	print('Finished aquasim')

class EnvAquasim():
	def __init__(self):
		self.ph = None
		self.th = threading.Thread(target = thread_aquasim, args = (self,))
		self.th.start()
		while True:
			if self.ph is not None:
				# wait of loading aquasimc
				break

	def init(self):
# Initialize simulation environment
# (@Top) >> 3: Calc >> 1: Simulation >> 2: Initialize >> B: Back  >> B: Back 
# >> (@Top)
		self.ph.stdin.write("""\
3
1
2
B
B
""")
		self.ph.stdin.flush()
		pass

	def step(self):
# Run simulation and export the dataset of observation
#
# (@Top) >> 3: Calc >> 1: Simulation >> 3: Start Simulation >> B: Back  
# >> B: Back >> (@Top) >> 4: View >> 8: List to File >> ... >> B: Back
# >> (@Top)
		self.ph.stdin.write("""\
3
1
3
B
B
4
8
biomass
./tmp/tmp.txt
B
""" )
		self.ph.stdin.flush()
		pass

	def observe(self):
		pass

	def close_aquasim(self):
		self.ph.stdin.write("""\
1
E
N
""")
		self.ph.stdin.flush()
		#self.th.join()


class Testcase(unittest.TestCase):
	def test_001(self):
		env = EnvAquasim()
		env.init()
		for k1 in range(10):
			print(k1)
			env.step()
		env.close_aquasim()

if __name__ == "__main__":
	unittest.main()
	pass
