
import subprocess
import threading
import time
from datetime import datetime, timedelta

cmd = "aquasimc.exe MinoSatohLab_01.aqu"

ph = []
def f():
	p_ = subprocess.Popen(cmd.split(), 
		shell = False, 
		stdin = subprocess.PIPE,
		stdout = subprocess.DEVNULL,
		universal_newlines = True)
	ph.append(p_)
	p_.wait()
	pass

t = threading.Thread(target = f)
t.start()

while True:
	if len(ph) > 0:
		break

elapsed_time = []
for k2 in range(3):
	ph[0].stdin.write("""\
3
1
2
B
""")
	t_bgn = time.time()
	for k1 in range(1000):
		ph[0].stdin.write("""\
3
1
3
B
B
4
8
biomass
./tmp/tmp_%03d.txt
B
""" % k1)

	t_end = time.time()
	elapsed_time.append((t_end - t_bgn))


ph[0].stdin.write("""\
1
E
N
""")
ph[0].stdin.flush()

t.join()



print(elapsed_time)
