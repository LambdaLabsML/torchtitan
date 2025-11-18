import os
import sys
import socket

host = socket.gethostname()
world_size = os.getenv("WORLD_SIZE", "-1")
rank = os.getenv("RANK", "-1")
local_rank = os.getenv("LOCAL_RANK", "-1")

print(f"{world_size=} {rank=} {local_rank=} {host=} {sys.executable} {sys.argv}")

