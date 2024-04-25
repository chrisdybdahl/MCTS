import sys

sys.path.append("src")

from src.MyActorClient import MyActionClient
from src.MyHexActor import MyHexActor
from src.config import CLIENT_50_PATH

if __name__ == "__main__":
    actor = MyHexActor(CLIENT_50_PATH)
    client = MyActionClient(actor)
    client.run()
