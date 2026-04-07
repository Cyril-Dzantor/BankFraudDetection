import sys
import os

# Add the inner directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from app.api.v1.endpoints.network import get_network_topology

async def main():
    try:
        res = await get_network_topology()
        print("Success:", res is not None)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
