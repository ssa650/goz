import os
import uvicorn
from .app import app

if __name__ == "__main__":
    print("GOZ ready at http://127.0.0.1:" + os.getenv("PORT", "3210"))
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "3210")))
