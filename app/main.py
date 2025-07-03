from fastapi import FastAPI, Request, status
from .database import engine,Base
from .routers import auth, prompts, admin, users
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn

app = FastAPI()

Base.metadata.create_all(bind=engine)




@app.get("/healthy")
def health_check():
    return {'status': 'Healthy'}


app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(users.router)
app.include_router(prompts.router)

if __name__ == "__main__":
    uvicorn.run(app="app.main:app",host="127.0.0.1",port=8000,reload=True)