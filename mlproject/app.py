from routes import app

if __name__ == "__main__":
    import uvicorn
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    uvicorn.run(app, host="docker-compose down -v", port=9510)
