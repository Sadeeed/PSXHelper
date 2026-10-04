def main() -> None:
    import uvicorn

    uvicorn.run("psxhelper.main:app", host="127.0.0.1", port=8000, app_dir=".")
