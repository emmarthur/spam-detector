# Comment spam detector (catalog #595)

Joshua Eckroth — *Python Artificial Intelligence Projects for Beginners*, Chapter 4.

| Artifact | Path |
|----------|------|
| Book steps | [`BOOK_DIRECTIONS.md`](./BOOK_DIRECTIONS.md) |
| Upstream script | [`reference/SpamDetectorNN.py`](./reference/SpamDetectorNN.py) |
| Shared logic | [`spam_core.py`](./spam_core.py) |
| Web (FastAPI) | [`web/`](./web/) — port **3019** |

## Run the web demo

```powershell
Set-Location "c:\Users\emmak\Desktop\Content\CodingProjects\PortProjects\demos\spam-detector\web"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 3019
```

Open [http://127.0.0.1:3019/](http://127.0.0.1:3019/). The first launch trains the model in the background (TensorFlow); weights are cached under `web/data/cache/`.

## Portfolio

See [`PORTFOLIO_INTEGRATION.md`](./PORTFOLIO_INTEGRATION.md) for `my-website` env vars and project card fields.
