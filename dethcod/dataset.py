import subprocess
import sys
import zipfile
from pathlib import Path

import typer

app = typer.Typer()


def download_file(zip_link: str, output_file: Path) -> Path:
    subprocess.run(
        ["wget", "--continue", "--timestamping", "-O", str(output_file), zip_link],
        check=True,
    )

    return output_file


def extract_zip(zip_file_path: Path, data_folder: Path) -> list[Path]:
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(data_folder)

        return [data_folder / file_info.filename for file_info in zip_ref.infolist()]


@app.command()
def main(
    zip_link: str = "http://www.mattmahoney.net/dc/enwik8.zip",
    data_folder: Path = typer.Option(
        Path("data/dataset"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    file_name: str = "temp.zip",
):
    data_folder.mkdir(parents=True, exist_ok=True)

    zip_file_path = download_file(zip_link, data_folder / file_name)
    extracted_files = extract_zip(zip_file_path, data_folder)

    typer.echo("File downloaded and decompressed successfully.", file=sys.stderr)
    typer.echo("Extracted files:", file=sys.stderr)
    for extracted_file in extracted_files:
        typer.echo(extracted_file)

    return extracted_files


if __name__ == "__main__":
    app()
