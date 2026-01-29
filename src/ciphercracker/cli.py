from __future__ import annotations

from typing import List, Optional

import typer

from ciphercracker.classical import register_all
from ciphercracker.core.features import analyze_text
from ciphercracker.core.registry import crack_unknown, decrypt_known, list_plugins

app = typer.Typer(help="CipherCracker CLI: classical cipher tools + cryptanalysis helpers.")


@app.callback()
def _init():
    # Register plugins exactly once per CLI run
    register_all()


@app.command()
def plugins():
    """List all registered cipher plugins."""
    for name in list_plugins():
        typer.echo(name)


@app.command()
def analyze(
    text: str,
    iocmax: int = typer.Option(0, help="If >0, show IoC scan up to this key length."),
):
    info = analyze_text(text)
    for k, v in info.items():
        typer.echo(f"{k}: {v}")

    if iocmax > 0:
        from ciphercracker.core.features import ioc_scan

        typer.echo("\nTop IoC candidates:")
        for klen, val in ioc_scan(text, max_len=iocmax)[:10]:
            typer.echo(f"  k={klen:2d}  avg_ioc={val:.5f}")


@app.command()
def decrypt(
    cipher: str = typer.Option(..., "--cipher", "-c", help="Cipher plugin name (e.g., caesar, vigenere)."),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Key material for the cipher."),
    text: str = typer.Argument(..., help="Ciphertext to decrypt."),
):
    """Decrypt when you already know the cipher type and have the key."""
    try:
        pt = decrypt_known(cipher, text, key)
    except Exception as e:
        raise typer.BadParameter(str(e))
    typer.echo(pt)


def _effective_score(r) -> float:
    # Prefer "best_full" if present (periodic_sub uses it)
    if r.meta and isinstance(r.meta, dict) and "best_full" in r.meta:
        try:
            return float(r.meta["best_full"])
        except Exception:
            pass
    return float(r.score)


@app.command()
def crack(
    text: str = typer.Argument(...),
    top: int = typer.Option(5, "--top", "-t"),
    cipher: Optional[List[str]] = typer.Option(
        None,
        "--cipher",
        "-c",
        help="Limit to specific plugin(s). Can repeat: -c periodic_substitution -c vigenere",
    ),
    iocmax: int = typer.Option(0, "--iocmax", help="If >0, show IoC scan up to this key length."),
):
    # Normalize include set safely even if Typer gives a string for some reason
    if cipher is None:
        include = None
    elif isinstance(cipher, str):
        include = {cipher.lower().strip()}
    else:
        include = {c.lower().strip() for c in cipher}

    # Validate filter names so it can't silently run the wrong thing
    if include is not None:
        available = set(list_plugins())
        unknown = sorted(include - available)
        if unknown:
            raise typer.BadParameter(
                f"Unknown cipher(s): {', '.join(unknown)}. Available: {', '.join(sorted(available))}"
            )

    if iocmax > 0:
        from ciphercracker.core.features import ioc_scan

        typer.echo("\nTop IoC candidates:")
        for klen, val in ioc_scan(text, max_len=iocmax)[:10]:
            typer.echo(f"  k={klen:2d}  avg_ioc={val:.5f}")
        typer.echo("")

    # Request extra candidates so registry-side de-dupe doesn't shrink results too much
    fetch_n = max(top * 5, top)
    results = crack_unknown(text, top_n=fetch_n, include=include)

    if not results:
        typer.echo("No candidates produced. Input may be too short or not supported yet.")
        raise typer.Exit(code=0)

    # Registry already de-dupes and sorts; just take top N
    results = results[:top]

    for i, r in enumerate(results, start=1):
        shown_score = _effective_score(r)

        typer.echo(
            f"#{i}  cipher={r.cipher_name}  score={shown_score:.2f}  conf={r.confidence:.2f}  key={r.key}"
        )
        if r.notes:
            typer.echo(f"    notes: {r.notes}")
        if r.meta:
            typer.echo(f"    meta: {r.meta}")
        if r.plaintext:
            typer.echo(r.plaintext)
        typer.echo("-" * 60)


def main():
    app()


if __name__ == "__main__":
    main()
