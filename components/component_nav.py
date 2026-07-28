#!/usr/bin/env python
"""The component-card nav shared by every dashboard page.

The same three cards sit at the top of each dashboard so they stay put as you
move between them, with the card for the page you are on highlighted. Defining
the set here means the titles and blurbs live in one place instead of being
restated in all three ``generate_html.py`` scripts.

``docs/index.html`` is hand-written and carries its own copy of this markup (it
is the one page with no card highlighted); keep the two in step when the wording
of a card changes.
"""

# (slug, card title, card blurb). The slug is both the docs/ subfolder and the
# value passed as ``active``.
COMPONENTS = (
    (
        "rfi",
        "RFI classification",
        "RFI classification using spectral kurtosis of the DRAO RFI monitor data.",
    ),
    (
        "sensitivity",
        "CHORD Sensitivity",
        "Brightness temperature sensitivity maps achievable by CHORD.",
    ),
    (
        "extension_chord",
        "Extension CHORD",
        "Extension CHORD configurations and their beam patterns.",
    ),
)


def nav_html(active=None, prefix="../"):
    """Return the card-grid markup for a dashboard page.

    ``active`` is the slug of the page being generated, which gets the
    ``is-active`` highlight; pass None to render the grid with nothing
    highlighted. ``prefix`` is prepended to each href, so the default "../"
    suits a page one level below docs/.
    """
    cards = []
    for slug, title, blurb in COMPONENTS:
        is_active = slug == active
        cls = "card is-active" if is_active else "card"
        # aria-current is the screen-reader equivalent of the .is-active styling.
        current = ' aria-current="page"' if is_active else ""
        cards.append(
            f'    <a class="{cls}" href="{prefix}{slug}/index.html"{current}>\n'
            f"      <h2>{title}</h2>\n"
            f"      <p>{blurb}</p>\n"
            f"    </a>"
        )
    return (
        '  <nav class="cards" aria-label="Analysis components">\n'
        + "\n\n".join(cards)
        + "\n  </nav>"
    )
