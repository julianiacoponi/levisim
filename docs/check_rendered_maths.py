r"""
Fails if maths on the built pages was not recognised as maths (which Sphinx does not warn about)

Usage (from the repo root, after the site is built):
    sphinx-build -b html docs docs/_build/html
    python docs/check_rendered_maths.py docs/_build/html

MyST only starts a display block at a ``$$`` that follows a blank line. A ``$$`` on the line
straight after a paragraph or a list item is read as inline delimiters instead: the block is set
inline, one ``$`` is left over, and it pairs with the next ``$`` of the paragraph, so from there
on the prose is typeset as maths and the maths is printed as text. A ``|`` inside maths in a table
cell ends the cell, with the same result. The build finishes without a warning in both cases.

Both leave the same two traces in the HTML, which is what is looked for here, in the hand-written
pages and the gallery notebooks (not the API reference, whose docstrings may hold a literal ``$``):

- a ``$`` in the text of a page, outside maths and outside code;
- inline maths that begins or ends with a line break, i.e. a display block set inline.

To fix one: leave a blank line before the ``$$`` (and, inside a list item, indent the block to the
item's text); in a table cell write ``\lvert x \rvert`` rather than ``|x|``; put a dollar sign
that really is one in code font. This does not check that the LaTeX itself is valid.
"""
import sys
from html.parser import HTMLParser
from pathlib import Path

# generated from the docstrings, or not pages at all
SKIPPED = ('_autosummary', '_modules', '_static', '_sources', '_images')
VOID_TAGS = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr'}
CODE_TAGS = {'script', 'style', 'pre', 'code', 'svg', 'textarea'}


class MathsAudit(HTMLParser):
    """Collects, from the main content of one page, stray dollar signs and display maths set inline"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []  # one entry per open element: 'main', 'code', 'math' or ''
        self.inline_maths = None  # the text of the inline maths element being read
        self.problems = []

    def handle_starttag(self, tag, attrs):
        if tag in VOID_TAGS:
            return
        attrs = dict(attrs)
        classes = (attrs.get('class') or '').split()
        if attrs.get('role') == 'main':
            kind = 'main'
        elif tag in CODE_TAGS:
            kind = 'code'
        elif 'math' in classes:
            kind = 'math'
            if tag == 'span' and 'math' not in self.stack:
                self.inline_maths = ''
        else:
            kind = ''
        self.stack.append(kind)

    def handle_endtag(self, tag):
        if tag in VOID_TAGS or not self.stack:
            return
        kind = self.stack.pop()
        if kind == 'math' and 'math' not in self.stack and self.inline_maths is not None:
            maths = self.inline_maths.removeprefix('\\(').removesuffix('\\)')
            if maths.startswith('\n') or maths.endswith('\n'):
                self.problems.append(f'display maths set inline: {snippet(maths)}')
            self.inline_maths = None

    def handle_data(self, data):
        if 'main' not in self.stack or 'code' in self.stack:
            return
        if 'math' in self.stack:
            if self.inline_maths is not None:
                self.inline_maths += data
        elif '$' in data:
            self.problems.append(f'stray $ in the text: {snippet(data)}')


def snippet(text, length=70):
    """One line of at most `length` characters from `text`, for a report"""
    text = ' '.join(text.split())
    return text if len(text) <= length else text[:length] + '...'


def main(html_dir):
    html_dir = Path(html_dir)
    pages = sorted(
        page for page in html_dir.rglob('*.html')
        if not set(page.relative_to(html_dir).parts) & set(SKIPPED)
    )
    if not pages:
        print(f'No pages found under {html_dir}: build the site first.')
        return 1
    count = 0
    for page in pages:
        audit = MathsAudit()
        audit.feed(page.read_text(encoding='utf-8'))
        if audit.problems:
            count += len(audit.problems)
            print(f'{page.relative_to(html_dir)}:')
            print('\n'.join(f'  {problem}' for problem in audit.problems))
    if count:
        print(f'\n{count} places on the built pages where maths was not recognised as maths; '
              'see the top of docs/check_rendered_maths.py for the causes and the fixes.')
        return 1
    print(f'All maths recognised on the {len(pages)} pages checked.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'docs/_build/html'))
