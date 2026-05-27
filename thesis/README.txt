To produce a thesis with a single appendix WITHOUT index and glossary, use ubc_2010_spring_doe_jane.tex
To produce a thesis multiple appendices WITHOUT index and glossary, use ubc_2010_spring_doe_jane-appendices.tex
To produce a thesis with a single appendix WITH index and glossary, use ubc_2010_spring_doe_jane-index-glossary.tex 

NOTE: The template works with pdflatex NOT with latex. If you use .eps pictures and compile with latex .tex->.dvi->.ps->.pdf the template is currently not compiling i.e. you are on your own.

The followings need separate tools to be setup:
- bibliography: bibtex
- index: makeindex [optional]
- glossary: makeglossaries [optional]
These tools are installed with any LaTeX distribution.

If you use TeXnicCenter, makeindex and bibtex are already setup. makeglossaries is easily added. See instructions at
http://www.latex-community.org/index.php?option=com_content&view=article&id=263%3Aglossaries-nomenclature-lists-of-symbols-and-acronyms&catid=55%3Alatex-general&Itemid=112
There is an ERROR in the instructions:
- use makeglossaries.exe in C:\Program Files\MiKTeX 2.9\miktex\bin\x64
- the option is "%tm" NOT "%bm"
- on my installation makeindex crashes TeXNicCenter so I put it in the postprocessing (executable makeindex; Arguments: "%tm.idx" -t "%tm.ilg" -o "%tm.ind"); This comes after the Glossary postprocessor (Executable makeglossaries; Arguments: %tm)
Afterwards it all works!

As for writing properly, note the following on correctly formatting tables:
http://www.inf.ethz.ch/personal/markusp/teaching/guides/guide-tables.pdf

June 2, 2013 by Yves Lucet; updated July 25, 2023 to point to the multiple appendices template
