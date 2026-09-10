# Review-mode color handling for Overleaf/latexmk.
# The current manuscript still contains broad \color{blue} annotations from the
# previous revision pass. Render those as black for the baseline draft.
# Future highlighted edits should use the custom color name `reviewblue` only
# around the individual sentence or phrase being changed.
$pdflatex = 'pdflatex %O "\\AtBeginDocument{\\colorlet{blue}{black}\\definecolor{reviewblue}{RGB}{0,76,153}}\\input{%S}"';
