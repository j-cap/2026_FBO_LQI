# Internal review-mode handling for Overleaf/latexmk.
# Keep legacy blue markup black and render the current baseline overview figure
# in place of the temporary Fig. 1 placeholder. Once concept.pdf is exported
# from figures/concept.drawio, this temporary override can be switched to PDF
# and then removed when main.tex is cleaned for submission.
$pdflatex = 'pdflatex %O "\\AtBeginDocument{\\colorlet{blue}{black}\\definecolor{reviewblue}{RGB}{0,76,153}\\renewcommand{\\placeholderfigurewide}[1]{\\includegraphics[width=0.96\\textwidth]{figures/concept.png}}}\\input{%S}"';
