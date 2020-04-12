# md -> tex

pandoc -f markdown syllabus.md -s -o syllabus.tex --pdf-engine=lualatex -V documentclass=ltjarticle

# tex -> pdf

lualatex syllabus.tex